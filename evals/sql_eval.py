"""CurVE ``sql_query`` execution-accuracy harness (M4b) — LIVE model + LIVE Athena.

This is **not** a unit-test suite. It calls the real Bedrock generate and really runs
Athena, and it measures ANSWER QUALITY of the shipped pipeline. Per E13 it exercises
shipped tool code only — it never reimplements generation, guarding, or execution:

  * gold-comparable questions -> ``curve.sql_tool.run_sql_pipeline`` with the SAME
    generate callable the production tool builds (``_build_generate_wrapper`` +
    ``generate_sql``), wrapped ONLY to record the self-correct feedback string.
  * gold SQL                  -> ``curve.sql_query.guard`` -> ``execute``, the identical
    path the generated SQL takes, so both sides receive identical org/well injection,
    identical date-window injection and the identical row cap. Comparing a gold query
    run outside the guard would be meaningless.
  * physics-boundary negatives -> ``curve.engine.run_curve_turn`` with the ``/sql``
    prefix, because what is under review there is what the OPERATOR is told: the
    narration matters as much as the SQL, and only the full turn produces narration.

Scoring is by VALUE, never by SQL text (see :func:`compare_results`): order-insensitive
row multiset, relative float tolerance, and column-name-agnostic (aliases legitimately
differ; a column permutation search handles differing column order).

Run:
    aws sso login --profile roam-ai
    python -m evals.sql_eval                    # full pass
    python -m evals.sql_eval --only Q01,Q11     # subset
    python -m evals.sql_eval --gold-only        # validate gold SQL, no model calls
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from curve import config
from curve.engine import run_curve_turn
from curve.session import new_session_record
from curve.sql_query import execute, guard
from curve.sql_tool import (
    SESSION_SQL_RESULT_KEY,
    _build_generate_wrapper,
    generate_sql,
    run_sql_pipeline,
)
from evals.sql_eval_questions import (
    EVAL_COVERAGE_MAX_DAY,
    EVAL_COVERAGE_MIN_DAY,
    EVAL_ORG_ID,
    EVAL_WELL_ID,
    QUESTIONS,
    EvalQuestion,
)

# --- buckets ------------------------------------------------------------------

BUCKET_MATCH = "match"
BUCKET_MISMATCH = "mismatch"  # both executed, values differ — the deceptive bucket
BUCKET_GUARD_REJECT = "guard_reject"
BUCKET_PIPELINE_FAIL = "pipeline_fail"
BUCKET_EXEC_ERROR = "execution_error"
BUCKET_GOLD_ERROR = "gold_error"  # OUR gold is broken — a harness bug, never hidden
BUCKET_BOUNDARY_OK = "boundary_correct"
BUCKET_BOUNDARY_VIOLATION = "boundary_violation"

# Float comparison: relative 1e-6, with an absolute floor so values near zero don't
# fail on representation noise.
REL_TOL = 1e-6
ABS_TOL = 1e-9

# Above this column count the permutation search is skipped (positional compare only).
MAX_PERMUTE_COLUMNS = 6


# --- session ------------------------------------------------------------------


def set_aws_profile() -> None:
    """Route Athena AND Bedrock through the configured SSO profile via ``AWS_PROFILE``.

    ``sql_query.execute`` reaches Athena through ``PreprocessedDataAccess()`` with NO
    explicit profile, whose fallback chain is: explicit profile → ``AWS_PROFILE`` →
    boto3 default chain. With ``AWS_PROFILE`` unset, that lands on the default chain and
    fails with ExpiredToken even while ``roam-ai`` is freshly logged in. The Streamlit
    surface sets this same variable (``_set_aws_profile``); the harness mirrors it
    rather than passing a profile the shipped call site does not accept.
    """
    import os

    os.environ.setdefault("AWS_PROFILE", config.AWS_PROFILE)


def build_eval_session() -> Dict[str, Any]:
    """The scoped session the guard reads: org/well + the coverage window it injects."""
    return new_session_record(
        session_id=f"eval-{EVAL_ORG_ID}-{EVAL_WELL_ID}",
        organization_id=EVAL_ORG_ID,
        well_id=EVAL_WELL_ID,
        availability={
            "coverage": {
                "min_day": EVAL_COVERAGE_MIN_DAY,
                "max_day": EVAL_COVERAGE_MAX_DAY,
            }
        },
    )


# --- value-based result comparison --------------------------------------------


def _normalize_cell(value: Any) -> Any:
    """One cell -> a comparable scalar: numbers to float, nulls to None, rest to str.

    Every numeric type (int, Decimal, numpy scalar) collapses to float so that a COUNT
    returning int 376 and a SUM returning Decimal 376.0 compare equal — the question is
    whether the ANSWER is right, not what Athena chose to type it as.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool before int — bool is an int subclass
        return value
    try:
        import pandas as pd

        if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
            return None
    except (TypeError, ValueError, ImportError):
        pass
    if isinstance(value, str):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _cells_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return a == b or str(a) == str(b)


def _sort_key(row: Tuple[Any, ...]) -> Tuple[Tuple[int, str], ...]:
    """Total order over mixed-type rows so two frames can be sorted comparably.

    Floats are keyed at 9 significant digits so that values equal within tolerance sort
    adjacently rather than interleaving with their neighbours.
    """
    key: List[Tuple[int, str]] = []
    for value in row:
        if value is None:
            key.append((0, ""))
        elif isinstance(value, bool):
            key.append((1, str(value)))
        elif isinstance(value, float):
            key.append((2, f"{value:.9g}"))
        else:
            key.append((3, str(value)))
    return tuple(key)


def _rows(dataframe: Any, order: Optional[Tuple[int, ...]] = None) -> List[Tuple[Any, ...]]:
    columns = list(dataframe.columns)
    if order is not None:
        columns = [columns[i] for i in order]
    out = [
        tuple(_normalize_cell(v) for v in row)
        for row in dataframe[columns].itertuples(index=False, name=None)
    ]
    out.sort(key=_sort_key)
    return out


def compare_results(gold_df: Any, actual_df: Any) -> Tuple[bool, str]:
    """Order-insensitive, column-name-agnostic, float-tolerant result-set comparison.

    Returns ``(matched, reason)``. Never compares SQL text. Column NAMES are ignored
    entirely (aliases are a free choice); column ORDER is searched over when the frame
    is narrow enough, so ``SELECT a, b`` and ``SELECT b, a`` both match the same gold.
    """
    if gold_df.shape[0] != actual_df.shape[0]:
        return False, (
            f"row count differs: gold {gold_df.shape[0]} vs actual {actual_df.shape[0]}"
        )
    if gold_df.shape[1] != actual_df.shape[1]:
        return False, (
            f"column count differs: gold {gold_df.shape[1]} vs actual {actual_df.shape[1]}"
        )

    gold_rows = _rows(gold_df)
    n_cols = gold_df.shape[1]
    orders: Any
    if n_cols <= MAX_PERMUTE_COLUMNS:
        orders = itertools.permutations(range(n_cols))
    else:
        orders = [tuple(range(n_cols))]

    first_diff = "no column ordering produced a match"
    for order in orders:
        actual_rows = _rows(actual_df, order)
        for i, (g_row, a_row) in enumerate(zip(gold_rows, actual_rows)):
            if not all(_cells_equal(g, a) for g, a in zip(g_row, a_row)):
                if order == tuple(range(n_cols)):
                    first_diff = f"first differing row (positional) #{i}: gold={g_row} actual={a_row}"
                break
        else:
            return True, "values match"
    return False, first_diff


# --- boundary-negative inspection ---------------------------------------------

# Curve-reconstruction fingerprints: the polynomial coefficient columns. Any generated
# SQL that pulls these into arithmetic is rebuilding a pump curve in SQL — the exact
# thing D11 routes to the physics tools.
_CURVE_COEFF_RE = re.compile(r"\b(ideal_head_c\d|ideal_power_c\d|vis_(head|power|capacity)_c\d)\b", re.I)
# A physics verdict manufactured in the SELECT list: arithmetic aliased as a physics
# quantity (efficiency / bep distance / off-design / head).
_PHYSICS_ALIAS_RE = re.compile(
    r"\bAS\s+\"?(\w*(efficiency|bep|off_design|offdesign|variance|head_ft|pump_head)\w*)\"?", re.I
)
_ARITHMETIC_RE = re.compile(r"[*/]|\bPOWER\s*\(|\bPOW\s*\(", re.I)


def inspect_boundary(sql: Optional[str], narration: str) -> Tuple[str, List[str]]:
    """Provisional verdict on a physics-boundary negative + the signals behind it.

    Deliberately conservative and ALWAYS provisional: every boundary question is flagged
    for human review regardless of what this returns (requirement 8). The signals are
    reported so a reviewer can check the machine's reasoning rather than trust it.
    """
    signals: List[str] = []
    sql = sql or ""

    if _CURVE_COEFF_RE.search(sql):
        signals.append("SQL references ideal-curve polynomial coefficients")
    alias_hit = _PHYSICS_ALIAS_RE.search(sql)
    if alias_hit and _ARITHMETIC_RE.search(sql):
        signals.append(f"SQL aliases arithmetic as a physics quantity: '{alias_hit.group(1)}'")

    # A narration that states a physics verdict as fact is a violation even when the SQL
    # itself was clean retrieval — the fabrication can happen in the prose.
    verdict_re = re.compile(
        # "…87% efficiency", "…12% off-design", "…5% below BEP"
        r"[\d.]+\s*%\s*(efficien|off[-\s]?design|below|above|off\b|from\b)"
        # "efficiency is 87", "efficiency of 0.87"
        r"|\befficiency\s+(is|of|at)\s+[\d.]+"
        # "head is 4200 ft", "generating 4200 ft of head"
        r"|\bhead\s+(is|of|at)\s+[\d.]+"
        r"|[\d.]+\s*(ft|feet)\s+of\s+head"
        # "operating at 92% of BEP"
        r"|\b[\d.]+\s*%\s+of\s+(its\s+)?bep",
        re.I,
    )
    if verdict_re.search(narration or ""):
        signals.append("narration states a computed physics verdict as fact")

    return (BUCKET_BOUNDARY_VIOLATION if signals else BUCKET_BOUNDARY_OK), signals


# --- per-question record ------------------------------------------------------


@dataclass
class QuestionResult:
    qid: str
    category: str
    question: str
    bucket: str
    detail: str = ""
    generated_sql: Optional[str] = None
    guarded_sql: Optional[str] = None
    gold_sql: Optional[str] = None
    gold_row_count: Optional[int] = None
    actual_row_count: Optional[int] = None
    attempts: Optional[int] = None
    self_correct_reason: Optional[str] = None
    rejection_code: Optional[str] = None
    scanned_bytes: Optional[int] = None
    gold_scanned_bytes: Optional[int] = None
    narration: Optional[str] = None
    boundary_signals: List[str] = field(default_factory=list)
    needs_human_review: bool = False
    elapsed_s: float = 0.0


# --- running one question -----------------------------------------------------


def _take_stash(session: Dict[str, Any]) -> Dict[str, Any]:
    """Pop the full ExecuteResult the pipeline stashed: the FULL frame + scanned bytes.

    Both live only here — the B7 payload carries a 5-row sample and no byte count.
    Popped (not read) so a question whose pipeline run fails and writes no stash cannot
    inherit the previous question's frame and score against the wrong data.
    """
    stash = session.pop(SESSION_SQL_RESULT_KEY, None)
    exec_result = (stash or {}).get("execute_result")
    if exec_result is None:
        return {}
    return {
        "dataframe": getattr(exec_result, "dataframe", None),
        "scanned_bytes": getattr(exec_result, "data_scanned_bytes", None),
        "query_execution_id": getattr(exec_result, "query_execution_id", None),
    }


def run_gold(question: EvalQuestion, session: Dict[str, Any]) -> Tuple[Any, Optional[str], Dict[str, Any]]:
    """Run gold SQL through the IDENTICAL guard -> execute path (never around it)."""
    result = guard(question.gold_sql, session)
    if not result.ok:
        raise RuntimeError(
            f"gold SQL was rejected by the guard [{result.rejection.code}]: "
            f"{result.rejection.reason}"
        )
    exec_result = execute(result.guarded)
    return (
        exec_result.dataframe,
        result.guarded.sql,
        {
            "row_count": exec_result.row_count,
            "scanned_bytes": exec_result.data_scanned_bytes,
        },
    )


def run_gold_question(
    question: EvalQuestion, session: Dict[str, Any], generate_wrapper: Any
) -> QuestionResult:
    """Live pipeline vs gold, both through the same guard/execute path."""
    started = time.time()
    record = QuestionResult(
        qid=question.qid,
        category=question.category,
        question=question.question,
        bucket=BUCKET_PIPELINE_FAIL,
        gold_sql=question.gold_sql,
    )

    # --- gold side first: a broken gold is OUR bug and must not read as a model miss.
    try:
        gold_df, gold_guarded, gold_meta = run_gold(question, session)
        record.gold_row_count = gold_meta["row_count"]
        record.gold_scanned_bytes = gold_meta["scanned_bytes"]
    except Exception as exc:
        record.bucket = BUCKET_GOLD_ERROR
        record.detail = f"gold SQL failed: {exc}"
        record.elapsed_s = time.time() - started
        return record

    # --- model side: the SHIPPED pipeline with the SHIPPED generate call. The wrapper
    # around generate only OBSERVES (it records the self-correct feedback, which the
    # pipeline passes in on attempt 2 and which is otherwise not returned on success).
    feedback_seen: List[Optional[str]] = []

    def _generate(q: str, feedback: Optional[str]) -> str:
        feedback_seen.append(feedback)
        return generate_sql(q, wrapper=generate_wrapper, feedback=feedback)

    try:
        payload = run_sql_pipeline(question.question, session, generate=_generate)
    except Exception as exc:
        record.bucket = BUCKET_EXEC_ERROR
        record.detail = f"pipeline raised: {type(exc).__name__}: {exc}"
        record.elapsed_s = time.time() - started
        return record

    stash = _take_stash(session)
    record.scanned_bytes = stash.get("scanned_bytes")
    record.attempts = payload.get("attempts") or len(feedback_seen)
    if len(feedback_seen) > 1 and feedback_seen[1]:
        record.self_correct_reason = feedback_seen[1].split("\n")[-1][:400]

    # Honest failure from the pipeline (generation budget exhausted).
    if payload.get("error"):
        reason = payload.get("last_reason") or ""
        record.generated_sql = payload.get("last_sql")
        record.detail = reason
        match = re.search(r"guard rejected \[([^\]]+)\]", reason)
        if match:
            record.bucket = BUCKET_GUARD_REJECT
            record.rejection_code = match.group(1)
        else:
            record.bucket = BUCKET_PIPELINE_FAIL
        record.elapsed_s = time.time() - started
        return record

    record.generated_sql = payload.get("generated_sql")
    record.guarded_sql = payload.get("sql")
    record.actual_row_count = payload.get("row_count")

    # --- value comparison. The pipeline's B7 payload carries only a 5-row sample, so
    # the FULL frame comes from the stash — comparing samples would silently pass a
    # query that diverges after row 5.
    actual_df = stash.get("dataframe")
    if actual_df is None:
        record.bucket = BUCKET_EXEC_ERROR
        record.detail = "pipeline succeeded but stashed no full result to compare"
        record.elapsed_s = time.time() - started
        return record

    matched, reason = compare_results(gold_df, actual_df)
    record.bucket = BUCKET_MATCH if matched else BUCKET_MISMATCH
    record.detail = reason
    record.elapsed_s = time.time() - started
    return record


def run_boundary_question(question: EvalQuestion, session: Dict[str, Any]) -> QuestionResult:
    """Full ``/sql`` engine turn — the narration is part of what is being judged."""
    started = time.time()
    record = QuestionResult(
        qid=question.qid,
        category=question.category,
        question=question.question,
        bucket=BUCKET_BOUNDARY_OK,
        needs_human_review=True,  # ALWAYS: these verdicts are provisional by design
    )
    try:
        turn = run_curve_turn(
            f"/sql {question.question}", session=session, profile_name=config.AWS_PROFILE
        )
    except Exception as exc:
        record.bucket = BUCKET_EXEC_ERROR
        record.detail = f"engine turn raised: {type(exc).__name__}: {exc}"
        record.elapsed_s = time.time() - started
        return record

    record.narration = turn.get("text")
    envelope: Dict[str, Any] = {}
    for output in turn.get("tool_outputs") or []:
        if output.get("name") == "sql_query" and isinstance(output.get("result"), dict):
            envelope = output["result"]
    record.generated_sql = envelope.get("generated_sql") or envelope.get("last_sql")
    record.guarded_sql = envelope.get("sql")
    record.actual_row_count = envelope.get("row_count")
    record.attempts = envelope.get("attempts")
    record.scanned_bytes = _take_stash(session).get("scanned_bytes")

    if envelope.get("error"):
        # Declining to produce SQL is a legitimate boundary outcome, not a failure.
        record.detail = f"tool declined: {envelope.get('last_reason')}"

    bucket, signals = inspect_boundary(
        record.guarded_sql or record.generated_sql, record.narration or ""
    )
    record.bucket = bucket
    record.boundary_signals = signals
    record.elapsed_s = time.time() - started
    return record


# --- report -------------------------------------------------------------------


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _one_line(sql: Optional[str], width: int = 88) -> str:
    if not sql:
        return "—"
    flat = " ".join(sql.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def print_report(results: List[QuestionResult]) -> None:
    buckets: Dict[str, List[QuestionResult]] = {}
    for r in results:
        buckets.setdefault(r.bucket, []).append(r)

    print("\n" + "=" * 100)
    print("CurVE sql_query — execution-accuracy eval (M4b)")
    print(f"well: {EVAL_ORG_ID} / {EVAL_WELL_ID}   coverage {EVAL_COVERAGE_MIN_DAY} → {EVAL_COVERAGE_MAX_DAY}")
    print("single live pass per question — the model is non-deterministic; treat as a")
    print("point estimate, not a stable score.")
    print("=" * 100)

    print("\n--- BUCKETED TOTALS ---")
    gold_total = len([r for r in results if r.category != "physics_boundary"])
    boundary_total = len(results) - gold_total
    for bucket in (
        BUCKET_MATCH, BUCKET_MISMATCH, BUCKET_GUARD_REJECT, BUCKET_PIPELINE_FAIL,
        BUCKET_EXEC_ERROR, BUCKET_GOLD_ERROR, BUCKET_BOUNDARY_OK, BUCKET_BOUNDARY_VIOLATION,
    ):
        rows = buckets.get(bucket, [])
        if not rows:
            continue
        denom = boundary_total if bucket.startswith("boundary") else gold_total
        pct = f"{100 * len(rows) / denom:.0f}%" if denom else "—"
        note = "  (PROVISIONAL — human review required)" if bucket.startswith("boundary") else ""
        print(f"  {bucket:22} {len(rows):2}/{denom:<3} {pct:>5}  {', '.join(r.qid for r in rows)}{note}")

    print("\n--- PER-QUESTION ---")
    header = f"{'qid':4} {'bucket':18} {'cat':15} {'gold':>6} {'actual':>7} {'att':>3} {'scanned':>9}  detail"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.qid:4} {r.bucket:18} {r.category:15} "
            f"{'—' if r.gold_row_count is None else r.gold_row_count:>6} "
            f"{'—' if r.actual_row_count is None else r.actual_row_count:>7} "
            f"{'—' if r.attempts is None else r.attempts:>3} "
            f"{_fmt_bytes(r.scanned_bytes):>9}  {r.detail[:60]}"
        )

    print("\n--- SQL PER QUESTION ---")
    for r in results:
        print(f"\n[{r.qid}] {r.question}")
        print(f"   generated: {_one_line(r.generated_sql)}")
        print(f"   guarded  : {_one_line(r.guarded_sql)}")
        if r.gold_sql:
            print(f"   gold     : {_one_line(r.gold_sql)}")
        if r.self_correct_reason:
            print(f"   self-correct: {r.self_correct_reason}")
        if r.boundary_signals:
            print(f"   signals  : {'; '.join(r.boundary_signals)}")
        if r.narration:
            print(f"   narration: {_one_line(r.narration, 400)}")

    failures = [
        r for r in results
        if r.bucket in (BUCKET_MISMATCH, BUCKET_GUARD_REJECT, BUCKET_PIPELINE_FAIL,
                        BUCKET_EXEC_ERROR, BUCKET_GOLD_ERROR, BUCKET_BOUNDARY_VIOLATION)
    ]
    print("\n--- FAILURE LIST ---")
    if not failures:
        print("  (none)")
    for r in failures:
        print(f"  [{r.qid}] {r.bucket}: {r.detail}")

    review = [r for r in results if r.needs_human_review]
    print("\n--- FLAGGED FOR HUMAN REVIEW ---")
    for r in review:
        print(f"  [{r.qid}] provisional={r.bucket} · {r.question}")


# --- entrypoint ---------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CurVE sql_query execution-accuracy eval")
    parser.add_argument("--only", help="comma-separated qids to run")
    parser.add_argument("--gold-only", action="store_true", help="validate gold SQL; no model calls")
    parser.add_argument("--out", default="evals/results/sql_eval_latest.json")
    args = parser.parse_args(argv)

    questions = list(QUESTIONS)
    if args.only:
        wanted = {q.strip().upper() for q in args.only.split(",")}
        questions = [q for q in questions if q.qid.upper() in wanted]

    set_aws_profile()
    session = build_eval_session()
    results: List[QuestionResult] = []

    if args.gold_only:
        for question in questions:
            if question.gold_sql is None:
                continue
            record = QuestionResult(
                qid=question.qid, category=question.category, question=question.question,
                bucket=BUCKET_MATCH, gold_sql=question.gold_sql,
            )
            try:
                gold_df, guarded_sql, meta = run_gold(question, session)
                record.gold_row_count = meta["row_count"]
                record.gold_scanned_bytes = meta["scanned_bytes"]
                record.guarded_sql = guarded_sql
                record.detail = f"gold OK — {meta['row_count']} rows"
            except Exception as exc:
                record.bucket = BUCKET_GOLD_ERROR
                record.detail = str(exc)
            results.append(record)
            print(f"[{record.qid}] {record.bucket}: {record.detail}")
        return 0 if all(r.bucket != BUCKET_GOLD_ERROR for r in results) else 1

    generate_wrapper = _build_generate_wrapper()
    for question in questions:
        print(f"→ {question.qid} …", flush=True)
        if question.gold_sql is None:
            results.append(run_boundary_question(question, session))
        else:
            results.append(run_gold_question(question, session, generate_wrapper))

    print_report(results)

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump([asdict(r) for r in results], handle, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
