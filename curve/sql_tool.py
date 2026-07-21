"""CurVE ``sql_query`` tool pipeline (M3) — NL→SQL over the allowlist, end-to-end.

This is the tool's **internal** pipeline. The outer Converse model never writes SQL;
it only calls ``sql_query`` with a natural-language ``question``. This module then:

    build schema context  (schema_catalog.render_schema_prompt)
      -> generate SQL       (a nested, tool-LESS Bedrock call via curve.wrapper)
      -> validate           (M2 guard -> M2 explain on the GUARDED sql)
      -> self-correct        (feed the structured reason back into ONE regen; cap 2)
      -> execute once        (M2 execute; stash the full result on the session)
      -> return the B7 payload {sql, columns, row_count, sample_rows}

Hard boundaries (locked decisions):
  * B4  — one tool; SQL generation is internal, never the outer model's job.
  * B6  — EXPLAIN pre-validate; self-correct capped at **2 generations total**.
  * B7  — trimmed model payload; the FULL ExecuteResult is stashed for M4 (no re-run).
  * D11 — retrieval over STORED COLUMNS only; physics-derived quantities are routed
          away (the prompt forbids them; no physics formulas live here).

Athena is reached ONLY through M2's ``guard`` -> ``explain`` -> ``execute`` (in that
order, on guarded output). This module never touches ``_GUARD_TOKEN`` and never
reimplements validation or scoping.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from curve import config
from curve.schema_catalog import render_schema_prompt
from curve.session import save_session
from curve.sql_query import execute, explain, guard
from curve.wrapper import CurveBedrockWrapper

# Self-correct budget (B6): at most this many generate attempts total. Attempt 1 is
# the first shot; a single regeneration is allowed after a guard/EXPLAIN failure.
MAX_SQL_GENERATIONS = 2

# Where the full ExecuteResult is stashed on the session record so M4 can offer CSV
# with no re-run. M3 owns these keys; M4 reads them.
#
# Two keys, deliberately:
#   * SESSION_SQL_RESULT_KEY  — the LAST execution only (original M3 contract; the eval
#     harness and any single-result consumer still read this).
#   * SESSION_SQL_RESULTS_KEY — an ORDERED LIST, one entry per execution in this turn.
#     Added after a live /sql turn showed the outer model calling sql_query TWICE in one
#     turn (forced on iteration 1, then `auto` lets it call again to refine). The single
#     slot held only the second execution, so the first rendered block offered the second
#     query's CSV under the second query's Athena id. A one-deep slot cannot represent an
#     N-execution turn; the list can.
SESSION_SQL_RESULT_KEY = "sql_last_result"
SESSION_SQL_RESULTS_KEY = "sql_results"


# --- generation prompt --------------------------------------------------------

# System prompt for the nested, tool-less generate call. Kept lean and deterministic.
SQL_GEN_SYSTEM_PROMPT = (
    "You are a careful Amazon Athena (Trino SQL) generator for CurVE, a physics-"
    "validation tool over ESP well data. Given a fixed schema and one operator "
    "question, emit EXACTLY ONE safe, read-only SQL query that answers it. You output "
    "SQL only — never prose, never an explanation."
)

# The hard rules appended after the schema block. These enforce the M2 carry-forwards
# so the model does not fight the guard (org/well auto-injected; join on observation_day;
# fully-qualify; no LIMIT/date; stored columns only). No physics formulas appear here.
_GENERATE_RULES = f"""\
GENERATION RULES — follow every one EXACTLY:
1. Output a SINGLE read-only query: one SELECT, or one WITH … SELECT. No INSERT /
   UPDATE / DELETE / DDL, no semicolon-separated or stacked statements.
2. Query ONLY the five tables above, and FULLY-QUALIFY every table exactly as written
   (catalog.database.table). An under-qualified table name is rejected.
3. NEVER write an organization_id or well_id predicate anywhere — not in WHERE, not in
   a JOIN … ON. Org and well scoping is injected automatically. Emitting one is rejected.
4. To relate two scoped tables (e.g. telemetry and production), JOIN ON observation_day
   ONLY. Do NOT join on well_id or organization_id — that is rejected.
5. Do NOT add a LIMIT to bound cost or "just to be safe" — a row cap of
   {config.SQL_QUERY_LIMIT_CAP} is injected for you automatically. EXCEPTION: when the
   question itself names a number of rows ("the top 5 pumps", "the 3 highest days",
   "the latest reading", "the single largest"), DO write an explicit
   LIMIT <that number> — it is part of the correct answer, not a cost guard. Such a
   LIMIT is accepted as long as it is a plain integer literal that is not greater than
   {config.SQL_QUERY_LIMIT_CAP}; never write LIMIT ALL, a parameter, or an expression.
   Do NOT add an observation_day date filter unless the question names a specific
   window. The org/well scope and the default date window are injected for you.
6. Retrieve STORED COLUMNS only. Do not attempt physics-derived quantities (best-
   efficiency-point position, off-design variance, pump efficiency, affinity-law
   scaling, ideal-curve reconstruction) — those are NOT columns in these tables. If the
   question asks for one, return the closest stored-column retrieval instead.
7. For the latest recommendation, ORDER BY timestamp DESC and take the top row.
8. Output ONLY the SQL. No prose, no commentary, no markdown code fences.\
"""


def build_generate_prompt(question: str, feedback: Optional[str] = None) -> str:
    """Compose the full NL→SQL generate prompt: schema + hard rules + question.

    ``feedback`` (set only on the self-correct regeneration) carries the previous
    attempt's SQL and the structured rejection / EXPLAIN reason so the model can fix
    the specific fault instead of guessing.
    """
    parts: List[str] = [render_schema_prompt(), _GENERATE_RULES]
    if feedback:
        parts.append(
            "YOUR PREVIOUS ATTEMPT WAS REJECTED. Fix the specific problem and try "
            "again, still obeying every rule above.\n" + feedback
        )
    parts.append(f"QUESTION: {question}\n\nSQL:")
    return "\n\n".join(parts)


# --- extraction ---------------------------------------------------------------


def _extract_text(response: Dict[str, Any]) -> str:
    """Concatenate the text blocks of a Converse response (skips thinking blocks)."""
    message = (response.get("output") or {}).get("message") or {}
    parts = [b["text"] for b in message.get("content", []) if "text" in b]
    return "\n".join(parts).strip()


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of possibly fenced / prose-wrapped model text.

    Handles ```sql … ``` fences, bare ``` … ``` fences, and leading prose before the
    first SELECT / WITH. Strips a trailing semicolon. Returns "" if nothing SQL-like
    is present (the guard then rejects the empty string — fail-closed).
    """
    t = (text or "").strip()
    fence = re.search(r"```(?:sql)?\s*(.+?)```", t, re.S | re.I)
    if fence:
        t = fence.group(1).strip()
    start = re.search(r"\b(WITH|SELECT)\b", t, re.I)
    if start:
        t = t[start.start():]
    return t.strip().rstrip(";").strip()


# --- JSON-safe sampling -------------------------------------------------------


def _jsonify(value: Any) -> Any:
    """Coerce one cell to a JSON-serializable scalar for the Converse toolResult.

    Decimals -> float, everything else non-primitive -> str, NaN/NaT/None -> None.
    """
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        # NaN is not valid JSON; emit null.
        return None if value != value else value
    # pandas.Timestamp, numpy scalars, etc. — stringify; also catch pandas NaT/NA.
    try:
        import pandas as pd  # lazy: keep the cred-free import path light

        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _sample_rows(dataframe: Any, n: int = 5) -> List[Dict[str, Any]]:
    """Top ``n`` rows of the result frame as JSON-safe dicts for the model payload."""
    head = dataframe.head(n)
    records = head.to_dict("records")
    return [{k: _jsonify(v) for k, v in row.items()} for row in records]


# --- the pipeline -------------------------------------------------------------

# A generate callable: (question, feedback) -> raw SQL string. Injected so the pipeline
# can be unit-tested with no Bedrock; the tool wires the real nested Converse call.
GenerateFn = Callable[[str, Optional[str]], str]


def run_sql_pipeline(
    question: str,
    session: Dict[str, Any],
    *,
    generate: GenerateFn,
    max_generations: int = MAX_SQL_GENERATIONS,
) -> Dict[str, Any]:
    """Generate → validate → self-correct (cap 2) → execute once → B7 payload.

    ``generate(question, feedback)`` returns candidate SQL (already fence-stripped by
    the caller, or raw — ``extract_sql`` is applied here defensively). On every
    guard / EXPLAIN failure the structured reason + the offending SQL is fed back into
    ONE regeneration. When the generation budget is exhausted, returns an HONEST
    structured failure (``{error, last_sql, last_reason}``) — never fabricated rows,
    never an empty-looking success.

    On success, executes ONCE via M2, stashes the full ExecuteResult on the session
    (``SESSION_SQL_RESULT_KEY``) for M4, and returns the trimmed model payload.
    """
    feedback: Optional[str] = None
    last_sql = ""
    last_reason = "no attempt ran"

    for attempt in range(1, max_generations + 1):
        candidate = extract_sql(generate(question, feedback))
        last_sql = candidate

        # Validate step 1: guard (parse + allowlist + scope injection + LIMIT policy).
        result = guard(candidate, session)
        if not result.ok:
            rej = result.rejection
            last_reason = f"guard rejected [{rej.code}]: {rej.reason}"
            feedback = (
                f"Attempt {attempt} SQL:\n{candidate}\n\n"
                f"It was REJECTED by the safety guard — code={rej.code}: {rej.reason}"
            )
            continue

        guarded = result.guarded

        # Validate step 2: EXPLAIN the GUARDED sql (zero-scan pre-validate).
        exp = explain(guarded)
        if not exp["ok"]:
            last_sql = guarded.sql
            last_reason = f"EXPLAIN failed: {exp['error']}"
            feedback = (
                f"Attempt {attempt} SQL (after safe rewrite):\n{guarded.sql}\n\n"
                f"Athena EXPLAIN rejected it: {exp['error']}"
            )
            continue

        # Validated → execute ONCE and stash the full result for M4 (no re-run).
        exec_result = execute(guarded)
        _stash_result(session, question, candidate, guarded, exec_result)
        return {
            "sql": guarded.sql,
            "columns": exec_result.columns,
            "row_count": exec_result.row_count,
            "sample_rows": _sample_rows(exec_result.dataframe),
            # diagnostics (also surfaced to the CLI/report via the full tool envelope):
            "generated_sql": candidate,
            "tables": list(guarded.tables),
            "attempts": attempt,
            "self_corrected": attempt > 1,
        }

    # Budget exhausted — honest structured failure (B6 cap-exceeded).
    return {
        "error": (
            f"sql_query could not produce a valid scoped query within "
            f"{max_generations} generations."
        ),
        "last_sql": last_sql,
        "last_reason": last_reason,
    }


def _stash_result(
    session: Dict[str, Any],
    question: str,
    generated_sql: str,
    guarded: Any,
    exec_result: Any,
) -> None:
    """Stash the FULL ExecuteResult on the session so M4 can offer CSV with no re-run.

    Written to BOTH the last-execution slot (original contract) and the per-execution
    list, so a turn that runs sql_query more than once keeps every result rather than
    letting the later execution clobber the earlier one.
    """
    if session is None:
        return
    record = {
        "question": question,
        "generated_sql": generated_sql,
        "guarded_sql": guarded.sql,
        "tables": list(guarded.tables),
        "execute_result": exec_result,  # holds the full DataFrame + metadata
        "query_execution_id": exec_result.query_execution_id,
    }
    session[SESSION_SQL_RESULT_KEY] = record
    session.setdefault(SESSION_SQL_RESULTS_KEY, []).append(record)
    # Persist through the active store (in-memory dict today; DDB later).
    try:
        if session.get("session_id"):
            save_session(session)
    except Exception:
        pass  # best-effort persistence; the in-place mutation already took effect


# --- the real nested generate call (Bedrock via curve.wrapper) ----------------


def _build_generate_wrapper() -> CurveBedrockWrapper:
    """A tool-LESS Converse wrapper for SQL generation.

    Thinking is OFF: SQL generation wants determinism and we do not carry
    reasoningContent across a single-shot generate. Profile comes from config so the
    live CLI (SSO profile ``roam-ai``) works; in-Lambda the default chain applies.
    """
    return CurveBedrockWrapper(
        profile_name=config.AWS_PROFILE,
        enable_thinking=False,
        temperature=0.0,
        # This model rejects temperature + topP together; with thinking off we send
        # only temperature (v1's thinking-on path omits topP already).
        top_p=None,
    )


def generate_sql(
    question: str,
    *,
    wrapper: CurveBedrockWrapper,
    feedback: Optional[str] = None,
) -> str:
    """One nested, tool-less Converse turn: prompt in, SQL text out (fence-stripped)."""
    prompt = build_generate_prompt(question, feedback)
    response = wrapper.converse(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": SQL_GEN_SYSTEM_PROMPT}],
        tool_config=None,  # tool-LESS — this call only writes SQL
    )
    return extract_sql(_extract_text(response))


# --- the tool entrypoint ------------------------------------------------------


def sql_query(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """``sql_query`` tool fn: NL question -> scoped SQL result (B7 payload).

    The engine injects the full session under ``tool_input['session']`` (org/well +
    coverage the guard needs). ``question`` is the model-supplied NL string. Builds the
    nested generate wrapper, runs the internal pipeline, and returns the trimmed
    payload (or an honest failure). The full result is stashed on the session by the
    pipeline.
    """
    question = (tool_input or {}).get("question") or ""
    session = (tool_input or {}).get("session")
    if not question.strip():
        return {"error": "sql_query requires a non-empty 'question'."}
    if not isinstance(session, dict) or not session.get("organization_id"):
        return {
            "error": "sql_query has no scoped session (organization_id/well_id) — "
            "cannot run a safe query."
        }

    wrapper = _build_generate_wrapper()

    def _gen(q: str, feedback: Optional[str]) -> str:
        return generate_sql(q, wrapper=wrapper, feedback=feedback)

    return run_sql_pipeline(question, session, generate=_gen)


# --- toolSpec (A1: written for eventual natural routing, muted while forced) ---

SQL_QUERY_SPEC = {
    "toolSpec": {
        "name": "sql_query",
        "description": (
            "Retrieve raw stored well data by translating a natural-language question "
            "into a single safe, read-only SQL query over the ESP data warehouse "
            "(preprocessed telemetry and daily production for the selected well, the "
            "latest ML setpoint recommendations, RRC well depth, and the multi-"
            "manufacturer ideal-pump reference library). Use this for direct data "
            "lookups, counts, sums, averages, min/max, distinct values, and ad-hoc "
            "column retrieval that the specialized physics tools do not cover — e.g. "
            "'how many days did we produce oil last month', 'what is the max motor "
            "frequency on record', 'list the pump models for this series', 'what is "
            "this well's depth'. This is RETRIEVAL of stored columns only: it does not "
            "compute physics-derived quantities (best-efficiency-point position, off-"
            "design variance, efficiency, affinity scaling) — route those to the "
            "physics tools. The well is already set up for this session; supply only "
            "the question."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The operator's natural-language data question, verbatim. "
                            "The tool writes and runs the SQL itself; do not write SQL."
                        ),
                    }
                },
                "required": ["question"],
            }
        },
    }
}
