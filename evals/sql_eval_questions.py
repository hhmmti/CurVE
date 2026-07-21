"""CurVE ``sql_query`` eval set — canonical questions + gold SQL (M4b).

**This file is the reviewable artifact.** Everything a human needs to audit the eval
lives here: the question as an operator would phrase it, the gold SQL, and a note
saying why that gold is the right answer (and where it is ambiguous).

Authoring rules — gold SQL obeys the SAME rules the model must follow, because both
sides are executed through the identical ``guard`` -> ``execute`` path:

  * NO ``organization_id`` / ``well_id`` predicate anywhere (the guard injects them;
    emitting one is a ``model_supplied_scope`` rejection for gold too).
  * Every table FULLY-QUALIFIED as ``catalog.database.table``.
  * A single ``SELECT`` or ``WITH … SELECT``.
  * No ``LIMIT`` except where the question itself names a row count.
  * No ``observation_day`` filter except where the question names a window.

Test well: ``greenlakeenergy`` / ``JAVELINA CHARMER 27-28 #4302H`` — the one well
present on all four surfaces. Session coverage 2024-03-30 → 2026-07-15, so every
window here sits inside 2024-03 → 2026-07. Live volumes probed 2026-07-20: production
376 rows (2024-03-30 → 2026-07-17), telemetry 34,692 rows, pump library 699 rows /
16 manufacturers, rrc depth 10,671 ft.

**Known schema defect this eval surfaced (2026-07-20).** ``schema_catalog`` states the
production grain is "one row per (organization_id, well_id, observation_day)". It is
not: 376 rows span only 356 distinct days, with 2024-04-01 carrying 9 rows. Any
per-day aggregate is therefore ambiguous — dedupe or not — and both the gold author
and the model were misled by the stated grain (see Q02, Q04, Q14). The gloss needs
fixing; until it is, treat per-day production aggregates as advisory.

**Row-cap hazard, deliberately avoided:** the guard injects ``LIMIT 10000`` and
telemetry holds 34,692 rows. A question returning raw telemetry rows would be
truncated to an arbitrary, unordered 10,000 on BOTH sides — a coin-flip mismatch that
measures nothing. Every question here therefore returns an aggregate, a grouped series
(≤ ~470 rows), or an explicitly ordered top-N.

Categories:
  ``production`` · ``telemetry`` · ``recommendation`` · ``well_depth`` ·
  ``pump_library`` · ``join`` · ``date_window`` · ``row_count`` ·
  ``physics_boundary`` (negatives — no gold; see :class:`EvalQuestion.gold_sql`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# --- the test session ---------------------------------------------------------

EVAL_ORG_ID = "greenlakeenergy"
EVAL_WELL_ID = "JAVELINA CHARMER 27-28 #4302H"
# The session coverage window the guard injects when the model supplies no date filter.
# Both gold and generated SQL receive this identical injection.
EVAL_COVERAGE_MIN_DAY = "2024-03-30"
EVAL_COVERAGE_MAX_DAY = "2026-07-15"

# Fully-qualified table names, spelled out so gold SQL is readable as-is. These mirror
# schema_catalog's config-driven FQNs; the harness asserts they agree at import time.
TELEMETRY = "roam_prd_products.esp_optimization_v2.esp_telemetry_preprocessed"
PRODUCTION = "roam_prd_products.esp_optimization_v2.esp_production_preprocessed"
RECOMMENDATIONS = "roam_prd_ddb.default.esp_setpoint_recommendations_v2"
WELL_DEPTH = "roam_dev_products.well_depth_dev.rrc_well_depth"
PUMP_LIBRARY = "roam_dev_products.esp_ideal_pump_dev.ideal_pump_library_v1"


@dataclass(frozen=True)
class EvalQuestion:
    """One canonical question. ``gold_sql=None`` marks a physics-boundary negative.

    A boundary negative has no correct SQL answer by construction: D11 says the
    quantity it asks for is not a stored column. The expected outcome is that the tool
    declines or returns only column-level retrieval — never a fabricated physics
    number. Those are scored by inspection (and always flagged for human review), not
    by result-set comparison.
    """

    qid: str
    category: str
    question: str
    gold_sql: Optional[str]
    note: str


QUESTIONS: List[EvalQuestion] = [
    # --- production aggregates ------------------------------------------------
    EvalQuestion(
        qid="Q01",
        category="production",
        question="How much oil has this well produced in total?",
        gold_sql=f"SELECT SUM(alloc_oil_vol) AS total_oil_bbl FROM {PRODUCTION}",
        note="Simplest possible aggregate. alloc_oil_vol is bbl. Guard injects org/well "
        "+ the coverage window, so 'in total' means 'across the covered window'.",
    ),
    EvalQuestion(
        qid="Q02",
        category="production",
        question="What is the average daily water volume for this well?",
        gold_sql=f"SELECT AVG(alloc_water_vol) AS avg_water_bbl FROM {PRODUCTION}",
        note="Gold is a plain row-wise AVG (492.06 bbl). CAVEAT, found during the "
        "2026-07-20 run: production is NOT one row per day as schema_catalog claims — "
        "374 rows span 354 days, and 2024-04-01 alone has 9 rows. A row-wise AVG "
        "therefore over-weights duplicated days and is not strictly the 'average daily' "
        "volume the operator asked for. Gold and the model agree, so this scores a "
        "match — a case where execution accuracy and semantic correctness diverge. Fix "
        "the grain gloss before treating this question's match as meaningful.",
    ),
    EvalQuestion(
        qid="Q03",
        category="row_count",
        question="Which single day did this well produce the most oil, and how much?",
        gold_sql=(
            f"SELECT observation_day, alloc_oil_vol\n"
            f"FROM {PRODUCTION}\n"
            f"ORDER BY alloc_oil_vol DESC\n"
            f"LIMIT 1"
        ),
        note="Row-count ask #1 — 'single day' requires LIMIT 1, which the amended rule 5 "
        "now permits. Verified tie-free live: top oil day is 2026-04-17 at 444.76 bbl vs "
        "382.98 for #2, so the top-1 is unambiguous.",
    ),
    EvalQuestion(
        qid="Q04",
        category="production",
        question="How many days do we have production records for on this well?",
        gold_sql=f"SELECT COUNT(DISTINCT observation_day) AS n_days FROM {PRODUCTION}",
        note="CORRECTED after the 2026-07-20 run — this note previously (wrongly) said "
        "COUNT(*) and COUNT(DISTINCT observation_day) were interchangeable here. They "
        "are not: within the injected window COUNT(DISTINCT observation_day) = 354 while "
        "COUNT(*) = 374, because production carries duplicate rows per day. DISTINCT is "
        "the right gold for 'how many days', and a model writing COUNT(*) SHOULD be "
        "scored a mismatch. This question now genuinely discriminates.",
    ),
    # --- telemetry aggregates / series ---------------------------------------
    EvalQuestion(
        qid="Q05",
        category="telemetry",
        question="What is the highest motor frequency this pump has ever run at?",
        gold_sql=f"SELECT MAX(motor_frequency_hz) AS max_hz FROM {TELEMETRY}",
        note="Tests that the model picks the MEASURED motor_frequency_hz, not the "
        "commanded motor_frequency_setpoint_hz — a real gloss-driven distinction.",
    ),
    EvalQuestion(
        qid="Q06",
        category="telemetry",
        question="What is the average pump intake pressure on this well?",
        gold_sql=f"SELECT AVG(pump_intake_pressure_psi) AS avg_pip_psi FROM {TELEMETRY}",
        note="Straight telemetry aggregate. AVG ignores NULLs on both sides identically.",
    ),
    EvalQuestion(
        qid="Q07",
        category="telemetry",
        question="How many telemetry readings do we have for this well?",
        gold_sql=f"SELECT COUNT(*) AS n_readings FROM {TELEMETRY}",
        note="COUNT returns one row, so the injected LIMIT 10000 cannot truncate it — "
        "this is the safe way to ask about the 34,692-row telemetry table.",
    ),
    EvalQuestion(
        qid="Q08",
        category="telemetry",
        question="Show me the average motor frequency for each day.",
        gold_sql=(
            f"SELECT observation_day, AVG(motor_frequency_hz) AS avg_hz\n"
            f"FROM {TELEMETRY}\n"
            f"GROUP BY observation_day"
        ),
        note="A grouped time series (~470 day-rows, well under the cap). No ORDER BY in "
        "gold on purpose — the comparison is order-insensitive, so a model that sorts "
        "still matches.",
    ),
    # --- recommendation -------------------------------------------------------
    EvalQuestion(
        qid="Q09",
        category="recommendation",
        question="What motor frequency is the model currently recommending for this well?",
        gold_sql=(
            f"SELECT optimal_setpoint_recommendation.max_oil.motor_frequency_hz AS hz\n"
            f"FROM {RECOMMENDATIONS}\n"
            f"ORDER BY timestamp DESC\n"
            f"LIMIT 1"
        ),
        note="Row-count ask #2 ('currently' -> latest -> LIMIT 1) AND a struct-access "
        "test. AMBIGUITY, flagged deliberately: the struct carries both a max_oil and a "
        "total_economics branch. On this well both currently read 62.0 Hz, so either "
        "choice matches — a model that picks total_economics is NOT penalised here, and "
        "this question would need re-authoring on a well where they diverge.",
    ),
    # --- well depth (rrc, well-only scoping) ----------------------------------
    EvalQuestion(
        qid="Q10",
        category="well_depth",
        question="How deep is this well?",
        gold_sql=f'SELECT "api depth" AS depth_ft FROM {WELL_DEPTH}',
        note="Tests the spaced-identifier quoting gotcha AND the heterogeneous-scoping "
        "rule: rrc is well_id-scoped ONLY, so the guard must inject no org predicate. "
        "Live value 10,671 ft.",
    ),
    # --- pump library (unscoped reference table) ------------------------------
    EvalQuestion(
        qid="Q11",
        category="row_count",
        question="Which five pumps in the library have the highest best-efficiency flow rate?",
        gold_sql=(
            f"SELECT esp_model, bep_bpd\n"
            f"FROM {PUMP_LIBRARY}\n"
            f"ORDER BY bep_bpd DESC\n"
            f"LIMIT 5"
        ),
        note="Row-count ask #3 — the canonical case the amended rule 5 exists for. Also "
        "tests that NO org/well predicate is added to the unscoped reference table. "
        "Tie-free at the boundary: rank 5 = 74,166 bpd vs rank 6 = 71,634 bpd.",
    ),
    EvalQuestion(
        qid="Q12",
        category="pump_library",
        question="How many pump models does each manufacturer have in the library?",
        gold_sql=(
            f"SELECT manufacturer, COUNT(*) AS n_models\n"
            f"FROM {PUMP_LIBRARY}\n"
            f"GROUP BY manufacturer"
        ),
        note="Grouped count over the reference table (16 manufacturers, 699 models). "
        "Order-insensitive comparison means no ORDER BY is needed on either side.",
    ),
    EvalQuestion(
        qid="Q13",
        category="pump_library",
        question="How many pumps in the library are marked obsolete?",
        gold_sql=(
            f"SELECT COUNT(*) AS n_obsolete FROM {PUMP_LIBRARY} WHERE is_obsolete = true"
        ),
        note="Boolean filter on the surfaced-never-dropped is_obsolete flag (~209 of 699 "
        "per the catalog note).",
    ),
    # --- two-table join -------------------------------------------------------
    EvalQuestion(
        qid="Q14",
        category="join",
        question="For each day, show me the oil produced alongside the average motor frequency.",
        gold_sql=(
            f"SELECT p.observation_day, p.alloc_oil_vol, AVG(t.motor_frequency_hz) AS avg_hz\n"
            f"FROM {PRODUCTION} p\n"
            f"JOIN {TELEMETRY} t ON p.observation_day = t.observation_day\n"
            f"GROUP BY p.observation_day, p.alloc_oil_vol"
        ),
        note="The two-table join. Rule 4 allows joining on observation_day ONLY — org "
        "and well are injected into BOTH scopes by the guard, so an observation_day-only "
        "join is still correctly scoped. Joining on well_id would be a rejection.",
    ),
    # --- explicit date window -------------------------------------------------
    EvalQuestion(
        qid="Q15",
        category="date_window",
        question="How much oil did this well produce in June 2026?",
        gold_sql=(
            f"SELECT SUM(alloc_oil_vol) AS june_oil_bbl\n"
            f"FROM {PRODUCTION}\n"
            f"WHERE observation_day BETWEEN '2026-06-01' AND '2026-06-30'"
        ),
        note="The model-supplied window case: the guard RESPECTS a supplied "
        "observation_day filter instead of injecting coverage. Inside the 2024-03 → "
        "2026-07 envelope; live answer 4,390.72 bbl over 28 days. A model that writes "
        "the window with >= / < or with LIKE '2026-06%' still lands on the same rows.",
    ),
    # --- physics-boundary negatives (D11): NO gold SQL ------------------------
    EvalQuestion(
        qid="B01",
        category="physics_boundary",
        question="Is this pump running near its best efficiency point right now?",
        gold_sql=None,
        note="BEP POSITION. bep_bpd is a stored column, but 'is the pump NEAR it' needs "
        "the pump-to-well connection plus a curve comparison — not a stored column. "
        "Correct: retrieve stored context (recent rate, and/or library bep_bpd) and hand "
        "the physics judgement to the physics tools. Violation: an SQL-computed verdict.",
    ),
    EvalQuestion(
        qid="B02",
        category="physics_boundary",
        question="How far off-design is this well operating?",
        gold_sql=None,
        note="OFF-DESIGN VARIANCE. Requires an ideal curve evaluated at the operating "
        "point; no stored column holds it. Correct: decline or return operating-point "
        "columns. Violation: arithmetic presented as an off-design percentage.",
    ),
    EvalQuestion(
        qid="B03",
        category="physics_boundary",
        question="What is the pump's efficiency at the moment?",
        gold_sql=None,
        note="EFFICIENCY. Named explicitly in D11 and in rule 6 as not-a-column. The "
        "tempting violation is hydraulic-power arithmetic (ΔP × rate ÷ motor_power_kw) "
        "inside SQL — that is the energy_efficiency physics tool's job, with its own "
        "trust label and provenance.",
    ),
    EvalQuestion(
        qid="B04",
        category="physics_boundary",
        question="What head is the pump generating at its current flow rate?",
        gold_sql=None,
        note="IDEAL-CURVE RECONSTRUCTION. The ideal_head_c1..c6 polynomial coefficients "
        "ARE stored columns, which makes this the sharpest trap in the set: evaluating "
        "that polynomial in SQL is exactly the physics D11 routes away. Any generated "
        "SQL doing arithmetic on ideal_head_c* is a violation.",
    ),
]


GOLD_QUESTIONS = [q for q in QUESTIONS if q.gold_sql is not None]
BOUNDARY_QUESTIONS = [q for q in QUESTIONS if q.gold_sql is None]
