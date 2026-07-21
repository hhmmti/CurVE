"""M4b — the eval harness's SCORER (offline; no model, no Athena).

The harness itself measures the live pipeline, so it cannot be unit-tested end to end.
What CAN and must be tested is the thing the whole report rests on: the result-set
comparator. If ``compare_results`` is wrong, every bucket in the report is wrong — a
false ``match`` is worse than no eval at all.

Covers the three properties the brief requires of scoring: order-insensitive,
float-tolerant, column-name-agnostic — and, just as important, the cases that MUST
still fail (genuinely different values).
"""

from __future__ import annotations

import pandas as pd
import pytest

from evals.sql_eval import (
    BUCKET_BOUNDARY_OK,
    BUCKET_BOUNDARY_VIOLATION,
    compare_results,
    inspect_boundary,
)
from evals.sql_eval_questions import BOUNDARY_QUESTIONS, GOLD_QUESTIONS, QUESTIONS


# --- the comparator: things that must MATCH -----------------------------------


def test_identical_frames_match():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert compare_results(df, df.copy())[0]


def test_row_order_is_ignored():
    gold = pd.DataFrame({"day": ["2026-01-01", "2026-01-02"], "v": [1.0, 2.0]})
    actual = gold.iloc[::-1].reset_index(drop=True)
    assert compare_results(gold, actual)[0]


def test_column_names_are_ignored():
    gold = pd.DataFrame({"total_oil_bbl": [1234.5]})
    actual = pd.DataFrame({"sum_of_oil": [1234.5]})  # different alias, same answer
    assert compare_results(gold, actual)[0]


def test_column_order_is_searched_over():
    gold = pd.DataFrame({"day": ["2026-01-01"], "hz": [60.0]})
    actual = pd.DataFrame({"hz": [60.0], "day": ["2026-01-01"]})
    assert compare_results(gold, actual)[0]


def test_float_noise_within_relative_tolerance_matches():
    gold = pd.DataFrame({"v": [1234.5678901]})
    actual = pd.DataFrame({"v": [1234.5678901 * (1 + 1e-9)]})
    assert compare_results(gold, actual)[0]


def test_int_and_decimal_representations_of_the_same_answer_match():
    """COUNT(*) -> int 376 and SUM(...) -> Decimal('376.0') are the same answer."""
    from decimal import Decimal

    gold = pd.DataFrame({"n": [376]})
    actual = pd.DataFrame({"n": [Decimal("376.000000000")]})
    assert compare_results(gold, actual)[0]


def test_nulls_match_nulls():
    gold = pd.DataFrame({"depth": [None]}, dtype=object)
    actual = pd.DataFrame({"depth": [float("nan")]})
    assert compare_results(gold, actual)[0]


# --- the comparator: things that must NOT match -------------------------------


def test_different_values_do_not_match():
    gold = pd.DataFrame({"v": [1234.5]})
    actual = pd.DataFrame({"v": [1235.5]})
    matched, reason = compare_results(gold, actual)
    assert not matched and "differing row" in reason


def test_float_difference_beyond_tolerance_does_not_match():
    gold = pd.DataFrame({"v": [1000.0]})
    actual = pd.DataFrame({"v": [1000.01]})  # 1e-5 relative — beyond the 1e-6 tolerance
    assert not compare_results(gold, actual)[0]


def test_row_count_difference_is_reported_not_ignored():
    gold = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    actual = pd.DataFrame({"v": [1.0, 2.0]})
    matched, reason = compare_results(gold, actual)
    assert not matched and "row count differs: gold 3 vs actual 2" in reason


def test_column_count_difference_is_reported():
    gold = pd.DataFrame({"a": [1.0], "b": [2.0]})
    actual = pd.DataFrame({"a": [1.0]})
    matched, reason = compare_results(gold, actual)
    assert not matched and "column count differs" in reason


def test_null_does_not_silently_match_a_number():
    gold = pd.DataFrame({"depth": [10671.0]})
    actual = pd.DataFrame({"depth": [None]}, dtype=object)
    assert not compare_results(gold, actual)[0]


def test_permutation_search_cannot_manufacture_a_false_match():
    """Same values in both frames but paired to the WRONG keys must not match."""
    gold = pd.DataFrame({"day": ["d1", "d2"], "v": [1.0, 2.0]})
    actual = pd.DataFrame({"day": ["d1", "d2"], "v": [2.0, 1.0]})  # values swapped
    assert not compare_results(gold, actual)[0]


# --- boundary inspection ------------------------------------------------------


def test_curve_coefficient_arithmetic_is_flagged_as_a_violation():
    sql = (
        "SELECT ideal_head_c1 + ideal_head_c2 * 1000 AS head_ft "
        "FROM roam_dev_products.esp_ideal_pump_dev.ideal_pump_library_v1"
    )
    bucket, signals = inspect_boundary(sql, "The pump generates 4200 ft of head.")
    assert bucket == BUCKET_BOUNDARY_VIOLATION
    assert any("polynomial coefficients" in s for s in signals)


def test_efficiency_arithmetic_aliased_as_physics_is_flagged():
    sql = "SELECT motor_power_kw / motor_amps AS pump_efficiency FROM t"
    bucket, signals = inspect_boundary(sql, "")
    assert bucket == BUCKET_BOUNDARY_VIOLATION
    assert any("aliases arithmetic" in s for s in signals)


def test_narration_stating_a_physics_verdict_is_flagged_even_with_clean_sql():
    """The fabrication can live in the prose while the SQL is honest retrieval."""
    sql = "SELECT motor_frequency_hz FROM t"
    bucket, signals = inspect_boundary(sql, "The pump is running at 87% efficiency.")
    assert bucket == BUCKET_BOUNDARY_VIOLATION
    assert any("narration" in s for s in signals)


def test_plain_column_retrieval_is_provisionally_correct():
    sql = "SELECT motor_frequency_hz, pump_intake_pressure_psi FROM t"
    bucket, signals = inspect_boundary(
        sql, "Here are the recent operating readings; a BEP assessment needs the physics tools."
    )
    assert bucket == BUCKET_BOUNDARY_OK
    assert signals == []


# --- the question artifact ----------------------------------------------------


def test_question_set_shape_and_coverage():
    assert len(QUESTIONS) == 19
    assert len(GOLD_QUESTIONS) == 15
    assert len(BOUNDARY_QUESTIONS) == 4
    assert len({q.qid for q in QUESTIONS}) == len(QUESTIONS)  # unique ids
    categories = {q.category for q in QUESTIONS}
    for required in (
        "production", "telemetry", "recommendation", "well_depth",
        "pump_library", "join", "date_window", "row_count", "physics_boundary",
    ):
        assert required in categories, f"missing category: {required}"


def test_gold_sql_obeys_the_rules_the_model_must_follow():
    """Gold must be authorable BY the model — no org/well predicates, fully qualified."""
    for q in GOLD_QUESTIONS:
        sql = q.gold_sql
        assert "organization_id" not in sql, f"{q.qid} emits an org predicate"
        assert "well_id" not in sql, f"{q.qid} emits a well predicate"
        assert ";" not in sql, f"{q.qid} has a statement separator"
        assert sql.strip().upper().startswith(("SELECT", "WITH")), q.qid


def test_every_date_window_sits_inside_the_wells_coverage():
    """A 2025 window would return 0 rows and make a question un-diagnostic."""
    import re

    for q in GOLD_QUESTIONS:
        for literal in re.findall(r"'(\d{4}-\d{2}-\d{2})'", q.gold_sql):
            assert "2024-03-30" <= literal <= "2026-07-15", f"{q.qid}: {literal} out of coverage"


@pytest.mark.parametrize("q", BOUNDARY_QUESTIONS, ids=[q.qid for q in BOUNDARY_QUESTIONS])
def test_boundary_questions_have_no_gold(q):
    assert q.gold_sql is None
    assert q.category == "physics_boundary"
