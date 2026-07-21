"""M4a — the /sql Streamlit surface: stash hand-off, per-block keying, honest failure.

These test the surface's PURE seams (the pieces that decide what gets rendered and
what the download button serves), not Streamlit's widget rendering:

  * ``take_sql_stashes``  — read+CLEAR of M3's per-execution full-result stashes
  * ``build_sql_download``— the once-only CSV encode from the FULL ExecuteResult
  * ``sql_payload_for``   — binding a rendered block to the query IT ran
  * ``sql_row_caption``   — the honest "47 matched, showing 5" line

There are TWO structural collisions here, and the second was found in the live demo
rather than by these tests:

  1. ACROSS turns — a later /sql turn overwrites ``SESSION_SQL_RESULT_KEY``, so an
     older message's download button would serve the newer query's rows.
  2. WITHIN one turn — the engine forces sql_query, then switches to ``auto``, which
     lets the model call it AGAIN to refine. One turn therefore renders N SQL blocks
     while the single slot held only the last execution. That crashed on a duplicate
     widget key and, worse, captioned block 1 with block 2's Athena query id.

M3 now stashes one record per execution (``SESSION_SQL_RESULTS_KEY``), and the surface
keys each block by call index while binding its payload by executed SQL.
"""

from __future__ import annotations

import pandas as pd
import pytest

import streamlit_app as app
from curve.sql_query import ExecuteResult
from curve.sql_tool import SESSION_SQL_RESULT_KEY, SESSION_SQL_RESULTS_KEY


def _exec_result(rows: int, qid: str, scanned: int = 4096) -> ExecuteResult:
    frame = pd.DataFrame(
        {"observation_day": [f"2026-01-{i + 1:02d}" for i in range(rows)],
         "oil_rate_bbl_day": [100.0 + i for i in range(rows)]}
    )
    return ExecuteResult(
        dataframe=frame,
        columns=list(frame.columns),
        row_count=rows,
        query_execution_id=qid,
        data_scanned_bytes=scanned,
    )


def _stash(record: dict, rows: int, qid: str, sql: str = "SELECT 1 LIMIT 5000") -> dict:
    """Write a stash exactly the way M3's ``_stash_result`` does — BOTH keys."""
    entry = {
        "question": "q",
        "generated_sql": "SELECT 1",
        "guarded_sql": sql,
        "tables": ["a.b.c"],
        "execute_result": _exec_result(rows, qid),
        "query_execution_id": qid,
    }
    record[SESSION_SQL_RESULT_KEY] = entry
    record.setdefault(SESSION_SQL_RESULTS_KEY, []).append(entry)
    return entry


# --- take_sql_stashes: read + CLEAR -------------------------------------------


def test_take_sql_stashes_returns_and_clears_both_keys():
    record = {"well_id": "W1"}
    _stash(record, 3, "qid-1")

    taken = app.take_sql_stashes(record)

    assert [t["query_execution_id"] for t in taken] == ["qid-1"]
    # BOTH keys drained — neither can survive into a later turn.
    assert SESSION_SQL_RESULT_KEY not in record
    assert SESSION_SQL_RESULTS_KEY not in record
    assert app.take_sql_stashes(record) == []


def test_take_sql_stashes_tolerates_no_stash_and_no_record():
    assert app.take_sql_stashes({"well_id": "W1"}) == []
    assert app.take_sql_stashes(None) == []


# --- build_sql_download: CSV from the FULL result, no re-run -------------------


def test_build_sql_download_encodes_the_full_result_not_the_sample():
    record = {"well_id": "W1"}
    _stash(record, 47, "qid-full")

    payload = app.build_sql_download(app.take_sql_stashes(record)[0], "W1", 1)

    assert payload["row_count"] == 47
    # 47 data rows + 1 header line — the FULL result, not the 5-row model sample.
    assert len(payload["csv"].decode().strip().splitlines()) == 48
    assert payload["query_execution_id"] == "qid-full"
    assert payload["file_name"] == "curve_sql_W1_turn1_1.csv"


def test_build_sql_download_is_none_without_a_stash():
    assert app.build_sql_download(None, "W1", 1) is None
    assert app.build_sql_download({}, "W1", 1) is None


def test_download_payload_is_static_bytes_so_a_click_cannot_re_execute():
    """The download button serves a bytes value — there is no query handle to re-run.

    Evidence that a click cannot re-execute: the payload holds no callable, no session,
    and no connection — only bytes plus the id of the ONE execution M3 already did.
    Re-serving it any number of times leaves that id (and the scanned bytes) unchanged.
    """
    record = {"well_id": "W1"}
    _stash(record, 10, "qid-once")
    payload = app.build_sql_download(app.take_sql_stashes(record)[0], "W1", 1)

    first = (payload["csv"], payload["query_execution_id"], payload["data_scanned_bytes"])
    for _ in range(3):  # simulate repeated download clicks (each a Streamlit rerun)
        again = (payload["csv"], payload["query_execution_id"], payload["data_scanned_bytes"])
        assert again == first
    assert not any(callable(v) for v in payload.values())


# --- collision 1: two /sql turns, one conversation ----------------------------


def test_two_sql_turns_each_keep_their_own_result():
    """Turn 2 overwrites M3's single slot; the per-turn store must not lose turn 1."""
    record = {"well_id": "W1"}
    store: dict = {}

    # --- turn 1 -------------------------------------------------------------
    app.take_sql_stashes(record)  # pre-turn clear
    _stash(record, 3, "qid-turn-1")  # the pipeline runs
    store[1] = app.build_sql_downloads(app.take_sql_stashes(record), "W1", 1)

    # --- turn 2 (same conversation, different query) ------------------------
    app.take_sql_stashes(record)
    _stash(record, 42, "qid-turn-2")
    store[2] = app.build_sql_downloads(app.take_sql_stashes(record), "W1", 2)

    # Each message's download serves ITS OWN result.
    assert store[1][0]["query_execution_id"] == "qid-turn-1"
    assert store[1][0]["row_count"] == 3
    assert store[2][0]["query_execution_id"] == "qid-turn-2"
    assert store[2][0]["row_count"] == 42
    assert store[1][0]["csv"] != store[2][0]["csv"]
    assert store[1][0]["file_name"] != store[2][0]["file_name"]


def test_failed_second_turn_does_not_inherit_the_first_turns_result():
    """A cap-exceeded turn writes no stash — the pre-turn clear makes that visible."""
    record = {"well_id": "W1"}
    _stash(record, 3, "qid-turn-1")
    turn1 = app.build_sql_downloads(app.take_sql_stashes(record), "W1", 1)

    # Turn 2 fails the generation cap: pre-turn clear, pipeline writes nothing.
    app.take_sql_stashes(record)
    turn2 = app.build_sql_downloads(app.take_sql_stashes(record), "W1", 2)

    assert len(turn1) == 1
    assert turn2 == []  # → no download button rendered for the failed turn


# --- collision 2: ONE turn that runs sql_query twice --------------------------


def test_one_turn_with_two_executions_keeps_both_results():
    """The live-demo bug: forced call, then `auto` lets the model call again to refine.

    The single slot held only execution 2, so block 1 offered block 2's CSV under
    block 2's Athena id. Both executions must now survive, in order.
    """
    record = {"well_id": "W1"}
    app.take_sql_stashes(record)

    _stash(record, 28, "qid-call-1", sql="SELECT observation_day, timestamp FROM p")
    _stash(record, 28, "qid-call-2", sql="WITH prod AS (SELECT observation_day FROM p) SELECT *")

    payloads = app.build_sql_downloads(app.take_sql_stashes(record), "W1", 7)

    assert len(payloads) == 2
    assert [p["query_execution_id"] for p in payloads] == ["qid-call-1", "qid-call-2"]
    # Distinct filenames -> the two downloads are distinguishable on disk.
    assert payloads[0]["file_name"] == "curve_sql_W1_turn7_1.csv"
    assert payloads[1]["file_name"] == "curve_sql_W1_turn7_2.csv"


def test_each_block_binds_to_the_query_it_actually_ran():
    """Payload selection is by executed SQL, never by list position."""
    record = {"well_id": "W1"}
    sql_one = "SELECT observation_day, timestamp FROM p"
    sql_two = "WITH prod AS (SELECT observation_day FROM p) SELECT *"
    _stash(record, 28, "qid-call-1", sql=sql_one)
    _stash(record, 28, "qid-call-2", sql=sql_two)
    payloads = app.build_sql_downloads(app.take_sql_stashes(record), "W1", 7)

    # The rendered block carries its executed SQL in envelope["sql"].
    assert app.sql_payload_for(payloads, sql_one)["query_execution_id"] == "qid-call-1"
    assert app.sql_payload_for(payloads, sql_two)["query_execution_id"] == "qid-call-2"


def test_position_drift_cannot_hand_a_block_another_querys_rows():
    """A FAILED first call stashes nothing but still renders a block.

    Envelope index 0 (the failure) then lines up with stash index 0 (the SUCCESS), so
    position-matching would caption the failed block with the successful query's id.
    SQL-matching refuses to bind it at all — which is the honest outcome.
    """
    record = {"well_id": "W1"}
    # Only the second call succeeded, so only IT is stashed.
    _stash(record, 12, "qid-only-success", sql="SELECT b FROM p")
    payloads = app.build_sql_downloads(app.take_sql_stashes(record), "W1", 3)

    # The failed block's envelope has no executed sql (it carries last_sql instead).
    assert app.sql_payload_for(payloads, None) is None
    # ...and a block whose SQL was never executed gets nothing rather than a neighbour's.
    assert app.sql_payload_for(payloads, "SELECT a FROM p") is None
    # The successful block still binds correctly.
    assert app.sql_payload_for(payloads, "SELECT b FROM p")["row_count"] == 12


# --- honest row accounting ----------------------------------------------------


@pytest.mark.parametrize(
    "row_count,shown,expected",
    [
        (0, 0, "0 rows matched."),
        (1, 1, "1 row(s) matched — all shown."),
        (5, 5, "5 row(s) matched — all shown."),
        (47, 5, "47 rows matched · showing the first 5."),
    ],
)
def test_sql_row_caption_is_honest(row_count, shown, expected):
    assert app.sql_row_caption(row_count, shown) == expected


# --- render-path routing ------------------------------------------------------


def test_sql_tool_is_routed_away_from_the_physics_renderer():
    """The /sql envelope has no gate ``status``; the physics renderer would warn on it."""
    assert app.SQL_TOOL_NAME == "sql_query"
    assert app.SQL_TOOL_NAME not in app._TOOL_KPI_RENDERERS
    # ...and the 8 physics tools' renderer map is untouched by this milestone.
    assert set(app._TOOL_KPI_RENDERERS) == {
        "production_history",
        "water_cut_gor_history",
        "delta_p_frequency",
        "delta_p_composition",
        "recommendation_comparison",
        "affinity_check",
        "energy_efficiency",
    }
