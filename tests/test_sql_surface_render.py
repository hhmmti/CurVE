"""M4a — the rendered /sql block, driven through Streamlit's own AppTest harness.

Complements ``test_sql_surface.py`` (pure seams) by running the block as a real
Streamlit script: what elements appear, whether "Show query" is collapsed, which SQL it
shows, and — the DoD item — that a download CLICK (a full script rerun) executes nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from curve import sql_query as sql_query_module

DRIVER = str(Path(__file__).parent / "fixtures" / "sql_surface_driver.py")


@pytest.fixture(autouse=True)
def _restore_execute():
    """The driver wraps ``curve.sql_query.execute`` to count calls — put it back."""
    original = sql_query_module.execute
    yield
    sql_query_module.execute = original


def _run(mode: str) -> AppTest:
    at = AppTest.from_file(DRIVER, default_timeout=30)
    at.session_state["mode"] = mode
    at.run()
    assert not at.exception, at.exception
    return at


def _expanders(at: AppTest):
    return [(e.label, e.proto.expanded) for e in at.expander]


# --- success turn -------------------------------------------------------------


def test_success_turn_renders_top5_rowcount_collapsed_sql_and_download():
    at = _run("success")

    # B7: exactly one inline table, showing the top 5 of 47.
    assert len(at.dataframe) == 1
    assert at.dataframe[0].value.shape == (5, 2)

    # Honest row accounting.
    captions = [c.value for c in at.caption]
    assert any("47 rows matched · showing the first 5." == c for c in captions)

    # C10: the expander exists and is DEFAULT-COLLAPSED.
    assert _expanders(at) == [("Show query", False)]

    # B7/C10: it shows the EXECUTED (guarded/injected) SQL — scope + cap visible —
    # and not the pre-guard generated SQL.
    shown_sql = at.code[0].value
    assert "organization_id = 'ORG'" in shown_sql and "well_id = 'W1'" in shown_sql
    assert "LIMIT 5000" in shown_sql
    assert shown_sql != "SELECT observation_day, oil_rate_bbl_day FROM esp_production_preprocessed"

    # B7: the CSV download is present.
    downloads = at.get("download_button")
    assert len(downloads) == 1
    assert downloads[0].label == "⬇ Download full result (CSV)"


def test_sql_block_carries_no_trust_label_and_no_kpi_cards():
    """Requirement 5: retrieval must not read as validated physics."""
    at = _run("success")

    assert len(at.get("metric")) == 0  # no physics-style KPI cards
    rendered = " ".join(
        [c.value for c in at.caption]
        + [m.value for m in at.markdown]
        + [i.value for i in at.info]
    )
    for badge in ("✅ Validated", "🟡 Estimated", "🟣 Proxy", "🔬 Research prototype"):
        assert badge not in rendered
    assert "Not a physics validation" in rendered


# --- the DoD evidence: a download click does not re-execute -------------------


def test_download_click_reruns_the_script_without_executing_the_query():
    """AppTest exposes no ``click()`` for ``download_button``, so this drives the thing
    a click actually causes — a full script rerun with session state preserved — and
    checks that nothing is re-executed and the same bytes come back."""
    at = _run("success")
    assert at.session_state["run_count"] == 1
    assert at.session_state["execute_calls"] == []

    # Streamlit serves the download from its media-file manager, keyed by a hash of the
    # bytes — so an unchanged URL across the rerun means unchanged, re-served bytes.
    before_url = at.get("download_button")[0].proto.url
    before_csv = at.session_state["_curve_sql_results"][7][0]["csv"]

    at.run()  # the rerun a download click triggers
    assert not at.exception, at.exception

    # The script DID rerun…
    assert at.session_state["run_count"] == 2
    # …and executed nothing: no Athena call was made to serve the CSV.
    assert at.session_state["execute_calls"] == []
    # The served result is byte-identical, and the Athena execution id is unchanged —
    # still the id of the single M3 execution.
    assert at.get("download_button")[0].proto.url == before_url
    payload = at.session_state["_curve_sql_results"][7][0]
    assert payload["csv"] == before_csv
    assert payload["query_execution_id"] == "qid-executed-once"
    assert payload["data_scanned_bytes"] == 8192
    assert len(payload["csv"].decode().strip().splitlines()) == 48  # full 47 rows


# --- regression: ONE turn, TWO sql_query executions ---------------------------


def test_two_executions_in_one_turn_render_without_a_duplicate_widget_key():
    """The live-demo crash: DuplicateWidgetID on key='sql_download_7'.

    The engine forces sql_query, then flips to ``auto``, which let the model call it a
    second time to refine its query. Both envelopes render, and both used to ask for the
    same download-button key — a hard crash that killed the page mid-render.
    """
    at = _run("two_calls")  # _run already asserts no exception was raised

    # Both blocks rendered: two tables, two "Show query" expanders, two downloads.
    assert len(at.dataframe) == 2
    assert _expanders(at) == [("Show query", False), ("Show query", False)]
    downloads = at.get("download_button")
    assert len(downloads) == 2
    # The keys are distinct — this is what the crash was about.
    assert downloads[0].proto.id != downloads[1].proto.id


def test_each_block_downloads_its_own_query_not_the_last_one():
    """The silent half of the bug: block 1 was captioned with block 2's Athena id."""
    at = _run("two_calls")

    payloads = at.session_state["_curve_sql_results"][7]
    assert [p["query_execution_id"] for p in payloads] == ["qid-call-1", "qid-call-2"]

    # Block 1's caption names call 1; block 2's names call 2. Previously BOTH said
    # qid-call-2, because both read the same single-slot payload.
    captions = [c.value for c in at.caption]
    assert any("qid-call-1" in c for c in captions)
    assert any("qid-call-2" in c for c in captions)

    # The two blocks serve DIFFERENT bytes (call 1 has a timestamp column, call 2 does not).
    assert payloads[0]["csv"] != payloads[1]["csv"]
    assert b"timestamp" in payloads[0]["csv"]
    assert b"timestamp" not in payloads[1]["csv"]


# --- honest failure -----------------------------------------------------------


def test_failure_turn_renders_reason_and_last_sql_with_no_table_or_download():
    at = _run("failure")

    # The reason is stated plainly.
    assert len(at.error) == 1
    body = at.error[0].value
    assert "could not produce a valid scoped query" in body
    assert "table_not_allowed" in body

    # The last attempted SQL is in the (still collapsed) expander.
    assert _expanders(at) == [("Show query", False)]
    assert at.code[0].value == "SELECT * FROM not_allowed.some_table"

    # Nothing that could read as a result.
    assert len(at.dataframe) == 0
    assert len(at.get("download_button")) == 0
    assert len(at.get("metric")) == 0
    # No empty-looking success ("0 rows matched" would imply the query ran).
    assert not any("matched" in c.value for c in at.caption)
