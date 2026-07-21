"""AppTest driver — renders the M4a /sql block in a REAL Streamlit script run.

Used by ``tests/test_sql_surface_render.py`` to exercise the rendered block (and the
download-click rerun) without needing AWS, a well, or a Bedrock turn. ``curve.sql_query
.execute`` is counted here so the test can prove a download click executes nothing.
"""

import pandas as pd
import streamlit as st

import streamlit_app as app
from curve import sql_query as sql_query_module
from curve.sql_query import ExecuteResult

# Count every execute() call made during this script run. A download click reruns the
# whole script; if the CSV were served by re-querying, this would go above zero.
# Patched once (guarded), and the test restores the original — no leak into the process.
_EXECUTE_CALLS = st.session_state.setdefault("execute_calls", [])
if not getattr(sql_query_module.execute, "_curve_counted", False):
    _real_execute = sql_query_module.execute

    def _counting_execute(*args, **kwargs):
        _EXECUTE_CALLS.append(1)
        return _real_execute(*args, **kwargs)

    _counting_execute._curve_counted = True  # type: ignore[attr-defined]
    sql_query_module.execute = _counting_execute  # type: ignore[assignment]

st.session_state["run_count"] = st.session_state.get("run_count", 0) + 1
mode = st.session_state.get("mode", "success")

_SQL = (
    "SELECT observation_day, oil_rate_bbl_day\n"
    "FROM roam_prd_products.esp_optimization_v2.esp_production_preprocessed\n"
    "WHERE organization_id = 'ORG' AND well_id = 'W1'\n"
    "LIMIT 5000"
)

if mode == "two_calls":
    # Reproduces the live-demo turn: ONE question, TWO sql_query executions. Renders
    # through _render_answer so the real per-block call-index keying is exercised —
    # this is the path that raised DuplicateWidgetID on key='sql_download_7'.
    frames = [
        pd.DataFrame({"observation_day": ["2026-06-01"], "timestamp": ["2026-06-01 00:00:00"],
                      "alloc_oil_vol": [0.0]}),
        pd.DataFrame({"observation_day": ["2026-06-01"], "alloc_oil_vol": [0.0]}),
    ]
    sqls = [
        "SELECT observation_day, timestamp, alloc_oil_vol FROM p LIMIT 10000",
        "WITH prod AS (SELECT observation_day, alloc_oil_vol FROM p) SELECT * FROM prod LIMIT 10000",
    ]
    store = st.session_state.setdefault(app.SQL_RESULTS_STATE_KEY, {})
    if 7 not in store:
        stashes = [
            {
                "guarded_sql": sqls[i],
                "execute_result": ExecuteResult(
                    dataframe=frames[i], columns=list(frames[i].columns), row_count=28,
                    query_execution_id=f"qid-call-{i + 1}", data_scanned_bytes=8064,
                ),
            }
            for i in range(2)
        ]
        store[7] = app.build_sql_downloads(stashes, "W1", 7)

    entry = {
        "role": "assistant",
        "narration": "Here is the day-by-day June production.",
        "tool_outputs": [
            {"name": "sql_query", "result": {
                "sql": sqls[i], "generated_sql": sqls[i],
                "columns": list(frames[i].columns), "row_count": 28,
                "sample_rows": frames[i].to_dict("records"),
            }}
            for i in range(2)
        ],
        "elapsed_s": 1.0,
        "sql_turn_id": 7,
    }
    app._render_answer(entry, dev_mode=False)

elif mode == "failure":
    app._render_sql_result(
        {
            "error": "sql_query could not produce a valid scoped query within 2 generations.",
            "last_sql": "SELECT * FROM not_allowed.some_table",
            "last_reason": "guard rejected [table_not_allowed]: table is not in the allowlist",
        },
        9,
    )
else:
    frame = pd.DataFrame(
        {
            "observation_day": [f"2026-01-{i + 1:02d}" for i in range(47)],
            "oil_rate_bbl_day": [100.0 + i for i in range(47)],
        }
    )
    result = ExecuteResult(
        dataframe=frame,
        columns=list(frame.columns),
        row_count=47,
        query_execution_id="qid-executed-once",
        data_scanned_bytes=8192,
    )
    # Encode once, on the FIRST run only — mirroring the app, where the turn handler
    # (not the render path) builds the payload. Later reruns must reuse these bytes.
    store = st.session_state.setdefault(app.SQL_RESULTS_STATE_KEY, {})
    if 7 not in store:
        store[7] = app.build_sql_downloads(
            [{"guarded_sql": _SQL, "execute_result": result}], "W1", 7
        )

    app._render_sql_result(
        {
            "sql": _SQL,
            "generated_sql": "SELECT observation_day, oil_rate_bbl_day FROM esp_production_preprocessed",
            "columns": list(frame.columns),
            "row_count": 47,
            "sample_rows": frame.head(5).to_dict("records"),
        },
        7,
    )
