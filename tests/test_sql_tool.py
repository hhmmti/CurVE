"""CurVE M3 tests — the sql_query pipeline, its tool registration, and /sql gating.

Runs with NO AWS credentials: the pipeline's Athena boundary (M2's guard / explain /
execute) is monkeypatched, and the nested generate call is injected. Covers:

  * generation extraction + JSON-safe sampling
  * validate → self-correct (guard reject AND EXPLAIN fail) with a 2-generation cap
  * honest-failure payload on cap-exceeded; execute-once + stash on success
  * the 9th-tool registration (sql_query absent from the v1 8-tool config)
  * the two-config /sql topology (forced generation turn → auto narration turn)
  * REGRESSION: the non-/sql toolConfig is byte-identical to v1 and a physics
    question still routes to its physics tool untouched
"""

import copy
from decimal import Decimal

import pandas as pd
import pytest

from curve import engine, sql_tool
from curve.engine import run_curve_turn
from curve.sql_query import ExecuteResult, GuardedSQL, GuardResult, Rejection
from curve.tools import (
    TOOL_REGISTRY,
    build_sql_tool_config,
    build_tool_config,
)

SESSION = {
    "session_id": "sess-sql",
    "organization_id": "org-acme",
    "well_id": "W-12",
    "availability": {"coverage": {"min_day": "2024-01-01", "max_day": "2024-12-31"}},
}


# --- scripted Converse plumbing (records tool_config per call) -----------------


def _end_turn(text="done"):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


def _tool_use_turn(name, tool_input, tool_use_id="tu-1"):
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}}
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


class RecordingWrapper:
    """Replays scripted responses; records the messages + tool_config of each call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.messages_seen = []
        self.tool_configs = []

    def converse(self, messages, system, tool_config=None):
        self.messages_seen.append(copy.deepcopy(messages))
        self.tool_configs.append(copy.deepcopy(tool_config))
        return self._responses.pop(0) if self._responses else _end_turn("(dry)")


# --- extraction + jsonify -----------------------------------------------------


def test_extract_sql_handles_sql_fence():
    txt = "Here you go:\n```sql\nSELECT 1 FROM t\n```\nHope that helps"
    assert sql_tool.extract_sql(txt) == "SELECT 1 FROM t"


def test_extract_sql_handles_bare_fence_and_prose():
    assert sql_tool.extract_sql("```\nSELECT a FROM t;\n```") == "SELECT a FROM t"
    assert sql_tool.extract_sql("The query is SELECT a FROM t;") == "SELECT a FROM t"
    assert sql_tool.extract_sql("WITH c AS (SELECT 1) SELECT * FROM c") == (
        "WITH c AS (SELECT 1) SELECT * FROM c"
    )


def test_extract_sql_empty_when_no_sql():
    assert sql_tool.extract_sql("I cannot answer that.") == "I cannot answer that."
    assert sql_tool.extract_sql("") == ""


def test_jsonify_coerces_decimal_nan_and_timestamp():
    assert sql_tool._jsonify(Decimal("12.5")) == 12.5
    assert sql_tool._jsonify(float("nan")) is None
    assert sql_tool._jsonify(pd.NaT) is None
    assert isinstance(sql_tool._jsonify(pd.Timestamp("2024-01-01")), str)
    assert sql_tool._jsonify(None) is None
    assert sql_tool._jsonify("x") == "x"
    assert sql_tool._jsonify(3) == 3


# --- guard/explain/execute doubles --------------------------------------------


def _ok_guard(sql="SELECT 1 FROM t", tables=("cat.db.t",)):
    return GuardResult(ok=True, guarded=GuardedSQL(sql=sql, tables=tables))


def _reject(code="model_supplied_scope", reason="model emitted a 'well_id' predicate"):
    return GuardResult(ok=False, rejection=Rejection(code=code, reason=reason))


def _exec_result(rows=7):
    df = pd.DataFrame({"oil": [float(i) for i in range(rows)]})
    return ExecuteResult(
        dataframe=df, columns=["oil"], row_count=rows, query_execution_id="qid-1"
    )


# --- pipeline: success --------------------------------------------------------


def test_pipeline_success_first_attempt(monkeypatch):
    monkeypatch.setattr(sql_tool, "guard", lambda sql, session: _ok_guard(sql="GUARDED SQL"))
    monkeypatch.setattr(sql_tool, "explain", lambda g: {"ok": True, "error": None})
    calls = {"execute": 0}

    def _exec(g):
        calls["execute"] += 1
        return _exec_result(rows=7)

    monkeypatch.setattr(sql_tool, "execute", _exec)

    session = copy.deepcopy(SESSION)
    out = sql_tool.run_sql_pipeline(
        "how much oil", session, generate=lambda q, fb: "SELECT sum(oil) FROM t"
    )

    assert out["sql"] == "GUARDED SQL"
    assert out["columns"] == ["oil"]
    assert out["row_count"] == 7
    assert len(out["sample_rows"]) == 5  # top-5 only
    assert out["attempts"] == 1 and out["self_corrected"] is False
    assert calls["execute"] == 1  # executed exactly once
    # full ExecuteResult stashed on the session for M4 (no re-run).
    stash = session[sql_tool.SESSION_SQL_RESULT_KEY]
    assert stash["guarded_sql"] == "GUARDED SQL"
    assert isinstance(stash["execute_result"], ExecuteResult)


# --- pipeline: self-correct ---------------------------------------------------


def test_pipeline_self_corrects_on_guard_rejection(monkeypatch):
    guard_results = [_reject(), _ok_guard(sql="FIXED SQL")]
    monkeypatch.setattr(sql_tool, "guard", lambda sql, session: guard_results.pop(0))
    monkeypatch.setattr(sql_tool, "explain", lambda g: {"ok": True, "error": None})
    monkeypatch.setattr(sql_tool, "execute", lambda g: _exec_result(rows=2))

    seen = []

    def _gen(q, feedback):
        seen.append(feedback)
        return "JOIN ON well_id BAD" if len(seen) == 1 else "GOOD"

    out = sql_tool.run_sql_pipeline("q", copy.deepcopy(SESSION), generate=_gen)

    assert out["sql"] == "FIXED SQL"
    assert out["attempts"] == 2 and out["self_corrected"] is True
    assert seen[0] is None  # first attempt has no feedback
    # the rejection reason + offending SQL are fed back into the single regeneration.
    assert "model_supplied_scope" in seen[1] and "well_id" in seen[1]
    assert "JOIN ON well_id BAD" in seen[1]


def test_pipeline_self_corrects_on_explain_failure(monkeypatch):
    monkeypatch.setattr(sql_tool, "guard", lambda sql, session: _ok_guard(sql=f"G::{sql}"))
    explains = [{"ok": False, "error": "COLUMN_NOT_FOUND: foo"}, {"ok": True, "error": None}]
    monkeypatch.setattr(sql_tool, "explain", lambda g: explains.pop(0))
    monkeypatch.setattr(sql_tool, "execute", lambda g: _exec_result(rows=1))

    seen = []
    monkeypatch.setattr(
        sql_tool, "extract_sql", lambda t: t
    )  # generate returns clean sql already

    def _gen(q, feedback):
        seen.append(feedback)
        return "first" if not seen[:-1] else "second"

    out = sql_tool.run_sql_pipeline("q", copy.deepcopy(SESSION), generate=_gen)
    assert out["attempts"] == 2 and out["self_corrected"] is True
    assert "EXPLAIN rejected" in seen[1] and "COLUMN_NOT_FOUND" in seen[1]


# --- pipeline: honest failure -------------------------------------------------


def test_pipeline_cap_exceeded_returns_honest_failure(monkeypatch):
    monkeypatch.setattr(sql_tool, "guard", lambda sql, session: _reject())
    monkeypatch.setattr(sql_tool, "explain", lambda g: pytest.fail("explain must not run"))
    monkeypatch.setattr(sql_tool, "execute", lambda g: pytest.fail("execute must not run"))

    out = sql_tool.run_sql_pipeline(
        "q", copy.deepcopy(SESSION), generate=lambda q, fb: "STILL BAD SQL"
    )

    assert "error" in out
    assert "2 generations" in out["error"]
    assert out["last_sql"] == "STILL BAD SQL"
    assert "model_supplied_scope" in out["last_reason"]
    assert "sql" not in out and "row_count" not in out  # no fabricated success


def test_pipeline_never_executes_when_all_rejected(monkeypatch):
    monkeypatch.setattr(sql_tool, "guard", lambda sql, session: _reject())
    calls = {"execute": 0}
    monkeypatch.setattr(sql_tool, "execute", lambda g: calls.__setitem__("execute", 1))
    session = copy.deepcopy(SESSION)
    sql_tool.run_sql_pipeline("q", session, generate=lambda q, fb: "bad")
    assert calls["execute"] == 0
    assert sql_tool.SESSION_SQL_RESULT_KEY not in session  # nothing stashed on failure


# --- tool fn guards -----------------------------------------------------------


def test_sql_query_tool_requires_question():
    assert "error" in sql_tool.sql_query({"session": SESSION, "question": "  "})


def test_sql_query_tool_requires_scoped_session():
    out = sql_tool.sql_query({"question": "how much oil", "session": None})
    assert "error" in out and "session" in out["error"]


# --- registration + the two-config topology -----------------------------------


def test_registry_is_eight_and_excludes_sql_query():
    assert len(TOOL_REGISTRY) == 8
    assert "sql_query" not in TOOL_REGISTRY


def test_non_sql_toolconfig_is_v1_eight_tools():
    cfg = build_tool_config(TOOL_REGISTRY)
    names = [t["toolSpec"]["name"] for t in cfg["tools"]]
    assert len(names) == 8
    assert "sql_query" not in names
    assert "toolChoice" not in cfg  # v1 default — never forces


def test_build_sql_tool_config_forced_and_auto_shapes():
    forced = build_sql_tool_config(force=True)
    assert [t["toolSpec"]["name"] for t in forced["tools"]] == ["sql_query"]
    assert forced["toolChoice"] == {"tool": {"name": "sql_query"}}
    auto = build_sql_tool_config(force=False)
    assert auto["toolChoice"] == {"auto": {}}


# --- engine /sql gating -------------------------------------------------------


def test_non_sql_turn_uses_untouched_config(monkeypatch):
    """REGRESSION: a normal question uses the exact v1 8-tool config, no forcing."""
    wrapper = RecordingWrapper([_end_turn("hi")])
    out = run_curve_turn("how has this well produced?", wrapper=wrapper, session=SESSION)
    assert wrapper.tool_configs[0] == build_tool_config(TOOL_REGISTRY)
    assert "toolChoice" not in wrapper.tool_configs[0]
    # question not prefix-stripped
    assert wrapper.messages_seen[0][0]["content"][0]["text"] == "how has this well produced?"
    assert out["tool_trace"] == []


def test_non_sql_question_routes_to_its_physics_tool_untouched():
    """REGRESSION: a physics question still dispatches to its physics tool."""
    ran = {}
    stub_registry = {
        "production_history": {
            "spec": {"toolSpec": {"name": "production_history", "description": "d",
                                  "inputSchema": {"json": {"type": "object", "properties": {}}}}},
            "fn": lambda ti: ran.setdefault("input", ti) or {"values": {"ok": True}},
        }
    }
    wrapper = RecordingWrapper(
        [_tool_use_turn("production_history", {"start_date": "2024-01-01"}), _end_turn("ok")]
    )
    out = run_curve_turn(
        "how has this well produced?", wrapper=wrapper, tools=stub_registry, session=SESSION
    )
    assert out["tool_trace"] == ["production_history"]
    # no toolChoice forced on any turn of a non-/sql question
    assert all("toolChoice" not in (c or {}) for c in wrapper.tool_configs)
    # the physics tool got org/well injected but NOT a 'session' blob (that is sql-only)
    assert "session" not in ran["input"] and ran["input"]["well_id"] == "W-12"


def test_sql_prefix_strips_forces_then_narrates(monkeypatch):
    """/sql: prefix stripped; turn-0 forced sql_query; turn-1 auto narration; session injected."""
    captured = {}

    def _stub_fn(tool_input):
        captured["input"] = tool_input
        return {"sql": "SELECT 1", "columns": ["c"], "row_count": 1, "sample_rows": [{"c": 1}]}

    monkeypatch.setattr(
        engine, "SQL_QUERY_ENTRY", {"spec": {"toolSpec": {"name": "sql_query"}}, "fn": _stub_fn}
    )

    wrapper = RecordingWrapper(
        [_tool_use_turn("sql_query", {"question": "how much oil last week?"}),
         _end_turn("Last week: 100 bbl.")]
    )
    out = run_curve_turn("/sql how much oil last week?", wrapper=wrapper, session=SESSION)

    # prefix stripped from the first user message
    assert wrapper.messages_seen[0][0]["content"][0]["text"] == "how much oil last week?"
    # turn 0 forces sql_query; turn 1 switches to auto for narration
    assert wrapper.tool_configs[0] == build_sql_tool_config(force=True)
    assert wrapper.tool_configs[1] == build_sql_tool_config(force=False)
    # the full session is injected into the sql tool (guard needs coverage + org/well)
    assert captured["input"]["session"]["well_id"] == "W-12"
    assert captured["input"]["question"] == "how much oil last week?"
    assert out["tool_trace"] == ["sql_query"]
    assert out["text"] == "Last week: 100 bbl."


def test_sql_prefix_bare_and_newline_variants(monkeypatch):
    monkeypatch.setattr(
        engine, "SQL_QUERY_ENTRY",
        {"spec": {"toolSpec": {"name": "sql_query"}}, "fn": lambda ti: {"row_count": 0}},
    )
    # "/sql\n<q>" strips to "<q>"
    wrapper = RecordingWrapper(
        [_tool_use_turn("sql_query", {"question": "count rows"}), _end_turn("0")]
    )
    run_curve_turn("/sql\ncount rows", wrapper=wrapper, session=SESSION)
    assert wrapper.messages_seen[0][0]["content"][0]["text"] == "count rows"
    assert wrapper.tool_configs[0]["toolChoice"] == {"tool": {"name": "sql_query"}}


def test_sql_substring_is_not_treated_as_prefix():
    """A question that merely contains '/sql' mid-text is NOT a /sql turn."""
    wrapper = RecordingWrapper([_end_turn("normal")])
    run_curve_turn("what does /sql do?", wrapper=wrapper, session=SESSION)
    assert "toolChoice" not in (wrapper.tool_configs[0] or {})
    assert wrapper.messages_seen[0][0]["content"][0]["text"] == "what does /sql do?"
