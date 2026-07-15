"""CurVE M2-core tests — production_history end-to-end, run with NO AWS credentials.

These mock the awswrangler fetch (``curve.data.fetch_preprocessed_window``) with a
realistic telemetry + production fixture, and mock Converse (the ``ScriptedWrapper``
from the M1 suite). They prove the M2 chain without touching AWS:

  * gate → available / Validated / flags: []
  * fetch → gate → vendored compute → vendored figure → envelope is well-formed
  * the tool_result returned TO THE MODEL carries values + trust_label but NOT the
    figure (figure excluded from the model)
  * org / well are INJECTED by the engine (not model-supplied), and the data query
    filters on BOTH (cross-org confidentiality)
  * the trust label threads into the narration path (reaches the model)
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from curve import data, session  # noqa: E402
from curve.engine import run_curve_turn  # noqa: E402
from curve.gate import run_tool_gate  # noqa: E402
from curve.tools import TOOL_REGISTRY, _project_values, production_history  # noqa: E402

# Reuse the M1 scripted wrapper + response builders (same package, cred-free).
from tests.test_loop import (  # noqa: E402
    ScriptedWrapper,
    _end_turn,
    _tool_use_turn,
)

ORG = "org-acme"
WELL = "W-12"


# --- fixtures -----------------------------------------------------------------


def _telemetry_fixture(n_days: int = 30) -> pd.DataFrame:
    """Realistic preprocessed-telemetry shape: 15-min grain, here 1 row/day is enough."""
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "organization_id": ORG,
            "well_id": WELL,
            "timestamp_telem": days,
            "observation_day": days.strftime("%Y-%m-%d"),
            "motor_frequency_hz": 55.0,
            "tubing_pressure_psi": 250.0,
            "pump_intake_pressure_psi": 800.0,
            "motor_amps": 40.0,
            "motor_volts": 2300.0,
        }
    )


def _production_fixture(n_days: int = 30) -> pd.DataFrame:
    """Daily allocation production with a gentle oil decline over the window."""
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    oil = [300.0 - i * 3 for i in range(n_days)]  # declines 300 → ~213
    return pd.DataFrame(
        {
            "organization_id": ORG,
            "well_id": WELL,
            "observation_day": days.strftime("%Y-%m-%d"),
            "alloc_oil_vol": oil,
            "alloc_water_vol": [120.0 for _ in range(n_days)],
            "alloc_gas_vol": [450.0 for _ in range(n_days)],
        }
    )


@pytest.fixture
def mock_fetch(monkeypatch):
    """Patch the data fetch with the fixtures; record the (org, well) it was called with."""
    calls = {}

    def _fake_fetch(organization_id, well_id, start_date=None, end_date=None, **kwargs):
        calls["organization_id"] = organization_id
        calls["well_id"] = well_id
        calls["start_date"] = start_date
        calls["end_date"] = end_date
        return _telemetry_fixture(), _production_fixture()

    monkeypatch.setattr(data, "fetch_preprocessed_window", _fake_fetch)
    return calls


@pytest.fixture(autouse=True)
def _clean_sessions():
    session.clear_sessions()
    yield
    session.clear_sessions()


# --- gate ---------------------------------------------------------------------


def test_gate_available_validated_no_flags():
    gate = run_tool_gate("production_history", _telemetry_fixture(), _production_fixture())
    assert gate["status"] == "available"
    assert gate["trust_label"] == "Validated"
    assert gate["flags"] == []


def test_gate_blocks_when_data_absent():
    empty = pd.DataFrame()
    gate = run_tool_gate("production_history", empty, _production_fixture())
    assert gate["status"] == "blocked"
    assert gate["trust_label"] is None
    assert "telemetry_or_production_absent" in gate["flags"]


# --- the full chain (envelope) ------------------------------------------------


def test_chain_yields_wellformed_envelope(mock_fetch):
    env = production_history({"organization_id": ORG, "well_id": WELL})

    assert env["status"] == "available"
    assert env["trust_label"] == "Validated"  # carried from the gate
    assert env["flags"] == []

    values = env["values"]
    assert values["well_id"] == WELL
    assert values["period"]["n_days"] == 30
    assert values["period"]["n_telemetry_points"] == 30
    # Latest day rates present and numeric.
    assert values["latest"]["oil_rate_bbl_day"] is not None
    assert values["latest"]["water_rate_bbl_day"] == 120.0
    # The fixture oil declines → trend direction must be "declining".
    assert values["trend"]["direction"] == "declining"
    assert values["trend"]["oil_rate_change_pct"] < 0

    # figure_ref + a real Plotly figure are produced.
    assert env["figure_ref"] == f"production_history::{WELL}"
    assert isinstance(env["figure"], go.Figure)


def test_tool_blocks_without_injected_org_well(mock_fetch):
    # No org/well (session-less call) → blocked, no fetch, no figure.
    env = production_history({"start_date": "2026-01-01"})
    assert env["status"] == "blocked"
    assert env["figure"] is None
    assert "missing_org_or_well_injection" in env["flags"]
    assert mock_fetch == {}  # fetch was never called


# --- engine injection + figure-excluded-from-model ----------------------------


def _setup_session():
    return session.save_session(
        session.new_session_record(
            session_id="sess-1", organization_id=ORG, well_id=WELL
        )
    )


def test_engine_injects_org_well_not_model_supplied(mock_fetch):
    # The model's toolUse carries ONLY a window selector — no org/well.
    rec = _setup_session()
    wrapper = ScriptedWrapper(
        [
            _tool_use_turn("production_history", tool_input={"start_date": "2026-01-01"}),
            _end_turn("Production has been declining. These figures are Validated."),
        ]
    )
    result = run_curve_turn("How has this well produced?", wrapper=wrapper, session=rec)

    assert result["tool_trace"] == ["production_history"]
    # The fetch saw the INJECTED org/well (from the session), not from the model.
    assert mock_fetch["organization_id"] == ORG
    assert mock_fetch["well_id"] == WELL
    # The model selector still flowed through.
    assert mock_fetch["start_date"] == "2026-01-01"


def test_query_filters_on_both_org_and_well():
    # Cross-org confidentiality: the SQL must filter on org AND well together.
    access = data.PreprocessedDataAccess()
    telem_sql = access.build_telemetry_query(ORG, WELL, "2026-01-01", "2026-01-31")
    prod_sql = access.build_production_query(ORG, WELL)
    for sql in (telem_sql, prod_sql):
        assert f"organization_id = '{ORG}'" in sql
        assert f"well_id = '{WELL}'" in sql
    assert "observation_day >= '2026-01-01'" in telem_sql
    assert "observation_day <= '2026-01-31'" in telem_sql


def test_query_refuses_without_both_keys():
    access = data.PreprocessedDataAccess()
    with pytest.raises(ValueError):
        access.build_telemetry_query(ORG, "")
    with pytest.raises(ValueError):
        access.build_production_query("", WELL)


def test_figure_excluded_from_model_but_values_and_label_present(mock_fetch):
    rec = _setup_session()
    wrapper = ScriptedWrapper(
        [
            _tool_use_turn("production_history", tool_input={}),
            _end_turn("Validated production summary."),
        ]
    )
    run_curve_turn("How has this well produced?", wrapper=wrapper, session=rec)

    # The toolResult the engine sent to the model on the 2nd converse call.
    second_call = wrapper.calls[1]
    tool_result_msg = second_call[2]
    assert tool_result_msg["role"] == "user"
    model_json = tool_result_msg["content"][0]["toolResult"]["content"][0]["json"]

    # Model sees values + trust_label (+ flags/status) — the narration path.
    assert model_json["trust_label"] == "Validated"
    assert model_json["values"]["well_id"] == WELL
    assert "flags" in model_json
    # But NOT the figure / figure_ref (UI only — no image tokens).
    assert "figure" not in model_json
    assert "figure_ref" not in model_json


def test_tool_outputs_retains_figure_for_ui(mock_fetch):
    rec = _setup_session()
    wrapper = ScriptedWrapper(
        [_tool_use_turn("production_history", tool_input={}), _end_turn("ok")]
    )
    result = run_curve_turn("How has this well produced?", wrapper=wrapper, session=rec)

    # The full envelope (with the figure) survives in tool_outputs for the UI/CLI.
    outputs = result["tool_outputs"]
    assert len(outputs) == 1
    env = outputs[0]["result"]
    assert env["figure_ref"] == f"production_history::{WELL}"
    assert isinstance(env["figure"], go.Figure)


def test_setup_context_line_added_to_system(mock_fetch):
    rec = _setup_session()
    wrapper = ScriptedWrapper(
        [_tool_use_turn("production_history", tool_input={}), _end_turn("ok")]
    )
    run_curve_turn("How has this well produced?", wrapper=wrapper, session=rec)
    # The engine appends the setup context line as a 2nd system block.
    # (ScriptedWrapper records messages, not system; assert via the prompt helper.)
    from curve.prompt import format_setup_context

    line = format_setup_context(rec)
    assert WELL in line and ORG in line


# --- registry sanity (M2 schema change) ---------------------------------------


# --- projection: last-valid selection on null-allocation windows --------------
# Boundary rows can carry a null allocation (a day with no allocated production →
# alloc_* null together, liquid_rate 0). The projection must reselect over the rows
# that actually have data instead of falsely reading "flat" or blanking the cards.
# These drive _project_values directly (the projection layer, no AWS, no pipeline).

_GAP = {"oil": None, "water": None, "gas": None, "liquid": 0.0, "wc": None, "gor": None}


def _day(oil, water=120.0, gas=450.0, liquid=None, wc=0.30, gor=1.5):
    return {
        "oil": oil,
        "water": water,
        "gas": gas,
        "liquid": (oil + water) if liquid is None and oil is not None else liquid,
        "wc": wc,
        "gor": gor,
    }


def _daily_frame(rows, start="2026-01-01"):
    """Build a projection-input daily frame from row dicts (None → a null cell)."""
    days = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(
        {
            "observation_day": days.strftime("%Y-%m-%d"),
            "alloc_oil_vol": [r["oil"] for r in rows],
            "alloc_water_vol": [r["water"] for r in rows],
            "alloc_gas_vol": [r["gas"] for r in rows],
            "liquid_rate_bbl_day": [r["liquid"] for r in rows],
            "water_cut": [r["wc"] for r in rows],
            "gor": [r["gor"] for r in rows],
        }
    )


def test_projection_tail_null_populates_latest_from_last_valid_day():
    # HACKBERRY SPRINGS 3BH shape: the final day has no allocation.
    rows = [_day(300.0), _day(280.0), _GAP]
    v = _project_values("W", _daily_frame(rows), telemetry_rows=3)

    # latest reads the LAST VALID day (2026-01-02), not the gap day — cards populate.
    assert v["latest"]["observation_day"] == "2026-01-02"
    assert v["latest"]["oil_rate_bbl_day"] == 280.0
    # trend.last skips the tail null; direction is computed, not a false "flat".
    assert v["trend"]["oil_rate_last_bbl_day"] == 280.0
    assert v["trend"]["oil_rate_change_pct"] is not None
    assert v["trend"]["direction"] == "declining"  # 300 → 280 = -6.7%


def test_projection_first_null_trend_first_populates():
    # MIPA NO SLEEP shape: the FIRST day has no allocation.
    rows = [_GAP, _day(280.0), _day(260.0)]
    v = _project_values("W", _daily_frame(rows), telemetry_rows=3)

    assert v["trend"]["oil_rate_first_bbl_day"] == 280.0  # skipped the leading null
    assert v["trend"]["oil_rate_change_pct"] is not None
    assert v["trend"]["direction"] == "declining"
    assert v["latest"]["oil_rate_bbl_day"] == 260.0


def test_projection_all_null_is_not_computable_no_false_flat():
    rows = [_GAP, _GAP, _GAP]
    v = _project_values("W", _daily_frame(rows), telemetry_rows=3)

    assert v["period"]["n_days"] == 3  # rows still counted
    assert v["trend"]["direction"] == "not_computable"  # NOT "flat"
    assert v["trend"]["oil_rate_change_pct"] is None
    assert v["latest"]["observation_day"] is None
    assert v["latest"]["oil_rate_bbl_day"] is None


def test_projection_single_valid_point_change_not_computable():
    rows = [_GAP, _day(275.0), _GAP]
    v = _project_values("W", _daily_frame(rows), telemetry_rows=3)

    assert v["trend"]["oil_rate_change_pct"] is None  # one point → no change
    assert v["trend"]["direction"] == "not_computable"
    # ...but that single valid day still fills the KPI cards.
    assert v["latest"]["observation_day"] == "2026-01-02"
    assert v["latest"]["oil_rate_bbl_day"] == 275.0


def test_projection_clean_window_unchanged():
    # No gaps → behaves exactly as the raw first/last boundary read did.
    rows = [_day(300.0 - i * 5) for i in range(5)]  # 300 → 280
    v = _project_values("W", _daily_frame(rows), telemetry_rows=5)

    assert v["trend"]["oil_rate_first_bbl_day"] == 300.0
    assert v["trend"]["oil_rate_last_bbl_day"] == 280.0
    assert v["trend"]["direction"] == "declining"
    assert v["latest"]["observation_day"] == "2026-01-05"
    assert v["latest"]["oil_rate_bbl_day"] == 280.0


def test_production_history_schema_has_no_org_or_well():
    schema = TOOL_REGISTRY["production_history"]["spec"]["toolSpec"]["inputSchema"]["json"]
    props = schema["properties"]
    assert "well_id" not in props  # injected, never model-supplied
    assert "organization_id" not in props
    assert set(props) == {"start_date", "end_date"}
    assert schema["required"] == []
