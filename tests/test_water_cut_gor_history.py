"""CurVE M3 tests — water_cut_gor_history end-to-end, plus the two M3 contracts it
carries: the shared error envelope and the gate field-name alias layer. Run with NO
AWS credentials (the awswrangler fetch is mocked, Converse is the scripted wrapper).

Covers:
  * gate → available / Validated / flags: [] (same Validated calcs as production_history)
  * fetch → gate → vendored compute → vendored water-cut/GOR figure → envelope
  * org/well INJECTED by the engine (not model-supplied); figure excluded from the model
  * the SHARED error envelope: water_cut_gor_history and production_history emit
    shape-identical success AND error envelopes
  * the gate ALIAS layer: GOR_scf_bbl → gor_scf_bbl, and pair-extensibility
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve import data, gate as gate_mod, session  # noqa: E402
from curve.engine import run_curve_turn  # noqa: E402
from curve.envelope import ENVELOPE_KEYS  # noqa: E402
from curve.gate import _apply_field_aliases, run_tool_gate  # noqa: E402
from curve.tools import (  # noqa: E402
    TOOL_REGISTRY,
    production_history,
    water_cut_gor_history,
)

from tests.test_loop import ScriptedWrapper, _end_turn, _tool_use_turn  # noqa: E402

ORG = "org-acme"
WELL = "W-12"


# --- fixtures (water cut watering up, GOR rising) -----------------------------


def _telemetry_fixture(n_days: int = 30) -> pd.DataFrame:
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
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    # Oil declines, water rises → water cut climbs; gas flat with declining oil → GOR rises.
    oil = [300.0 - i * 3 for i in range(n_days)]
    water = [100.0 + i * 4 for i in range(n_days)]
    return pd.DataFrame(
        {
            "organization_id": ORG,
            "well_id": WELL,
            "observation_day": days.strftime("%Y-%m-%d"),
            "alloc_oil_vol": oil,
            "alloc_water_vol": water,
            "alloc_gas_vol": [450.0 for _ in range(n_days)],
        }
    )


@pytest.fixture
def mock_fetch(monkeypatch):
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
    g = run_tool_gate("water_cut_gor_history", _telemetry_fixture(), _production_fixture())
    assert g["status"] == "available"
    assert g["trust_label"] == "Validated"
    assert g["flags"] == []


def test_gate_blocks_when_data_absent():
    g = run_tool_gate("water_cut_gor_history", pd.DataFrame(), _production_fixture())
    assert g["status"] == "blocked"
    assert g["trust_label"] is None
    assert "telemetry_or_production_absent" in g["flags"]


# --- the full chain (envelope) ------------------------------------------------


def test_chain_yields_wellformed_envelope(mock_fetch):
    env = water_cut_gor_history({"organization_id": ORG, "well_id": WELL})

    assert env["status"] == "available"
    assert env["trust_label"] == "Validated"  # carried from the gate
    assert env["flags"] == []

    values = env["values"]
    assert values["well_id"] == WELL
    assert values["period"]["n_days"] == 30
    assert values["period"]["n_telemetry_points"] == 30
    assert values["latest"]["water_cut"] is not None
    assert values["latest"]["gor"] is not None
    # Water cut climbs over the window → watering_up, positive change.
    assert values["trend"]["direction"] == "watering_up"
    assert values["trend"]["water_cut_change_pct"] > 0

    assert env["figure_ref"] == f"water_cut_gor_history::{WELL}"
    assert isinstance(env["figure"], go.Figure)


def test_tool_blocks_without_injected_org_well(mock_fetch):
    env = water_cut_gor_history({"start_date": "2026-01-01"})
    assert env["status"] == "blocked"
    assert env["figure"] is None
    assert "missing_org_or_well_injection" in env["flags"]
    assert mock_fetch == {}  # fetch never called


# --- engine injection + figure-excluded-from-model ----------------------------


def _setup_session():
    return session.save_session(
        session.new_session_record(session_id="sess-1", organization_id=ORG, well_id=WELL)
    )


def test_engine_injects_org_well_not_model_supplied(mock_fetch):
    rec = _setup_session()
    wrapper = ScriptedWrapper(
        [
            _tool_use_turn("water_cut_gor_history", tool_input={"start_date": "2026-01-01"}),
            _end_turn("Water cut is rising. These figures are Validated."),
        ]
    )
    result = run_curve_turn("How has water cut changed?", wrapper=wrapper, session=rec)

    assert result["tool_trace"] == ["water_cut_gor_history"]
    assert mock_fetch["organization_id"] == ORG
    assert mock_fetch["well_id"] == WELL
    assert mock_fetch["start_date"] == "2026-01-01"


def test_figure_excluded_from_model_but_values_and_label_present(mock_fetch):
    rec = _setup_session()
    wrapper = ScriptedWrapper(
        [
            _tool_use_turn("water_cut_gor_history", tool_input={}),
            _end_turn("Validated water-cut/GOR summary."),
        ]
    )
    run_curve_turn("How has water cut changed?", wrapper=wrapper, session=rec)

    second_call = wrapper.calls[1]
    model_json = second_call[2]["content"][0]["toolResult"]["content"][0]["json"]
    assert model_json["trust_label"] == "Validated"
    assert model_json["values"]["well_id"] == WELL
    assert "flags" in model_json
    assert "figure" not in model_json
    assert "figure_ref" not in model_json


def test_tool_outputs_retains_figure_for_ui(mock_fetch):
    rec = _setup_session()
    wrapper = ScriptedWrapper(
        [_tool_use_turn("water_cut_gor_history", tool_input={}), _end_turn("ok")]
    )
    result = run_curve_turn("How has water cut changed?", wrapper=wrapper, session=rec)
    env = result["tool_outputs"][0]["result"]
    assert env["figure_ref"] == f"water_cut_gor_history::{WELL}"
    assert isinstance(env["figure"], go.Figure)


# --- registry sanity ----------------------------------------------------------


def test_schema_has_no_org_or_well():
    schema = TOOL_REGISTRY["water_cut_gor_history"]["spec"]["toolSpec"]["inputSchema"]["json"]
    props = schema["properties"]
    assert "well_id" not in props
    assert "organization_id" not in props
    assert set(props) == {"start_date", "end_date"}
    assert schema["required"] == []


def test_bubble_point_screen_is_gone():
    assert "bubble_point_screen" not in TOOL_REGISTRY


# --- shared error envelope (M3 contract) --------------------------------------


def test_success_envelopes_shape_identical(mock_fetch):
    ph = production_history({"organization_id": ORG, "well_id": WELL})
    wcg = water_cut_gor_history({"organization_id": ORG, "well_id": WELL})
    assert set(ph) == ENVELOPE_KEYS
    assert set(wcg) == ENVELOPE_KEYS
    assert set(ph) == set(wcg)


def test_error_envelopes_shape_identical_and_no_fabrication(mock_fetch):
    # Same failure (no injected org/well) → both tools emit the same-shape envelope.
    ph = production_history({"start_date": "2026-01-01"})
    wcg = water_cut_gor_history({"start_date": "2026-01-01"})
    for env in (ph, wcg):
        assert set(env) == ENVELOPE_KEYS
        assert env["status"] == "blocked"
        assert env["trust_label"] is None
        assert env["figure"] is None
        assert env["figure_ref"] is None
        assert env["values"] == {}  # no fabricated numbers
    assert set(ph) == set(wcg)


def test_data_failure_returns_structured_error_not_exception(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("athena exploded")

    monkeypatch.setattr(data, "fetch_preprocessed_window", _boom)
    for tool in (production_history, water_cut_gor_history):
        env = tool({"organization_id": ORG, "well_id": WELL})
        assert set(env) == ENVELOPE_KEYS
        assert env["status"] == "error"
        assert env["trust_label"] is None
        assert env["figure"] is None
        assert any("data_or_compute_error" in f for f in env["flags"])
        assert env["values"] == {"well_id": WELL}  # identity only, no numbers


# --- gate alias layer (M3 contract) -------------------------------------------


def test_alias_surfaces_canonical_from_legacy():
    df = pd.DataFrame({"GOR_scf_bbl": [100.0, 200.0]})
    out = _apply_field_aliases(df)
    assert "gor_scf_bbl" in out.columns  # canonical surfaced
    assert "GOR_scf_bbl" in out.columns  # legacy left in place (non-destructive)
    assert list(out["gor_scf_bbl"]) == [100.0, 200.0]


def test_alias_does_not_clobber_existing_canonical():
    df = pd.DataFrame({"GOR_scf_bbl": [1.0], "gor_scf_bbl": [999.0]})
    out = _apply_field_aliases(df)
    assert list(out["gor_scf_bbl"]) == [999.0]  # canonical wins, untouched


def test_alias_is_pair_extensible(monkeypatch):
    # Prompt #4's recipe: add a pair, no mechanism change. Simulate the delta_p_* pair.
    monkeypatch.setitem(gate_mod._FIELD_ALIASES, "delta_P_pump_psi", "delta_p_pump_psi")
    df = pd.DataFrame({"delta_P_pump_psi": [12.0]})
    out = _apply_field_aliases(df)
    assert "delta_p_pump_psi" in out.columns
    assert list(out["delta_p_pump_psi"]) == [12.0]
