"""CurVE tests — the two ΔP history tools (delta_p_frequency / delta_p_composition)
end-to-end, plus the cross-cutting contracts they establish: the shared ΔP
input-resolution layer (depth/SG precedence + source flags + trust label), the
Estimated label threaded through the gate, the PIP measured-or-missing coverage
handling, the operator override, and the alias pair. Run with NO AWS credentials
(awswrangler fetch + rrc depth lookup are monkeypatched; Converse is scripted).

Verified across all three label outcomes:
  * Estimated + source flags — depth defaulted / overridden, SG defaulted.
  * Validated — every input measured/real (rrc depth + measured SG).
  * not-ready / blocked — zero PIP coverage, or a missing x-axis (projection) input.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve import data, delta_p_inputs, gate as gate_mod, session, well_depth  # noqa: E402
from curve.delta_p_inputs import resolve_delta_p_inputs  # noqa: E402
from curve.engine import run_curve_turn  # noqa: E402
from curve.envelope import ENVELOPE_KEYS  # noqa: E402
from curve.gate import _FIELD_ALIASES, run_delta_p_tool_gate  # noqa: E402
from curve.tools import (  # noqa: E402
    TOOL_REGISTRY,
    delta_p_composition,
    delta_p_frequency,
)
from curve.well_depth import build_depth_query  # noqa: E402

from tests.test_loop import ScriptedWrapper, _end_turn, _tool_use_turn  # noqa: E402

ORG = "org-acme"
WELL = "W-12"


# --- fixtures -----------------------------------------------------------------


def _telemetry_fixture(n_days: int = 30, pip: float = 800.0, pip_pattern=None) -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    intake = pip_pattern if pip_pattern is not None else [pip] * n_days
    return pd.DataFrame(
        {
            "organization_id": ORG,
            "well_id": WELL,
            "timestamp_telem": days,
            "observation_day": days.strftime("%Y-%m-%d"),
            "motor_frequency_hz": [55.0 + i * 0.1 for i in range(n_days)],
            "tubing_pressure_psi": 250.0,
            "pump_intake_pressure_psi": intake,
            "motor_amps": 40.0,
            "motor_volts": 2300.0,
        }
    )


def _production_fixture(n_days: int = 30) -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "organization_id": ORG,
            "well_id": WELL,
            "observation_day": days.strftime("%Y-%m-%d"),
            "alloc_oil_vol": [300.0 - i * 2 for i in range(n_days)],
            "alloc_water_vol": [100.0 + i * 3 for i in range(n_days)],
            "alloc_gas_vol": [450.0] * n_days,
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
def _no_rrc(monkeypatch):
    """Default: no rrc depth on record (depth falls to the CurVE default) + no AWS."""
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _clean_sessions():
    session.clear_sessions()
    yield
    session.clear_sessions()


# --- resolution layer: precedence + flags + trust -----------------------------


def test_resolve_default_depth_estimated():
    r = resolve_delta_p_inputs(rrc_depth_ft=None)
    assert r.depth_ft == delta_p_inputs.CURVE_DEFAULT_DEPTH_FT  # 10,000 ft CurVE default
    assert r.depth_source == "defaulted"
    assert r.sg_source == "defaulted"
    assert r.trust_label == "Estimated"
    assert "depth_defaulted" in r.flags and "sg_defaulted" in r.flags


def test_resolve_rrc_real_depth():
    r = resolve_delta_p_inputs(rrc_depth_ft=8200.0)
    assert r.depth_ft == 8200.0
    assert r.depth_source == "rrc"
    assert "depth_from_rrc" in r.flags
    # SG still defaults → overall Estimated (worst input wins).
    assert r.trust_label == "Estimated"


def test_resolve_override_supersedes_and_is_estimated():
    r = resolve_delta_p_inputs(rrc_depth_ft=8200.0, depth_override=12345.0)
    assert r.depth_ft == 12345.0
    assert r.depth_source == "user_supplied"  # a hand-typed depth is an estimate
    assert "depth_user_supplied" in r.flags
    assert r.trust_label == "Estimated"


def test_resolve_all_real_is_validated():
    r = resolve_delta_p_inputs(
        rrc_depth_ft=8200.0, sg_oil_measured=0.82, sg_water_measured=1.02
    )
    assert r.depth_source == "rrc" and r.sg_source == "measured"
    assert r.trust_label == "Validated"
    assert "depth_from_rrc" in r.flags and "sg_from_measured" in r.flags


# --- rrc depth query (ported app read) ----------------------------------------


def test_depth_query_targets_app_table_and_column():
    q = build_depth_query(WELL)
    assert '"API Depth"' in q
    assert '"well_depth_dev"."rrc_well_depth"' in q
    assert f"well_id = '{WELL}'" in q


# --- the full chain: Estimated + flags ----------------------------------------


def test_delta_p_frequency_estimated_envelope(mock_fetch):
    env = delta_p_frequency({"organization_id": ORG, "well_id": WELL})
    assert set(env) == ENVELOPE_KEYS
    assert env["status"] == "available"
    assert env["trust_label"] == "Estimated"  # depth defaulted, SG defaulted
    assert "depth_defaulted" in env["flags"]
    assert "sg_defaulted" in env["flags"]
    values = env["values"]
    assert values["well_id"] == WELL
    assert values["latest"]["delta_p_pump_psi"] is not None
    assert values["latest"]["motor_frequency_hz"] is not None
    assert values["inputs"]["depth_ft"] == delta_p_inputs.CURVE_DEFAULT_DEPTH_FT
    assert values["period"]["pip_coverage"]["rows_with_pip"] == 30
    assert env["figure_ref"] == f"delta_p_frequency::{WELL}"
    assert isinstance(env["figure"], go.Figure)


def test_delta_p_composition_estimated_envelope(mock_fetch):
    env = delta_p_composition({"organization_id": ORG, "well_id": WELL})
    assert env["status"] == "available"
    assert env["trust_label"] == "Estimated"
    latest = env["values"]["latest"]
    # the pressure-component decomposition is present
    for key in (
        "pump_intake_pressure_psi",
        "delta_p_hyd_psi",
        "tubing_pressure_psi",
        "p_dis_downhole_psi",
        "delta_p_pump_psi",
    ):
        assert latest[key] is not None
    assert isinstance(env["figure"], go.Figure)


def test_rrc_depth_tier_fires_when_present(monkeypatch, mock_fetch):
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: 9100.0)
    env = delta_p_frequency({"organization_id": ORG, "well_id": WELL})
    assert env["values"]["inputs"]["depth_ft"] == 9100.0
    assert env["values"]["inputs"]["depth_source"] == "rrc"
    assert "depth_from_rrc" in env["flags"]
    assert env["trust_label"] == "Estimated"  # SG still default


def test_all_real_yields_validated_end_to_end(monkeypatch, mock_fetch):
    # rrc depth present + measured per-well SG injected (setup context) → Validated.
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: 9100.0)
    env = delta_p_frequency(
        {
            "organization_id": ORG,
            "well_id": WELL,
            "resolved_inputs": {"sg_oil_measured": 0.82, "sg_water_measured": 1.02},
        }
    )
    assert env["status"] == "available"
    assert env["trust_label"] == "Validated"
    assert "depth_from_rrc" in env["flags"] and "sg_from_measured" in env["flags"]


# --- operator override changes the computed ΔP --------------------------------


def test_override_changes_delta_p_and_flags_label_stays_estimated(mock_fetch):
    base = delta_p_frequency({"organization_id": ORG, "well_id": WELL})
    overridden = delta_p_frequency(
        {
            "organization_id": ORG,
            "well_id": WELL,
            "resolved_inputs": {"depth_override": 20000.0},
        }
    )
    base_dp = base["values"]["latest"]["delta_p_pump_psi"]
    ovr_dp = overridden["values"]["latest"]["delta_p_pump_psi"]
    # Deeper override → larger hydrostatic → larger ΔP_pump.
    assert ovr_dp > base_dp
    assert overridden["values"]["inputs"]["depth_ft"] == 20000.0
    assert "depth_user_supplied" in overridden["flags"]
    assert overridden["trust_label"] == "Estimated"  # override is still an estimate


# --- PIP measured-or-missing: coverage + zero-coverage block -------------------


def test_zero_pip_coverage_blocks(monkeypatch):
    def _fetch_zero(*a, **k):
        return _telemetry_fixture(pip=0.0), _production_fixture()

    monkeypatch.setattr(data, "fetch_preprocessed_window", _fetch_zero)
    env = delta_p_frequency({"organization_id": ORG, "well_id": WELL})
    assert env["status"] == "blocked"
    assert env["trust_label"] is None
    assert "pip_coverage_zero" in env["flags"]
    assert env["figure"] is None
    assert env["values"] == {"well_id": WELL}  # identity only, no fabricated numbers


def test_partial_pip_coverage_flag(monkeypatch):
    # Half the days have PIP, half are missing (0 → measured-or-missing nulls them).
    pattern = [800.0 if i % 2 == 0 else 0.0 for i in range(30)]

    def _fetch_partial(*a, **k):
        return _telemetry_fixture(pip_pattern=pattern), _production_fixture()

    monkeypatch.setattr(data, "fetch_preprocessed_window", _fetch_partial)
    env = delta_p_frequency({"organization_id": ORG, "well_id": WELL})
    assert env["status"] == "available"
    assert env["trust_label"] == "Estimated"
    assert any("pip_coverage_partial" in f for f in env["flags"])
    assert env["values"]["period"]["pip_coverage"]["rows_with_pip"] == 15


def test_missing_x_axis_input_blocks(monkeypatch):
    # Drop motor_frequency_hz → delta_p_frequency's projection input is missing.
    def _fetch_no_freq(*a, **k):
        telem = _telemetry_fixture().drop(columns=["motor_frequency_hz"])
        return telem, _production_fixture()

    monkeypatch.setattr(data, "fetch_preprocessed_window", _fetch_no_freq)
    env = delta_p_frequency({"organization_id": ORG, "well_id": WELL})
    assert env["status"] == "blocked"
    assert any("missing_x_axis_input: motor_frequency_hz" in f for f in env["flags"])


# --- guards: org/well injection + structured errors ---------------------------


def test_blocks_without_injected_org_well(mock_fetch):
    env = delta_p_frequency({"start_date": "2026-01-01"})
    assert env["status"] == "blocked"
    assert "missing_org_or_well_injection" in env["flags"]
    assert mock_fetch == {}


def test_data_failure_is_structured_error(monkeypatch):
    monkeypatch.setattr(
        data, "fetch_preprocessed_window", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    for tool in (delta_p_frequency, delta_p_composition):
        env = tool({"organization_id": ORG, "well_id": WELL})
        assert set(env) == ENVELOPE_KEYS
        assert env["status"] == "error"
        assert any("data_or_compute_error" in f for f in env["flags"])
        assert env["values"] == {"well_id": WELL}


# --- envelope parity with the existing tools ----------------------------------


def test_envelope_shape_parity(mock_fetch):
    freq = delta_p_frequency({"organization_id": ORG, "well_id": WELL})
    comp = delta_p_composition({"organization_id": ORG, "well_id": WELL})
    assert set(freq) == ENVELOPE_KEYS
    assert set(comp) == ENVELOPE_KEYS
    assert set(freq) == set(comp)


# --- engine injection (overrides are NOT model args) --------------------------


def test_engine_injects_resolved_inputs(mock_fetch):
    rec = session.save_session(
        session.new_session_record(
            session_id="sess-dp",
            organization_id=ORG,
            well_id=WELL,
            resolved_inputs={"depth_override": 15000.0},
        )
    )
    wrapper = ScriptedWrapper(
        [
            _tool_use_turn("delta_p_frequency", tool_input={}),
            _end_turn("ΔP is Estimated — depth was operator-supplied."),
        ]
    )
    result = run_curve_turn("How is ΔP responding to frequency?", wrapper=wrapper, session=rec)
    env = result["tool_outputs"][0]["result"]
    assert env["values"]["inputs"]["depth_ft"] == 15000.0
    assert "depth_user_supplied" in env["flags"]


def test_schema_exposes_only_time_window():
    for tool in ("delta_p_frequency", "delta_p_composition"):
        schema = TOOL_REGISTRY[tool]["spec"]["toolSpec"]["inputSchema"]["json"]
        props = schema["properties"]
        assert set(props) == {"start_date", "end_date"}
        for forbidden in ("well_id", "organization_id", "depth", "well_depth_ft", "sg_oil", "pip"):
            assert forbidden not in props


# --- alias pair (static, prompt #4) -------------------------------------------


def test_delta_p_alias_pair_present():
    assert _FIELD_ALIASES.get("delta_P_pump_psi") == "delta_p_pump_psi"


def test_registry_has_both_delta_p_tools():
    assert "delta_p_frequency" in TOOL_REGISTRY
    assert "delta_p_composition" in TOOL_REGISTRY


# --- gate unit: NO proxy path, Estimated threading ----------------------------


def test_gate_blocks_on_zero_coverage_without_proxy():
    analyzed = pd.DataFrame(
        {
            "delta_p_pump_psi": [float("nan")] * 3,
            "pump_intake_pressure_psi": [float("nan")] * 3,
            "motor_frequency_hz": [55.0] * 3,
            "p_dis_downhole_psi": [1000.0] * 3,
            "delta_p_hyd_psi": [700.0] * 3,
            "tubing_pressure_psi": [250.0] * 3,
        }
    )
    resolved = resolve_delta_p_inputs(rrc_depth_ft=None)
    cov = delta_p_inputs.pip_coverage(analyzed)
    g = run_delta_p_tool_gate("delta_p_frequency", analyzed, resolved, cov)
    assert g["status"] == "blocked"
    assert "pip_coverage_zero" in g["flags"]
