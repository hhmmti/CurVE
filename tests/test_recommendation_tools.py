"""CurVE tests — the three recommendation-dependent tools (recommendation_comparison /
affinity_check / energy_efficiency) end-to-end, plus the cross-cutting M3 contracts
they establish: CurVE's second data path (the Athena-mirror recommendation read), v1's
first hard block (recommendation-absence), v1's first Proxy label, and the general
weakest-wins trust precedence. Run with NO AWS credentials (the recommendation fetch +
rrc depth lookup are monkeypatched; Converse is scripted).

Verified outcomes:
  * recommendation_comparison — Validated relative to the payload.
  * affinity_check — Validated/Estimated per provenance (ΔP depth/SG).
  * energy_efficiency — weakest-wins across liquid + ΔP + power, with the first Proxy
    label on the amp×volt power path.
  * recommendation-absence blocks all three (not-ready / recommendation_absent).
  * envelope shape parity with the existing tools.
"""

import json

import plotly.graph_objects as go
import pytest

from curve import recommendations, session, well_depth  # noqa: E402
from curve.engine import run_curve_turn  # noqa: E402
from curve.envelope import ENVELOPE_KEYS  # noqa: E402
from curve.gate import NOT_READY, weakest_trust  # noqa: E402
from curve.tools import (  # noqa: E402
    TOOL_REGISTRY,
    affinity_check,
    energy_efficiency,
    recommendation_comparison,
)

from tests.test_loop import ScriptedWrapper, _end_turn, _tool_use_turn  # noqa: E402

ORG = "org-acme"
WELL = "W-12"

_CURRENT = "{motor_frequency_hz=58.0, tubing_pressure_psi=240.0, production={oil=300.0, water=100.0, gas=450.0}}"
_MODEL = "{max_oil={motor_frequency_hz=62.0, tubing_pressure_psi=260.0, production={oil=340.0, water=110.0, gas=480.0}}}"


def _summary(**overrides):
    base = {
        "pump_intake_pressure_psi_1d_avg": 800.0,
        "tubing_pressure_psi_1d_avg": 240.0,
        "alloc_oil_vol_1d_ago": 300.0,
        "alloc_water_vol_1d_ago": 100.0,
        "motor_amps_1d_avg": 40.0,
        "motor_volts_1d_avg": 2300.0,
    }
    base.update(overrides)
    return json.dumps(base)


def _rec_row(current=_CURRENT, model=_MODEL, summary=None, uuid="rec-123"):
    return {
        "organization_id": ORG,
        "well_id": WELL,
        "uuid": uuid,
        "timestamp": "2026-06-01T00:00:00",
        "current_setpoint": current,
        "model_setpoint_recommendations": model,
        "summary_data_json": summary if summary is not None else _summary(),
    }


@pytest.fixture
def mock_rec(monkeypatch):
    """Default: a full recommendation present, no rrc depth (depth → default)."""
    row = _rec_row()
    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", lambda o, w, **k: row)
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: None)
    return row


@pytest.fixture
def mock_no_rec(monkeypatch):
    """No recommendation exists for the well/session."""
    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", lambda o, w, **k: None)
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _clean_sessions():
    session.clear_sessions()
    yield
    session.clear_sessions()


_TI = {"organization_id": ORG, "well_id": WELL, "resolved_inputs": {}}


# --- recommendation source: Athena-mirror read + VE-parity parse ---------------


def test_mirror_query_targets_ddb_mirror_table():
    q = recommendations.build_latest_recommendation_query(ORG, WELL)
    assert "esp_setpoint_recommendations_v2" in q
    assert "current_setpoint" in q and "model_setpoint_recommendations" in q
    assert "summary_data_json" in q and "uuid" in q
    assert f"well_id = '{WELL}'" in q and f"organization_id = '{ORG}'" in q
    assert "ORDER BY timestamp DESC LIMIT 1" in q


def test_mirror_query_requires_both_keys():
    with pytest.raises(ValueError):
        recommendations.build_latest_recommendation_query("", WELL)


def test_ve_power_matches_3phase_pf085():
    # VE method: amps × volts × √3 × 0.85 / 1000.
    assert recommendations.ve_power_kw_from_amps_volts(40.0, 2300.0) == pytest.approx(
        40.0 * 2300.0 * 1.7321 * 0.85 / 1000.0
    )
    assert recommendations.ve_power_kw_from_amps_volts(0, 2300) is None
    assert recommendations.ve_power_kw_from_amps_volts(None, 2300) is None


def test_summary_blob_consumes_ve_key_not_synthesized():
    row = _rec_row(summary=_summary(motor_power_kw_1d_avg=150.0))
    blob = recommendations.summary_blob(row)
    parsed = json.loads(blob)  # it is the VE's blob, parseable as-is
    assert parsed["motor_power_kw_1d_avg"] == 150.0


# --- recommendation_comparison: Validated relative to payload ------------------


def test_recommendation_comparison_validated(mock_rec):
    env = recommendation_comparison(dict(_TI))
    assert set(env) == ENVELOPE_KEYS
    assert env["status"] == "available"
    assert env["trust_label"] == "Validated"  # relative to the payload
    assert env["flags"] == []
    values = env["values"]
    assert values["recommendation_uuid"] == "rec-123"
    assert values["motor_frequency_hz"]["current"] == 58.0
    assert values["motor_frequency_hz"]["recommended"] == 62.0
    assert values["motor_frequency_hz"]["delta"] == pytest.approx(4.0)
    assert isinstance(env["figure"], go.Figure)
    assert env["figure_ref"] == f"recommendation_comparison::{WELL}"


# --- recommendation_comparison: % delta (M5 step-4 enrichment) -----------------


def test_recommendation_comparison_carries_delta_pct(mock_rec):
    # % delta beside each absolute delta: delta / current × 100 (normal case).
    env = recommendation_comparison(dict(_TI))
    freq = env["values"]["motor_frequency_hz"]
    # freq: current 58, delta +4 → +6.9%. Existing keys untouched.
    assert freq["current"] == 58.0 and freq["delta"] == pytest.approx(4.0)
    assert freq["delta_pct"] == pytest.approx(4.0 / 58.0 * 100.0, abs=0.1)
    # every compared field carries the new key
    for field in (
        "motor_frequency_hz", "tubing_pressure_psi", "liquid_rate_bpd",
        "oil_rate_bpd", "water_rate_bpd", "gas_rate",
    ):
        assert "delta_pct" in env["values"][field]


def test_recommendation_comparison_delta_pct_guards_zero_current(monkeypatch):
    # current == 0 (or null) → delta_pct None, never a divide-by-zero / inf.
    from curve import tools

    proj = tools._project_recommendation_comparison_values(
        WELL,
        {"uuid": "rec-z"},
        {
            "cur_motor_frequency_hz": 0.0, "rec_motor_frequency_hz": 5.0, "delta_motor_frequency_hz": 5.0,
            "cur_tubing_pressure_psi": None, "rec_tubing_pressure_psi": 10.0, "delta_tubing_pressure_psi": 10.0,
        },
        "max_oil",
    )
    assert proj["motor_frequency_hz"]["delta_pct"] is None  # current == 0
    assert proj["tubing_pressure_psi"]["delta_pct"] is None  # current null


# --- affinity_check: Validated/Estimated per provenance -----------------------


def test_affinity_check_estimated_when_depth_defaulted(mock_rec):
    env = affinity_check(dict(_TI))
    assert env["status"] == "available"
    # ΔP pressure check rides on defaulted depth/SG → Estimated (weakest term).
    assert env["trust_label"] == "Estimated"
    assert "depth_defaulted" in env["flags"] and "sg_defaulted" in env["flags"]
    assert env["values"]["flow_check"]["available"] is True
    assert isinstance(env["figure"], go.Figure)


def test_affinity_check_validated_when_all_real(monkeypatch, mock_rec):
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: 9100.0)
    ti = {
        "organization_id": ORG,
        "well_id": WELL,
        "resolved_inputs": {"sg_oil_measured": 0.82, "sg_water_measured": 1.02},
    }
    env = affinity_check(ti)
    assert env["status"] == "available"
    assert env["trust_label"] == "Validated"  # rrc depth + measured SG → all real
    assert "depth_from_rrc" in env["flags"] and "sg_from_measured" in env["flags"]


# --- energy_efficiency: weakest-wins + first Proxy label ----------------------


def test_energy_efficiency_proxy_on_amp_volt_power(mock_rec):
    # liquid (Validated) + ΔP (Estimated, defaulted depth/SG) + power (amp×volt → Proxy)
    # → overall Proxy (the worked example; weakest-wins, first Proxy label in v1).
    env = energy_efficiency(dict(_TI))
    assert env["status"] == "available"
    assert env["trust_label"] == "Proxy"
    assert "power_term_proxy" in env["flags"]
    assert "liquid_validated" in env["flags"]
    terms = env["values"]["term_provenance"]
    assert terms == {"liquid": "Validated", "delta_p": "Estimated", "power": "Proxy"}
    assert env["values"]["current"]["power_source"] == "amp_x_volt"
    assert isinstance(env["figure"], go.Figure)


def test_energy_efficiency_estimated_on_direct_power(monkeypatch):
    # Direct motor_power_kw channel → power term Validated; ΔP still Estimated →
    # overall Estimated (weakest is ΔP, never inflated to Validated).
    row = _rec_row(summary=_summary(motor_power_kw_1d_avg=150.0))
    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", lambda o, w, **k: row)
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: None)
    env = energy_efficiency(dict(_TI))
    assert env["status"] == "available"
    assert env["trust_label"] == "Estimated"
    assert env["values"]["term_provenance"]["power"] == "Validated"
    assert "power_term_validated" in env["flags"]


def test_energy_efficiency_validated_when_all_real_and_direct(monkeypatch):
    row = _rec_row(summary=_summary(motor_power_kw_1d_avg=150.0))
    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", lambda o, w, **k: row)
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: 9100.0)
    ti = {
        "organization_id": ORG,
        "well_id": WELL,
        "resolved_inputs": {"sg_oil_measured": 0.82, "sg_water_measured": 1.02},
    }
    env = energy_efficiency(ti)
    assert env["trust_label"] == "Validated"  # all three terms real


def test_energy_efficiency_blocks_when_no_power_source(monkeypatch):
    # No power channel and no amps/volts → power term not-ready → tool blocks.
    row = _rec_row(
        summary=_summary(motor_amps_1d_avg=None, motor_volts_1d_avg=None)
    )
    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", lambda o, w, **k: row)
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: None)
    env = energy_efficiency(dict(_TI))
    assert env["status"] == "blocked"
    assert "power_source_absent" in env["flags"]
    assert env["trust_label"] is None


def test_energy_efficiency_blocks_when_intake_missing(monkeypatch):
    # PIP measured-or-missing on the rec path: no intake → ΔP NaN → ΔP term not-ready.
    row = _rec_row(summary=_summary(pump_intake_pressure_psi_1d_avg=None))
    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", lambda o, w, **k: row)
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: None)
    env = energy_efficiency(dict(_TI))
    assert env["status"] == "blocked"
    assert "delta_p_absent" in env["flags"]


# --- recommendation-absence hard block (v1's first) ---------------------------


def test_absence_blocks_all_three(mock_no_rec):
    for tool in (recommendation_comparison, affinity_check, energy_efficiency):
        env = tool(dict(_TI))
        assert set(env) == ENVELOPE_KEYS
        assert env["status"] == "not-ready"
        assert env["flags"] == ["recommendation_absent"]
        assert env["trust_label"] is None
        assert env["figure"] is None
        assert env["values"] == {"well_id": WELL}  # identity only, no fabrication


def test_absence_is_distinct_from_coverage_and_estimated(mock_no_rec):
    env = energy_efficiency(dict(_TI))
    assert "pip_coverage_zero" not in env["flags"]  # distinct from #4 coverage-block
    assert env["trust_label"] != "Estimated"  # a separate, named label state


# --- weakest-wins precedence (general, not hardcoded) -------------------------


def test_weakest_trust_precedence():
    assert weakest_trust(["Validated", "Estimated", "Proxy"]) == "Proxy"
    assert weakest_trust(["Validated", "Estimated"]) == "Estimated"
    assert weakest_trust(["Validated", "Validated", "Validated"]) == "Validated"
    assert weakest_trust(["Validated", "Estimated", None]) == NOT_READY
    assert weakest_trust([]) == NOT_READY


# --- guards: org/well injection + structured errors ---------------------------


def test_blocks_without_injected_org_well(monkeypatch):
    called = {"n": 0}

    def _fetch(*a, **k):
        called["n"] += 1
        return _rec_row()

    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", _fetch)
    for tool in (recommendation_comparison, affinity_check, energy_efficiency):
        env = tool({"resolved_inputs": {}})
        assert env["status"] == "blocked"
        assert "missing_org_or_well_injection" in env["flags"]
    assert called["n"] == 0  # never fetched without both keys


def test_data_failure_is_structured_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(recommendations, "fetch_latest_recommendation", _boom)
    monkeypatch.setattr(well_depth, "fetch_well_depth_ft", lambda *a, **k: None)
    for tool in (recommendation_comparison, affinity_check, energy_efficiency):
        env = tool(dict(_TI))
        assert set(env) == ENVELOPE_KEYS
        assert env["status"] == "error"
        assert any("data_or_compute_error" in f for f in env["flags"])


# --- Converse spec: no model-facing args (rec is session-injected) ------------


def test_specs_expose_no_args():
    for tool in _REC_TOOL_NAMES:
        schema = TOOL_REGISTRY[tool]["spec"]["toolSpec"]["inputSchema"]["json"]
        assert schema["properties"] == {}
        for forbidden in ("well_id", "organization_id", "recommendation", "current_setpoint"):
            assert forbidden not in schema["properties"]


_REC_TOOL_NAMES = ("recommendation_comparison", "affinity_check", "energy_efficiency")


# --- envelope shape parity with existing tools --------------------------------


def test_envelope_shape_parity(mock_rec):
    envs = [tool(dict(_TI)) for tool in (recommendation_comparison, affinity_check, energy_efficiency)]
    for env in envs:
        assert set(env) == ENVELOPE_KEYS


# --- engine threads the session (rec re-fetched per session) ------------------


def test_engine_routes_recommendation_comparison(mock_rec):
    rec = session.save_session(
        session.new_session_record(
            session_id="sess-rec", organization_id=ORG, well_id=WELL, resolved_inputs={}
        )
    )
    wrapper = ScriptedWrapper(
        [
            _tool_use_turn("recommendation_comparison", tool_input={}),
            _end_turn("The model recommends raising frequency — Validated vs the payload."),
        ]
    )
    result = run_curve_turn("What change does the model recommend?", wrapper=wrapper, session=rec)
    env = result["tool_outputs"][0]["result"]
    assert env["status"] == "available"
    assert env["trust_label"] == "Validated"
    # The figure is stripped from the model-facing result.
    model_facing = result["tool_outputs"][0]["result"]
    assert "figure" in model_facing  # full envelope kept for the UI
