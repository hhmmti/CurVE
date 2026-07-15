"""CurVE M4 / 4a — the connection-resolution layer (ideal-catalog fetch + BEP-narrow +
manual pump pick + per-well coverage report), run with NO AWS credentials.

Covered:
  * BEP narrowing reproduces the app's TWO-STAGE AND formula (flow-range containment
    AND BEP proximity ±tol) — verified against crafted rows that pass one stage only.
  * Selectability: rows with missing/NaN curve coeffs are EXCLUDED from selectable
    candidates but COUNTED (never silently dropped) — guardrail 3.
  * Obsolete rows are SURFACED with a flag, NOT dropped — guardrail 3 / app deviation.
  * pump_id (key) vs esp_model (display) and bep_bpd vs min/max_recommended stay
    distinct in the pick + candidates — guardrail 4.
  * The pump pick flows end-to-end as a SETUP-INJECTED value through the engine (the
    same path as org/well), never a Converse tool argument; ``None`` when unpicked.
  * make_pump_pick raises rather than substitute a default when the pump is absent.
  * total fluid = median liquid rate (oil+water), not oil; fetch validates the schema.
"""

import numpy as np
import pandas as pd
import pytest

from curve import ideal_catalog, session  # noqa: E402
from curve.engine import run_curve_turn  # noqa: E402

from tests.test_loop import ScriptedWrapper, _end_turn, _tool_use_turn  # noqa: E402

ORG = "permian_resources"
WELL = "HACKBERRY SPRINGS 3BH"


# --- fixtures -----------------------------------------------------------------


def _coeffs(prefix: str, missing: bool = False) -> dict:
    """Six coeffs for head/power; one NaN when ``missing`` (→ not selectable)."""
    vals = {f"ideal_{prefix}_c{i}": float(i) for i in range(1, 7)}
    if missing:
        vals[f"ideal_{prefix}_c3"] = np.nan
    return vals


def _row(pump_id, esp_model, bep, lo, hi, *, obsolete=False, missing_coeffs=False, series="513", mfr="ChampionX"):
    row = {
        "pump_id": pump_id,
        "esp_model": esp_model,
        "manufacturer": mfr,
        "series": series,
        "bep_bpd": bep,
        "min_recommended_bpd": lo,
        "max_recommended_bpd": hi,
        "is_obsolete": "true" if obsolete else "false",
    }
    row.update(_coeffs("head", missing=missing_coeffs))
    row.update(_coeffs("power"))
    return row


def _catalog() -> pd.DataFrame:
    # Well total fluid will be 1000 bpd. Design rows around that:
    return pd.DataFrame(
        [
            # A: passes BOTH stages (1000 in [800,1300] AND |1000-1000|/1000=0 <= .25).
            _row("CX-A", "P6-1000", bep=1000, lo=800, hi=1300),
            # B: flow-range OK but BEP too far (bep 2000 → ±25% = [1500,2500], 1000 out).
            _row("CX-B", "P6-2000", bep=2000, lo=500, hi=3000),
            # C: BEP-close but OUTSIDE flow-range (bep 1000 OK, but min 1100 > 1000).
            _row("CX-C", "P6-1000B", bep=1000, lo=1100, hi=1400),
            # D: passes both, but OBSOLETE — must be surfaced+flagged, not dropped.
            _row("CX-D", "P6-1000-OBS", bep=1050, lo=820, hi=1280, obsolete=True),
            # E: passes both, but MISSING a head coeff — not selectable, counted.
            _row("CX-E", "P6-1000-NOCRV", bep=980, lo=800, hi=1300, missing_coeffs=True),
        ]
    )


def _analyzed(liquid_vals) -> pd.DataFrame:
    return pd.DataFrame({"liquid_rate_bbl_day": liquid_vals})


# --- BEP narrowing parity (the app's two-stage AND) ---------------------------


def test_narrow_is_two_stage_and_matches_app_formula():
    narrowed = ideal_catalog.narrow_candidates(_catalog(), total_fluid_bpd=1000.0, bep_tolerance=0.25)
    ids = set(narrowed["pump_id"])
    # A, D, E pass BOTH stages; B fails BEP proximity; C fails flow-range containment.
    assert ids == {"CX-A", "CX-D", "CX-E"}
    assert "CX-B" not in ids  # flow-range OK but BEP too far → excluded (stage 2)
    assert "CX-C" not in ids  # BEP-close but outside flow-range → excluded (stage 1)


def test_narrow_empty_on_nonpositive_rate():
    assert ideal_catalog.narrow_candidates(_catalog(), 0.0).empty
    assert ideal_catalog.narrow_candidates(_catalog(), None).empty


def test_tolerance_widening_admits_more():
    # At ±25% the bep=2000 pump is out; at a wide enough tolerance it comes in.
    narrow = ideal_catalog.narrow_candidates(_catalog(), 1000.0, bep_tolerance=0.25)
    wide = ideal_catalog.narrow_candidates(_catalog(), 1000.0, bep_tolerance=0.60)
    assert "CX-B" not in set(narrow["pump_id"])
    assert "CX-B" in set(wide["pump_id"])  # 1000 >= 2000*(1-.6)=800 AND in [500,3000]


# --- selectability + obsolete (the CurVE deviations) --------------------------


def test_missing_coeff_row_excluded_but_counted():
    report = ideal_catalog.build_coverage_report(_catalog(), 1000.0, WELL, 0.25)
    cand_ids = {c["pump_id"] for c in report["candidates"]}
    assert "CX-E" not in cand_ids  # missing head coeff → not a selectable candidate
    assert report["n_excluded_missing_coeffs"] == 1  # but counted, not hidden
    # The full BEP-compatible set still includes it, flagged not selectable.
    bep_ids = {c["pump_id"] for c in report["bep_compatible"]}
    assert "CX-E" in bep_ids
    assert next(c for c in report["bep_compatible"] if c["pump_id"] == "CX-E")["selectable"] is False


def test_obsolete_surfaced_not_dropped():
    report = ideal_catalog.build_coverage_report(_catalog(), 1000.0, WELL, 0.25)
    obs = [c for c in report["candidates"] if c["is_obsolete"]]
    assert [c["pump_id"] for c in obs] == ["CX-D"]  # obsolete IS a selectable candidate
    assert report["n_obsolete_surfaced"] == 1


def test_coverage_report_shape_and_counts():
    report = ideal_catalog.build_coverage_report(_catalog(), 1000.0, WELL, 0.25)
    assert report["well_id"] == WELL
    assert report["total_fluid_bpd"] == 1000.0
    assert report["bep_tolerance"] == 0.25
    assert report["n_catalog"] == 5
    assert report["n_bep_compatible"] == 3  # A, D, E
    assert report["n_candidates"] == 2  # A, D (E excluded for coeffs)
    assert {c["pump_id"] for c in report["candidates"]} == {"CX-A", "CX-D"}


def test_no_candidates_when_telemetry_absent():
    report = ideal_catalog.build_coverage_report(_catalog(), None, WELL, 0.25)
    assert report["n_candidates"] == 0
    assert report["candidates"] == []
    assert report["total_fluid_bpd"] is None  # never a fabricated rate


# --- pump_id (key) vs esp_model (display); bep vs min/max (guardrail 4) --------


def test_pick_keeps_key_and_display_distinct():
    pick = ideal_catalog.make_pump_pick(_catalog(), "CX-A", bep_tolerance=0.25)
    assert pick["pump_id"] == "CX-A"  # the KEY
    assert pick["esp_model"] == "P6-1000"  # the DISPLAY label — distinct from the key
    assert pick["bep_bpd"] == 1000.0
    assert pick["min_recommended_bpd"] == 800.0 and pick["max_recommended_bpd"] == 1300.0
    assert pick["source"] == "manual"  # §1 ladder: v1 = manual
    assert pick["selectable"] is True and pick["is_obsolete"] is False


def test_pick_obsolete_carries_flag():
    pick = ideal_catalog.make_pump_pick(_catalog(), "CX-D")
    assert pick["is_obsolete"] is True  # operator may run it; flagged, not refused here


def test_pick_unknown_raises_no_silent_default():
    with pytest.raises(KeyError):
        ideal_catalog.make_pump_pick(_catalog(), "NOPE")  # never substitutes a default


# --- total fluid = median liquid rate (oil+water), not oil --------------------


def test_total_fluid_is_median_liquid_rate():
    assert ideal_catalog.resolve_total_fluid_bpd(_analyzed([900, 1000, 1100])) == 1000.0
    assert ideal_catalog.resolve_total_fluid_bpd(_analyzed([])) is None
    assert ideal_catalog.resolve_total_fluid_bpd(pd.DataFrame({"other": [1]})) is None


# --- bep tolerance is setup-injected (rides resolved_inputs) ------------------


def test_bep_tolerance_from_context():
    assert ideal_catalog.bep_tolerance_from_context({"bep_tolerance": 0.4}) == 0.4
    assert ideal_catalog.bep_tolerance_from_context({}) == ideal_catalog.DEFAULT_BEP_TOLERANCE
    assert ideal_catalog.bep_tolerance_from_context(None) == ideal_catalog.DEFAULT_BEP_TOLERANCE
    assert ideal_catalog.bep_tolerance_from_context({"bep_tolerance": 0}) == ideal_catalog.DEFAULT_BEP_TOLERANCE


# --- fetch schema validation (resolve-or-stop, guardrail 11) ------------------


def test_query_targets_resolved_catalog_location():
    q = ideal_catalog.build_catalog_query()
    assert "esp_ideal_pump_dev.ideal_pump_library_v1" in q
    assert ideal_catalog.IDEAL_CATALOG_CATALOG == "roam_dev_products"


def test_validate_required_columns_raises_on_missing():
    bad = _catalog().drop(columns=["bep_bpd"])
    with pytest.raises(ValueError, match="does not line up"):
        ideal_catalog._validate_required_columns(bad)


# --- the pump pick flows through the engine as a setup-injected value ----------


def _echo_registry():
    """A tool that echoes the tool_input the engine handed it (to inspect injection)."""

    def echo(tool_input):
        return {"received": dict(tool_input)}

    spec = {
        "toolSpec": {
            "name": "echo_tool",
            "description": "echo",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    }
    return {"echo_tool": {"spec": spec, "fn": echo}}


def test_pump_pick_injected_into_tool_call_not_a_model_arg():
    session.clear_sessions()
    pick = ideal_catalog.make_pump_pick(_catalog(), "CX-A", bep_tolerance=0.25)
    record = session.new_session_record(
        session_id="s1", organization_id=ORG, well_id=WELL
    )
    ideal_catalog.set_pump_on_session(record, pick)  # rides the M3 override path

    wrapper = ScriptedWrapper([_tool_use_turn("echo_tool", tool_input={}), _end_turn("done")])
    result = run_curve_turn(
        "where am I on the curve?",
        wrapper=wrapper,
        tools=_echo_registry(),
        session=record,
    )
    received = result["tool_outputs"][0]["result"]["received"]
    # Injected top-level, exactly like org/well — not a model-supplied Converse arg.
    assert received["pump"]["pump_id"] == "CX-A"
    assert received["organization_id"] == ORG and received["well_id"] == WELL


def test_pump_none_injected_when_unpicked():
    session.clear_sessions()
    record = session.new_session_record(
        session_id="s2", organization_id=ORG, well_id=WELL
    )  # pump defaults to None — honest "no connection yet"
    wrapper = ScriptedWrapper([_tool_use_turn("echo_tool", tool_input={}), _end_turn("done")])
    result = run_curve_turn(
        "curve?", wrapper=wrapper, tools=_echo_registry(), session=record
    )
    assert result["tool_outputs"][0]["result"]["received"]["pump"] is None
