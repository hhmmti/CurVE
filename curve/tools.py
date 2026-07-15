"""CurVE tools (M3).

``production_history`` and ``water_cut_gor_history`` are **real, end-to-end**: real
Athena telemetry + production → per-tool gate → vendored physics → vendored Plotly →
the shared ``{status, values, trust_label, flags, figure_ref, figure}`` envelope
(``curve.envelope``; CurVE-decisions §3 D3). Both are connection-free Validated tools
and share the data path, the gate keys, and the success/error envelope shape.
``curve_position`` remains an **M1 stub** (real spec, mock body) — real in M4. The
``bubble_point_screen`` stub was retired in M3 (v2 tool, not in the v1 roster).

Naming convention (documented in README):
  * snake_case
  * capability-named — the verb/noun of the question the tool answers
    (``production_history``, ``water_cut_gor_history``, ``curve_position``), not the
    implementation module it will eventually call. The model routes on the
    capability, so the name + description must read like an operator's intent.

THE ENVELOPE (CurVE-decisions §3 D2/D3):
  A real tool returns ``{values, trust_label, flags, figure_ref, figure}``. The
  engine sends ONLY ``{values, trust_label, flags, status}`` back to the model — the
  ``figure`` (a Plotly object) and ``figure_ref`` go to the UI, never into the model
  (no image tokens; the model narrates from ``values`` only). ``trust_label`` is
  carried faithfully from the gate — never hardcoded.

ORG / WELL ARE INJECTED, NOT MODEL-SUPPLIED:
  ``production_history``'s ``inputSchema`` exposes ONLY the time-window selector. The
  engine backend-injects ``organization_id`` + ``well_id`` from the session record
  (CurVE-decisions §3 D3). The model never sees or supplies them.
"""

from typing import Any, Callable, Dict, Optional

import pandas as pd

from compute import ideal_curve_overlay, ml_recommendation_calcs
from compute.affinity_validator import (
    MINIMAL_FREQ_CHANGE_HZ,
    compute_affinity_law_validator,
)
from compute.energy_efficiency import compute_energy_efficiency_diagnostic
from compute.preprocessed_calcs import WELL_DEPTH_FT
from plotting.affinity_charts import build_affinity_law_panel
from plotting.curve_position_charts import (
    build_curve_position_family,
    build_curve_position_overlay,
)
from plotting.energy_charts import build_energy_power_cards
from plotting.preprocessed_charts import (
    build_allocation_temporal,
    build_delta_p_composition,
    build_delta_p_pump_vs_frequency,
    build_water_cut_gor_analysis,
)
from plotting.recommendation_compare_charts import build_recommendation_comparison_bars

from . import (
    curve_position as curve_position_layer,
    data,
    delta_p_inputs,
    ideal_catalog,
    recommendations,
    well_depth,
)
from services import preprocessed_pipeline_service
from .envelope import error_envelope, success_envelope
from .gate import (
    recommendation_absence_block,
    run_affinity_check_gate,
    run_curve_position_gate,
    run_delta_p_tool_gate,
    run_energy_efficiency_gate,
    run_recommendation_comparison_gate,
    run_tool_gate,
)

# Pipeline fns from the services layer.
prepare_daily_data = preprocessed_pipeline_service.prepare_daily_data
run_preprocessed_analysis = preprocessed_pipeline_service.run_preprocessed_analysis

# Keys the engine strips from a tool result before it reaches the model. The Plotly
# figure(s) and the UI ref never go back to the model (CurVE-decisions §3 D2). ``figures``
# (plural) is curve_position's multi-figure slot — stripped exactly like the singular
# ``figure`` so the model-facing envelope stays values + trust_label only.
NON_MODEL_RESULT_KEYS = {"figure", "figure_ref", "figures"}


# --- real tool: production_history --------------------------------------------


def _round(value: Any, ndigits: int = 1) -> Optional[float]:
    """Round to a JSON-friendly float, mapping NaN/None → None."""
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


# --- history projection helpers (last-valid selection) ------------------------
# A window boundary row can carry a null allocation (a day with no allocated
# production nulls oil/water/gas together and drives liquid_rate to 0). Taking the
# raw first/last row therefore misreads a data gap: the KPI cards blank out and the
# trend reports a false "flat". These helpers reselect over the rows that actually
# have data — per-series for the trend endpoints, and the last row with any
# allocation for the "latest" block — mirroring the ΔP tools' "last valid reading"
# semantics. The null grain is ROW-LEVEL (a missing allocation nulls the alloc_*
# volumes together), so liquid_rate is NOT a validity anchor — the alloc_* are.

_ALLOC_RATE_COLS = ("alloc_oil_vol", "alloc_water_vol", "alloc_gas_vol")


def _valid_endpoints(daily: pd.DataFrame, field: str):
    """First/last non-null values of ``field`` + the count of non-null observations.

    Returns ``(first, last, n_valid)``. ``n_valid == 0`` → no data (both None);
    ``n_valid == 1`` → a single observation (first == last, a change is not
    computable); ``n_valid >= 2`` → two real endpoints to compute a change from.
    """
    if field not in daily.columns:
        return None, None, 0
    valid = daily[field].dropna()
    if valid.empty:
        return None, None, 0
    return valid.iloc[0], valid.iloc[-1], int(len(valid))


def _rate_trend(daily: pd.DataFrame, field: str, ndigits: int = 1):
    """First/last/percent-change of ``field`` over its non-null observations only.

    ``change_pct`` is None (not computable) unless there are ≥2 non-null points and a
    non-zero base — so a data gap or a single point never masquerades as a computed 0%.
    """
    first, last, n_valid = _valid_endpoints(daily, field)
    change_pct = None
    if n_valid >= 2 and first not in (None, 0):
        change_pct = round((last - first) / first * 100.0, 1)
    return _round(first, ndigits), _round(last, ndigits), change_pct


def _last_valid_rate_row(daily: pd.DataFrame):
    """The last row carrying real allocation (any ``alloc_*`` non-null), else None.

    The KPI ``latest`` anchor — the last day the well actually reported an allocated
    rate — so a tail-null day no longer blanks the cards (last-valid semantics,
    matching the ΔP tools). Anchors on the alloc_* volumes, never liquid_rate (which
    is 0, not null, on a gap row).
    """
    cols = [c for c in _ALLOC_RATE_COLS if c in daily.columns]
    if not cols or daily.empty:
        return None
    valid = daily[daily[cols].notna().any(axis=1)]
    if valid.empty:
        return None
    return valid.iloc[-1]


def _project_values(
    well_id: str,
    daily: pd.DataFrame,
    telemetry_rows: int,
) -> Dict[str, Any]:
    """Project the compute output → the model-facing ``values`` for narration + KPIs.

    These are the fields proposed for review (CurVE-decisions §3 D3 per-tool spec):
    period coverage, the latest day's rates/fluid character, and the window trend.
    The model narrates from these; it does not restate every raw number.
    """
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")
    n_days = len(daily)

    period = {
        "start": str(daily["observation_day"].min().date()) if n_days else None,
        "end": str(daily["observation_day"].max().date()) if n_days else None,
        "n_days": int(n_days),
        "n_telemetry_points": int(telemetry_rows),
    }

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    if n_days:
        # KPI cards read the last day that actually reported allocation (last-valid),
        # not the raw last row — so a tail-null day no longer blanks them.
        valid_row = _last_valid_rate_row(daily)

        def _lv(field: str, ndigits: int = 1) -> Optional[float]:
            return _round(valid_row.get(field), ndigits) if valid_row is not None else None

        latest = {
            "observation_day": (
                str(valid_row["observation_day"].date()) if valid_row is not None else None
            ),
            "oil_rate_bbl_day": _lv("alloc_oil_vol"),
            "water_rate_bbl_day": _lv("alloc_water_vol"),
            "gas_rate_mcf_day": _lv("alloc_gas_vol"),
            "liquid_rate_bbl_day": _lv("liquid_rate_bbl_day"),
            "water_cut": _lv("water_cut", 3),
            "gor": _lv("gor"),
        }
        # Trend endpoints are the first/last NON-NULL oil observations (per-series).
        oil_first, oil_last, change_pct = _rate_trend(daily, "alloc_oil_vol")
        # Honest not-computable when there aren't two real endpoints — never a false "flat".
        direction = "not_computable"
        if change_pct is not None:
            if change_pct <= -5:
                direction = "declining"
            elif change_pct >= 5:
                direction = "rising"
            else:
                direction = "flat"
        trend = {
            "oil_rate_first_bbl_day": oil_first,
            "oil_rate_last_bbl_day": oil_last,
            "oil_rate_change_pct": change_pct,
            "direction": direction,
        }

    return {"well_id": well_id, "period": period, "latest": latest, "trend": trend}


def production_history(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``production_history``: fetch → gate (before compute) → compute → figure → envelope.

    ``tool_input`` carries the backend-injected ``organization_id`` + ``well_id``
    (from the session record) merged with the model's window selectors
    (``start_date`` / ``end_date``). The well/org are NOT model-supplied.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    start_date = tool_input.get("start_date")
    end_date = tool_input.get("end_date")

    # Defensive: the engine injects org/well from the session. If they are absent
    # (e.g. a session-less call), do not fetch — report it rather than scan broadly.
    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    try:
        # Data access (the M2 seam). The join + feature engineering is the vendored
        # service, called only after the gate clears.
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id, start_date, end_date
        )

        # Gate BEFORE compute (code-enforced). Presence + readiness → status/label/flags.
        gate = run_tool_gate("production_history", telemetry_df, production_df)
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        # Vendored physics (join → feature-engineer) and the vendored V1 figure.
        analyzed, meta = run_preprocessed_analysis(
            telemetry_df, production_df, well_depth_ft=WELL_DEPTH_FT
        )
        daily = prepare_daily_data(analyzed)
        figure = build_allocation_temporal(daily, title=f"Production History — {well_id}")

        values = _project_values(
            well_id, daily, telemetry_rows=int(meta.get("telemetry_rows", len(analyzed)))
        )
    except Exception as exc:  # data/compute failure → structured envelope, not a raw exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # carried from the gate, not hardcoded
        flags=gate["flags"],
        figure_ref=f"production_history::{well_id}",
        figure=figure,  # → UI only; the engine strips this before the model sees it
    )


# --- real tool: water_cut_gor_history -----------------------------------------


def _project_wcg_values(
    well_id: str,
    daily: pd.DataFrame,
    telemetry_rows: int,
) -> Dict[str, Any]:
    """Project the compute output → the model-facing ``values`` for the water-cut/GOR story.

    Same shape as ``_project_values`` (period / latest / trend) but the fluid-character
    fields the operator's water-cut/GOR-over-time question is about: latest water cut +
    GOR + liquid rate, and the window trend in each. The model narrates from these.
    """
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")
    n_days = len(daily)

    period = {
        "start": str(daily["observation_day"].min().date()) if n_days else None,
        "end": str(daily["observation_day"].max().date()) if n_days else None,
        "n_days": int(n_days),
        "n_telemetry_points": int(telemetry_rows),
    }

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    if n_days:
        # Last day with real allocation (last-valid) — un-blanks the cards on tail-null wells.
        valid_row = _last_valid_rate_row(daily)

        def _lv(field: str, ndigits: int = 1) -> Optional[float]:
            return _round(valid_row.get(field), ndigits) if valid_row is not None else None

        latest = {
            "observation_day": (
                str(valid_row["observation_day"].date()) if valid_row is not None else None
            ),
            "water_cut": _lv("water_cut", 3),
            "gor": _lv("gor"),
            "liquid_rate_bbl_day": _lv("liquid_rate_bbl_day"),
            "oil_rate_bbl_day": _lv("alloc_oil_vol"),
            "water_rate_bbl_day": _lv("alloc_water_vol"),
            "gas_rate_mcf_day": _lv("alloc_gas_vol"),
        }
        # Per-series first/last NON-NULL endpoints — either boundary may be the null one.
        wc_first, wc_last, wc_change = _rate_trend(daily, "water_cut", 3)
        gor_first, gor_last, gor_change = _rate_trend(daily, "gor")
        # Honest not-computable when water cut has no two real endpoints — never a false "flat".
        direction = "not_computable"
        if wc_change is not None:
            if wc_change >= 5:
                direction = "watering_up"
            elif wc_change <= -5:
                direction = "drying_out"
            else:
                direction = "flat"
        trend = {
            "water_cut_first": wc_first,
            "water_cut_last": wc_last,
            "water_cut_change_pct": wc_change,
            "gor_first_scf_bbl": gor_first,
            "gor_last_scf_bbl": gor_last,
            "gor_change_pct": gor_change,
            "direction": direction,
        }

    return {"well_id": well_id, "period": period, "latest": latest, "trend": trend}


def water_cut_gor_history(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``water_cut_gor_history``: fetch → gate (before compute) → compute → figure → envelope.

    Mirrors ``production_history`` 1:1 — same data path, same per-tool gate (Validated
    fluid-character calcs), same shared envelope. Answers the operator's water-cut /
    GOR-over-time question. Org/well are backend-injected, never model-supplied.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    start_date = tool_input.get("start_date")
    end_date = tool_input.get("end_date")

    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    try:
        # Re-fetch its own telemetry+production frame via the SAME data path (per-session
        # re-read; the known double-fetch latency is logged Bucket-C debt, not special-cased).
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id, start_date, end_date
        )

        gate = run_tool_gate("water_cut_gor_history", telemetry_df, production_df)
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        # Vendored physics (wraps the existing water-cut/GOR/liquid compute fns inside
        # engineer_features) and the vendored V2 water-cut-vs-GOR figure.
        analyzed, meta = run_preprocessed_analysis(
            telemetry_df, production_df, well_depth_ft=WELL_DEPTH_FT
        )
        daily = prepare_daily_data(analyzed)
        figure = build_water_cut_gor_analysis(
            daily, title=f"Water Cut & GOR History — {well_id}"
        )

        values = _project_wcg_values(
            well_id, daily, telemetry_rows=int(meta.get("telemetry_rows", len(analyzed)))
        )
    except Exception as exc:  # data/compute failure → structured envelope, not a raw exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # carried from the gate, not hardcoded
        flags=gate["flags"],
        figure_ref=f"water_cut_gor_history::{well_id}",
        figure=figure,  # → UI only; the engine strips this before the model sees it
    )


# --- real tools: delta_p_frequency + delta_p_composition (ΔP history) ----------
# Both sit on the SAME vendored delta_p_pump preprocessed compute and differ only in
# projection / x-axis / figure. They are the FIRST CurVE tools to carry the Estimated
# trust label end-to-end (depth + SG resolved by curve.delta_p_inputs) and the first to
# resolve a real depth (curve.well_depth rrc read) with an operator override. PIP is
# measured-or-missing — ΔP is computed on PIP-present rows; PIP-absent rows are excluded
# with a coverage flag; zero PIP coverage hard-blocks. Org/well/overrides are
# backend-injected (not model args); the Converse spec exposes only the time window.


def _dp_period(daily: pd.DataFrame, coverage: Dict[str, Any]) -> Dict[str, Any]:
    """Period coverage block shared by both ΔP tools, including PIP coverage."""
    n_days = len(daily)
    return {
        "start": str(daily["observation_day"].min().date()) if n_days else None,
        "end": str(daily["observation_day"].max().date()) if n_days else None,
        "n_days": int(n_days),
        "n_telemetry_points": int(coverage.get("n_present", 0)),
        "pip_coverage": {
            "rows_with_pip": int(coverage.get("n_present", 0)),
            "rows_total": int(coverage.get("n_total", 0)),
            "fraction": coverage.get("fraction", 0.0),
        },
    }


def _trend_of(daily: pd.DataFrame, field_name: str, ndigits: int = 1):
    """First/last/percent-change of a daily field (None-safe)."""
    if not len(daily):
        return None, None, None
    first = _round(daily.iloc[0].get(field_name), ndigits)
    last = _round(daily.iloc[-1].get(field_name), ndigits)
    change_pct = None
    if first not in (None, 0) and last is not None:
        change_pct = round((last - first) / first * 100.0, 1)
    return first, last, change_pct


def _affinity_normalized_dp(hz_first, hz_last, observed_dp_change_pct) -> Dict[str, Any]:
    """Frequency-normalized ΔP drift (endpoints-only, affinity-referenced).

    Compares the observed ΔP change over the window against the change the affinity law
    predicts from the speed change alone (ΔP ∝ N², fixed-point / first-order):

      * ``affinity_expected_delta_p_change_pct`` = ((N_last/N_first)² − 1) × 100
      * ``observed_delta_p_change_pct``          = the already-computed observed change
      * ``affinity_adjusted_delta_p_pct``        = observed − affinity-expected (headline
        drift against the speed change)

    This is NOT a ΔP-on-Hz regression slope. It stays Estimated (first-order). When the
    speed barely moved (|Δf| < ``MINIMAL_FREQ_CHANGE_HZ``) or an endpoint is missing, the
    affinity-derived values are ``None`` with a reason — reporting a "drift" with no speed
    change would just re-label the raw observed change as spurious drift.
    """
    block = {
        "affinity_expected_delta_p_change_pct": None,
        "observed_delta_p_change_pct": observed_dp_change_pct,
        "affinity_adjusted_delta_p_pct": None,
        "reason": None,
    }
    if hz_first in (None, 0) or hz_last is None or observed_dp_change_pct is None:
        block["reason"] = "frequency or ΔP endpoints unavailable"
        return block
    if abs(hz_last - hz_first) < MINIMAL_FREQ_CHANGE_HZ:
        block["reason"] = f"no material speed change (|Δf| < {MINIMAL_FREQ_CHANGE_HZ} Hz)"
        return block
    speed_ratio = hz_last / hz_first
    expected = round((speed_ratio ** 2 - 1.0) * 100.0, 1)
    block["affinity_expected_delta_p_change_pct"] = expected
    block["affinity_adjusted_delta_p_pct"] = round(observed_dp_change_pct - expected, 1)
    return block


def _project_dp_frequency_values(
    well_id: str,
    daily: pd.DataFrame,
    resolved: "delta_p_inputs.DeltaPInputs",
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """ΔP-vs-frequency narration payload: ΔP_pump + motor frequency, trend, provenance."""
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    affinity_normalized: Dict[str, Any] = {}
    if len(daily):
        last = daily.iloc[-1]
        latest = {
            "observation_day": str(last["observation_day"].date()),
            "delta_p_pump_psi": _round(last.get("delta_p_pump_psi")),
            "motor_frequency_hz": _round(last.get("motor_frequency_hz")),
            "pump_intake_pressure_psi": _round(last.get("pump_intake_pressure_psi")),
            "p_dis_downhole_psi": _round(last.get("p_dis_downhole_psi")),
        }
        dp_first, dp_last, dp_change = _trend_of(daily, "delta_p_pump_psi")
        hz_first, hz_last, hz_change = _trend_of(daily, "motor_frequency_hz")
        direction = "flat"
        if dp_change is not None:
            direction = "rising" if dp_change >= 5 else "declining" if dp_change <= -5 else "flat"
        trend = {
            "delta_p_pump_first_psi": dp_first,
            "delta_p_pump_last_psi": dp_last,
            "delta_p_pump_change_pct": dp_change,
            "motor_frequency_first_hz": hz_first,
            "motor_frequency_last_hz": hz_last,
            "motor_frequency_change_pct": hz_change,
            "direction": direction,
        }
        # Frequency-normalized ΔP drift: observed ΔP change vs the (N_last/N_first)²
        # affinity-expected change (replaces the model-inferred inverse read with data).
        affinity_normalized = _affinity_normalized_dp(hz_first, hz_last, dp_change)

    return {
        "well_id": well_id,
        "period": _dp_period(daily, coverage),
        "latest": latest,
        "trend": trend,
        "affinity_normalized": affinity_normalized,
        "inputs": resolved.as_values(),
    }


# Rounding-noise band on the friction residual: p_dis_downhole = tubing + hyd (compute
# omits friction), so the residual closes to ~0. A residual below −tol is a real
# non-closing decomposition (bad inputs / sign error), not rounding → not-decomposable.
_FRICTION_RESIDUAL_TOL_PSI = 1.0


def _dp_composition_split(latest: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit friction term + %-composition split + ΔP_pump/PIP drawdown ratio.

    Decomposes ΔP_pump into three additive terms that close the balance:
      * ``hydrostatic_psi``  = ΔP_hyd (0.433·SG·depth)
      * ``backpressure_psi`` = tubing_pressure − PIP (net surface-head-over-intake term;
        legitimately negative when intake exceeds surface tubing pressure)
      * ``friction_psi``     = the RESIDUAL that closes ΔP_pump − hydrostatic − backpressure.
        Labeled as a residual, NOT a first-principles friction calc — it absorbs friction
        and any unmodeled discharge term (≈0 while compute omits friction).

    ``composition_pct`` is each term's share of ΔP_pump (sums to ~100; a share can exceed
    100 / go negative when the hydrostatic column dominates and intake offsets it).
    ``delta_p_intake_ratio`` = ΔP_pump / PIP is drawdown severity (guarded on PIP==0/null).
    A materially-negative residual means the terms don't close → not-decomposable (honest
    ``None`` + reason, never a negative friction number).
    """
    dp = latest.get("delta_p_pump_psi")
    hyd = latest.get("delta_p_hyd_psi")
    tub = latest.get("tubing_pressure_psi")
    pip = latest.get("pump_intake_pressure_psi")

    # Drawdown severity needs only ΔP_pump + PIP — independent of the decomposition.
    ratio = None
    if pip not in (None, 0) and dp is not None:
        ratio = round(dp / pip, 2)

    block = {
        "hydrostatic_psi": hyd,
        "backpressure_psi": None,
        "friction_psi": None,
        "delta_p_pump_psi": dp,
        "composition_pct": {"hydrostatic": None, "friction": None, "backpressure": None},
        "delta_p_intake_ratio": ratio,
        "decomposable": False,
        "reason": None,
    }

    if dp in (None, 0) or hyd is None or tub is None or pip is None:
        block["reason"] = "ΔP_pump or a component input unavailable"
        return block

    backpressure = round(tub - pip, 1)
    friction = round(dp - hyd - backpressure, 1)
    block["backpressure_psi"] = backpressure

    if friction < -_FRICTION_RESIDUAL_TOL_PSI:
        # Terms don't close — report rather than emit a negative friction / bogus split.
        block["reason"] = "decomposition does not close (residual materially negative)"
        return block

    friction = max(friction, 0.0)  # clamp rounding-noise negatives to a clean 0
    block["friction_psi"] = friction
    block["composition_pct"] = {
        "hydrostatic": round(hyd / dp * 100.0, 1),
        "friction": round(friction / dp * 100.0, 1),
        "backpressure": round(backpressure / dp * 100.0, 1),
    }
    block["decomposable"] = True
    return block


def _project_dp_composition_values(
    well_id: str,
    daily: pd.DataFrame,
    resolved: "delta_p_inputs.DeltaPInputs",
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """ΔP-composition narration payload: the pressure-component decomposition + trend."""
    daily = daily.copy()
    daily["observation_day"] = pd.to_datetime(daily["observation_day"], errors="coerce")
    daily = daily.dropna(subset=["observation_day"]).sort_values("observation_day")

    latest: Dict[str, Any] = {}
    trend: Dict[str, Any] = {}
    composition: Dict[str, Any] = {}
    if len(daily):
        last = daily.iloc[-1]
        latest = {
            "observation_day": str(last["observation_day"].date()),
            "pump_intake_pressure_psi": _round(last.get("pump_intake_pressure_psi")),
            "delta_p_hyd_psi": _round(last.get("delta_p_hyd_psi")),
            "tubing_pressure_psi": _round(last.get("tubing_pressure_psi")),
            "p_dis_downhole_psi": _round(last.get("p_dis_downhole_psi")),
            "delta_p_pump_psi": _round(last.get("delta_p_pump_psi")),
        }
        dp_first, dp_last, dp_change = _trend_of(daily, "delta_p_pump_psi")
        hyd_first, hyd_last, hyd_change = _trend_of(daily, "delta_p_hyd_psi")
        direction = "flat"
        if dp_change is not None:
            direction = "rising" if dp_change >= 5 else "declining" if dp_change <= -5 else "flat"
        trend = {
            "delta_p_pump_first_psi": dp_first,
            "delta_p_pump_last_psi": dp_last,
            "delta_p_pump_change_pct": dp_change,
            "delta_p_hyd_first_psi": hyd_first,
            "delta_p_hyd_last_psi": hyd_last,
            "delta_p_hyd_change_pct": hyd_change,
            "direction": direction,
        }
        # Explicit friction residual + %-composition split + drawdown-severity ratio.
        composition = _dp_composition_split(latest)

    return {
        "well_id": well_id,
        "period": _dp_period(daily, coverage),
        "latest": latest,
        "trend": trend,
        "composition": composition,
        "inputs": resolved.as_values(),
    }


def _run_delta_p_tool(
    tool_input: Dict[str, Any],
    *,
    tool_name: str,
    figure_builder: Callable[..., Any],
    value_projector: Callable[..., Dict[str, Any]],
    title: str,
) -> Dict[str, Any]:
    """Shared body for both ΔP history tools: fetch → resolve inputs → compute → gate
    (PIP coverage + projection presence) → figure → envelope.

    The two tools differ only in ``figure_builder`` (the vendored ΔP plot), the
    ``value_projector`` (the per-tool ``values`` projection / x-axis) and ``title``.
    Org/well + the resolved-input overrides arrive backend-injected in ``tool_input``.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    start_date = tool_input.get("start_date")
    end_date = tool_input.get("end_date")
    # Operator-controlled depth/SG overrides — setup-injected like org/well, never a
    # model-facing tool arg (the spec exposes only the time window).
    resolved_inputs_ctx = tool_input.get("resolved_inputs")

    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    try:
        # Re-fetch this tool's own preprocessed frame via the SAME data path.
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id, start_date, end_date
        )
        # Presence gate BEFORE compute — don't compute on absent data.
        if telemetry_df is None or len(telemetry_df) == 0 or production_df is None or len(production_df) == 0:
            return error_envelope("blocked", ["telemetry_or_production_absent"], well_id=well_id)

        # Resolve depth (real rrc → override → default) + SG (override → default). The
        # real rrc depth read is the ported app query (curve.well_depth).
        rrc_depth_ft = well_depth.fetch_well_depth_ft(well_id)
        resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)

        # Vendored ΔP_pump preprocessed compute, with the resolved depth/SG. PIP is
        # measured-or-missing inside the service (null/zero intake → NaN ΔP).
        analyzed, _meta = run_preprocessed_analysis(
            telemetry_df,
            production_df,
            well_depth_ft=resolved.depth_ft,
            sg_oil=resolved.sg_oil,
            sg_water=resolved.sg_water,
        )

        # PIP coverage — READ from the computed frame (the #2 intake-state truth).
        coverage = delta_p_inputs.pip_coverage(analyzed)

        # Gate: zero-coverage / missing-projection → blocked; else Estimated + flags.
        gate = run_delta_p_tool_gate(tool_name, analyzed, resolved, coverage)
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        # Compute ΔP only on PIP-present rows; the vendored figure builds from them.
        present = delta_p_inputs.pip_present_rows(analyzed, coverage)
        figure = figure_builder(present, title=f"{title} — {well_id}")
        daily = prepare_daily_data(present)
        values = value_projector(well_id, daily, resolved, coverage)
    except Exception as exc:  # data/compute failure → structured envelope, not an exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # Estimated (or Validated) — carried from the gate
        flags=gate["flags"],
        figure_ref=f"{tool_name}::{well_id}",
        figure=figure,  # → UI only; the engine strips it before the model sees it
    )


def probe_delta_p_readiness(
    tool_name: str,
    telemetry_df: pd.DataFrame,
    production_df: pd.DataFrame,
    well_id: str,
    resolved_inputs_ctx: Optional[Dict[str, Any]],
    rrc_depth_ft: Optional[float],
) -> Dict[str, Any]:
    """Front-load a ΔP tool's gate WITHOUT building a figure (Streamlit setup step).

    Resolves depth/SG (real rrc + any overrides), runs the vendored ΔP compute, reads
    PIP coverage, and runs the per-tool gate — returning the same ``{status,
    trust_label, flags}`` head the tool will carry, plus the coverage + resolved-input
    provenance. Lets the readiness surface show the Estimated label, the PIP coverage,
    and which depth tier fired before the operator asks a question.
    """
    if telemetry_df is None or len(telemetry_df) == 0 or production_df is None or len(production_df) == 0:
        return {
            "gate": {"status": "blocked", "trust_label": None, "flags": ["telemetry_or_production_absent"]},
            "coverage": {"n_present": 0, "n_total": 0, "fraction": 0.0},
            "resolved": None,
        }
    resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)
    analyzed, _meta = run_preprocessed_analysis(
        telemetry_df,
        production_df,
        well_depth_ft=resolved.depth_ft,
        sg_oil=resolved.sg_oil,
        sg_water=resolved.sg_water,
    )
    coverage = delta_p_inputs.pip_coverage(analyzed)
    gate = run_delta_p_tool_gate(tool_name, analyzed, resolved, coverage)
    return {
        "gate": gate,
        "coverage": {k: coverage[k] for k in ("n_present", "n_total", "fraction") if k in coverage},
        "resolved": resolved.as_values(),
    }


def delta_p_frequency(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``delta_p_frequency``: ΔP_pump vs operating frequency over time (V3).

    Pressure-lift response to motor frequency — the pump's measured differential
    pressure against the frequency that drove it. Estimated trust (hydrostatic depth /
    SG); blocked when PIP coverage is zero or motor frequency is absent.
    """
    return _run_delta_p_tool(
        tool_input,
        tool_name="delta_p_frequency",
        figure_builder=build_delta_p_pump_vs_frequency,
        value_projector=_project_dp_frequency_values,
        title="ΔP Pump vs Motor Frequency",
    )


def delta_p_composition(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``delta_p_composition``: ΔP_pump pressure-component decomposition over time (V10).

    Decomposes downhole discharge pressure into its parts — pump intake, hydrostatic
    column, tubing pressure, and the resulting discharge — so the operator sees what
    builds the ΔP_pump. Same Estimated trust + PIP coverage handling as
    ``delta_p_frequency``; blocked when PIP coverage is zero or the pressure components
    are absent.
    """
    return _run_delta_p_tool(
        tool_input,
        tool_name="delta_p_composition",
        figure_builder=build_delta_p_composition,
        value_projector=_project_dp_composition_values,
        title="Temporal ΔP Composition",
    )


# --- recommendation-dependent tools (M3) --------------------------------------
# The three tools on CurVE's SECOND data path (the recommendation payload, read from
# the Athena mirror per CurVE-decisions §9). They share v1's FIRST hard block — the
# recommendation-absence block — and never synthesize a recommendation when none
# exists. The recommendation is session-controlled context (fetched per turn from the
# injected org/well), NEVER a model-facing Converse arg. Org/well + depth/SG overrides
# arrive backend-injected in ``tool_input``; the Converse spec exposes no args.


def _rec_method(tool_input: Dict[str, Any]) -> str:
    """The recommendation goal/method branch — session-controlled, not a model arg.

    Read from the session's ``resolved_inputs`` (like depth/SG overrides); defaults to
    the app's ``max_oil`` goal. The VE keys the recommended setpoint on the goal
    function — this is that goal.
    """
    ctx = tool_input.get("resolved_inputs") or {}
    return ctx.get("recommendation_method") or recommendations.DEFAULT_RECOMMENDATION_METHOD


def _resolve_current_power_kw(summary_map: Dict[str, Any]):
    """Resolve current-state power (kW) + its provenance from the rec summary snapshot.

    Returns ``(power_kw, source, term_label)``:
      * direct ``motor_power_kw`` channel (``…_1d_avg`` / bare) → term **Validated**;
      * else amp×volt 3-phase via the VE's method (PF 0.85) → term **Proxy** (v1's
        first Proxy label);
      * else ``(None, None, None)`` — no power source (the tool blocks).
    PIP-style measured-or-missing: power is never defaulted/fabricated.
    """
    if not isinstance(summary_map, dict):
        return None, None, None

    direct = _first_num(
        summary_map.get("motor_power_kw_1d_avg"),
        summary_map.get("motor_power_kw"),
    )
    if direct is not None and direct > 0:
        return direct, "motor_power_kw", "Validated"

    amps = _first_num(
        summary_map.get("motor_amps_1d_avg"),
        summary_map.get("motor_amps_1h_avg"),
        summary_map.get("motor_amps"),
    )
    volts = _first_num(
        summary_map.get("motor_volts_1d_avg"),
        summary_map.get("motor_volts_1h_avg"),
        summary_map.get("motor_volts"),
    )
    proxy_kw = recommendations.ve_power_kw_from_amps_volts(amps, volts)
    if proxy_kw is not None and proxy_kw > 0:
        return proxy_kw, "amp_x_volt", "Proxy"

    return None, None, None


def _first_num(*candidates: Any) -> Optional[float]:
    """First finite positive-or-any float among candidates, else None."""
    for value in candidates:
        v = _round(value, 6)
        if v is not None:
            return v
    return None


# --- real tool: recommendation_comparison -------------------------------------


def _project_recommendation_comparison_values(
    well_id: str, latest_row: Dict[str, Any], compare_row: Dict[str, Any], method: str
) -> Dict[str, Any]:
    """Narration payload for recommendation_comparison — current vs recommended deltas.

    Faithful extraction relative to the payload (no physics). Surfaces the rec uuid for
    traceability and the per-metric current/recommended/delta for frequency, tubing
    pressure, and the production rates.
    """
    def _triple(cur_key: str, rec_key: str, delta_key: str, nd: int = 1):
        current = _round(compare_row.get(cur_key), nd)
        recommended = _round(compare_row.get(rec_key), nd)
        delta = _round(compare_row.get(delta_key), nd)
        # % delta beside the absolute delta; guard current == 0/null → None (never a
        # divide-by-zero or an infinite %). Existing keys are untouched.
        delta_pct = None
        if current not in (None, 0) and delta is not None:
            delta_pct = round(delta / current * 100.0, 1)
        return {
            "current": current,
            "recommended": recommended,
            "delta": delta,
            "delta_pct": delta_pct,
        }

    return {
        "well_id": well_id,
        "recommendation_uuid": latest_row.get("uuid"),
        "method": method,
        "motor_frequency_hz": _triple(
            "cur_motor_frequency_hz", "rec_motor_frequency_hz", "delta_motor_frequency_hz"
        ),
        "tubing_pressure_psi": _triple(
            "cur_tubing_pressure_psi", "rec_tubing_pressure_psi", "delta_tubing_pressure_psi"
        ),
        "liquid_rate_bpd": _triple(
            "cur_liquid_rate_bpd", "rec_liquid_rate_bpd", "delta_liquid_rate_bpd"
        ),
        "oil_rate_bpd": _triple("cur_oil", "rec_oil", "delta_oil"),
        "water_rate_bpd": _triple("cur_water", "rec_water", "delta_water"),
        "gas_rate": _triple("cur_gas", "rec_gas", "delta_gas"),
    }


def recommendation_comparison(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``recommendation_comparison``: fetch rec (Athena mirror) → absence gate →
    faithful extract → figure → envelope. **Validated relative to the payload** — no
    physics; it reports what the recommendation says, not a field-validation of it.

    Blocks ``not-ready / recommendation_absent`` when no recommendation exists for the
    well/session (v1's first hard block); never synthesizes one.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    method = _rec_method(tool_input)
    try:
        latest_row = recommendations.fetch_latest_recommendation(organization_id, well_id)
        if not recommendations.has_recommendation(latest_row):
            gate = recommendation_absence_block()
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        compare_row = recommendations.extract_operating_point(latest_row, method)
        gate = run_recommendation_comparison_gate(compare_row)
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        figure = build_recommendation_comparison_bars(
            compare_row, title=f"Current vs Recommended — {well_id}"
        )
        values = _project_recommendation_comparison_values(
            well_id, latest_row, compare_row, method
        )
    except Exception as exc:  # data/parse failure → structured envelope, not an exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # Validated (rel. payload) — carried from gate
        flags=gate["flags"],
        figure_ref=f"recommendation_comparison::{well_id}",
        figure=figure,
    )


# --- real tool: affinity_check ------------------------------------------------


def _project_affinity_values(
    well_id: str,
    diagnostic: Dict[str, Any],
    compare_row: Dict[str, Any],
    resolved: "delta_p_inputs.DeltaPInputs",
) -> Dict[str, Any]:
    """Narration payload for affinity_check — speed ratio + per-check agreement."""
    flow = diagnostic.get("flow_check") or {}
    pressure = diagnostic.get("pressure_check") or {}
    power = diagnostic.get("power_check") or {}
    return {
        "well_id": well_id,
        "mode": diagnostic.get("mode"),
        "speed_ratio": _round(diagnostic.get("speed_ratio"), 3),
        "frequency_delta_hz": _round(diagnostic.get("frequency_delta_hz"), 2),
        "frequency_change_label": diagnostic.get("frequency_change_label"),
        "overall_agreement": diagnostic.get("overall_label"),
        "flow_check": {
            "available": flow.get("available"),
            "agreement": flow.get("agreement_label"),
            "predicted_liquid_rate_bpd": _round(flow.get("affinity_predicted_liquid_rate_bpd")),
            "recommended_liquid_rate_bpd": _round(flow.get("ml_recommended_liquid_rate_bpd")),
            "difference_pct": _round(flow.get("difference_pct"), 1),
        },
        "pressure_check": {
            "available": pressure.get("available"),
            "agreement": pressure.get("agreement_label"),
            "difference_pct": _round(pressure.get("difference_pct"), 1),
        },
        "power_check": {
            "available": power.get("available"),
            "agreement": power.get("agreement_label"),
            "power_source": power.get("power_source_label"),
        },
        "inputs": resolved.as_values(),
    }


def affinity_check(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``affinity_check``: affinity-law validation from the recommended freq + rates.

    Flow check rides on payload freq + liquid rate (Validated); the pressure check uses
    ΔP whose depth/SG provenance is carried by the resolution layer (Estimated unless
    every input is real). Labeled by the #4 provenance rule (weakest-wins): **Validated**
    if all inputs measured/from-payload, **Estimated** if any default feeds in (Proxy if
    proxy power feeds in). Blocks ``recommendation_absent`` when no recommendation exists.

    This is affinity *check / validation* — NOT an affinity *recommendation*; it never
    proposes a setpoint.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    resolved_inputs_ctx = tool_input.get("resolved_inputs")
    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    method = _rec_method(tool_input)
    try:
        latest_row = recommendations.fetch_latest_recommendation(organization_id, well_id)
        if not recommendations.has_recommendation(latest_row):
            gate = recommendation_absence_block()
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        compare_row = recommendations.extract_operating_point(latest_row, method)

        # Resolve depth/SG (real rrc → override → default) so the ΔP pressure check can
        # run with explicit provenance — the same resolution layer the ΔP tools use.
        rrc_depth_ft = well_depth.fetch_well_depth_ft(well_id)
        resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)
        summary = recommendations.summary_blob(latest_row)
        compare_row = ml_recommendation_calcs.augment_with_delta_p_pump(
            compare_row, summary, resolved.depth_ft, resolved.sg_oil, resolved.sg_water
        )

        cur_dp = _round(compare_row.get("cur_delta_p_pump_psi"))
        rec_dp = _round(compare_row.get("rec_delta_p_pump_psi"))
        pressure_available = cur_dp is not None and rec_dp is not None

        # Optional power check — current/recommended direct power only (recommended amps
        # are not predicted). Usually unavailable → flow/pressure-only modes.
        summary_map = ml_recommendation_calcs.parse_json_summary_data(summary)
        cur_power, cur_src, cur_term = _resolve_current_power_kw(summary_map)
        rec_power = _first_num(compare_row.get("rec_motor_power_kw"))
        power_term_label = None
        power_source_label = None
        if cur_power is not None and rec_power is not None:
            power_term_label = cur_term
            power_source_label = cur_src
        else:
            cur_power = rec_power = None  # don't run a half-populated power check

        diagnostic = compute_affinity_law_validator(
            current_frequency_hz=compare_row.get("cur_motor_frequency_hz"),
            recommended_frequency_hz=compare_row.get("rec_motor_frequency_hz"),
            current_liquid_rate_bpd=compare_row.get("cur_liquid_rate_bpd"),
            recommended_liquid_rate_bpd=compare_row.get("rec_liquid_rate_bpd"),
            current_delta_p_psi=compare_row.get("cur_delta_p_pump_psi") if pressure_available else None,
            recommended_delta_p_psi=compare_row.get("rec_delta_p_pump_psi") if pressure_available else None,
            current_power=cur_power,
            recommended_power=rec_power,
            power_source_label=power_source_label,
        )

        gate = run_affinity_check_gate(
            compare_row,
            resolved,
            pressure_available=pressure_available,
            power_term_label=power_term_label,
        )
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        figure = build_affinity_law_panel(diagnostic)
        values = _project_affinity_values(well_id, diagnostic, compare_row, resolved)
    except Exception as exc:
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # weakest-wins per provenance — carried from gate
        flags=gate["flags"],
        figure_ref=f"affinity_check::{well_id}",
        figure=figure,
    )


# --- real tool: energy_efficiency ---------------------------------------------


def _project_energy_values(
    well_id: str,
    current_diag: Dict[str, Any],
    rec_diag: Dict[str, Any],
    resolved: "delta_p_inputs.DeltaPInputs",
    terms: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Narration payload for energy_efficiency — current-state efficiency + term provenance."""
    def _state(diag: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "available": diag.get("available"),
            "mode": diag.get("mode"),
            "power_source": diag.get("power_source_label"),
            "liquid_rate_bpd": _round(diag.get("liquid_rate_bpd")),
            "delta_p_psi": _round(diag.get("delta_p_psi")),
            "hydraulic_kw_estimate": _round(diag.get("hydraulic_kw_estimate"), 2),
            "motor_power_kw": _round(diag.get("motor_power_kw"), 2),
            "proxy_power_kw": _round(diag.get("proxy_power_kw"), 2),
            "direct_power_efficiency_pct": _round(diag.get("direct_power_efficiency_pct"), 1),
            "proxy_power_efficiency_pct": _round(diag.get("proxy_power_efficiency_pct"), 1),
            "specific_power_kwh_per_liquid_bbl": _round(
                diag.get("specific_power_kwh_per_liquid_bbl"), 3
            ),
        }

    return {
        "well_id": well_id,
        "current": _state(current_diag),
        "recommended": _state(rec_diag),
        "term_provenance": {
            "liquid": terms.get("liquid"),
            "delta_p": terms.get("delta_p"),
            "power": terms.get("power"),
        },
        "inputs": resolved.as_values(),
    }


def _energy_diag_for_state(compare_row: Dict[str, Any], summary_map: Dict[str, Any], state: str):
    """Compute one state's (current/recommended) energy diagnostic + power provenance.

    Current state: measured liquid/oil + Estimated ΔP + power (direct→Validated /
    amp×volt→Proxy). Recommended state: model liquid/oil + recommended ΔP + direct
    power only (best-effort; usually unavailable). Returns ``(diagnostic, power_term)``.
    """
    if state == "current":
        liquid = compare_row.get("cur_liquid_rate_bpd")
        oil = compare_row.get("cur_oil")
        dp = compare_row.get("cur_delta_p_pump_psi")
        power_kw, power_src, power_term = _resolve_current_power_kw(summary_map)
    else:
        liquid = compare_row.get("rec_liquid_rate_bpd")
        oil = compare_row.get("rec_oil")
        dp = compare_row.get("rec_delta_p_pump_psi")
        # Recommended power: direct channel only (amps/volts are not predicted).
        power_kw = _first_num(compare_row.get("rec_motor_power_kw"))
        power_src = "motor_power_kw" if power_kw is not None else None
        power_term = "Validated" if power_kw is not None else None

    diagnostic = compute_energy_efficiency_diagnostic(
        liquid_rate_bpd=liquid,
        delta_p_psi=dp,
        motor_power_kw=power_kw if power_src == "motor_power_kw" else None,
        proxy_power_kw=power_kw if power_src == "amp_x_volt" else None,
        oil_rate_bpd=oil,
        power_source_label=power_src,
    )
    return diagnostic, power_term


def energy_efficiency(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``energy_efficiency``: hydraulic-power efficiency for the current operating
    state, anchored by the recommendation payload's measured snapshot.

    Three terms with INDEPENDENT provenance, folded weakest-wins (v1's first multi-tier
    label): liquid (measured → **Validated**) + ΔP (depth/SG via the resolution layer →
    **Estimated**) + power (direct ``motor_power_kw`` → **Validated**; amp×volt proxy →
    **Proxy**, v1's first Proxy label). Overall label = the WEAKEST term — never the
    strongest. Blocks ``recommendation_absent`` when no recommendation exists, and
    blocks when a required term (liquid/ΔP/power) is unresolved.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    resolved_inputs_ctx = tool_input.get("resolved_inputs")
    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    method = _rec_method(tool_input)
    try:
        latest_row = recommendations.fetch_latest_recommendation(organization_id, well_id)
        if not recommendations.has_recommendation(latest_row):
            gate = recommendation_absence_block()
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        compare_row = recommendations.extract_operating_point(latest_row, method)

        # ΔP needs depth/SG (Estimated provenance) + the measured intake from the summary
        # snapshot (measured-or-missing). Same resolution layer the ΔP tools use.
        rrc_depth_ft = well_depth.fetch_well_depth_ft(well_id)
        resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)
        summary = recommendations.summary_blob(latest_row)
        compare_row = ml_recommendation_calcs.augment_with_delta_p_pump(
            compare_row, summary, resolved.depth_ft, resolved.sg_oil, resolved.sg_water
        )
        summary_map = ml_recommendation_calcs.parse_json_summary_data(summary)

        # Per-term provenance (current state is the headline; weakest-wins decides label).
        liquid = _round(compare_row.get("cur_liquid_rate_bpd"))
        cur_dp = _round(compare_row.get("cur_delta_p_pump_psi"))
        _power_kw, _power_src, power_term = _resolve_current_power_kw(summary_map)
        liquid_term = "Validated" if liquid is not None and liquid > 0 else None
        delta_p_term = resolved.trust_label if cur_dp is not None and cur_dp > 0 else None

        gate = run_energy_efficiency_gate(
            liquid_term_label=liquid_term,
            delta_p_term_label=delta_p_term,
            power_term_label=power_term,
            resolved_inputs=resolved,
        )
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        current_diag, _ = _energy_diag_for_state(compare_row, summary_map, "current")
        rec_diag, _ = _energy_diag_for_state(compare_row, summary_map, "recommended")
        figure = build_energy_power_cards(current_diag, rec_diag)
        values = _project_energy_values(
            well_id,
            current_diag,
            rec_diag,
            resolved,
            {"liquid": liquid_term, "delta_p": delta_p_term, "power": power_term},
        )
    except Exception as exc:
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    return success_envelope(
        values=values,
        trust_label=gate["trust_label"],  # weakest-wins across the three terms
        flags=gate["flags"],
        figure_ref=f"energy_efficiency::{well_id}",
        figure=figure,
    )


# --- recommendation readiness probe (Streamlit front-load) ---------------------


def probe_recommendation_readiness(
    tool_name: str,
    organization_id: str,
    well_id: str,
    resolved_inputs_ctx: Optional[Dict[str, Any]],
    rrc_depth_ft: Optional[float],
    latest_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Front-load a recommendation tool's gate WITHOUT building a figure (setup step).

    Returns the same ``{status, trust_label, flags}`` head the tool will carry, so the
    availability report shows the recommendation-absence block (honest per-well) and the
    trust label before the operator asks a question. ``latest_row`` is fetched once at
    setup and shared across the three tools (per-session re-read).
    """
    if not recommendations.has_recommendation(latest_row):
        return {"gate": recommendation_absence_block()}

    method = (resolved_inputs_ctx or {}).get("recommendation_method") or recommendations.DEFAULT_RECOMMENDATION_METHOD
    compare_row = recommendations.extract_operating_point(latest_row, method)

    if tool_name == "recommendation_comparison":
        return {"gate": run_recommendation_comparison_gate(compare_row)}

    resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)
    summary = recommendations.summary_blob(latest_row)
    compare_row = ml_recommendation_calcs.augment_with_delta_p_pump(
        compare_row, summary, resolved.depth_ft, resolved.sg_oil, resolved.sg_water
    )
    summary_map = ml_recommendation_calcs.parse_json_summary_data(summary)

    if tool_name == "affinity_check":
        cur_dp = _round(compare_row.get("cur_delta_p_pump_psi"))
        rec_dp = _round(compare_row.get("rec_delta_p_pump_psi"))
        pressure_available = cur_dp is not None and rec_dp is not None
        cur_power, _src, cur_term = _resolve_current_power_kw(summary_map)
        rec_power = _first_num(compare_row.get("rec_motor_power_kw"))
        power_term_label = cur_term if (cur_power is not None and rec_power is not None) else None
        return {
            "gate": run_affinity_check_gate(
                compare_row, resolved,
                pressure_available=pressure_available,
                power_term_label=power_term_label,
            )
        }

    if tool_name == "energy_efficiency":
        liquid = _round(compare_row.get("cur_liquid_rate_bpd"))
        cur_dp = _round(compare_row.get("cur_delta_p_pump_psi"))
        _pk, _ps, power_term = _resolve_current_power_kw(summary_map)
        liquid_term = "Validated" if liquid is not None and liquid > 0 else None
        delta_p_term = resolved.trust_label if cur_dp is not None and cur_dp > 0 else None
        return {
            "gate": run_energy_efficiency_gate(
                liquid_term_label=liquid_term,
                delta_p_term_label=delta_p_term,
                power_term_label=power_term,
                resolved_inputs=resolved,
            )
        }

    raise KeyError(f"Unknown recommendation tool: {tool_name}")


# --- M4 / 4a: connection-resolution coverage probe ----------------------------
# Front-loads the well ↔ pump connection layer at setup, mirroring the readiness
# probes above: it does the orchestration (fetch telemetry/production → vendored
# analysis → total fluid → BEP-narrow the re-read catalog) and returns the per-well
# coverage report. It does NOT pick a pump (the operator does) and it does NOT touch
# curve_position (that is 4b). The report is a de-risking check, never a gate.


def probe_connection_coverage(
    organization_id: str,
    well_id: str,
    catalog_df: pd.DataFrame,
    resolved_inputs_ctx: Optional[Dict[str, Any]] = None,
    *,
    telemetry_df: Optional[pd.DataFrame] = None,
    production_df: Optional[pd.DataFrame] = None,
    rrc_depth_ft: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the per-well pump-connection coverage report (setup-step orchestration).

    Fetches telemetry + production (unless pre-fetched and passed in to avoid a double
    read), runs the vendored preprocessed analysis to get the representative total fluid
    (median liquid rate — independent of depth/SG), then BEP-narrows the re-read catalog
    with the setup-injected tolerance. Returns
    ``{report, total_fluid_bpd, n_candidates}``; ``report`` is the full
    :func:`curve.ideal_catalog.build_coverage_report` dict (total fluid, candidate count,
    candidate list, obsolete/excluded counts). Telemetry-absent → total fluid ``None`` →
    zero candidates, surfaced honestly (never a fabricated rate or a default pump).
    """
    if telemetry_df is None or production_df is None:
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id
        )

    total_fluid_bpd: Optional[float] = None
    if telemetry_df is not None and len(telemetry_df) and production_df is not None and len(production_df):
        # Liquid rate is independent of depth/SG; resolve them only to satisfy the
        # vendored analysis signature (defaults are fine for the total-fluid read).
        resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)
        analyzed, _meta = run_preprocessed_analysis(
            telemetry_df,
            production_df,
            well_depth_ft=resolved.depth_ft,
            sg_oil=resolved.sg_oil,
            sg_water=resolved.sg_water,
        )
        total_fluid_bpd = ideal_catalog.resolve_total_fluid_bpd(analyzed)

    bep_tolerance = ideal_catalog.bep_tolerance_from_context(resolved_inputs_ctx)
    report = ideal_catalog.build_coverage_report(
        catalog_df, total_fluid_bpd, well_id, bep_tolerance
    )
    return {
        "report": report,
        "total_fluid_bpd": total_fluid_bpd,
        "n_candidates": report["n_candidates"],
    }


# --- real tool: curve_position (M4 / 4b) --------------------------------------
# The headline M4 tool — "where am I on the pump curve, and how far off design am I."
# FIRST tool that is BOTH connection-dependent (the 4a manual pump pick → the picked
# pump's ideal curve) AND ΔP-input-dependent (the operating point's head from the M3
# delta_p_inputs layer). Trust = Estimated(catalog model) ⊓ ΔP-tier, weakest-wins. The
# pump pick rides the SETUP-INJECTION path (session['pump'] → tool_input['pump']); the
# stage count + operating Hz ride resolved_inputs — neither is a model-facing arg.


def curve_position(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Real ``curve_position``: pick → curve reconstruct → operating point → position → envelope.

    Two hard blocks read off the injected pick state (NOT re-derived): no pump picked →
    blocked ``pump_connection_unresolved`` ("connect a pump"); picked pump's row has
    null/incomplete curve coeffs → blocked ``pump_curve_coeffs_absent`` ("no ideal curve
    for this pump"). A missing operating point from the ΔP layer (zero PIP coverage) is
    the inherited not-ready, not a new block. Obsolete is a FLAG, never a block/downgrade.
    """
    organization_id = tool_input.get("organization_id")
    well_id = tool_input.get("well_id")
    pump = tool_input.get("pump")
    resolved_inputs_ctx = tool_input.get("resolved_inputs")

    if not organization_id or not well_id:
        return error_envelope("blocked", ["missing_org_or_well_injection"])

    # Hard block 1 — connection unresolved (no pump picked). Not proxyable: a pump pick
    # is the v1 connection ladder's manual rung; absence is honest, never a default pump.
    if not pump or not pump.get("pump_id"):
        return error_envelope(
            "blocked", ["pump_connection_unresolved: connect a pump"], well_id=well_id
        )
    # Hard block 2 — picked pump has null/incomplete curve coeffs (coverage gap). The
    # 4a pick already resolved selectability; we READ it, not re-derive selectable_mask.
    if not pump.get("selectable"):
        return error_envelope(
            "blocked",
            ["pump_curve_coeffs_absent: no ideal curve for this pump"],
            well_id=well_id,
        )

    # Stage count + operating Hz — setup-injected manual inputs (NOT well_configuration).
    scaling = curve_position_layer.resolve_scaling_inputs(resolved_inputs_ctx)
    if not scaling.ready:
        return error_envelope("blocked", scaling.flags, well_id=well_id)

    try:
        # Re-read the catalog (per-session) and pull the picked pump's full coeff row.
        catalog = ideal_catalog.fetch_ideal_catalog()
        pump_row = curve_position_layer.lookup_pump_row(catalog, pump["pump_id"])
        if pump_row is None:
            return error_envelope(
                "blocked", ["picked_pump_not_in_catalog"], well_id=well_id
            )

        # Telemetry + production via the SAME data path the ΔP tools use.
        telemetry_df, production_df = data.fetch_preprocessed_window(
            organization_id, well_id
        )
        if telemetry_df is None or len(telemetry_df) == 0 or production_df is None or len(production_df) == 0:
            return error_envelope("blocked", ["telemetry_or_production_absent"], well_id=well_id)

        # Resolve depth/SG (real rrc → override → default) for the ΔP compute, then run
        # the vendored preprocessed analysis (writes delta_p_pump_psi + liquid_rate).
        rrc_depth_ft = well_depth.fetch_well_depth_ft(well_id)
        resolved = delta_p_inputs.resolve_from_context(resolved_inputs_ctx, rrc_depth_ft)
        analyzed, _meta = run_preprocessed_analysis(
            telemetry_df,
            production_df,
            well_depth_ft=resolved.depth_ft,
            sg_oil=resolved.sg_oil,
            sg_water=resolved.sg_water,
        )

        # PIP coverage (measured-or-missing) + operating point (flow = 4a total fluid;
        # ΔP = median over PIP-present rows; head via the mixture SG).
        coverage = delta_p_inputs.pip_coverage(analyzed)
        present = delta_p_inputs.pip_present_rows(analyzed, coverage)
        operating_flow = ideal_catalog.resolve_total_fluid_bpd(analyzed)
        sg_for_dp = curve_position_layer.representative_sg(present)
        operating_point = curve_position_layer.build_operating_point(
            analyzed, operating_flow, present, sg_for_dp
        )

        # Gate: inherited ΔP not-ready (zero PIP / no op point) else Estimated ⊓ ΔP-tier,
        # with the obsolete flag carried (never a block/downgrade).
        gate = run_curve_position_gate(
            resolved, coverage, operating_point, obsolete=bool(pump.get("is_obsolete"))
        )
        if gate["status"] != "available":
            return error_envelope(gate["status"], gate["flags"], well_id=well_id)

        # Reconstruct the well-scaled curve + the affinity family + the position values.
        curve_df, family_curves, position = curve_position_layer.evaluate_position(
            pump_row, pump, scaling, operating_point, sg_for_dp
        )
        daily = prepare_daily_data(present)
        if "amp_x_volt" in daily.columns:
            observed = ideal_curve_overlay.compute_observed_proxies(daily)
        else:  # no electrical channel for the efficiency proxy — figures still draw ΔP.
            observed = daily.copy()
            observed["eff_real_proxy_ratio"] = float("nan")
        _label = (
            f"{well_id} | {pump.get('esp_model')} | "
            f"{scaling.frequency_hz:.1f} Hz | {scaling.stages} stages"
        )
        # TWO figures, ordered: single-frequency overlay first, then the affinity family
        # sweep — both mark the same operating point. UI only; stripped from the model.
        figures = [
            build_curve_position_overlay(
                observed, curve_df, operating_point,
                title=f"Operating Point vs Ideal Curve — {_label}",
            ),
            build_curve_position_family(
                observed, family_curves, operating_point,
                selected_frequency_hz=scaling.frequency_hz,
                title=f"Frequency Family (affinity sweep) — {_label}",
            ),
        ]
        values = {"well_id": well_id, **position, "inputs": resolved.as_values()}
    except Exception as exc:  # data/compute failure → structured envelope, not an exception
        return error_envelope("error", [f"data_or_compute_error: {exc}"], well_id=well_id)

    # curve_position carries a FIGURES LIST (overlay, family) rather than the singular
    # figure/figure_ref slot. ``figures`` is out-of-band + stripped from the model exactly
    # as ``figure`` is (see NON_MODEL_RESULT_KEYS); the model-facing envelope stays
    # {status, values, trust_label, flags}.
    return {
        "status": "available",
        "values": values,
        "trust_label": gate["trust_label"],  # Estimated ⊓ ΔP-tier (weakest-wins) — from gate
        "flags": gate["flags"],
        "figures": figures,  # → UI only; the engine strips this before the model sees it
    }


# --- real tool specs (Converse toolConfig shape) ------------------------------
# Converse expects inputSchema wrapped under a "json" key.

_PRODUCTION_HISTORY_SPEC = {
    "toolSpec": {
        "name": "production_history",
        "description": (
            "Retrieve historical production and telemetry for the selected well over "
            "a time window (oil/water/gas allocation rates, fluid character, recent "
            "trend). Use for questions about how the well has produced or behaved "
            "over time, recent trends, or 'what has this well been doing'. The well "
            "is already set up for this session — supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_WATER_CUT_GOR_HISTORY_SPEC = {
    "toolSpec": {
        "name": "water_cut_gor_history",
        "description": (
            "Retrieve the selected well's water-cut and gas-oil-ratio (GOR) history "
            "over a time window — how the produced-fluid character has changed "
            "(watering up / drying out, GOR trend) alongside liquid rate. Use for "
            "questions about water cut, GOR, gas-oil ratio, fluid mix, or 'is this "
            "well watering up'. Note: water cut (not water rate) and GOR (gas-OIL "
            "ratio, not gas-liquid). The well is already set up for this session — "
            "supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_DELTA_P_FREQUENCY_SPEC = {
    "toolSpec": {
        "name": "delta_p_frequency",
        "description": (
            "Retrieve the selected well's pump differential pressure (ΔP across the "
            "pump, discharge minus intake) plotted against motor frequency over a time "
            "window — the pressure-lift response to operating frequency. Use for "
            "questions about pump ΔP, pressure rise/lift across the pump, head vs "
            "frequency, or 'how is the pump's differential pressure responding to "
            "frequency'. Note: ΔP_pump (discharge − intake), not stage/TDH ΔP, and "
            "intake (PIP) is measured-only. The well is already set up for this "
            "session — supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_DELTA_P_COMPOSITION_SPEC = {
    "toolSpec": {
        "name": "delta_p_composition",
        "description": (
            "Retrieve the selected well's pump pressure decomposition over a time "
            "window — how downhole discharge pressure is built from pump intake (PIP), "
            "the hydrostatic column, and tubing pressure, and the resulting ΔP across "
            "the pump. Use for questions about the pressure components, what makes up "
            "the pump's differential pressure, hydrostatic vs tubing contribution, or "
            "'break down the pump pressure'. Intake (PIP) is measured-only; depth/SG "
            "for the hydrostatic term are resolved with provenance. The well is already "
            "set up for this session — supply only the time window."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date YYYY-MM-DD), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date YYYY-MM-DD), optional.",
                    },
                },
                "required": [],
            }
        },
    }
}

_RECOMMENDATION_COMPARISON_SPEC = {
    "toolSpec": {
        "name": "recommendation_comparison",
        "description": (
            "Retrieve the ML recommendation for the selected well and compare the "
            "recommended operating point against the current one (motor frequency, "
            "tubing pressure, and the resulting oil/water/gas/liquid rates, with the "
            "delta for each). Use for questions about what the model recommends, the "
            "recommended setpoint, current-vs-recommended, or 'what change is being "
            "suggested'. This reports the recommendation faithfully; it is not a "
            "physics validation of it. The well is already set up for this session — "
            "no arguments are needed."
        ),
        "inputSchema": {
            "json": {"type": "object", "properties": {}, "required": []}
        },
    }
}

_AFFINITY_CHECK_SPEC = {
    "toolSpec": {
        "name": "affinity_check",
        "description": (
            "Validate the ML recommendation against the pump Affinity Laws — whether "
            "the recommended change in motor frequency is consistent with the implied "
            "change in flow (and pump ΔP) under first-order affinity scaling (flow ∝ "
            "speed, head ∝ speed²). Use for questions about whether the recommendation "
            "obeys the affinity laws, is physically consistent, or 'does the frequency "
            "change match the flow change'. This is a sanity CHECK of the recommendation, "
            "not a new recommendation. The well is already set up for this session — no "
            "arguments are needed."
        ),
        "inputSchema": {
            "json": {"type": "object", "properties": {}, "required": []}
        },
    }
}

_ENERGY_EFFICIENCY_SPEC = {
    "toolSpec": {
        "name": "energy_efficiency",
        "description": (
            "Assess the selected well's energy efficiency at its current operating "
            "point — hydraulic power vs input power, the resulting efficiency, and "
            "specific power (kWh per barrel). Use for questions about pump/energy "
            "efficiency, power consumption, kWh per barrel, or 'how efficiently is this "
            "well running'. Power uses a direct motor-power channel when available, "
            "otherwise an amp×volt proxy (labeled accordingly). The well is already set "
            "up for this session — no arguments are needed."
        ),
        "inputSchema": {
            "json": {"type": "object", "properties": {}, "required": []}
        },
    }
}

_CURVE_POSITION_SPEC = {
    "toolSpec": {
        "name": "curve_position",
        "description": (
            "Determine where the selected well's pump is operating on its ideal "
            "performance curve right now, and how far off design it is — the operating "
            "point (flow + pump ΔP) overlaid on the well-scaled ideal curve, the "
            "variance from the ideal head at that flow, and the position relative to "
            "the best-efficiency point (BEP) and the recommended operating window. Use "
            "for 'where am I on the pump curve', 'am I near BEP', 'how far off design "
            "is this pump', or operating-point-vs-pump-curve questions. The well, the "
            "installed pump, and its stage count / operating frequency are already set "
            "up for this session — no arguments are needed."
        ),
        "inputSchema": {
            "json": {"type": "object", "properties": {}, "required": []}
        },
    }
}

# --- registry -----------------------------------------------------------------
# name -> {"spec": <toolSpec dict>, "fn": <callable>}. The engine builds toolConfig
# from the specs and dispatches tool_use by name to fn.

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "production_history": {"spec": _PRODUCTION_HISTORY_SPEC, "fn": production_history},
    "water_cut_gor_history": {
        "spec": _WATER_CUT_GOR_HISTORY_SPEC,
        "fn": water_cut_gor_history,
    },
    "delta_p_frequency": {"spec": _DELTA_P_FREQUENCY_SPEC, "fn": delta_p_frequency},
    "delta_p_composition": {
        "spec": _DELTA_P_COMPOSITION_SPEC,
        "fn": delta_p_composition,
    },
    "recommendation_comparison": {
        "spec": _RECOMMENDATION_COMPARISON_SPEC,
        "fn": recommendation_comparison,
    },
    "affinity_check": {"spec": _AFFINITY_CHECK_SPEC, "fn": affinity_check},
    "energy_efficiency": {"spec": _ENERGY_EFFICIENCY_SPEC, "fn": energy_efficiency},
    "curve_position": {"spec": _CURVE_POSITION_SPEC, "fn": curve_position},
}


def build_tool_config(registry: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the Converse ``toolConfig`` from a registry."""
    registry = registry if registry is not None else TOOL_REGISTRY
    return {"tools": [entry["spec"] for entry in registry.values()]}
