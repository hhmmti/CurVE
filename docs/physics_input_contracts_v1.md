# Physics Input Contracts V1

> [!note] CurVE v1 deltas (read me first)
> This file is **vendored verbatim** from the physics app's `documentation/` and documents the **app as-built**. It is the calculation-level **spec** the gate is built from — it is *not* loaded at runtime (the gate is `services/data_availability_gate.py`). CurVE consumes the same compute/gate, so contracts hold at the calculation level **except** the deltas below. Do not edit the app contracts to match CurVE; record divergence here.
>
> **CurVE v1 overrides the app contracts in these spots:**
> - **Well-depth default:** `5000 ft` (Hydrostatic pressure correction contract) → **`10,000 ft`** for CurVE. Source chain: `rrc_mining → enrichment → 10,000 ft default`; labeled Estimated when defaulted.
> - **PIP intake:** `tubing_pressure_psi / 0.45` (Pump delta-P preprocessed + recommendation contracts) → **no proxy; measured-or-missing (regression rejected — see `pip_reg_report`)**. The `predict_pip` global OLS was weak (LOWO R² ≈ −0.01, ~37% median error), so no proxy is wired. Both seams cleaned of the `/0.45` fallback: `services/preprocessed_pipeline_service.py → run_preprocessed_analysis_from_joined()` (preprocessed) and the rec-path `compute/ml_recommendation_calcs.py → augment_with_delta_p_pump()` intake fallback. Missing PIP → flagged a missing, operator-suppliable required input (`delta_p_ready=False`, added to `delta_p_missing_inputs`, `delta_p_intake_source=missing`), never backfilled with a constant.
> - **Naming debt (§5):** **live in the CurVE gate now, not deferred** — the gate maps `delta_P_*`↔`delta_p_*`, `GOR_*`↔`gor_*` across pipeline paths. The global rename stays deferred.
> - **Gate envelope:** the app gate returns summary-shaped output; the CurVE gate emits the per-tool `{status, trust_label, flags}` envelope (M2 adapter). Contract *inputs/labels* here are still authoritative; only the return shape differs.

## 1. Purpose

These contracts define the inputs required by the physics and engineering calculations that are already active in the ESP app today. They are intended to make the current calculation layer explicit before a future data availability and metadata acquisition gate is added. This document does not implement that gate and does not change app behavior. It is a V1 registry and should be updated as active calculations evolve or as new calculations are promoted from candidate status into the app.

## 2. Contract Status Rules

| Status | Meaning |
|---|---|
| Direct | Input is available directly from telemetry, production, recommendation payload, ideal catalog, metadata, or user-provided value |
| Manual required | Input is missing from data but can be supplied by the user |
| Fallback allowed | Input is missing but an approved fallback exists |
| Proxy allowed | Input is missing but an approved proxy exists |
| Blocked | Input is missing and no approved resolution path exists |

## 3. Output Trust Labels

| Output Label | Meaning |
|---|---|
| Validated | Uses direct required inputs with no proxy/fallback affecting the calculation |
| Estimated | Uses direct inputs plus accepted assumptions or manual values |
| Proxy | Uses a proxy measurement instead of the true physical measurement |
| Research prototype | Screening/research output that should not be treated as validated engineering calculation |

## 4. Active Calculation Contracts

### Liquid rate

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active |
| Purpose | Compute total liquid production rate used by downstream pressure, proxy, segmentation, and catalog-selection logic. |
| Required inputs | `alloc_oil_vol` and `alloc_water_vol` in preprocessed flow; `oil_flow_rate` and `water_flow_rate` in legacy core flow; `cur_oil` and `cur_water` or `rec_oil` and `rec_water` in recommendation extraction. |
| Optional inputs | None. |
| Allowed substitutes | Equivalent oil and water rate fields from the active pipeline path may substitute for each other if they represent the same physical quantity and units. |
| Allowed fallback / proxy | None. Missing oil or water should not be silently proxied. |
| Blocking missingness | Both component liquid-rate inputs missing for the active calculation path. |
| Warning-only missingness | One component is present but the other is missing or coerced to zero by the source pipeline logic. |
| Output columns | `liquid_rate_bbl_day`, `cur_liquid_rate_bpd`, `rec_liquid_rate_bpd`, `delta_liquid_rate_bpd`. |
| Output label | Validated |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()`; `compute/ml_recommendation_calcs.py` → `extract_compare_row()`. |
| Notes | In the recommendation compare-row path, liquid rate is computed as oil plus water from current and recommended payload branches. |

### GOR

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active |
| Purpose | Compute gas-oil ratio for fluid characterization, segmentation, and later gas-related diagnostics. |
| Required inputs | `alloc_gas_vol` plus `alloc_oil_vol` in preprocessed flow; `gas_flow_rate` plus `oil_flow_rate` in legacy core flow. |
| Optional inputs | None. |
| Allowed substitutes | Equivalent gas-rate and oil-rate fields from the active pipeline path if they carry the same units. |
| Allowed fallback / proxy | None. |
| Blocking missingness | Gas and oil inputs both missing for the selected path. |
| Warning-only missingness | Oil is zero, which makes GOR undefined and should yield `NaN` rather than an exception. |
| Output columns | `gor`, `GOR_scf_bbl`. |
| Output label | Validated |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()`; scalar helper in `compute/physics_common.py` → `calc_gor()`. |
| Notes | Zero-oil rows are valid input states but produce undefined GOR by design. |

### Water cut

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active |
| Purpose | Compute produced-water fraction for mixture property estimation, segmentation, and recommendation delta-P enrichment. |
| Required inputs | Water rate plus liquid rate for the active pipeline path. |
| Optional inputs | None. |
| Allowed substitutes | Liquid rate may be supplied directly if already computed from oil plus water in the same units. |
| Allowed fallback / proxy | None. |
| Blocking missingness | Water rate missing and no liquid rate available for the same row/state. |
| Warning-only missingness | Liquid rate equals zero, which makes water cut undefined and should yield `NaN`; clipping to `[0, 1]` remains part of the active implementation where applied. |
| Output columns | `water_cut`, `cur_water_cut`, `rec_water_cut`. |
| Output label | Validated |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()` and `create_segments()`; `compute/ml_recommendation_calcs.py` → `augment_with_delta_p_pump()`. |
| Notes | Recommendation-path water cut is derived separately for current and recommended states from oil plus water totals. |

### Mixture specific gravity

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active fallback |
| Purpose | Estimate mixed-fluid specific gravity needed for hydrostatic pressure correction and pressure conversion. |
| Required inputs | Water cut plus `sg_oil` and `sg_water`. |
| Optional inputs | None. |
| Allowed substitutes | Per-well validated SG values from metadata or direct user-provided values may replace defaults. |
| Allowed fallback / proxy | Approved fallback is the current app default/manual SG path: `sg_oil=0.85`, `sg_water=1.00` when validated well-specific values are unavailable. |
| Blocking missingness | Water cut unresolved and no manual fluid-property assumption supplied. |
| Warning-only missingness | SG values are default/manual rather than validated per-well properties; water cut is `NaN` and implementation falls back to oil SG. |
| Output columns | `sg_mixture`, `SG`, `cur_sg_mix`, `rec_sg_mix`. |
| Output label | Estimated |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()`; `compute/ml_recommendation_calcs.py` → `augment_with_delta_p_pump()`; helper in `compute/physics_common.py` → `calc_mixture_sg()`. |
| Notes | This is not a laboratory fluid-property model; it is a weighted SG estimate using current app assumptions. |

### Hydrostatic pressure correction

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active fallback |
| Purpose | Estimate hydrostatic pressure from the fluid column between surface pressure reference and downhole pump depth. |
| Required inputs | Mixture SG and `well_depth_ft`. |
| Optional inputs | None. |
| Allowed substitutes | Validated metadata depth or manual engineer-entered depth may substitute for the app default depth. |
| Allowed fallback / proxy | Approved fallback is canonical default depth `5000.0 ft`; SG may also be default/manual as defined in the mixture-SG contract. |
| Blocking missingness | Mixture SG unresolved and no approved assumption path; depth unresolved and no default/manual value available. |
| Warning-only missingness | Depth is default/manual; SG is default/manual; resulting hydrostatic pressure should be treated as estimated rather than validated. |
| Output columns | `delta_p_hyd_psi`, `delta_P_hyd_psi`, `cur_delta_p_hyd_psi`, `rec_delta_p_hyd_psi`. |
| Output label | Estimated |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()`; `compute/ml_recommendation_calcs.py` → `augment_with_delta_p_pump()` and `enrich_surface_points_for_panel2()`; helper in `compute/physics_common.py` → `calc_hydrostatic_pressure_psi()`. |
| Notes | The active formula is purely hydrostatic and does not model multiphase slip or friction losses. |

### Downhole discharge pressure

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active fallback |
| Purpose | Estimate pump discharge pressure downhole as the intermediate state required for pump delta-P. |
| Required inputs | `tubing_pressure_psi` and hydrostatic pressure correction. |
| Optional inputs | None. |
| Allowed substitutes | Equivalent tubing-pressure field from the active recommendation or preprocessed path. |
| Allowed fallback / proxy | No tubing-pressure proxy is approved. Hydrostatic pressure may already be estimated under its own contract. |
| Blocking missingness | `tubing_pressure_psi` missing. |
| Warning-only missingness | Hydrostatic term is estimated because depth or SG came from manual/default inputs. |
| Output columns | `p_dis_downhole_psi`, `P_dis_downhole_psi`, `cur_p_dis_downhole_psi`, `rec_p_dis_downhole_psi`. |
| Output label | Estimated |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()`; `compute/ml_recommendation_calcs.py` → `augment_with_delta_p_pump()`; helper in `compute/physics_common.py` → `calc_discharge_pressure_downhole_psi()`. |
| Notes | If tubing pressure is direct but hydrostatic correction is estimated, the output remains estimated. |

### Pump delta-P, preprocessed analysis

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active |
| Purpose | Estimate observed pressure rise across the pump for preprocessed telemetry and production analysis. |
| Required inputs | Downhole discharge pressure and `pump_intake_pressure_psi`. |
| Optional inputs | None. |
| Allowed substitutes | None for direct intake pressure beyond the active pipeline field. |
| Allowed fallback / proxy | PIP: no proxy; measured-or-missing (regression rejected — see `pip_reg_report`). The legacy `tubing_pressure_psi / 0.45` intake estimate was removed; null/zero intake is marked missing upstream, never backfilled with a constant. |
| Blocking missingness | Intake pressure missing (no proxy path; operator-suppliable); discharge pressure missing. |
| Warning-only missingness | Discharge pressure is estimated because hydrostatic inputs were estimated. |
| Output columns | `delta_p_pump_psi`, `delta_P_pump_psi`. |
| Output label | Estimated |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()`; intake measured-or-missing in `services/preprocessed_pipeline_service.py` → `run_preprocessed_analysis_from_joined()`; helper in `compute/physics_common.py` → `calc_pump_delta_p_psi()`. |
| Notes | Intake is measured-or-missing; the preprocessed calculation returns `NaN` delta-P when intake pressure is missing (no proxy). |

### Pump delta-P, recommendation analysis

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active |
| Purpose | Estimate current and recommended pump delta-P values for recommendation comparison and ideal-curve overlay anchoring. |
| Required inputs | Current tubing pressure 1-day average, current intake pressure 1-day average, current oil and water allocation, recommended tubing pressure, recommended oil, recommended water, `well_depth_ft`, `sg_oil`, `sg_water`. |
| Optional inputs | Direct recommendation-state intake pressure if it becomes available later. |
| Allowed substitutes | Manual or metadata depth and SG values may replace defaults. |
| Allowed fallback / proxy | PIP: no proxy; measured-or-missing (regression rejected — see `pip_reg_report`). The `current tubing pressure / 0.45` intake fallback was removed; missing current intake is flagged a missing required input, never proxied. Approved recommendation assumption is `recommended intake = current intake`. |
| Blocking missingness | Current intake pressure missing (no proxy path; operator-suppliable); recommended tubing pressure missing; current and recommended oil plus water totals unresolved. |
| Warning-only missingness | Recommendation intake reused current intake; SG or depth came from default/manual values. |
| Output columns | `cur_delta_p_pump_psi`, `rec_delta_p_pump_psi`, `delta_delta_p_pump_psi`, `cur_pump_intake_pressure_psi`, `rec_pump_intake_pressure_psi`, `delta_p_ready`, `delta_p_missing_inputs`, `delta_p_intake_fallback_used`, `delta_p_intake_source`. |
| Output label | Estimated |
| Current implementation location | `compute/ml_recommendation_calcs.py` → `augment_with_delta_p_pump()`; orchestration in `services/ml_recommendation_service.py` → `build_analysis_from_latest_row()` and `build_grid_analysis_payload()`. |
| Notes | This contract is recommendation-specific and intentionally carries source and readiness metadata alongside the computed delta-P values. |

### Scenario surface delta-P enrichment

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active fallback |
| Purpose | Ensure scenario-grid points have a usable pump delta-P field for panel projection and curve comparison even when the source surface payload is incomplete. |
| Required inputs | `tubing_pressure_psi` per scenario row plus either direct `scenario_delta_p_pump_psi` or enough context to engineer it: scenario water cut or scenario oil and water rates or default water cut, recommendation/current intake context, `well_depth_ft`, `sg_oil`, `sg_water`. |
| Optional inputs | Direct `scenario_water_cut`, `scenario_oil_bpd`, `scenario_water_bpd`, and direct `scenario_delta_p_pump_psi`. |
| Allowed substitutes | `rec_water_cut` or `cur_water_cut` from compare-row context may substitute for missing scenario water cut. |
| Allowed fallback / proxy | Approved fallback is to keep direct `scenario_delta_p_pump_psi` when present; otherwise engineer it from tubing pressure, hydrostatic correction, and recommendation/current intake context. Depth and SG may use active defaults/manual values. |
| Blocking missingness | Both direct scenario delta-P and the engineered input path are unresolved. |
| Warning-only missingness | Direct scenario delta-P absent and engineered fallback used; scenario water cut inferred from rates or compare-row defaults; recommendation intake borrowed from current intake. |
| Output columns | `scenario_delta_p_pump_psi`, `scenario_flow_bpd`, `scenario_water_cut`, `scenario_oil_bpd`, `scenario_water_bpd`. |
| Output label | Estimated |
| Current implementation location | `compute/ml_recommendation_calcs.py` → `enrich_surface_points_for_panel2()`. |
| Notes | When the source surface payload already contains `scenario_delta_p_pump_psi`, that direct value remains authoritative. |

### Electrical power proxy

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active proxy |
| Purpose | Provide a simple electrical loading indicator for trend comparison when validated motor power is not available in the current app path. |
| Required inputs | `motor_amps` and `motor_volts`, or the first detected amp/current and voltage columns in the legacy core path. |
| Optional inputs | None. |
| Allowed substitutes | Equivalent telemetry columns representing current and voltage. |
| Allowed fallback / proxy | This output is itself the approved proxy. No further fallback is defined. |
| Blocking missingness | Amp/current input missing or voltage input missing. |
| Warning-only missingness | Candidate telemetry columns were auto-detected rather than explicitly named in the legacy core path. |
| Output columns | `amp_x_volt`. |
| Output label | Proxy |
| Current implementation location | `compute/preprocessed_calcs.py` → `engineer_features()`; `compute/core_calcs.py` → `engineer_features()`. |
| Notes | This is not true three-phase motor power and must not be presented as validated kW or HP. |

### Hydraulic HP proxy

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active proxy |
| Purpose | Provide a relative hydraulic-load indicator for ideal-versus-observed overlay work when validated hydraulic horsepower is not separately computed. |
| Required inputs | `liquid_rate_bbl_day` and `delta_p_pump_psi`. |
| Optional inputs | None. |
| Allowed substitutes | None beyond the active liquid-rate and delta-P outputs. |
| Allowed fallback / proxy | This output is itself a proxy and inherits any estimated/fallback quality from pump delta-P. |
| Blocking missingness | Liquid rate missing or pump delta-P missing. |
| Warning-only missingness | Pump delta-P came from an estimated intake or estimated hydrostatic term. |
| Output columns | `hhp_proxy`. |
| Output label | Proxy |
| Current implementation location | `compute/ideal_curve_overlay.py` → `compute_observed_proxies()`. |
| Notes | This is not validated hydraulic horsepower; it is an overlay-oriented proxy. |

### Electrical HP / BHP proxy

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active proxy |
| Purpose | Approximate electrical horsepower demand for overlay comparison when direct power telemetry is unavailable. |
| Required inputs | `amp_x_volt` and power factor assumption `pf`. |
| Optional inputs | None. |
| Allowed substitutes | A direct measured power factor may replace the default when supported later. |
| Allowed fallback / proxy | Approved proxy path uses `amp_x_volt` with default `pf = 0.90` and `WATTS_PER_HP`. |
| Blocking missingness | `amp_x_volt` missing. |
| Warning-only missingness | Default power factor used rather than a measured value. |
| Output columns | `bhp_proxy`. |
| Output label | Proxy |
| Current implementation location | `compute/ideal_curve_overlay.py` → `compute_observed_proxies()`. |
| Notes | This is an electrical HP proxy only and should not be interpreted as validated brake horsepower. |

### Observed efficiency proxy ratio

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active proxy |
| Purpose | Provide a relative observed efficiency indicator for overlay comparison using currently available proxy terms. |
| Required inputs | `hhp_proxy` and `bhp_proxy`. |
| Optional inputs | None. |
| Allowed substitutes | None. |
| Allowed fallback / proxy | This output is itself a proxy and inherits the proxy/estimated status of both numerator and denominator. |
| Blocking missingness | `hhp_proxy` missing or `bhp_proxy` missing. |
| Warning-only missingness | `bhp_proxy` is zero or invalid, which yields `NaN`; upstream delta-P or power-factor assumptions affect the ratio. |
| Output columns | `eff_real_proxy_ratio`. |
| Output label | Proxy |
| Current implementation location | `compute/ideal_curve_overlay.py` → `compute_observed_proxies()`. |
| Notes | This is not validated true ESP efficiency and must remain explicitly labeled as a proxy ratio. |

### Ideal pump curve generation

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Catalog-based |
| Purpose | Generate ideal pump performance curves for the selected catalog pump, operating frequency, and stage count. |
| Required inputs | Catalog polynomial coefficients `ideal_head_c1...c6` and `ideal_power_c1...c6`, catalog flow anchors, `frequency_hz`, and `stages`. |
| Optional inputs | `sg_for_dp` for pressure conversion. |
| Allowed substitutes | Equivalent catalog coefficient fields from the same pump library schema. |
| Allowed fallback / proxy | `sg_for_dp` defaults to `1.0` if not supplied. |
| Blocking missingness | Required pump-row coefficients missing; frequency or stage count unresolved. |
| Warning-only missingness | SG for pressure conversion left at default `1.0`. |
| Output columns | `flow_bpd`, `head_ft`, `delta_p_psi`, `bhp_hp`, `eff_ideal_ratio`. |
| Output label | Estimated |
| Current implementation location | `compute/ideal_curve_overlay.py` → `build_ideal_curve_for_frequency()` and `build_multi_frequency_curves()`; orchestration in `services/ideal_curve_service.py` → `build_ideal_payload()` and `services/ml_recommendation_service.py` → `build_analysis_from_latest_row()`. |
| Notes | This is catalog-model output, not field-validated pump behavior. |

### Ideal head-to-delta-P conversion

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Catalog-based |
| Purpose | Convert ideal catalog head into pressure units so ideal curves can be compared against observed or recommended pump delta-P. |
| Required inputs | Ideal head in feet and `sg_for_dp`. |
| Optional inputs | None. |
| Allowed substitutes | Any caller-supplied fluid-specific SG appropriate for the comparison context. |
| Allowed fallback / proxy | Approved fallback is `sg_for_dp = 1.0` when no fluid-specific SG is provided. |
| Blocking missingness | Head missing. |
| Warning-only missingness | SG left at default `1.0`, so converted pressure should be treated as estimated. |
| Output columns | `delta_p_psi`. |
| Output label | Estimated |
| Current implementation location | `compute/ideal_curve_overlay.py` → `build_ideal_curve_for_frequency()`; helper in `compute/physics_common.py` → `calc_pressure_psi_from_head_ft()`. |
| Notes | This is an ideal-curve conversion step, not a field measurement. |

### Ideal pump efficiency ratio

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Catalog-based |
| Purpose | Compute the catalog-based ideal efficiency ratio for the generated ideal pump curve. |
| Required inputs | Ideal flow, ideal head, ideal BHP, and the catalog efficiency denominator constant. |
| Optional inputs | None. |
| Allowed substitutes | None in V1. |
| Allowed fallback / proxy | None beyond the catalog model itself. |
| Blocking missingness | Ideal BHP missing or invalid across the curve. |
| Warning-only missingness | BHP equals zero for a point, which yields `NaN` at that point. |
| Output columns | `eff_ideal_ratio`. |
| Output label | Estimated |
| Current implementation location | `compute/ideal_curve_overlay.py` → `build_ideal_curve_for_frequency()`. |
| Notes | This is a catalog-based ratio and should not be confused with observed field efficiency. |

### Fluid segmentation

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active fallback |
| Purpose | Group rows into fluid-behavior buckets for diagnostics, summarization, and segment-based analysis. |
| Required inputs | Water cut and GOR for the selected path. |
| Optional inputs | None. |
| Allowed substitutes | None beyond the active computed water-cut and GOR fields. |
| Allowed fallback / proxy | Undefined-category assignment is the approved fallback behavior when water cut or GOR missing. |
| Blocking missingness | None at the pipeline level because the implementation can assign undefined groups. |
| Warning-only missingness | Water cut or GOR missing, causing `undefined` segmentation; segmentation thresholds differ between `preprocessed_calcs.py` and `core_calcs.py`. |
| Output columns | `seg_liquid_comp`, `seg_gas`, `segment`. |
| Output label | Estimated |
| Current implementation location | `compute/preprocessed_calcs.py` → `create_segmentation()`; `compute/core_calcs.py` → `create_segments()`. |
| Notes | This is diagnostic grouping logic, not a primary physics measurement. |

### Pump candidate narrowing

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Catalog-based |
| Purpose | Reduce the ideal pump catalog to pumps whose recommended operating range is compatible with the well's liquid rate. |
| Required inputs | `liquid_rate_bpd`, `min_recommended_bpd`, `max_recommended_bpd`, `bep_bpd`. |
| Optional inputs | `bep_tolerance` with current default `0.25`. |
| Allowed substitutes | None beyond equivalent catalog flow-range columns. |
| Allowed fallback / proxy | No fallback for missing liquid rate. |
| Blocking missingness | Liquid rate missing or non-positive; required catalog range columns missing. |
| Warning-only missingness | None in V1 beyond empty-match outcomes. |
| Output columns | Filtered catalog rows rather than a single metric column. |
| Output label | Estimated |
| Current implementation location | `services/ideal_curve_service.py` → `narrow_catalog()`; diagnostic messaging in `build_narrowing_message()`. |
| Notes | This is diagnostic candidate-selection logic using catalog rules, not a direct physical measurement. |

### Recommendation operating point extraction

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Recommendation-based |
| Purpose | Extract current and recommended operating-point values from the recommendation payload for summary, comparison, and curve anchoring. |
| Required inputs | `model_setpoint_recommendations`, `current_setpoint`, selected recommendation method branch, and payload fields for frequency, tubing pressure, oil, water, and gas. |
| Optional inputs | None. |
| Allowed substitutes | Alternative recommendation branch selected by `method` if the payload supports it. |
| Allowed fallback / proxy | No proxy path is approved for missing recommendation payload fields. |
| Blocking missingness | Missing current or recommendation payload text; missing selected method branch; missing core fields needed for the comparison row. |
| Warning-only missingness | Missing oil or water in the current implementation may collapse to zero in liquid-rate arithmetic; missing individual deltas become `NaN`. |
| Output columns | `cur_motor_frequency_hz`, `rec_motor_frequency_hz`, `delta_motor_frequency_hz`, `cur_tubing_pressure_psi`, `rec_tubing_pressure_psi`, `delta_tubing_pressure_psi`, `cur_oil`, `rec_oil`, `cur_water`, `rec_water`, `cur_gas`, `rec_gas`, `cur_liquid_rate_bpd`, `rec_liquid_rate_bpd`, delta fields. |
| Output label | Validated |
| Current implementation location | `compute/ml_recommendation_calcs.py` → `extract_compare_row()` and `build_summary_table()`; orchestration in `services/ml_recommendation_service.py` → `build_analysis_from_latest_row()`. |
| Notes | Trust is direct relative to the recommendation payload, not relative to field validation of the recommendation itself. |

### BEP position diagnostic

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active diagnostic / Catalog-based |
| Purpose | Evaluate current and recommended operating-point flow positions relative to the selected pump's catalog BEP and recommended operating flow range. |
| Required inputs | `cur_liquid_rate_bpd`, `rec_liquid_rate_bpd`, `bep_bpd`, `min_recommended_bpd`, `max_recommended_bpd`, and selected pump candidate row. |
| Optional inputs | Pump label and pump source metadata for diagnostics/reporting. |
| Allowed substitutes | None in V1. |
| Allowed fallback / proxy | None. Diagnostic is unavailable when BEP/range/current/recommended flow inputs are missing or invalid. |
| Blocking missingness | Missing/invalid current flow (`<=0`), missing/invalid recommended flow (`<=0`), missing/invalid BEP (`<=0`), missing min or max recommended flow, or invalid catalog range where min > max. |
| Warning-only missingness | Missing pump-source metadata when the diagnostic can still compute from valid core inputs. |
| Output columns / keys | `bep_diagnostic.available`, `bep_diagnostic.reason_unavailable`, `bep_diagnostic.pump_label`, `bep_diagnostic.pump_source`, `bep_diagnostic.bep_bpd`, `bep_diagnostic.min_recommended_bpd`, `bep_diagnostic.max_recommended_bpd`, `bep_diagnostic.current`, `bep_diagnostic.recommended`, `bep_diagnostic.movement`. |
| Output label | Estimated (catalog-based diagnostic) |
| Current implementation location | `compute/ideal_curve_overlay.py` → `compute_bep_position_diagnostic()`; integration in `services/ml_recommendation_service.py` → `build_analysis_from_latest_row()`; UI in `ml_recommendation_page.py` → `BEP / Operating Range` tab. |
| Notes | Catalog-based BEP position diagnostic only; not a true field efficiency measurement and not an optimization rule. |

### Affinity Law validator

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active diagnostic / Estimated |
| Purpose | Compare ML-recommended operating-point changes against first-order field-level Affinity Law expectations using current and recommended points. |
| Required inputs | `current_motor_frequency_hz`, `recommended_motor_frequency_hz`, `current_liquid_rate_bpd`, `recommended_liquid_rate_bpd`. |
| Optional inputs | `current_delta_p_pump_psi`, `recommended_delta_p_pump_psi`, direct `motor_power_kw`, `bhp_proxy`, `amp_x_volt`. |
| Allowed substitutes | Equivalent compare-row fields may be used when names differ, as long as they represent the same current/recommended quantities and units. |
| Allowed fallback / proxy | Power proxy is allowed via `bhp_proxy` or `amp_x_volt`; delta-P check is optional and skipped when delta-P inputs are missing; no fallback for missing frequency or missing current/recommended liquid rate. |
| Blocking missingness | Missing/invalid frequency inputs (`<=0`) or missing/invalid current/recommended liquid-rate inputs (`<=0`). |
| Warning-only missingness | Missing delta-P or power inputs run reduced mode (`Pressure` or `Flow-only`) rather than blocking. |
| Output columns / keys | `affinity_law_diagnostic.available`, `reason_unavailable`, `mode`, `trust_label`, `current_frequency_hz`, `recommended_frequency_hz`, `frequency_delta_hz`, `speed_ratio`, `frequency_change_label`, `flow_check`, `pressure_check`, `power_check`, `overall_label`, `notes`, `gate`. |
| Output label | Estimated / Diagnostic; Proxy when proxy power input is used. |
| Current implementation location | `compute/affinity_validator.py` → `compute_affinity_law_validator()`; gate contract in `services/data_availability_gate.py`; service integration in `services/ml_recommendation_service.py`; UI in `ml_recommendation_page.py` → `Affinity Law Check` tab. |
| Notes | Field-level sanity check only. This does not modify ideal-curve generation, recommendation optimization logic, or act as a full simulator. |

### Energy / Efficiency diagnostic

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active diagnostic |
| Purpose | Provide a compact diagnostic of hydraulic power estimate, direct/proxy efficiency, and specific power for current/recommended recommendation contexts without changing optimizer behavior. |
| Required inputs | `liquid_rate_bpd`, `delta_p_pump_psi`, and one power-source path: direct `motor_power_kw` OR proxy `bhp_proxy` OR proxy-convertible `amp_x_volt`. |
| Optional inputs | `oil_rate_bpd`, `power_source_label`. |
| Allowed substitutes | Direct `motor_power_kw` is preferred. If direct power is missing, approved proxy power sources may substitute. |
| Allowed fallback / proxy | `bhp_proxy` or `amp_x_volt`-derived power is allowed and must be labeled `Proxy`. |
| Blocking missingness | Missing/invalid `liquid_rate_bpd`; missing/invalid `delta_p_pump_psi`; missing all power-source paths; selected power source <= 0. |
| Warning-only missingness | `oil_rate_bpd` missing (oil-basis specific power omitted); proxy power path used; hydraulic side estimated from engineered delta-P basis. |
| Output keys | `energy_efficiency_diagnostic.current.*` and `energy_efficiency_diagnostic.recommended.*` including: `available`, `reason_unavailable`, `mode`, `power_source_label`, `trust_label`, `liquid_rate_bpd`, `oil_rate_bpd`, `delta_p_psi`, `hydraulic_hp_estimate`, `hydraulic_kw_estimate`, `motor_power_kw`, `proxy_power_kw`, `direct_power_efficiency_pct`, `proxy_power_efficiency_pct`, `specific_power_kwh_per_liquid_bbl`, `specific_power_kwh_per_oil_bbl`, `notes`, `gate`. |
| Output label | `Estimated` for direct-power mode (hydraulic side is engineered); `Proxy` for proxy-power mode; `Unavailable` when required inputs are unresolved. |
| Current implementation location | `compute/energy_efficiency.py` → `compute_energy_efficiency_diagnostic()`; gate contract in `services/data_availability_gate.py`; integration in `services/ml_recommendation_service.py`; UI in `ml_recommendation_page.py` → `Energy / Efficiency` tab. |
| Notes | Hydraulic power is pump-delta-P-based (`(Q*deltaP)/HHP_DENOMINATOR`) and is not a full TDH/system model. Proxy-power efficiency is diagnostic-only and must not be presented as measured ESP/system efficiency. |

### Gas-Interference Trend Screen

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active screening prototype |
| Purpose | Provide a screening-only gas-interference risk diagnostic using historical preprocessed trends in intake pressure, gas/fluid signals, and pump-behavior signals. |
| Required inputs | Enough historical rows, `pump_intake_pressure_psi`, gas indicator (`gor` OR derivable from `alloc_gas_vol` + `alloc_oil_vol`), and pump-behavior signal (`delta_p_pump_psi` OR `liquid_rate_bbl_day`). |
| Optional inputs | `water_cut`, `motor_amps`, `amp_x_volt`, `pump_intake_temperature_f`, `motor_temperature_f`. |
| Allowed substitutes | `gor` may be derived from `alloc_gas_vol / alloc_oil_vol`; pump-behavior may fall back to `liquid_rate_bbl_day` when `delta_p_pump_psi` is unavailable. |
| Allowed fallback / proxy | Approved fallback paths: `derive_gor_from_alloc` and `pump_behavior_from_liquid_rate`. Optional motor-load may use `amp_x_volt` proxy when direct amps are missing. |
| Blocking missingness | Insufficient historical rows, missing `pump_intake_pressure_psi`, missing gas indicator path, or missing both `delta_p_pump_psi` and `liquid_rate_bbl_day`. |
| Warning-only missingness | Missing optional sensors should degrade to reduced mode and lower confidence, not block execution. Unreliable intake-temperature data should be ignored rather than forced. |
| Output keys | `available`, `reason_unavailable`, `mode`, `risk_label`, `trust_label`, `time_window_summary`, `trend_statistics`, `evidence_table`, `triggered_evidence`, `missing_optional_signals`, `notes`, `gate`. |
| Output label | Research prototype / Screening prototype |
| Current implementation location | `compute/gas_interference_screen.py` → `compute_gas_interference_trend_screen()`; gate contract in `services/data_availability_gate.py`; integration in `services/ml_recommendation_service.py`; UI in `ml_recommendation_page.py` → `Gas / Bubble-Point Screening` tab. |
| Notes | Trend evidence only. This contract does not implement confirmed gas-lock detection, true downhole gas fraction calculation, bubble-point/NPSH/IPR analysis, or recommendation optimizer changes. |

### Bubble-Point / Gas Breakout Prototype

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active research prototype / manual-input path |
| Purpose | Provide a human-in-the-loop bubble-point screening diagnostic comparing a calculated or user-supplied bubble-point pressure against the ESP pump intake pressure (suction reference). |
| Required inputs | `pump_intake_pressure_psi` (primary V1 pressure reference); and either a user-supplied `selected_bubble_point_psi` OR a complete Standing-correlation input set: `R_so`, `gamma_g`, `T_f`, `API`. |
| Optional inputs | `user_supplied_bubble_point_psi`, `R_so` (solution GOR), `gamma_g` (gas SG), `T_f` (temperature °F), `API` (oil API gravity), `gor_proxy_candidate` (from summary stats), `sg_oil` (for API derivation), `pump_intake_temperature_f` and temperature summary statistics. |
| Allowed substitutes | API may be derived from `sg_oil` using `(141.5 / sg_oil) − 131.5` when direct API is unavailable. Temperature may be suggested from valid pump-intake-temperature summary keys (preferred: `pump_intake_temperature_f_0d_7d_avg` through `pump_intake_temperature_f_1d_avg`). |
| Allowed fallback / proxy | Approved fallback strategy: `standing_correlation_from_user_inputs` (use Standing-correlation fields when direct `selected_bubble_point_psi` is not supplied). Approved proxy: producing GOR may be offered as a candidate for `R_so`, but **only if explicitly user-confirmed** — it is not automatically used as true solution GOR. Default `gamma_g = 0.75` and default `T_f = 150 °F` are allowed as suggestions when user confirms them. |
| Blocking missingness | Missing `pump_intake_pressure_psi` with no resolution path; missing both `selected_bubble_point_psi` and a complete Standing-correlation input set without fallback strategy available. |
| Warning-only missingness | Temperature candidate missing from summary stats (falls back to default suggestion); API derived from SG rather than direct lab data; producing GOR used as R_so proxy after user confirmation. |
| Output keys | `available`, `reason_unavailable`, `mode` (`User-supplied P_b` / `Standing correlation` / `Unavailable`), `trust_label` (`Research prototype` / `Manual input`), `formula_text`, `input_table`, `calculated_bubble_point_psi`, `user_supplied_bubble_point_psi`, `selected_bubble_point_psi`, `pressure_reference_name`, `pressure_reference_psi`, `margin_psi`, `margin_pct`, `status_label` (`Above bubble point` / `Near bubble point` / `Below bubble point` / `Unavailable`), `source_notes`, `caution_notes`. |
| Output label | Research prototype (Standing correlation path) / Manual input (user-supplied P_b path) / Proxy / estimated (when proxy inputs are confirmed). |
| Current implementation location | `compute/bubble_point_screen.py` → `calculate_api_from_sg_oil()`, `calculate_standing_bubble_point_psi()`, `compare_bubble_point_to_pressure()`, `suggest_inputs_from_summary()`, `build_bubble_point_diagnostic()`; gate contract in `services/data_availability_gate.py` → `bubble_point_gas_breakout_prototype`; UI in `ml_recommendation_page.py` → `Gas / Bubble-Point Screening` tab, `Bubble-Point / Gas Breakout Prototype` section. |
| Notes | Standing correlation formula: `C_pb = (R_so/gamma_g)^0.83 × 10^(0.00091×T − 0.0125×API)`, `P_b = 18.2 × (C_pb − 1.4)`. Primary comparison is `P_b` vs `pump_intake_pressure_psi`. Producing GOR is not lab PVT `R_so`. Regional Permian bubble-point ranges are context only, not hidden defaults. Status bands: margin > +10% → Above bubble point; ±10% → Near bubble point; < −10% → Below bubble point. This contract does not implement NPSH, IPR, or reservoir/nodal analysis. |

### NPSH / Cavitation Screening Prototype

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Active screening prototype / manual-input path |
| Purpose | Provide a human-in-the-loop NPSH/cavitation screening diagnostic using pump intake pressure as the suction-pressure reference. Screens whether NPSHa margin over NPSHr is sufficient to avoid cavitation risk. |
| Required inputs | `pump_intake_pressure_psi` (gauge psi); `vapor_pressure_psi_abs` (absolute psia — user input preferred); `sg_mixture` (dimensionless — from payload or default); `npshr_ft` (ft — user input preferred). |
| Optional inputs | `bubble_point_psi_proxy_candidate` (from Order 10, as vapor-pressure proxy candidate if user-confirmed); `temperature_f` (context only); `notes_source` (provenance notes). |
| Allowed substitutes | `sg_mixture` may be derived from `cur_sg_mix` from recommendation compare-row, or default SG path (`sg_oil=0.85, sg_water=1.00`) when direct SG is missing. |
| Allowed fallback / proxy | Approved fallback `bubble_point_proxy_for_vapor_pressure`: Order 10 bubble-point P_b may be offered as a conservative proxy for vapor pressure **only if explicitly user-confirmed**; must trigger Proxy / Research prototype trust label. Approved fallback `placeholder_npshr_default`: generic 7.5 ft placeholder may be used **only if user-confirmed** and clearly labeled as a placeholder. Approved fallback `default_sg_values` for `sg_mixture`. |
| Blocking missingness | `pump_intake_pressure_psi` missing — no approved proxy path exists. |
| Warning-only missingness | `vapor_pressure_psi_abs` missing from data (manual required, not blocking); `npshr_ft` missing from data (manual required, not blocking); `sg_mixture` missing (default fallback allowed). |
| Output keys | `available`, `reason_unavailable`, `mode` (`Manual / Estimated` / `Proxy / Research prototype` / `Unavailable`), `trust_label`, `pip_gauge_psi`, `pip_abs_psi`, `vapor_pressure_psi_abs`, `sg_mixture`, `npshr_ft`, `npsha_ft`, `margin_ft`, `status_label` (`Safe` / `Watch` / `Risk` / `Unavailable`), `input_source_table`, `caution_notes`, `formula_text`. |
| Output label | `Manual / Estimated` when vapor pressure and NPSHr are user-confirmed; `Proxy / Research prototype` when bubble-point proxy or placeholder NPSHr is used; `Unavailable` when required pressure reference is missing. |
| Current implementation location | `compute/npsh_screen.py` → `calculate_npsha_ft()`, `calculate_npsh_margin_ft()`, `classify_npsh_margin()`, `build_npsh_diagnostic()`; gate contract in `services/data_availability_gate.py` → `npsh_cavitation_screening_prototype`; UI in `ml_recommendation_page.py` → `Gas / Bubble-Point Screening` tab, `NPSH / Cavitation Screening Prototype` section. |
| Notes | ESP V1 uses pump intake pressure (gauge) directly as the suction-pressure reference: `PIP_abs = PIP_gauge + 14.7`, `NPSHa_ft = ((PIP_abs − vapor_pressure_abs) × 2.31) / sg_mixture`, `Margin_ft = NPSHa_ft − NPSHr_ft`. This is NOT full suction-piping NPSH (no surface static head or pipe friction terms). No validated margin is produced without pump-specific NPSHr and confirmed vapor pressure. Status bands: Safe ≥ 5 ft, Watch 0–5 ft, Risk < 0 ft. |

### ML recommendation scenario grid normalization

| Contract Item | V1 Definition |
|---|---|
| Calculation status | Recommendation-based |
| Purpose | Normalize recommendation surface rows into a stable schema for grid exploration, selection, and downstream enrichment. |
| Required inputs | Surface payload rows with scenario identifiers and the canonical or alias fields for frequency, tubing pressure, economics, in-bounds status, and optional scenario flow and delta-P. |
| Optional inputs | `violated_boundaries`, direct `scenario_flow_bpd`, direct `scenario_delta_p_pump_psi`. |
| Allowed substitutes | Alias columns defined in the normalization map may substitute for canonical names. |
| Allowed fallback / proxy | Missing direct scenario flow or delta-P may remain `NaN` at normalization time and be enriched later in the active enrichment step. |
| Blocking missingness | Surface rows absent when grid analysis is requested. |
| Warning-only missingness | Individual optional scenario columns absent and deferred to enrichment; out-of-bounds points remain present but marked not selectable. |
| Output columns | `organization_id`, `well_id`, `recommendation_uuid`, `inserted_at`, `motor_frequency_hz`, `tubing_pressure_psi`, `total_economics`, `in_bounds`, `violated_boundaries`, `scenario_flow_bpd`, `scenario_delta_p_pump_psi`, `scenario_id`, `is_selectable`. |
| Output label | Validated |
| Current implementation location | `compute/ml_recommendation_calcs.py` → `normalize_recommendation_surface_rows()` and `build_recommendation_surface_grid_payload()`; orchestration in `services/ml_recommendation_service.py` → `build_grid_analysis_payload()`. |
| Notes | If later enrichment populates missing scenario fields, those enriched fields should be treated as estimated under their own contract rather than changing the normalization contract itself. |

## 5. Known Naming Debt

The current physics layer still carries a known naming split across active modules:

- `delta_P_pump_psi` vs `delta_p_pump_psi`
- `delta_P_hyd_psi` vs `delta_p_hyd_psi`
- `P_dis_downhole_psi` vs `p_dis_downhole_psi`
- `GOR_scf_bbl` vs `gor_scf_bbl`

This naming debt should not be resolved in Order 3. It should be considered explicitly when the future data availability gate is implemented so the gate can map equivalent fields safely across pipeline paths.

## 6. Contract Use in Future Gate

The future data availability and metadata acquisition gate should use these contracts as the calculation-level input registry. Each required input should be classified as `Direct`, `Manual required`, `Fallback allowed`, `Proxy allowed`, or `Blocked` before a calculation is run. A calculation should run only when all required inputs are resolved through one of the approved paths defined in its contract. The resulting output should then carry the trust label defined by the contract and adjusted for the path actually used.

## 7. Update Policy

This file should be updated whenever a new physics or engineering calculation becomes active in the app. Candidate calculations do not need full contracts until they are selected for implementation. Each new active calculation should have its contract defined before it is wired into the app runtime or into the future data availability gate.