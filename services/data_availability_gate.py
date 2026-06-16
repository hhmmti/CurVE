"""Minimal V1 data availability gate for active physics calculations.

This module is calculation-driven: it evaluates only the required fields for
selected calculations, classifies field resolution status, and returns
readiness reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ResolutionStatus = Literal[
    "Direct",
    "Manual required",
    "Fallback allowed",
    "Proxy allowed",
    "Blocked",
]

OutputTrustLabel = Literal[
    "Validated",
    "Estimated",
    "Proxy",
    "Research prototype",
]


@dataclass(frozen=True)
class RequiredField:
    """Required field contract for one calculation input."""

    name: str
    treat_zero_as_missing: bool = False
    manual_resolvable: bool = False
    fallback_strategy: Optional[str] = None
    fallback_fields: Sequence[str] = field(default_factory=tuple)
    proxy_strategy: Optional[str] = None
    proxy_fields: Sequence[str] = field(default_factory=tuple)
    notes: str = ""


@dataclass(frozen=True)
class CalculationContract:
    """V1 calculation contract used by the gate."""

    name: str
    output_label: OutputTrustLabel
    required_fields: Sequence[RequiredField]
    optional_fields: Sequence[str] = field(default_factory=tuple)
    notes: str = ""


@dataclass(frozen=True)
class FieldResolution:
    """Resolution result for one required field."""

    calculation: str
    required_field: str
    status: ResolutionStatus
    source_or_resolution: str
    required: bool
    output_label: OutputTrustLabel
    ready: bool
    notes: str = ""


@dataclass(frozen=True)
class CalculationReadinessReport:
    """Readiness report for one calculation contract."""

    calculation: str
    output_label: OutputTrustLabel
    ready: bool
    missing_fields: List[str]
    fallback_or_proxy_fields: List[str]
    notes: str
    field_resolutions: List[FieldResolution]


DEFAULT_AVAILABLE_FALLBACKS: Set[str] = {
    "default_sg_values",
    "default_well_depth_ft",
    "default_sg_for_dp",
}


def _is_usable_value(value: Any, treat_zero_as_missing: bool) -> bool:
    """Return True when a value is usable for V1 gate purposes."""
    if value is None:
        return False

    if isinstance(value, pd.Series):
        series = pd.to_numeric(value, errors="coerce")
        if series.notna().sum() == 0:
            return False
        if treat_zero_as_missing:
            return bool((series.fillna(0) != 0).any())
        return True

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        arr = pd.to_numeric(pd.Series(value), errors="coerce")
        if arr.notna().sum() == 0:
            return False
        if treat_zero_as_missing:
            return bool((arr.fillna(0) != 0).any())
        return True

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return False
        if not treat_zero_as_missing:
            return True
        numeric = value.apply(pd.to_numeric, errors="coerce")
        if numeric.notna().sum().sum() == 0:
            return False
        return bool((numeric.fillna(0) != 0).any().any())

    # Scalars and generic objects
    if pd.isna(value):
        return False
    if treat_zero_as_missing:
        try:
            return float(value) != 0.0
        except (TypeError, ValueError):
            return True
    return True


def _collect_candidate_sources(context: Dict[str, Any], field_name: str) -> List[Tuple[str, Any]]:
    """Collect possible field value sources from context."""
    candidates: List[Tuple[str, Any]] = []

    field_values = context.get("field_values", {}) or {}
    if field_name in field_values:
        candidates.append(("field_values", field_values[field_name]))

    if field_name in context:
        candidates.append(("context", context[field_name]))

    records = context.get("records", {}) or {}
    for record_name, record in records.items():
        if isinstance(record, dict) and field_name in record:
            candidates.append((f"records.{record_name}", record[field_name]))

    dataframes = context.get("dataframes", {}) or {}
    for df_name, df in dataframes.items():
        if isinstance(df, pd.DataFrame) and field_name in df.columns:
            candidates.append((f"dataframes.{df_name}[{field_name}]", df[field_name]))

    return candidates


def _field_directly_available(required: RequiredField, context: Dict[str, Any]) -> Tuple[bool, str]:
    """Check direct field availability and return source label."""
    resolved_fields = set(context.get("resolved_fields", []) or [])
    if required.name in resolved_fields:
        return True, "resolved_fields"

    for source_name, value in _collect_candidate_sources(context, required.name):
        if _is_usable_value(value, treat_zero_as_missing=required.treat_zero_as_missing):
            return True, source_name

    return False, ""


def _dependency_fields_available(
    field_names: Sequence[str],
    context: Dict[str, Any],
    treat_zero_as_missing: bool,
) -> bool:
    """Check whether dependency fields are available in direct/resolved context."""
    for name in field_names:
        dependency = RequiredField(name=name, treat_zero_as_missing=treat_zero_as_missing)
        ok, _ = _field_directly_available(dependency, context)
        if not ok:
            return False
    return True


def classify_field_resolution(required: RequiredField, context: Dict[str, Any]) -> Tuple[ResolutionStatus, str, str]:
    """Classify one required field into one of the V1 status labels."""
    direct_ok, direct_source = _field_directly_available(required, context)
    if direct_ok:
        return "Direct", direct_source, required.notes

    fallback_available = set(context.get("fallbacks_available", []) or []) | DEFAULT_AVAILABLE_FALLBACKS
    proxy_available = set(context.get("proxies_available", []) or [])
    manual_fields = set(context.get("manual_fields", []) or [])

    if required.fallback_strategy:
        strategy_ok = required.fallback_strategy in fallback_available
        deps_ok = _dependency_fields_available(
            required.fallback_fields,
            context,
            treat_zero_as_missing=required.treat_zero_as_missing,
        )
        if strategy_ok and (deps_ok or not required.fallback_fields):
            source = f"{required.fallback_strategy}"
            if required.fallback_fields:
                source = f"{source} via {', '.join(required.fallback_fields)}"
            return "Fallback allowed", source, required.notes

    if required.proxy_strategy:
        strategy_ok = required.proxy_strategy in proxy_available
        deps_ok = _dependency_fields_available(
            required.proxy_fields,
            context,
            treat_zero_as_missing=required.treat_zero_as_missing,
        )
        if strategy_ok and (deps_ok or not required.proxy_fields):
            source = f"{required.proxy_strategy}"
            if required.proxy_fields:
                source = f"{source} via {', '.join(required.proxy_fields)}"
            return "Proxy allowed", source, required.notes

    if required.manual_resolvable or required.name in manual_fields:
        return "Manual required", "manual input path", required.notes

    return "Blocked", "missing with no approved path", required.notes


def _contracts_registry() -> Dict[str, CalculationContract]:
    """Return minimal active contract registry for Order 4 gate foundation."""
    return {
        "liquid_rate": CalculationContract(
            name="Liquid rate",
            output_label="Validated",
            required_fields=(
                RequiredField("alloc_oil_vol", treat_zero_as_missing=False),
                RequiredField("alloc_water_vol", treat_zero_as_missing=False),
            ),
            notes="Liquid rate uses direct oil + water volumes for active path.",
        ),
        "gor": CalculationContract(
            name="GOR",
            output_label="Validated",
            required_fields=(
                RequiredField("alloc_gas_vol", treat_zero_as_missing=False),
                RequiredField("alloc_oil_vol", treat_zero_as_missing=False),
            ),
        ),
        "water_cut": CalculationContract(
            name="Water cut",
            output_label="Validated",
            required_fields=(
                RequiredField("alloc_water_vol", treat_zero_as_missing=False),
                RequiredField(
                    "liquid_rate_bbl_day",
                    treat_zero_as_missing=True,
                    fallback_strategy="derive_liquid_rate_from_alloc",
                    fallback_fields=("alloc_oil_vol", "alloc_water_vol"),
                    notes="Liquid rate can be derived from oil + water when not precomputed.",
                ),
            ),
        ),
        "mixture_specific_gravity": CalculationContract(
            name="Mixture specific gravity",
            output_label="Estimated",
            required_fields=(
                RequiredField("water_cut", treat_zero_as_missing=False),
                RequiredField(
                    "sg_oil",
                    fallback_strategy="default_sg_values",
                    manual_resolvable=True,
                    notes="Default SG_OIL is allowed when direct value is missing.",
                ),
                RequiredField(
                    "sg_water",
                    fallback_strategy="default_sg_values",
                    manual_resolvable=True,
                    notes="Default SG_WATER is allowed when direct value is missing.",
                ),
            ),
        ),
        "hydrostatic_pressure_correction": CalculationContract(
            name="Hydrostatic pressure correction",
            output_label="Estimated",
            required_fields=(
                RequiredField("sg_mixture", treat_zero_as_missing=False),
                RequiredField(
                    "well_depth_ft",
                    fallback_strategy="default_well_depth_ft",
                    manual_resolvable=True,
                    notes="Default depth is allowed when metadata depth is missing.",
                ),
            ),
        ),
        "downhole_discharge_pressure": CalculationContract(
            name="Downhole discharge pressure",
            output_label="Estimated",
            required_fields=(
                RequiredField("tubing_pressure_psi", treat_zero_as_missing=True),
                RequiredField("delta_p_hyd_psi", treat_zero_as_missing=False),
            ),
        ),
        "pump_delta_p_preprocessed": CalculationContract(
            name="Pump delta-P, preprocessed analysis",
            output_label="Estimated",
            required_fields=(
                RequiredField("p_dis_downhole_psi", treat_zero_as_missing=False),
                RequiredField(
                    "pump_intake_pressure_psi",
                    treat_zero_as_missing=True,
                    fallback_strategy="intake_from_tubing_over_0_45",
                    fallback_fields=("tubing_pressure_psi",),
                    notes="Service-layer intake fallback is approved for this path.",
                ),
            ),
        ),
        "pump_delta_p_recommendation": CalculationContract(
            name="Pump delta-P, recommendation analysis",
            output_label="Estimated",
            required_fields=(
                RequiredField("cur_tubing_pressure_psi", treat_zero_as_missing=True),
                RequiredField("rec_tubing_pressure_psi", treat_zero_as_missing=True),
                RequiredField("cur_oil", treat_zero_as_missing=False),
                RequiredField("cur_water", treat_zero_as_missing=False),
                RequiredField("rec_oil", treat_zero_as_missing=False),
                RequiredField("rec_water", treat_zero_as_missing=False),
                RequiredField(
                    "cur_pump_intake_pressure_psi",
                    treat_zero_as_missing=True,
                    fallback_strategy="intake_from_tubing_over_0_45",
                    fallback_fields=("cur_tubing_pressure_psi",),
                ),
                RequiredField(
                    "well_depth_ft",
                    fallback_strategy="default_well_depth_ft",
                    manual_resolvable=True,
                ),
                RequiredField(
                    "sg_oil",
                    fallback_strategy="default_sg_values",
                    manual_resolvable=True,
                ),
                RequiredField(
                    "sg_water",
                    fallback_strategy="default_sg_values",
                    manual_resolvable=True,
                ),
            ),
            optional_fields=("rec_pump_intake_pressure_psi",),
            notes="Recommendation intake may be assumed equal to current intake.",
        ),
        "electrical_power_proxy": CalculationContract(
            name="Electrical power proxy",
            output_label="Proxy",
            required_fields=(
                RequiredField("motor_amps", treat_zero_as_missing=True),
                RequiredField("motor_volts", treat_zero_as_missing=True),
            ),
            notes="amp_x_volt is a proxy, not validated three-phase power.",
        ),
        "ideal_pump_curve_generation": CalculationContract(
            name="Ideal pump curve generation",
            output_label="Estimated",
            required_fields=(
                RequiredField("ideal_head_c1"),
                RequiredField("ideal_head_c2"),
                RequiredField("ideal_head_c3"),
                RequiredField("ideal_head_c4"),
                RequiredField("ideal_head_c5"),
                RequiredField("ideal_head_c6"),
                RequiredField("ideal_power_c1"),
                RequiredField("ideal_power_c2"),
                RequiredField("ideal_power_c3"),
                RequiredField("ideal_power_c4"),
                RequiredField("ideal_power_c5"),
                RequiredField("ideal_power_c6"),
                RequiredField("motor_frequency_hz", treat_zero_as_missing=True),
                RequiredField("stages", treat_zero_as_missing=True),
            ),
            optional_fields=("sg_for_dp",),
            notes="sg_for_dp defaults to 1.0 when not provided.",
        ),
        "recommendation_operating_point_extraction": CalculationContract(
            name="Recommendation operating point extraction",
            output_label="Validated",
            required_fields=(
                RequiredField("model_setpoint_recommendations"),
                RequiredField("current_setpoint"),
            ),
            optional_fields=("method_used",),
        ),
        "affinity_law_validator": CalculationContract(
            name="Affinity Law validator",
            output_label="Estimated",
            required_fields=(
                RequiredField("current_motor_frequency_hz", treat_zero_as_missing=True),
                RequiredField("recommended_motor_frequency_hz", treat_zero_as_missing=True),
                RequiredField("current_liquid_rate_bpd", treat_zero_as_missing=True),
                RequiredField("recommended_liquid_rate_bpd", treat_zero_as_missing=True),
            ),
            optional_fields=(
                "current_delta_p_pump_psi",
                "recommended_delta_p_pump_psi",
                "motor_power_kw",
                "bhp_proxy",
                "amp_x_volt",
            ),
            notes=(
                "Field-level sanity check against affinity expectations. "
                "Delta-P and power are optional; power may use approved proxy inputs."
            ),
        ),
        "gas_interference_trend_screen": CalculationContract(
            name="Gas-Interference Trend Screen",
            output_label="Research prototype",
            required_fields=(
                RequiredField(
                    "historical_row_count",
                    treat_zero_as_missing=True,
                    notes="Screen requires enough historical rows; caller enforces min-row threshold.",
                ),
                RequiredField(
                    "pump_intake_pressure_psi",
                    treat_zero_as_missing=True,
                    notes="Core trend signal for intake-pressure decline screening.",
                ),
                RequiredField(
                    "gor",
                    treat_zero_as_missing=False,
                    fallback_strategy="derive_gor_from_alloc",
                    fallback_fields=("alloc_gas_vol", "alloc_oil_vol"),
                    notes="GOR may be derived from allocation gas/oil when direct GOR is missing.",
                ),
                RequiredField(
                    "delta_p_pump_psi",
                    treat_zero_as_missing=False,
                    fallback_strategy="pump_behavior_from_liquid_rate",
                    fallback_fields=("liquid_rate_bbl_day",),
                    notes="At least one pump-behavior signal is required: delta-P or liquid rate fallback path.",
                ),
            ),
            optional_fields=(
                "water_cut",
                "motor_amps",
                "amp_x_volt",
                "pump_intake_temperature_f",
                "motor_temperature_f",
                "liquid_rate_bbl_day",
                "alloc_gas_vol",
                "alloc_oil_vol",
            ),
            notes=(
                "Screening-only diagnostic contract. Optional signals can be missing without blocking "
                "execution, but mode should degrade to reduced confidence."
            ),
        ),
        "energy_efficiency_diagnostic": CalculationContract(
            name="Energy / Efficiency diagnostic",
            output_label="Estimated",
            required_fields=(
                RequiredField("liquid_rate_bpd", treat_zero_as_missing=True),
                RequiredField("delta_p_pump_psi", treat_zero_as_missing=True),
                RequiredField(
                    "motor_power_kw",
                    treat_zero_as_missing=True,
                    proxy_strategy="energy_power_from_proxy",
                    proxy_fields=("energy_proxy_power_kw",),
                    notes="Direct motor_power_kw is preferred; proxy power path is allowed when direct power is missing.",
                ),
            ),
            optional_fields=(
                "oil_rate_bpd",
                "power_source_label",
                "bhp_proxy",
                "amp_x_volt",
                "energy_proxy_power_kw",
            ),
            notes=(
                "Pump-delta-P-based energy diagnostic. Uses direct motor power when available, "
                "otherwise approved proxy power path."
            ),
        ),
        "bubble_point_gas_breakout_prototype": CalculationContract(
            name="Bubble-Point / Gas Breakout Prototype",
            output_label="Research prototype",
            required_fields=(
                RequiredField(
                    "pump_intake_pressure_psi",
                    treat_zero_as_missing=True,
                    notes=(
                        "Primary V1 pressure reference for bubble-point comparison: ESP suction pressure. "
                        "Must be available and positive."
                    ),
                ),
                RequiredField(
                    "selected_bubble_point_psi",
                    treat_zero_as_missing=True,
                    manual_resolvable=True,
                    fallback_strategy="standing_correlation_from_user_inputs",
                    fallback_fields=("R_so", "gamma_g", "T_f", "API"),
                    notes=(
                        "Either a user-supplied bubble-point pressure or a complete Standing-correlation "
                        "input set (R_so, gamma_g, T_f, API) must be provided. "
                        "User-supplied P_b is preferred when available (Manual input trust). "
                        "Standing inputs are user-confirmed; proxy/default inputs allowed only when "
                        "explicitly confirmed by the user."
                    ),
                ),
            ),
            optional_fields=(
                "R_so",
                "gamma_g",
                "T_f",
                "API",
                "user_supplied_bubble_point_psi",
                "gor_proxy_candidate",
                "sg_oil",
                "pump_intake_temperature_f",
                "temperature_summary_stats",
            ),
            notes=(
                "Research-prototype contract. Proxy/default inputs (producing GOR as R_so candidate, "
                "default gamma_g, default/derived temperature, sg_oil-derived API) are allowed only "
                "when explicitly user-confirmed. All outputs labeled Research prototype unless a "
                "user-supplied PVT P_b is clearly provided."
            ),
        ),
        "npsh_cavitation_screening_prototype": CalculationContract(
            name="NPSH / Cavitation Screening Prototype",
            output_label="Research prototype",
            required_fields=(
                RequiredField(
                    "pump_intake_pressure_psi",
                    treat_zero_as_missing=True,
                    notes=(
                        "Primary V1 suction-pressure reference: pump intake pressure (gauge psi). "
                        "Blocking when missing — no approved proxy path exists."
                    ),
                ),
                RequiredField(
                    "vapor_pressure_psi_abs",
                    treat_zero_as_missing=True,
                    manual_resolvable=True,
                    fallback_strategy="bubble_point_proxy_for_vapor_pressure",
                    notes=(
                        "Absolute vapor pressure (psia). User input is preferred. "
                        "Bubble-point P_b from Order 10 may be used as a conservative proxy "
                        "only if explicitly user-confirmed. "
                        "Manual required when no direct or proxy source is available."
                    ),
                ),
                RequiredField(
                    "sg_mixture",
                    treat_zero_as_missing=True,
                    fallback_strategy="default_sg_values",
                    manual_resolvable=True,
                    notes=(
                        "Mixture specific gravity. Uses current sg_mixture estimate when available; "
                        "default SG fallback allowed when missing."
                    ),
                ),
                RequiredField(
                    "npshr_ft",
                    treat_zero_as_missing=False,
                    manual_resolvable=True,
                    fallback_strategy="placeholder_npshr_default",
                    notes=(
                        "NPSH required by the pump (ft). User input is preferred. "
                        "A placeholder default (7.5 ft) may be used only if user-confirmed and clearly labeled. "
                        "Manual required when no direct or placeholder source is confirmed."
                    ),
                ),
            ),
            optional_fields=(
                "bubble_point_psi_proxy_candidate",
                "temperature_f",
                "notes_source",
            ),
            notes=(
                "NPSH screening prototype contract. ESP V1 uses pump intake pressure directly as the "
                "suction-pressure reference — this is not full suction-piping NPSH. "
                "No validated margin is produced without pump-specific NPSHr and confirmed vapor pressure. "
                "Output must be labeled Proxy / Research prototype when proxy/default inputs are used."
            ),
        ),
    }


ACTIVE_CONTRACTS_REGISTRY: Dict[str, CalculationContract] = _contracts_registry()


def evaluate_calculation_readiness(
    contract: CalculationContract,
    context: Dict[str, Any],
) -> CalculationReadinessReport:
    """Evaluate a single calculation contract against provided field context."""
    field_resolutions: List[FieldResolution] = []
    missing_fields: List[str] = []
    fallback_or_proxy_fields: List[str] = []

    for required in contract.required_fields:
        status, source, notes = classify_field_resolution(required, context)
        if status in {"Manual required", "Blocked"}:
            missing_fields.append(required.name)
        if status in {"Fallback allowed", "Proxy allowed"}:
            fallback_or_proxy_fields.append(required.name)

        field_resolutions.append(
            FieldResolution(
                calculation=contract.name,
                required_field=required.name,
                status=status,
                source_or_resolution=source,
                required=True,
                output_label=contract.output_label,
                ready=status != "Blocked",
                notes=notes,
            )
        )

    blocked_present = any(item.status == "Blocked" for item in field_resolutions)
    ready = not blocked_present

    return CalculationReadinessReport(
        calculation=contract.name,
        output_label=contract.output_label,
        ready=ready,
        missing_fields=missing_fields,
        fallback_or_proxy_fields=fallback_or_proxy_fields,
        notes=contract.notes,
        field_resolutions=field_resolutions,
    )


def run_data_availability_gate(
    context: Dict[str, Any],
    calculation_keys: Optional[Sequence[str]] = None,
) -> List[CalculationReadinessReport]:
    """Run readiness evaluation for selected calculations.

    Args:
        context: Gate context with optional keys:
            - field_values: dict[str, Any]
            - records: dict[str, dict]
            - dataframes: dict[str, pd.DataFrame]
            - resolved_fields: iterable[str]
            - manual_fields: iterable[str]
            - fallbacks_available: iterable[str]
            - proxies_available: iterable[str]
        calculation_keys: specific contract keys from ACTIVE_CONTRACTS_REGISTRY.
            When omitted, all contracts are evaluated.
    """
    if calculation_keys is None:
        keys = list(ACTIVE_CONTRACTS_REGISTRY.keys())
    else:
        keys = list(calculation_keys)

    reports: List[CalculationReadinessReport] = []
    for key in keys:
        if key not in ACTIVE_CONTRACTS_REGISTRY:
            raise KeyError(f"Unknown calculation key: {key}")
        reports.append(
            evaluate_calculation_readiness(ACTIVE_CONTRACTS_REGISTRY[key], context=context)
        )
    return reports


def build_readiness_rows(
    reports: Sequence[CalculationReadinessReport],
) -> List[Dict[str, Any]]:
    """Flatten readiness reports into row dictionaries for app/report use."""
    rows: List[Dict[str, Any]] = []
    for report in reports:
        for resolution in report.field_resolutions:
            rows.append(
                {
                    "calculation": resolution.calculation,
                    "required_field": resolution.required_field,
                    "status": resolution.status,
                    "source_or_resolution": resolution.source_or_resolution,
                    "required": resolution.required,
                    "output_label": resolution.output_label,
                    "ready": report.ready,
                    "notes": resolution.notes,
                }
            )
    return rows


def build_readiness_dataframe(
    reports: Sequence[CalculationReadinessReport],
) -> pd.DataFrame:
    """Convert readiness rows to a pandas DataFrame suitable for display/export."""
    rows = build_readiness_rows(reports)
    return pd.DataFrame(
        rows,
        columns=[
            "calculation",
            "required_field",
            "status",
            "source_or_resolution",
            "required",
            "output_label",
            "ready",
            "notes",
        ],
    )


def summarize_readiness(
    reports: Sequence[CalculationReadinessReport],
) -> Dict[str, Any]:
    """Build compact readiness summary for selected calculations."""
    total = len(reports)
    ready_count = sum(1 for r in reports if r.ready)
    blocked_count = total - ready_count

    manual_required_fields: List[str] = []
    fallback_or_proxy_fields: List[str] = []
    blocking_fields: List[str] = []

    for report in reports:
        for resolution in report.field_resolutions:
            field_id = f"{report.calculation}.{resolution.required_field}"
            if resolution.status == "Manual required":
                manual_required_fields.append(field_id)
            if resolution.status in {"Fallback allowed", "Proxy allowed"}:
                fallback_or_proxy_fields.append(field_id)
            if resolution.status == "Blocked":
                blocking_fields.append(field_id)

    return {
        "total_calculations_checked": total,
        "ready_calculations": ready_count,
        "blocked_calculations": blocked_count,
        "fields_requiring_manual_input": manual_required_fields,
        "fields_using_fallback_or_proxy": fallback_or_proxy_fields,
        "blocking_fields": blocking_fields,
    }


def demo_gate_for_preprocessed_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Small internal helper that demonstrates gate usage for preprocessed path."""
    context = {
        "dataframes": {"preprocessed": df},
        "fallbacks_available": {
            "derive_liquid_rate_from_alloc",
            "intake_from_tubing_over_0_45",
            "default_sg_values",
            "default_well_depth_ft",
            "default_sg_for_dp",
        },
        "proxies_available": set(),
    }
    reports = run_data_availability_gate(
        context=context,
        calculation_keys=[
            "liquid_rate",
            "gor",
            "water_cut",
            "mixture_specific_gravity",
            "hydrostatic_pressure_correction",
            "downhole_discharge_pressure",
            "pump_delta_p_preprocessed",
            "electrical_power_proxy",
        ],
    )
    return {
        "reports": reports,
        "rows": build_readiness_rows(reports),
        "summary": summarize_readiness(reports),
    }
