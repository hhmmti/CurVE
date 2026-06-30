"""CurVE — agentic physics-validation layer for ROAM's Virtual Engineer.

M1 (walking skeleton): a hand-rolled Bedrock Converse tool loop that routes an
operator question to the correct stub tool.
M2 core: ``production_history`` is real end-to-end — real Athena telemetry +
production → per-tool gate → vendored physics → vendored Plotly → the
``{values, trust_label + flags, figure_ref}`` envelope, with org/well injected from
a session record and the model narrating the trust basis. (Streamlit surface = 2b.)
"""

from .data import PreprocessedDataAccess, fetch_preprocessed_window
from .delta_p_inputs import (
    CURVE_DEFAULT_DEPTH_FT,
    DeltaPInputs,
    pip_coverage,
    resolve_delta_p_inputs,
    resolve_from_context,
)
from .engine import MAX_ITERATIONS, run_curve_turn
from .gate import run_delta_p_tool_gate, run_tool_gate
from .ideal_catalog import (
    DEFAULT_BEP_TOLERANCE,
    annotate_candidates,
    bep_tolerance_from_context,
    build_coverage_report,
    fetch_ideal_catalog,
    make_pump_pick,
    narrow_candidates,
    resolve_total_fluid_bpd,
    selectable_candidates,
    set_pump_on_session,
)
from .prompt import CURVE_SYSTEM_PROMPT, format_setup_context
from .session import (
    clear_sessions,
    load_session,
    new_session_record,
    save_session,
    use_store,
)
from .tools import NON_MODEL_RESULT_KEYS, TOOL_REGISTRY, build_tool_config
from .well_depth import fetch_well_depth_ft
from .wrapper import CurveBedrockWrapper

__all__ = [
    "run_curve_turn",
    "MAX_ITERATIONS",
    "CURVE_SYSTEM_PROMPT",
    "format_setup_context",
    "TOOL_REGISTRY",
    "NON_MODEL_RESULT_KEYS",
    "build_tool_config",
    "CurveBedrockWrapper",
    "PreprocessedDataAccess",
    "fetch_preprocessed_window",
    "run_tool_gate",
    "run_delta_p_tool_gate",
    "resolve_delta_p_inputs",
    "resolve_from_context",
    "pip_coverage",
    "DeltaPInputs",
    "CURVE_DEFAULT_DEPTH_FT",
    "fetch_well_depth_ft",
    "new_session_record",
    "save_session",
    "load_session",
    "clear_sessions",
    "use_store",
    "fetch_ideal_catalog",
    "resolve_total_fluid_bpd",
    "narrow_candidates",
    "annotate_candidates",
    "selectable_candidates",
    "build_coverage_report",
    "make_pump_pick",
    "bep_tolerance_from_context",
    "set_pump_on_session",
    "DEFAULT_BEP_TOLERANCE",
]
