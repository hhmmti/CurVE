"""CurVE — agentic physics-validation layer for ROAM's Virtual Engineer.

M1 (walking skeleton): a hand-rolled Bedrock Converse tool loop that routes an
operator question to the correct stub tool.
M2 core: ``production_history`` is real end-to-end — real Athena telemetry +
production → per-tool gate → vendored physics → vendored Plotly → the
``{values, trust_label + flags, figure_ref}`` envelope, with org/well injected from
a session record and the model narrating the trust basis. (Streamlit surface = 2b.)
"""

from .data import PreprocessedDataAccess, fetch_preprocessed_window
from .engine import MAX_ITERATIONS, run_curve_turn
from .gate import run_tool_gate
from .prompt import CURVE_SYSTEM_PROMPT, format_setup_context
from .session import (
    clear_sessions,
    load_session,
    new_session_record,
    save_session,
    use_store,
)
from .tools import NON_MODEL_RESULT_KEYS, TOOL_REGISTRY, build_tool_config
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
    "new_session_record",
    "save_session",
    "load_session",
    "clear_sessions",
    "use_store",
]
