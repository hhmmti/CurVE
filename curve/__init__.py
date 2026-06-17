"""CurVE — agentic physics-validation layer for ROAM's Virtual Engineer.

M1 (walking skeleton): a hand-rolled Bedrock Converse tool loop that routes an
operator question to the correct stub tool. No gate, no physics, no data, no UI.
"""

from .engine import MAX_ITERATIONS, run_curve_turn
from .prompt import CURVE_SYSTEM_PROMPT
from .tools import TOOL_REGISTRY, build_tool_config
from .wrapper import CurveBedrockWrapper

__all__ = [
    "run_curve_turn",
    "MAX_ITERATIONS",
    "CURVE_SYSTEM_PROMPT",
    "TOOL_REGISTRY",
    "build_tool_config",
    "CurveBedrockWrapper",
]
