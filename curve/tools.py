"""CurVE stub tools (M1).

M1 proves exactly one thing: *does the model route to the correct tool?* So these
tools carry **real** ``toolSpec``s (name, description, inputSchema) — routing is a
genuine test — but **mock** bodies that return a canned dict. No physics, no data,
no I/O. Real bodies wrap ``compute/`` + ``services/`` (the gate) in M2+.

Naming convention (documented in README):
  * snake_case
  * capability-named — the verb/noun of the question the tool answers
    (``production_history``, ``curve_position``, ``bubble_point_screen``), not the
    implementation module it will eventually call. The model routes on the
    capability, so the name + description must read like an operator's intent.

The three M1 stubs intentionally span the gate's structural classes so routing is
tested across them: a connection-free historical tool (``production_history``), a
connection-dependent diagnostic (``curve_position``), and a screening diagnostic
(``bubble_point_screen``).
"""

from typing import Any, Callable, Dict

# --- mock tool bodies ---------------------------------------------------------
# Each takes the model-supplied ``input`` dict and returns a canned result. The
# input is accepted (and echoed) only to make the mock observable; nothing real
# is computed.


def production_history(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return {"mock": "production_history output", "received_input": tool_input}


def curve_position(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return {"mock": "curve_position output", "received_input": tool_input}


def bubble_point_screen(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return {"mock": "bubble_point_screen output", "received_input": tool_input}


# --- real tool specs (Converse toolConfig shape) ------------------------------
# Converse expects inputSchema wrapped under a "json" key.

_PRODUCTION_HISTORY_SPEC = {
    "toolSpec": {
        "name": "production_history",
        "description": (
            "Retrieve historical production and telemetry for a single well over a "
            "time window (oil/water/gas rates, intake/tubing pressure, frequency, "
            "amps). Use for questions about how a well has produced or behaved over "
            "time, recent trends, or 'what has this well been doing'."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "well_id": {
                        "type": "string",
                        "description": "Identifier of the well to retrieve.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Window start (ISO date), optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Window end (ISO date), optional.",
                    },
                },
                "required": ["well_id"],
            }
        },
    }
}

_CURVE_POSITION_SPEC = {
    "toolSpec": {
        "name": "curve_position",
        "description": (
            "Determine where the pump is operating on its performance curve right "
            "now — the ideal-curve overlay (single + multi-frequency, ΔP) plus the "
            "BEP position. Use for 'where am I on the curve', 'am I near best "
            "efficiency point', or operating-point-vs-pump-curve questions. "
            "(In production this needs a resolved pump connection; mocked in M1.)"
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "well_id": {
                        "type": "string",
                        "description": "Identifier of the well.",
                    },
                    "frequency_hz": {
                        "type": "number",
                        "description": "Operating frequency to evaluate, optional.",
                    },
                },
                "required": ["well_id"],
            }
        },
    }
}

_BUBBLE_POINT_SCREEN_SPEC = {
    "toolSpec": {
        "name": "bubble_point_screen",
        "description": (
            "Screen whether the well is operating below bubble point at the pump "
            "intake — i.e. whether gas is breaking out of solution before the "
            "intake. Use for gas-breakout, free-gas, gas-interference, or "
            "'am I below bubble point' questions."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "well_id": {
                        "type": "string",
                        "description": "Identifier of the well to screen.",
                    },
                },
                "required": ["well_id"],
            }
        },
    }
}


# --- registry -----------------------------------------------------------------
# name -> {"spec": <toolSpec dict>, "fn": <callable>}. The engine builds toolConfig
# from the specs and dispatches tool_use by name to fn.

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "production_history": {"spec": _PRODUCTION_HISTORY_SPEC, "fn": production_history},
    "curve_position": {"spec": _CURVE_POSITION_SPEC, "fn": curve_position},
    "bubble_point_screen": {"spec": _BUBBLE_POINT_SCREEN_SPEC, "fn": bubble_point_screen},
}


def build_tool_config(registry: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the Converse ``toolConfig`` from a registry."""
    registry = registry if registry is not None else TOOL_REGISTRY
    return {"tools": [entry["spec"] for entry in registry.values()]}
