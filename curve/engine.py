"""CurVE hand-rolled Converse tool loop (M1).

This is the **shipped** loop: the CLI and the tests both call ``run_curve_turn`` —
neither reimplements it. The loop drives Bedrock Converse's native ``toolConfig``:

    converse → if stopReason == "tool_use": run the named tool(s), append a
    toolResult user message, loop → else return the final text.

HARD REQUIREMENT — preserve thinking across tool turns:
When extended thinking is on, the assistant turn that carries ``toolUse`` also
carries ``reasoningContent`` block(s). Bedrock validates that these are returned
**verbatim and in order** on the next turn. We achieve this by appending the
assistant message object from the response *as-is* (``output.message``) — we never
rebuild it, so reasoningContent blocks are neither dropped, reordered, nor mutated.

The loop is structured so a future ``fn(event: dict) -> dict`` VE action can wrap
this same entry without touching the loop. That action is NOT built in M1.
"""

from typing import Any, Callable, Dict, List, Optional

from .prompt import CURVE_SYSTEM_PROMPT, format_setup_context
from .tools import NON_MODEL_RESULT_KEYS, TOOL_REGISTRY, build_tool_config
from .wrapper import CurveBedrockWrapper

# Loop safety cap — "shallow multi-step" (CurVE-decisions §1 Decision 10). A hard
# stop so a misbehaving model can't spin indefinitely.
MAX_ITERATIONS = 5


def _extract_text(message: Dict[str, Any]) -> str:
    """Concatenate the text blocks of an assistant message (skips thinking/tools)."""
    parts = [block["text"] for block in message.get("content", []) if "text" in block]
    return "\n".join(parts).strip()


def _empty_usage() -> Dict[str, int]:
    return {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}


def _accumulate_usage(total: Dict[str, int], response: Dict[str, Any]) -> None:
    """Add one Converse response's ``usage`` block into the running total.

    A tool-using turn makes multiple converse calls; usage must be summed across
    ALL of them, not read off the last call (which would undercount).
    """
    usage = response.get("usage") or {}
    for key in ("inputTokens", "outputTokens", "totalTokens"):
        total[key] += usage.get(key, 0)


def _model_facing_result(result: Any) -> Any:
    """Strip the figure + figure_ref from a tool result before it reaches the model.

    The Plotly figure (and its UI ref) go to the UI, never back into the model
    (CurVE-decisions §3 D2 — no image tokens; narrate from ``values``). M1 stub
    results (plain dicts with neither key) pass through unchanged.
    """
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if k not in NON_MODEL_RESULT_KEYS}
    return result


def run_curve_turn(
    question: str,
    *,
    wrapper: Optional[CurveBedrockWrapper] = None,
    tools: Optional[Dict[str, Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    session: Optional[Dict[str, Any]] = None,
    profile_name: Optional[str] = None,
    region_name: str = "us-east-1",
    enable_thinking: bool = True,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one CurVE turn through the hand-rolled tool loop.

    Args:
        question: The operator's question (becomes the first user message).
        wrapper: A Converse wrapper exposing ``.converse(messages, system,
            tool_config)``. Defaults to a real :class:`CurveBedrockWrapper`. Tests
            inject a mock here so no AWS creds are needed.
        tools: Registry ``name -> {"spec", "fn"}``. Defaults to ``TOOL_REGISTRY``.
        system_prompt: Composed system prompt. Defaults to ``CURVE_SYSTEM_PROMPT``.
        session: The session record (CurVE-decisions §2 D8). When present, the engine
            (1) backend-injects ``organization_id`` + ``well_id`` into every tool
            call (the model never supplies them), and (2) adds a "setup complete;
            selected well: …" context line to the system prompt. When ``None`` (M1
            routing tests), neither happens — the loop behaves exactly as in M1.
        profile_name / region_name / enable_thinking: forwarded to the default
            wrapper when one isn't injected.
        max_iterations: Hard safety cap on Converse calls.
        verbose: Print per-iteration trace (used by the CLI's dev view).

    Returns:
        ``{"text", "tool_trace", "tool_outputs", "stop_reason", "iterations",
        "usage", "messages"}``. ``tool_trace`` is the ordered list of tool names
        called; ``tool_outputs`` is the ordered list of FULL tool envelopes (incl.
        ``figure`` / ``figure_ref``) for the UI/CLI to render; ``usage`` is the token
        usage summed across every converse call in this turn.
    """
    if wrapper is None:
        wrapper = CurveBedrockWrapper(
            profile_name=profile_name,
            region_name=region_name,
            enable_thinking=enable_thinking,
        )
    if tools is None:
        tools = TOOL_REGISTRY
    if system_prompt is None:
        system_prompt = CURVE_SYSTEM_PROMPT

    system = [{"text": system_prompt}]
    if session is not None:
        # Re-supply the setup state each turn (the well/org context line).
        system.append({"text": format_setup_context(session)})
    tool_config = build_tool_config(tools)
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"text": question}]}
    ]
    tool_trace: List[str] = []
    tool_outputs: List[Dict[str, Any]] = []
    usage = _empty_usage()

    for iteration in range(max_iterations):
        response = wrapper.converse(
            messages=messages, system=system, tool_config=tool_config
        )
        # Sum usage across every converse call in this turn (not just the last).
        _accumulate_usage(usage, response)
        assistant_message = response["output"]["message"]
        # Append the assistant message VERBATIM — this preserves reasoningContent
        # (thinking) blocks in order, which Bedrock requires on the next turn.
        messages.append(assistant_message)
        stop_reason = response.get("stopReason", "")

        if verbose:
            called = [
                b["toolUse"]["name"]
                for b in assistant_message.get("content", [])
                if "toolUse" in b
            ]
            print(f"  [iter {iteration + 1}] stop={stop_reason} tools={called}")

        if stop_reason != "tool_use":
            return {
                "text": _extract_text(assistant_message),
                "tool_trace": tool_trace,
                "tool_outputs": tool_outputs,
                "stop_reason": stop_reason,
                "iterations": iteration + 1,
                "usage": usage,
                "messages": messages,
            }

        # Run every toolUse block in this turn; collect a toolResult per block.
        tool_result_blocks: List[Dict[str, Any]] = []
        for block in assistant_message.get("content", []):
            tool_use = block.get("toolUse")
            if not tool_use:
                continue
            name = tool_use["name"]
            tool_trace.append(name)
            entry = tools.get(name)
            if entry is None:
                result: Dict[str, Any] = {"error": f"unknown tool: {name}"}
            else:
                # Backend-inject org/well from the session record (merged with the
                # model's selectors; backend wins — the model never supplies them).
                tool_input = dict(tool_use.get("input", {}) or {})
                if session is not None:
                    tool_input["organization_id"] = session.get("organization_id")
                    tool_input["well_id"] = session.get("well_id")
                result = entry["fn"](tool_input)

            # The FULL envelope (incl. figure) is recorded for the UI/CLI; only the
            # model-facing fields (no figure) are returned to the model.
            tool_outputs.append({"name": name, "result": result})
            tool_result_blocks.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": _model_facing_result(result)}],
                    }
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})

    # Safety cap reached — terminate and report rather than spin.
    return {
        "text": (
            "[CurVE] Stopped: reached the maximum of "
            f"{max_iterations} tool iterations without a final answer."
        ),
        "tool_trace": tool_trace,
        "tool_outputs": tool_outputs,
        "stop_reason": "max_iterations",
        "iterations": max_iterations,
        "usage": usage,
        "messages": messages,
    }
