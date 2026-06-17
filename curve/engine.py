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

from .prompt import CURVE_SYSTEM_PROMPT
from .tools import TOOL_REGISTRY, build_tool_config
from .wrapper import CurveBedrockWrapper

# Loop safety cap — "shallow multi-step" (CurVE-decisions §1 Decision 10). A hard
# stop so a misbehaving model can't spin indefinitely.
MAX_ITERATIONS = 5


def _extract_text(message: Dict[str, Any]) -> str:
    """Concatenate the text blocks of an assistant message (skips thinking/tools)."""
    parts = [block["text"] for block in message.get("content", []) if "text" in block]
    return "\n".join(parts).strip()


def run_curve_turn(
    question: str,
    *,
    wrapper: Optional[CurveBedrockWrapper] = None,
    tools: Optional[Dict[str, Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
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
        profile_name / region_name / enable_thinking: forwarded to the default
            wrapper when one isn't injected.
        max_iterations: Hard safety cap on Converse calls.
        verbose: Print per-iteration trace (used by the CLI's dev view).

    Returns:
        ``{"text", "tool_trace", "stop_reason", "iterations", "messages"}`` where
        ``tool_trace`` is the ordered list of tool names the model called.
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
    tool_config = build_tool_config(tools)
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"text": question}]}
    ]
    tool_trace: List[str] = []

    for iteration in range(max_iterations):
        response = wrapper.converse(
            messages=messages, system=system, tool_config=tool_config
        )
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
                "stop_reason": stop_reason,
                "iterations": iteration + 1,
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
                result = entry["fn"](tool_use.get("input", {}))
            tool_result_blocks.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": result}],
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
        "stop_reason": "max_iterations",
        "iterations": max_iterations,
        "messages": messages,
    }
