"""CurVE M1 loop mechanics tests — run with NO AWS credentials.

These mock the wrapper's ``converse`` (and, for the bug-fix guard, the underlying
boto3 client) so nothing reaches AWS. They exercise the **shipped** loop
(``run_curve_turn``) and the real ``CurveBedrockWrapper.converse`` — not
reimplementations.

Covered:
  * tool_use detection → tool call → toolResult append → loop → end_turn terminate
  * tool_trace correctness
  * reasoningContent (thinking) preserved verbatim across the tool turn
  * 5-iteration safety cap halts a forced infinite tool_use
  * the wrapper transmits a POPULATED inferenceConfig (guards the monorepo bug)
"""

import copy
import os
import sys

# Make the package importable when pytest is run from the repo root or tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve.engine import run_curve_turn  # noqa: E402
from curve.tools import TOOL_REGISTRY  # noqa: E402
from curve.wrapper import CurveBedrockWrapper  # noqa: E402


class ScriptedWrapper:
    """Stand-in for CurveBedrockWrapper that replays scripted Converse responses.

    Records a deep copy of the ``messages`` passed on each call so tests can assert
    what the engine appended between turns.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # list of messages snapshots, one per converse call

    def converse(self, messages, system, tool_config=None):
        self.calls.append(copy.deepcopy(messages))
        if self._responses:
            return self._responses.pop(0)
        # If the script runs dry, behave like a benign end_turn.
        return _end_turn("(script exhausted)")


def _reasoning_block(text="Routing to the production history tool."):
    return {
        "reasoningContent": {
            "reasoningText": {"text": text, "signature": "sig-abc123"}
        }
    }


def _tool_use_turn(name, tool_use_id="tu-1", tool_input=None, with_thinking=True):
    content = []
    if with_thinking:
        content.append(_reasoning_block())
    content.append({"text": "Let me pull that up."})
    content.append(
        {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input or {"well_id": "W-12"}}}
    )
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": "tool_use"}


def _end_turn(text="Here is the answer."):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
    }


def test_routes_runs_tool_and_terminates():
    wrapper = ScriptedWrapper(
        [_tool_use_turn("production_history"), _end_turn("Well W-12 produced steadily.")]
    )
    result = run_curve_turn("How has W-12 produced?", wrapper=wrapper)

    assert result["tool_trace"] == ["production_history"]
    assert result["stop_reason"] == "end_turn"
    assert result["text"] == "Well W-12 produced steadily."
    assert result["iterations"] == 2
    # Two converse calls: initial + after the toolResult.
    assert len(wrapper.calls) == 2


def test_tool_result_appended_between_turns():
    wrapper = ScriptedWrapper(
        [_tool_use_turn("bubble_point_screen"), _end_turn()]
    )
    run_curve_turn("Is W-12 below bubble point?", wrapper=wrapper)

    # The messages seen on the 2nd converse call: user q, assistant tool_use,
    # user toolResult.
    second_call = wrapper.calls[1]
    assert second_call[0]["role"] == "user"
    assert second_call[1]["role"] == "assistant"
    tool_result_msg = second_call[2]
    assert tool_result_msg["role"] == "user"
    block = tool_result_msg["content"][0]["toolResult"]
    assert block["toolUseId"] == "tu-1"
    assert block["content"][0]["json"]["mock"] == "bubble_point_screen output"


def test_reasoning_content_preserved_verbatim():
    wrapper = ScriptedWrapper(
        [_tool_use_turn("curve_position"), _end_turn()]
    )
    run_curve_turn("Where am I on the curve?", wrapper=wrapper)

    # On the 2nd call, the appended assistant message must still carry the
    # reasoningContent block, first and unmutated.
    assistant_msg = wrapper.calls[1][1]
    assert assistant_msg["role"] == "assistant"
    first_block = assistant_msg["content"][0]
    assert "reasoningContent" in first_block
    assert (
        first_block["reasoningContent"]["reasoningText"]["text"]
        == "Routing to the production history tool."
    )
    assert first_block["reasoningContent"]["reasoningText"]["signature"] == "sig-abc123"
    # Order preserved: thinking, then text, then toolUse.
    assert "text" in assistant_msg["content"][1]
    assert "toolUse" in assistant_msg["content"][2]


def test_safety_cap_halts_infinite_tool_use():
    # Every turn is tool_use → the loop must stop at the cap, not spin.
    class AlwaysToolUse:
        def __init__(self):
            self.n = 0

        def converse(self, messages, system, tool_config=None):
            self.n += 1
            return _tool_use_turn("production_history", tool_use_id=f"tu-{self.n}")

    wrapper = AlwaysToolUse()
    result = run_curve_turn("loop forever", wrapper=wrapper)

    assert result["stop_reason"] == "max_iterations"
    assert result["iterations"] == 5
    assert wrapper.n == 5  # exactly the cap — no spin
    assert len(result["tool_trace"]) == 5


def test_wrapper_transmits_populated_inference_config():
    """Guards the monorepo bug: inferenceConfig must reach converse non-empty."""

    class RecordingClient:
        def __init__(self):
            self.kwargs = None

        def converse(self, **kwargs):
            self.kwargs = kwargs
            return _end_turn("ok")

    client = RecordingClient()
    wrapper = CurveBedrockWrapper(client=client, enable_thinking=True)
    wrapper.converse(messages=[{"role": "user", "content": [{"text": "hi"}]}], system=[{"text": "sys"}])

    cfg = client.kwargs["inferenceConfig"]
    assert cfg, "inferenceConfig must be non-empty (the bug sent an empty dict)"
    assert cfg["maxTokens"] > 0
    # Thinking constraint: temperature pinned to 1.0, topP omitted.
    assert cfg["temperature"] == 1.0
    assert "topP" not in cfg
    assert client.kwargs["additionalModelRequestFields"]["thinking"]["type"] == "enabled"


def test_wrapper_inference_config_without_thinking():
    """With thinking off, configured temperature + topP transmit."""

    class RecordingClient:
        def converse(self, **kwargs):
            self.kwargs = kwargs
            return _end_turn("ok")

    client = RecordingClient()
    wrapper = CurveBedrockWrapper(client=client, enable_thinking=False, temperature=0.2, top_p=0.95)
    wrapper.converse(messages=[{"role": "user", "content": [{"text": "hi"}]}], system=[{"text": "sys"}])

    cfg = client.kwargs["inferenceConfig"]
    assert cfg["temperature"] == 0.2
    assert cfg["topP"] == 0.95
    assert "additionalModelRequestFields" not in client.kwargs


def test_real_registry_specs_are_converse_shaped():
    """The 3 stub tools carry real, Converse-shaped toolSpecs."""
    assert set(TOOL_REGISTRY) == {"production_history", "curve_position", "bubble_point_screen"}
    for name, entry in TOOL_REGISTRY.items():
        spec = entry["spec"]["toolSpec"]
        assert spec["name"] == name
        assert spec["description"]
        assert spec["inputSchema"]["json"]["type"] == "object"
        assert callable(entry["fn"])
