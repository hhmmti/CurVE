"""CurVE M1 CLI — drive the hand-rolled tool loop against live Bedrock.

Two modes:
  * interactive (default): type a question → runs ``run_curve_turn`` against live
    Bedrock → prints the answer and the ordered tool_trace.
  * batch (``--batch``): runs the placeholder ``TEST_QUESTIONS`` fixture, printing
    expected-vs-actual tool per question.

Both modes call the **shipped** ``run_curve_turn`` — the CLI does not reimplement
the loop. Live Bedrock requires AWS creds; run ``aws sso login --profile roam-ai``
first.

Usage:
    python cli.py                          # interactive, profile roam-ai
    python cli.py --batch                  # routing batch over the fixture
    python cli.py --profile roam-ai --region us-east-1
    python cli.py --no-thinking            # disable extended thinking
"""

import argparse
import sys

from curve.engine import run_curve_turn
from curve.test_questions import TEST_QUESTIONS
from curve.wrapper import CURVE_DEFAULT_PROFILE, CURVE_REGION

# --- ESTIMATED cost rates -----------------------------------------------------
# EDIT HERE to update pricing. These are ESTIMATES (USD per 1M tokens) for the
# `us.` cross-region inference profile: base Sonnet pricing ($3 in / $15 out) plus
# ~10% cross-region uplift. Not authoritative — confirm against current AWS pricing.
COST_PER_1M_INPUT_USD = 3.30
COST_PER_1M_OUTPUT_USD = 16.50


def _estimate_cost_usd(usage: dict) -> float:
    """Estimated USD cost for an aggregated usage block (an estimate, not a bill)."""
    in_cost = usage.get("inputTokens", 0) / 1_000_000 * COST_PER_1M_INPUT_USD
    out_cost = usage.get("outputTokens", 0) / 1_000_000 * COST_PER_1M_OUTPUT_USD
    return in_cost + out_cost


def _add_usage(total: dict, usage: dict) -> None:
    for key in ("inputTokens", "outputTokens", "totalTokens"):
        total[key] = total.get(key, 0) + usage.get(key, 0)


def _format_usage(label: str, usage: dict) -> str:
    return (
        f"{label}: in={usage.get('inputTokens', 0)} "
        f"out={usage.get('outputTokens', 0)} "
        f"total={usage.get('totalTokens', 0)} tokens  "
        f"~${_estimate_cost_usd(usage):.4f} (estimated)"
    )


def _run_batch(profile_name: str, region_name: str, enable_thinking: bool) -> int:
    print(f"CurVE routing batch — {len(TEST_QUESTIONS)} placeholder questions\n")
    passed = 0
    session_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    for question, expected in TEST_QUESTIONS:
        result = run_curve_turn(
            question,
            profile_name=profile_name,
            region_name=region_name,
            enable_thinking=enable_thinking,
        )
        actual = result["tool_trace"]
        ok = actual[:1] == [expected]
        passed += ok
        mark = "PASS" if ok else "FAIL"
        _add_usage(session_usage, result["usage"])
        print(f"[{mark}] {question}")
        print(f"       expected={expected!r}  actual_trace={actual}")
        print(f"       {_format_usage('turn', result['usage'])}")
    print(f"\n{passed}/{len(TEST_QUESTIONS)} routed to the expected tool first.")
    print(_format_usage("session total", session_usage))
    return 0 if passed == len(TEST_QUESTIONS) else 1


def _run_interactive(profile_name: str, region_name: str, enable_thinking: bool) -> int:
    print("CurVE M1 — interactive routing (Ctrl-D / 'exit' to quit)")
    print(f"profile={profile_name}  region={region_name}  thinking={enable_thinking}\n")
    session_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    while True:
        try:
            question = input("ask> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        result = run_curve_turn(
            question,
            profile_name=profile_name,
            region_name=region_name,
            enable_thinking=enable_thinking,
            verbose=True,
        )
        _add_usage(session_usage, result["usage"])
        print(f"\ntool_trace: {result['tool_trace']}")
        print(f"stop_reason: {result['stop_reason']}  iterations: {result['iterations']}")
        print(_format_usage("turn usage", result["usage"]))
        print(_format_usage("session total", session_usage))
        print(f"\n{result['text']}\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CurVE M1 tool-loop CLI.")
    parser.add_argument(
        "--profile",
        default=CURVE_DEFAULT_PROFILE,
        help=f"AWS profile (default: {CURVE_DEFAULT_PROFILE}).",
    )
    parser.add_argument(
        "--region", default=CURVE_REGION, help=f"AWS region (default: {CURVE_REGION})."
    )
    parser.add_argument(
        "--batch", action="store_true", help="Run the placeholder routing fixture."
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable extended thinking (default: enabled).",
    )
    args = parser.parse_args(argv)

    enable_thinking = not args.no_thinking
    if args.batch:
        return _run_batch(args.profile, args.region, enable_thinking)
    return _run_interactive(args.profile, args.region, enable_thinking)


if __name__ == "__main__":
    sys.exit(main())
