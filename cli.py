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


def _run_batch(profile_name: str, region_name: str, enable_thinking: bool) -> int:
    print(f"CurVE routing batch — {len(TEST_QUESTIONS)} placeholder questions\n")
    passed = 0
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
        print(f"[{mark}] {question}")
        print(f"       expected={expected!r}  actual_trace={actual}")
    print(f"\n{passed}/{len(TEST_QUESTIONS)} routed to the expected tool first.")
    return 0 if passed == len(TEST_QUESTIONS) else 1


def _run_interactive(profile_name: str, region_name: str, enable_thinking: bool) -> int:
    print("CurVE M1 — interactive routing (Ctrl-D / 'exit' to quit)")
    print(f"profile={profile_name}  region={region_name}  thinking={enable_thinking}\n")
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
        print(f"\ntool_trace: {result['tool_trace']}")
        print(f"stop_reason: {result['stop_reason']}  iterations: {result['iterations']}")
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
