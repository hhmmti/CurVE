"""CurVE CLI — drive the hand-rolled tool loop against live Bedrock.

Three modes:
  * interactive (default): type a question → runs ``run_curve_turn`` against live
    Bedrock → prints the answer and the ordered tool_trace.
  * batch (``--batch``): runs the placeholder ``TEST_QUESTIONS`` fixture, printing
    expected-vs-actual tool per question.
  * session ask (``--org`` + ``--well`` + ``--ask``, M2): set up a session for a
    real (org, well), ask a ``production_history`` question, and print the narration
    (with trust basis), the ``values``, the ``trust_label``, and confirm a
    ``figure_ref`` was produced (optionally write the Plotly figure to HTML, since
    the CLI can't render it inline).

All modes call the **shipped** ``run_curve_turn`` — the CLI does not reimplement the
loop. Live Bedrock + Athena require AWS creds; run ``aws sso login --profile
roam-ai`` first.

Usage:
    python cli.py                          # interactive, profile roam-ai
    python cli.py --batch                  # routing batch over the fixture
    python cli.py --profile roam-ai --region us-east-1
    python cli.py --no-thinking            # disable extended thinking
    # M2 session ask (real telemetry+production for one well):
    python cli.py --org <ORG_ID> --well <WELL_ID> \\
        --ask "How has this well produced over the last 90 days?" \\
        --html-out /tmp/production_history.html
"""

import argparse
import os
import sys

from curve import ideal_catalog
from curve.engine import run_curve_turn
from curve.session import new_session_record, save_session
from curve.test_questions import TEST_QUESTIONS
from curve.tools import probe_connection_coverage
from curve.wrapper import CURVE_DEFAULT_PROFILE, CURVE_REGION

# M4 / 4a demo wells for the connection-coverage check. The build-plan names the three
# wells in `permian_resources`; we default to that org per the task spec, but the local
# well_configuration_v2 snapshot (intership-experience/18) maps two of them to OTHER
# orgs — HACKBERRY SPRINGS 3BH → `usedc`, ANNIE OAKLEY 4231A 1L → `civitas`, only
# CHEDDAR FED COM 502H → `permian_resources`. FLAGGED, not reconciled (guardrail 11):
# if a well returns no data live, re-run that one with `--org <real_org> --well "<name>"`.
DEMO_WELLS = [
    {"organization_id": "permian_resources", "well_id": "HACKBERRY SPRINGS 3BH"},
    {"organization_id": "permian_resources", "well_id": "ANNIE OAKLEY 4231A 1L"},
    {"organization_id": "permian_resources", "well_id": "CHEDDAR FED COM 502H"},
]

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


def _run_session_ask(
    organization_id: str,
    well_id: str,
    question: str,
    profile_name: str,
    region_name: str,
    enable_thinking: bool,
    html_out: str = None,
) -> int:
    """M2: set up a session for (org, well) and ask one production_history question."""
    session = save_session(
        new_session_record(
            session_id=f"cli-{organization_id}-{well_id}",
            organization_id=organization_id,
            well_id=well_id,
        )
    )
    print("CurVE M2 — session ask (production_history, real telemetry+production)")
    print(f"profile={profile_name}  region={region_name}  thinking={enable_thinking}")
    print(f"org={organization_id}  well={well_id}")
    print(f"question: {question}\n")

    result = run_curve_turn(
        question,
        session=session,
        profile_name=profile_name,
        region_name=region_name,
        enable_thinking=enable_thinking,
        verbose=True,
    )

    print(f"\ntool_trace: {result['tool_trace']}")
    print(f"stop_reason: {result['stop_reason']}  iterations: {result['iterations']}")
    print(_format_usage("turn usage", result["usage"]))

    # Surface each tool's envelope: values, trust_label, flags, and the figure_ref.
    for output in result.get("tool_outputs", []):
        env = output["result"]
        if not isinstance(env, dict):
            continue
        print(f"\n--- tool: {output['name']} ---")
        print(f"status:      {env.get('status')}")
        print(f"trust_label: {env.get('trust_label')}")
        print(f"flags:       {env.get('flags')}")
        print(f"values:      {env.get('values')}")
        figure_ref = env.get("figure_ref")
        figure = env.get("figure")
        if figure_ref:
            print(f"figure_ref:  {figure_ref}  (figure produced: {figure is not None})")
            if html_out and figure is not None:
                figure.write_html(html_out)
                print(f"figure written to: {html_out}")
        else:
            print("figure_ref:  (none — tool was blocked/unavailable)")

    print(f"\n=== narration ===\n{result['text']}\n")
    return 0


def _run_coverage_check(
    wells: list,
    profile_name: str,
    region_name: str,
    bep_tolerance: float,
) -> int:
    """M4 / 4a: per-well pump-connection coverage check (the de-risking report).

    For each demo well: fetch the catalog (re-read), compute total fluid, BEP-narrow,
    and print total fluid + candidate count + the candidate list. Exercises the
    connection-resolution layer end-to-end with NO ``curve_position`` tool present.
    """
    if profile_name:
        os.environ.setdefault("AWS_PROFILE", profile_name)

    print("CurVE M4/4a — pump-connection coverage check")
    print(f"profile={profile_name}  region={region_name}  bep_tolerance=±{bep_tolerance*100:.0f}%\n")
    print("Resolved catalog: "
          f"{ideal_catalog.IDEAL_CATALOG_CATALOG}.{ideal_catalog.IDEAL_CATALOG_DATABASE}."
          f"{ideal_catalog.IDEAL_CATALOG_TABLE}")
    print("BEP filter (ported from app narrow_catalog): "
          "min_recommended_bpd <= rate <= max_recommended_bpd  AND  "
          "bep_bpd*(1-tol) <= rate <= bep_bpd*(1+tol)\n")

    try:
        catalog_df = ideal_catalog.fetch_ideal_catalog(profile_name=profile_name, region_name=region_name)
    except Exception as exc:
        print(f"Catalog fetch FAILED: {exc}")
        return 1
    print(f"Catalog rows fetched (obsolete KEPT, not dropped): {len(catalog_df)}\n")

    resolved_inputs_ctx = {"bep_tolerance": bep_tolerance}
    for well in wells:
        org, wid = well["organization_id"], well["well_id"]
        print(f"=== {wid}  (org {org}) ===")
        try:
            probe = probe_connection_coverage(
                org, wid, catalog_df, resolved_inputs_ctx=resolved_inputs_ctx
            )
        except Exception as exc:
            print(f"  coverage probe FAILED: {exc}\n")
            continue
        report = probe["report"]
        tf = report["total_fluid_bpd"]
        print(f"  total fluid (median liquid rate): {tf:.0f} bpd" if tf is not None
              else "  total fluid: — (telemetry absent → no candidates)")
        print(f"  BEP-compatible rows: {report['n_bep_compatible']}  "
              f"| selectable candidates: {report['n_candidates']}  "
              f"| excluded (missing coeffs): {report['n_excluded_missing_coeffs']}  "
              f"| obsolete surfaced: {report['n_obsolete_surfaced']}")
        for c in report["candidates"]:
            obs = "  ⚠ obsolete" if c["is_obsolete"] else ""
            bep = f"{c['bep_bpd']:.0f}" if c["bep_bpd"] is not None else "—"
            print(f"    • {c['esp_model']}  (pump_id={c['pump_id']}, {c['manufacturer']}/{c['series']}) "
                  f"BEP {bep} bpd{obs}")
        if not report["candidates"]:
            print("    (no selectable candidates — operator picks nothing; downstream blocks honestly)")
        print()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CurVE tool-loop CLI.")
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
    # M2 session-ask mode.
    parser.add_argument("--org", help="organization_id for the session (M2 ask mode).")
    parser.add_argument("--well", help="well_id for the session (M2 ask mode).")
    parser.add_argument(
        "--ask", help="A production_history question to ask the set-up session (M2)."
    )
    parser.add_argument(
        "--html-out", help="Optional path to write the produced Plotly figure as HTML."
    )
    # M4 / 4a connection-coverage check.
    parser.add_argument(
        "--coverage-check",
        action="store_true",
        help="Run the M4/4a pump-connection coverage check over the demo wells.",
    )
    parser.add_argument(
        "--bep-tolerance",
        type=float,
        default=ideal_catalog.DEFAULT_BEP_TOLERANCE,
        help=f"BEP tolerance fraction for narrowing (default {ideal_catalog.DEFAULT_BEP_TOLERANCE}).",
    )
    args = parser.parse_args(argv)

    enable_thinking = not args.no_thinking
    if args.coverage_check:
        # --org/--well (when both given) override the demo list with a single well.
        wells = (
            [{"organization_id": args.org, "well_id": args.well}]
            if (args.org and args.well)
            else DEMO_WELLS
        )
        return _run_coverage_check(wells, args.profile, args.region, args.bep_tolerance)
    if args.ask or args.org or args.well:
        if not (args.org and args.well and args.ask):
            parser.error("--ask mode requires all of --org, --well, and --ask.")
        return _run_session_ask(
            args.org,
            args.well,
            args.ask,
            args.profile,
            args.region,
            enable_thinking,
            html_out=args.html_out,
        )
    if args.batch:
        return _run_batch(args.profile, args.region, enable_thinking)
    return _run_interactive(args.profile, args.region, enable_thinking)


if __name__ == "__main__":
    sys.exit(main())
