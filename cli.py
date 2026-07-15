"""CurVE CLI — drive the hand-rolled tool loop against live Bedrock.

Two modes:
  * session ask (``--org`` + ``--well`` + ``--ask``): set up a session for a real
    (org, well), ask a question, and print the narration (with trust basis), the
    ``values``, the ``trust_label``, and confirm a ``figure_ref`` was produced
    (optionally write the Plotly figure to HTML, since the CLI can't render inline).
  * coverage check (``--coverage-check``): the M4/4a pump-connection coverage report
    over the demo wells (or one ``--org``/``--well``).

Both modes call the **shipped** ``run_curve_turn`` / connection layer — the CLI does
not reimplement anything. Live Bedrock + Athena require AWS creds; run ``aws sso
login --profile roam-ai`` first.

Usage:
    # session ask (real telemetry+production for one well):
    python cli.py --org <ORG_ID> --well <WELL_ID> \\
        --ask "How has this well produced over the last 90 days?" \\
        --html-out /tmp/production_history.html
    python cli.py --profile roam-ai --region us-east-1 --org ... --well ... --ask ...
    python cli.py --no-thinking --org ... --well ... --ask ...   # disable thinking
    python cli.py --coverage-check          # pump-connection coverage over demo wells
"""

import argparse
import os
import sys

from curve import ideal_catalog
from curve.cost import estimate_cost_usd
from curve.engine import run_curve_turn
from curve.session import new_session_record, save_session
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

def _format_usage(label: str, usage: dict) -> str:
    return (
        f"{label}: in={usage.get('inputTokens', 0)} "
        f"out={usage.get('outputTokens', 0)} "
        f"total={usage.get('totalTokens', 0)} tokens  "
        f"~${estimate_cost_usd(usage):.4f} (estimated)"
    )


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
    # Route the tool's internal Athena fetch through the intended SSO profile too
    # (mirrors _run_coverage_check): the data accessor falls back to AWS_PROFILE when no
    # explicit profile is passed, so without this it could resolve a stale default token.
    if profile_name:
        os.environ.setdefault("AWS_PROFILE", profile_name)
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
    parser.error(
        "no mode selected — pass --org/--well/--ask (session ask) or --coverage-check."
    )


if __name__ == "__main__":
    sys.exit(main())
