# CurVE v1 — scope

The v1 tool roster, what's deliberately out, and the known limitations we ship and
document rather than fix. The roster and descriptions below are grounded in
`curve.tools` (the `TOOL_REGISTRY` and each tool's `toolSpec`).

## The 8-tool roster

### Connection-free (7)

These run on the well's own telemetry/production or the ML recommendation payload — no
pump connection needed.

| Tool | What it answers |
|------|-----------------|
| `production_history` | Historical production + telemetry over a window — oil/water/gas allocation rates, fluid character, recent trend. |
| `water_cut_gor_history` | Water-cut and gas-oil-ratio (GOR) history over a window — is the well watering up, how the GOR is trending, alongside liquid rate. |
| `delta_p_frequency` | Pump differential pressure (ΔP, discharge − intake) plotted against motor frequency over time — the pressure-lift response to operating frequency. |
| `delta_p_composition` | The pump pressure decomposition over time — how downhole discharge pressure is built from intake (PIP), the hydrostatic column, and tubing pressure. |
| `affinity_check` | Validates the ML recommendation against the pump Affinity Laws — whether the recommended frequency change is consistent with the implied flow (and ΔP) change. A check, not a new recommendation. |
| `recommendation_comparison` | The ML recommendation vs the current operating point — motor frequency, tubing pressure, and resulting rates, with the delta for each. Reports the recommendation faithfully; not a physics validation of it. |
| `energy_efficiency` | Energy efficiency at the current operating point — hydraulic vs input power, resulting efficiency, and specific power (kWh/bbl). |

### Connection-dependent (1)

| Tool | What it answers |
|------|-----------------|
| `curve_position` | Where the well's pump is operating on its ideal performance curve right now, and how far off design — the operating point (flow + ΔP) on the well-scaled ideal curve, the variance from ideal head, and position relative to BEP and the recommended window. Single- and multi-frequency (affinity family) overlays, ΔP-based, with the BEP position bundled in. The installed pump is picked **manually** at setup. |

## Explicitly out of v1

- **Pump auto-connection** (`pump_config`) — the pump is picked manually at setup, not
  resolved automatically.
- **Composite / tapered curves** (series pumps whose heads add).
- **BHP overlays.**
- **Gas / bubble-point / NPSH screens.**
- **Multi-tool compose & cross-tool synthesis** — the loop calls the matching tool(s),
  but does not synthesize one tool's output into another's.
- **SQL generation.**
- **Fleet / cross-well / org-wide views** — one well per session.
- **Action-taking** — CurVE explains and validates; it never changes a setpoint.

## Known limitations (v1)

Shipped and documented; **not** fixed in v1.

1. **ΔP question routing.** A bare "what's my delta-P" can misroute to
   `delta_p_composition` instead of `delta_p_frequency`. The two `toolSpec` descriptions
   read alike, and both render through the identical `_render_dp_kpis` KPI-card row
   (`streamlit_app._TOOL_KPI_RENDERERS`), so there is little to disambiguate a generic
   ΔP ask. Slated for a v2 spec-narrowing fix. (To force one, phrase the question toward
   *frequency response* or toward a *component breakdown*.)

2. **`delta_p_composition` display.** The vendored compute builds discharge pressure as
   `p_dis = tubing_pressure + hydrostatic` with **no friction term**
   (`compute.physics_common.calc_discharge_pressure_downhole_psi`). CurVE's decomposition
   (`curve.tools._dp_composition_split`) then derives friction as the residual
   `ΔP_pump − hydrostatic − backpressure`, so that residual closes to ≈0 **by
   construction**. And because each term is shown as a percentage share of ΔP_pump, the
   hydrostatic share can render **>100%** with a **negative backpressure share** whenever
   intake pressure exceeds surface tubing pressure — which reads as broken even though
   the arithmetic is correct. The correct v2 framing is a **signed additive breakdown**
   (hydrostatic + surface backpressure − intake offset), not percentage shares.

## Frozen-prompt drift note

CurVE's base system prompt (`curve.prompt._VE_SYSTEM_PROMPT_SNAPSHOT`) is a **verbatim
frozen snapshot** of the Virtual Engineer's `VIRTUAL_ENGINEER_SYSTEM_PROMPT` from the
monorepo (`esp_resources_v2/llm/prompts.py`). This standalone repo does not live-import
it, so the copy **can drift** from the live VE prompt over time — a known drift point,
to be re-synced when CurVE lands in the monorepo (where the live constant can be imported
instead of snapshotted). CurVE's own `curve.prompt._CURVE_ADDENDUM` carries the additive
overrides — tool-use routing, the trust-basis instruction, and prose-only narration — on
top of the snapshot.
