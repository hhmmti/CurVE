# CurVE

CurVE is a physics-validation layer for ESP well optimization, delivered as an
agentic tool loop over Roam's well data. An operator asks a plain-language question
about a selected well; CurVE routes it to one of eight physics/data tools, runs the
calculation behind a data-availability **gate**, and answers in prose with an explicit
**trust label** — so the operator sees not just the number, but how much to trust it
and why. It runs the Bedrock Converse tool loop by hand (native `toolConfig`, no
LangChain) and stands alone: no monorepo or playground imports.

## Quickstart

Requires Python **3.12.8** (pinned in [`.python-version`](.python-version)) and the
`==`-pinned dependencies in [`requirements.txt`](requirements.txt). CurVE reads live
AWS Bedrock (the model) and Athena (the data), so you need AWS SSO credentials.

```bash
make install          # build .venv/ and install pinned requirements
make login-aws        # AWS SSO login (aws sso login --profile roam-ai)

# optional: override any deployment pin. With no .env, built-in defaults are used.
cp .env.example .env  # then edit; every var is optional (see CONFIG.md)
```

Then run either surface:

```bash
make run               # Streamlit chat surface (streamlit run streamlit_app.py)

# or the CLI, one question against one well:
python cli.py --org <ORG_ID> --well "<WELL_ID>" \
    --ask "How has this well produced over the last 90 days?" \
    --html-out /tmp/curve_figure.html      # the CLI can't render inline; write the figure
```

Both surfaces call the same shipped tool loop (`curve.engine.run_curve_turn`) — neither
reimplements it. The default model is `us.anthropic.claude-sonnet-4-6` on `us-east-1`
(both overridable; see [CONFIG.md](CONFIG.md)). `make smoke` runs the tests plus an
import/config boot check with no AWS needed.

## Trust-label model

Every tool result carries a `trust_label`, decided by the gate that runs **per
calculation, before compute** — never hardcoded, never inferred by the model. Three
tiers, strongest to weakest:

- **Validated** — measured, with no substitution (e.g. allocation + telemetry straight
  from the well).
- **Estimated** — a defaulted or derived input fed in (e.g. hydrostatic ΔP using a
  default well depth / SG).
- **Proxy** — a data-backed stand-in was used (e.g. amp×volt power when no direct
  `motor_power_kw` channel exists).

When a tool folds several inputs of different provenance, the overall label is
**weakest-wins** (`curve.gate.weakest_trust`): Validated liquid + Estimated ΔP + Proxy
power → **Proxy**, never the strongest term. When a required setup value is absent the
tool returns **`blocked`** (or **`not-ready`** for the recommendation-absence case)
rather than fabricating a number. *(A fourth label, "Research prototype", exists in the
precedence scaffolding and the UI badge map but no v1 tool emits it.)*

## Architecture at a glance

```
  setup                     gate-before-compute        Bedrock Converse loop
  ─────                     ───────────────────        ─────────────────────
  pick org / well           per-tool availability      hand-rolled tool loop
  → availability report     gate runs BEFORE any       (native toolConfig,
  → resolve / proxy inputs  physics; emits the          NO LangChain)
    (depth, SG, pump pick)  trust_label + flags        converse → tool_use?
                                                         → run tool → toolResult
                                                         → loop → final prose
                                   │
                                   ▼
                  physics (compute/) + Plotly (plotting/)
                                   │
                                   ▼
        envelope { status, values, trust_label, flags, figure_ref, figure }
```

- **Setup** (`streamlit_app.py` → `curve.tools.probe_*`): the operator picks an
  org/well (enumerated from `esp_well_configuration_v2`); CurVE front-loads each tool's
  gate into an availability report, resolves depth/SG (real RRC depth → override →
  default), and — for `curve_position` — offers a manual pump pick from the BEP-narrowed
  catalog. Org, well, resolved inputs, and the pump pick are **injected** into every tool
  call by the engine; the model never sees or supplies them.
- **Gate before compute** (`curve.gate`): each tool runs its availability gate *before*
  computing and carries the resulting `trust_label` faithfully into its envelope.
- **The loop** (`curve.engine.run_curve_turn` + `curve.wrapper.CurveBedrockWrapper`):
  a hand-rolled Bedrock Converse loop using the native `toolConfig` built from
  `curve.tools.TOOL_REGISTRY` — no LangChain/LangGraph. Extended thinking is on;
  `reasoningContent` is echoed back verbatim across tool turns as Bedrock requires.
- **Physics + charts**: pure physics in `compute/`, Plotly figures in `plotting/`.
- **The envelope** (`curve.envelope`): one shape for success and failure alike. The
  engine strips `figure` / `figure_ref` (and `curve_position`'s `figures` list) before
  the result returns to the model (`NON_MODEL_RESULT_KEYS`) — **figures render to the
  UI, never back into the model.** The model narrates from `values` only.

See [SCOPE.md](SCOPE.md) for the tool roster and what's in / out of v1.

## Try these

Grouped by the tool each question routes to (the well is set up for the session; you
ask only the question):

**production_history**
- "What's my oil rate done over the last 90 days on CHEDDAR FED COM 502H?"

**water_cut_gor_history**
- "Is ANNIE OAKLEY 4231A 1L watering up? Show me the water cut trend."

**curve_position**
- "Where am I sitting on the pump curve right now on HACKBERRY SPRINGS 3BH?"
- "How far off BEP is this pump running on PETERS 1102A 15MS?"

**recommendation_comparison**
- "What frequency does the model want versus what I'm running now on CHEDDAR FED COM 502H?"

**affinity_check**
- "Does the recommended frequency change line up with the affinity laws on ANNIE OAKLEY 4231A 1L?"

**energy_efficiency**
- "What's my specific power — kWh per barrel — on ANNIE OAKLEY 4231A 1L?"

**delta_p_frequency**
- "What's my delta-P across the pump right now on PETERS 1102A 15MS?"

**delta_p_composition**
- "Break down my TDH into lift, friction and wellhead pressure on HACKBERRY SPRINGS 3BH."

> Note: a bare "what's my delta-P" question can route to `delta_p_composition` instead
> of `delta_p_frequency` — the two tool specs read alike. See the known limitations in
> [SCOPE.md](SCOPE.md).
