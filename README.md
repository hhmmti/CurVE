# CurVE

CurVE is a net-new, **additive** physics-validation layer for ROAM's Virtual Engineer (VE).
It exposes ESP physics validation as agent **tools** so operators can understand and trust
the ML's setpoint recommendations. CurVE does **not** modify the existing VE — it ships
alongside it as a self-contained module.

## What's in this repo

This repo was stood up at milestone **M0** of the CurVE v1 build. It vendors the three
reusable physics layers from the existing physics-validation Streamlit app
(`intership-experience/8/real-ideal-analysis/app/`):

| Layer        | Role                                                                 |
|--------------|----------------------------------------------------------------------|
| `compute/`   | Pure physics functions (no I/O): ΔP, affinity, BEP, energy, screens. |
| `services/`  | Workflow orchestration, incl. the data-availability **gate**.        |
| `plotting/`  | Plotly figure generation only (no physics).                          |

The source app **freezes as reference** — active development moves here. The `data/` layer
is intentionally **not** vendored (deferred to M2; the VE's data path differs from the app's),
so some `services/` modules carry dangling `from data ...` imports. That is expected and is
documented in the audit, not fixed.

## Status / next steps

- **M0:** repo + vendored layers + compatibility audit. No agent loop, no Bedrock, no
  data layer. Audit: [`docs/m0-compat-audit.md`](docs/m0-compat-audit.md).
- **M1 (this commit):** the walking-skeleton **tool loop** — a CurVE-scoped Bedrock
  Converse wrapper, a hand-rolled tool loop with extended thinking, 3 stub tools, a
  composed system prompt, and a CLI. Proves exactly one thing: *does the model route to
  the correct tool?* No gate, no physics, no data, no UI.

## M1 — the tool loop

The M1 code lives under [`curve/`](curve/):

| File | Role |
|------|------|
| `curve/wrapper.py` | CurVE-scoped Bedrock Converse wrapper. Fixes the monorepo `inferenceConfig` bug (assembles the config dict fully, then puts it in the request). Extended thinking ON by default (forces `temperature=1.0`, omits `topP`). |
| `curve/engine.py` | The **shipped** hand-rolled Converse `toolConfig` loop. `run_curve_turn(question, ...) -> {text, tool_trace, ...}`. Preserves `reasoningContent` across tool turns; 5-iteration safety cap. |
| `curve/tools.py` | 3 stub tools — real `toolSpec`s, mock bodies. Registry + `toolConfig` builder. |
| `curve/prompt.py` | Composed system prompt = snapshot VE base + thin CurVE addendum. |
| `curve/test_questions.py` | Placeholder canonical-question fixture (the single swap point). |
| `cli.py` | Interactive + batch CLI over `run_curve_turn`. |
| `tests/test_loop.py` | Loop-mechanics tests; **run with no AWS creds** (Converse mocked). |

### Running the CLI (live Bedrock)

Live routing needs AWS credentials. Log in first:

```bash
aws sso login --profile roam-ai
pip install -r requirements.txt      # into the shared venv

python cli.py                        # interactive: type a question, see the tool_trace
python cli.py --batch                # routing batch over the placeholder fixture
python cli.py --profile roam-ai --region us-east-1
python cli.py --no-thinking          # disable extended thinking
```

The CLI and tests both call the shipped `run_curve_turn` — the loop is never
reimplemented. Only `converse` (in tests) and tool outputs (everywhere in M1) are mocked.

### Running the tests (no AWS creds)

```bash
python -m pytest tests/test_loop.py -q
```

These mock Converse, so they prove the loop mechanics — tool routing, `toolResult`
appending, **`reasoningContent` preservation**, the 5-iteration cap, and a **populated
`inferenceConfig`** (the bug-fix guard) — without any AWS call.

### Tool naming convention

Tools are **snake_case** and **capability-named** — named for the operator's intent /
the question they answer (`production_history`, `curve_position`, `bubble_point_screen`),
**not** for the implementation module they will eventually call. The model routes on the
capability, so the name and description must read like an operator's question.

### Snapshot-prompt caveat

`curve/prompt.py` holds a **verbatim snapshot** of the monorepo
`VIRTUAL_ENGINEER_SYSTEM_PROMPT` (the playground can't import the monorepo package). It
**can drift** from the live VE prompt — re-syncing against the live base is a
monorepo-time concern (when CurVE lands there, the live constant can be imported instead).
CurVE appends only a thin tool-use addendum; the base supplies the ESP expertise.

### Placeholder questions — swap note

`curve/test_questions.py` is a **placeholder** fixture headed accordingly. Replace it with
the canonical operator question list once authored (post-Donny meeting). It is the single
swap point — the CLI batch mode and future eval scoring read from it.

## Build plan (authoritative)

The full v1 milestone sequence lives in the vault (notes-only, read-only):

```
~/roam-ai/intership-experience/18/CurVe-build-plan-v1.md
```
