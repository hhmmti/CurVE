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

- **M0 (this commit):** repo + vendored layers + compatibility audit. No agent loop, no
  Bedrock, no data layer.
- Compatibility audit: [`docs/m0-compat-audit.md`](docs/m0-compat-audit.md) — per-layer
  CurVE-readiness verdicts and blockers.

## Build plan (authoritative)

The full v1 milestone sequence lives in the vault (notes-only, read-only):

```
~/roam-ai/intership-experience/18/CurVe-build-plan-v1.md
```
