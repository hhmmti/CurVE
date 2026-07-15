# CurVE — configuration

Every runtime pin is read from an environment variable with a built-in default equal to
the prior hardcoded value (`curve/config.py`). With no `.env` and no OS env set, CurVE
behaves exactly as before config was externalized.

**The authoritative variable list is [`.env.example`](.env.example)** — copy it to
`.env` and edit only the pins you want to change. Every variable is optional. This file
does not re-list them; it only calls out the things worth knowing before you run.

**Precedence:** a real OS environment variable wins over a line in `.env`, which wins
over the built-in default.

## Prod vs dev sources

Most Athena bindings default to **production** (`roam_prd_products` /
`roam_prd_ddb`) — telemetry, production, recommendations, well configuration. Two
default to **dev** catalogs, so be deliberate about them:

- **Well depth** — `CURVE_WELL_DEPTH_CATALOG` defaults to `roam_dev_products`
  (`well_depth_dev.rrc_well_depth`), the RRC-mined depth used for the hydrostatic ΔP term.
- **Ideal pump catalog** — `CURVE_IDEAL_CATALOG_CATALOG` defaults to `roam_dev_products`
  (`esp_ideal_pump_dev.ideal_pump_library_v1`), the ChampionX curve library that
  `curve_position` scales to the well.

If you point CurVE at a different environment, revisit these two dev-pinned sources
explicitly.

## AWS profile / SSO

CurVE uses one AWS SSO profile for **both** Bedrock and Athena. Profile precedence is
explicit → `AWS_PROFILE` env → the default credential chain (which keeps the role-based
path working if CurVE later runs in-Lambda):

- `CURVE_AWS_PROFILE` (default `roam-ai`) seeds the CLI's `--profile` and the Streamlit
  profile field; `make login-aws` logs this profile in (override with
  `CURVE_AWS_PROFILE=<name> make login-aws`).
- Both run surfaces export the chosen profile to `AWS_PROFILE` so a tool's internal
  Athena fetch resolves the same profile as the Bedrock loop.

## Dev panel

`CURVE_DEV_PANEL` (default **off**) sets the default state of the Streamlit "Developer
mode" checkbox, which exposes the agentic loop laid bare — tool trace, gate verdict,
token/cost, and the raw model-facing envelope. It stays toggleable per session; set
`CURVE_DEV_PANEL=true` to default it on.
