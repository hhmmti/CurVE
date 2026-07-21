# CurVE — run tooling. Portable: a local .venv/ (gitignored) is built by `make
# install`; no dev-machine paths. Override the SSO profile with CURVE_AWS_PROFILE.

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
CURVE_AWS_PROFILE ?= roam-ai

.PHONY: install login-aws run test smoke clean

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)          ## Build the venv and install pinned dependencies.
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

login-aws:                ## AWS SSO login for Bedrock + Athena.
	aws sso login --profile $(CURVE_AWS_PROFILE)

run:                      ## Launch the Streamlit surface.
	$(STREAMLIT) run streamlit_app.py

test:                     ## Run the test suite (no AWS needed; fetches are mocked).
	$(PY) -m pytest -q

smoke:                    ## Fast no-AWS regression: tests + import/config boot check.
	$(PY) -m pytest -q
	$(PY) -c "import streamlit_app; from curve import config; print('boot OK — model=%s region=%s profile=%s' % (config.BEDROCK_MODEL_ID, config.AWS_REGION, config.AWS_PROFILE))"

eval-sql-gold:            ## Validate the eval's gold SQL through guard+execute (no model calls).
	$(PY) -m evals.sql_eval --gold-only

eval-sql:                 ## LIVE sql_query execution-accuracy eval (real Bedrock + real Athena).
	$(PY) -m evals.sql_eval

clean:                    ## Remove caches and Streamlit artifacts.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
