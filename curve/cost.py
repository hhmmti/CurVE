"""Shared Bedrock token-cost estimator (used by the CLI and the Streamlit surface).

Relocated out of ``cli.py`` so both callers import one implementation. These are
ESTIMATES (USD per 1M tokens) for the ``us.`` cross-region inference profile: base
Sonnet pricing ($3 in / $15 out) plus ~10% cross-region uplift. Not authoritative —
confirm against current AWS pricing.
"""

from __future__ import annotations

from typing import Dict

# EDIT HERE to update pricing (estimates, not a bill).
COST_PER_1M_INPUT_USD = 3.30
COST_PER_1M_OUTPUT_USD = 16.50


def estimate_cost_usd(usage: Dict) -> float:
    """Estimated USD cost for an aggregated usage block (an estimate, not a bill)."""
    in_cost = usage.get("inputTokens", 0) / 1_000_000 * COST_PER_1M_INPUT_USD
    out_cost = usage.get("outputTokens", 0) / 1_000_000 * COST_PER_1M_OUTPUT_USD
    return in_cost + out_cost
