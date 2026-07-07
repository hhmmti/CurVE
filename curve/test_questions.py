# Canonical operator question list — CurVE v1 (authored 2026-07-05, from
# 18/CurVE-canonical-questions.md).
#
# Single-tool routing fixture: each question routes to exactly ONE tool, in
# authentic ESP operator language (web research + Donny's curve_position
# priority). Triple duty (CurVE-decisions §3 Decision 4): system-prompt routing
# hints, UI suggested-prompt chips, eval seeds. Also the M5 gap-pass probe set
# (one clean single-fire per tool) and the Keith demo script. Wells = the four
# live demo wells.
#
# Multi-tool FAQs (route to 2-3 tools) are parked in
# 18/CurVE-canonical-questions.md — they can't live here because this fixture is
# 1:1 (question -> one expected_tool). They land when routing-eval / compose gets
# a home (post-v1).

from typing import List, Tuple

# (question, expected_tool)
TEST_QUESTIONS: List[Tuple[str, str]] = [
    # --- production_history ---
    ("What's my oil rate done over the last 90 days on CHEDDAR FED COM 502H?", "production_history"),
    ("Give me the last 6 months of oil, water and gas rates on HACKBERRY SPRINGS 3BH.", "production_history"),
    # --- water_cut_gor_history ---
    ("Is ANNIE OAKLEY 4231A 1L watering up? Show me the water cut trend.", "water_cut_gor_history"),
    ("Is the gas-oil ratio trending up on CHEDDAR FED COM 502H? Looks like I'm gassing up.", "water_cut_gor_history"),
    # --- curve_position (Donny priority) ---
    ("Where am I sitting on the pump curve right now on HACKBERRY SPRINGS 3BH?", "curve_position"),
    ("How far off BEP is this pump running on PETERS 1102A 15MS?", "curve_position"),
    ("Am I inside the recommended operating range or out of range on HACKBERRY SPRINGS 3BH?", "curve_position"),
    # --- recommendation_comparison ---
    ("What frequency does the model want versus what I'm running now on CHEDDAR FED COM 502H?", "recommendation_comparison"),
    ("If I take the model's recommendation on ANNIE OAKLEY 4231A 1L, how much more oil do I get?", "recommendation_comparison"),
    # --- affinity_check ---
    ("Does the recommended frequency change line up with the affinity laws on ANNIE OAKLEY 4231A 1L?", "affinity_check"),
    # --- energy_efficiency ---
    ("What's my specific power — kWh per barrel — on ANNIE OAKLEY 4231A 1L?", "energy_efficiency"),
    ("How efficient is this pump running right now on PETERS 1102A 15MS?", "energy_efficiency"),
    # --- delta_p_frequency ---
    ("What's my delta-P across the pump right now on PETERS 1102A 15MS?", "delta_p_frequency"),
    # --- delta_p_composition ---
    ("Break down my TDH into lift, friction and wellhead pressure on HACKBERRY SPRINGS 3BH.", "delta_p_composition"),
]