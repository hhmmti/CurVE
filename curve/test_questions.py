# PLACEHOLDER — replace with canonical operator question list post-Donny meeting
#
# This fixture maps placeholder operator questions to the tool each *should* route
# to. It is the **single swap point** for the real canonical question list (which is
# authored later from operator/Nate input and serves triple duty: system-prompt
# routing hints, UI suggested-prompt chips, and eval seeds — CurVE-decisions §3
# Decision 4). Until then these are hand-written stand-ins spanning the live tools,
# used by the CLI batch mode to print expected-vs-actual routing.

from typing import List, Tuple

# (question, expected_tool)
TEST_QUESTIONS: List[Tuple[str, str]] = [
    ("How has well W-12 been producing over the last 90 days?", "production_history"),
    ("Show me the recent oil and water rate trend for well A-3.", "production_history"),
    ("How has the water cut and GOR changed on well W-12?", "water_cut_gor_history"),
    ("Is well A-3 watering up over the last few months?", "water_cut_gor_history"),
    ("Where is my pump operating on its curve right now for well W-12?", "curve_position"),
    ("Am I running near the best efficiency point on well B-7?", "curve_position"),
]
