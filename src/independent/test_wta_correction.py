"""Semantic check for the WTA team-event protocol correction."""

import pandas as pd

from src.independent.run_wta_2025_corrected import team_event_mask


def test_team_event_filter_covers_named_team_competitions_only():
    names = pd.Series([
        "United Cup",
        "Fed Cup WG F: USA vs BLR",
        "Billie Jean King Cup Finals",
        "Hopman Cup",
        "WTA Finals",
        "Australian Open",
        None,
    ])
    assert team_event_mask(names).tolist() == [True, True, True, True, False, False, False]
