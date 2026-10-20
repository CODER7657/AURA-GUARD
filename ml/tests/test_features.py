import numpy as np
import pandas as pd
import pytest

from auraguard.features import hourly_grid, make_supervised


def readings(hours: int = 400, stations=("A", "B"), drop=()) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=hours, freq="h")
    rows = []
    for i, s in enumerate(stations):
        for k, ts in enumerate(t):
            if (s, k) in drop:
                continue
            rows.append({"station": s, "time_gmt": ts, "utc_offset_h": -5, "value": float(1000 * i + k)})
    return pd.DataFrame(rows)


def test_grid_keeps_gaps_as_missing():
    grid = hourly_grid(readings(hours=10, stations=("A",), drop={("A", 3)}))
    assert len(grid) == 10
    assert np.isnan(grid.loc[3, "value"])


@pytest.mark.parametrize("h", [1, 6, 24])
def test_lags_and_target_line_up(h):
    # value = hour index, so every column can be checked by arithmetic
    table = make_supervised(hourly_grid(readings(stations=("A",))), h)
    row = table.iloc[200]
    t = row["lag_0"]
    assert row["y"] == t + h
    assert row["lag_5"] == t - 5
    assert row["same_hour_yesterday"] == t + h - 24
    assert row["same_hour_last_week"] == t + h - 168


def test_no_feature_uses_data_after_the_origin():
    """Changing every reading after hour 250 must not change any feature at origins up to 250."""
    clean = make_supervised(hourly_grid(readings(stations=("A",))), 6)
    later = readings(stations=("A",))
    later.loc[later["time_gmt"] > pd.Timestamp("2024-01-01") + pd.Timedelta(hours=250), "value"] = -999.0
    corrupted = make_supervised(hourly_grid(later), 6)

    cutoff = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=250)
    a = clean[clean["origin"] <= cutoff].drop(columns=["y"]).reset_index(drop=True)
    b = corrupted[corrupted["origin"] <= cutoff].drop(columns=["y"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_calendar_uses_local_time_of_the_target_hour():
    table = make_supervised(hourly_grid(readings(stations=("A",))), 6)
    row = table.iloc[0]  # origin 00:00 GMT, target 06:00 GMT = 01:00 local at UTC-5
    assert row["hour"] == 1


def test_rows_without_a_target_or_current_reading_are_dropped():
    table = make_supervised(hourly_grid(readings(hours=30, stations=("A",), drop={("A", 10)})), 1)
    assert not table["y"].isna().any()
    assert not table["lag_0"].isna().any()
    assert len(table) == 30 - 1 - 2  # last hour has no target; hour 10 is neither a target nor an origin


def test_horizon_bounds():
    with pytest.raises(ValueError):
        make_supervised(hourly_grid(readings(hours=50, stations=("A",))), 25)
