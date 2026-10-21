import numpy as np
import pandas as pd
import pytest

from auraguard import serve
from auraguard.features import hourly_grid, make_supervised


def readings(start="2024-01-01", end="2024-03-31 23:00", seed=0, stations=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="h")
    frames = []
    for i, s in enumerate(stations):
        cycle = 20 + 10 * i + 8 * np.sin(2 * np.pi * (t.hour - 7) / 24)
        frames.append(pd.DataFrame({"station": s, "time_gmt": t, "utc_offset_h": -6,
                                    "value": cycle + rng.normal(0, 2, len(t))}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def bundle():
    return serve.train(hourly_grid(readings()), horizons=(1, 6))


def test_bundle_has_a_model_and_interval_per_horizon(bundle):
    assert sorted(bundle["models"]) == [1, 6]
    assert all(q > 0 for q in bundle["halfwidth"].values())
    assert bundle["codes"] == {"A": 0, "B": 1, "C": 2}


def test_forecast_is_issued_from_each_stations_last_reading(bundle):
    newer = hourly_grid(readings("2024-04-01", "2024-04-20 23:00", seed=1))
    records = serve.forecast(bundle, newer)
    assert [r["station"] for r in records] == ["A", "B", "C"]
    for r in records:
        assert r["issued_at_utc"] == "2024-04-20 23:00"
        assert r["hours_behind_file_end"] == 0
        assert [f["horizon_h"] for f in r["forecasts"]] == [1, 6]
        for f in r["forecasts"]:
            assert f["interval90_low_ppb"] <= f["no2_ppb"] <= f["interval90_high_ppb"]
        six = r["forecasts"][1]
        assert six["target_time_utc"] == "2024-04-21 05:00"
        assert six["target_time_local"] == "2024-04-20 23:00"  # UTC-6


def test_a_missing_station_does_not_shift_the_others_codes(bundle):
    newer = readings("2024-04-01", "2024-04-20 23:00", seed=1)
    full = serve.forecast(bundle, hourly_grid(newer))
    no_a = serve.forecast(bundle, hourly_grid(newer[newer["station"] != "A"]))
    assert full[1:] == no_a  # B and C get exactly the same forecasts without A


def test_live_rows_keep_the_last_origin_without_a_target():
    grid = hourly_grid(readings("2024-01-01", "2024-01-10 23:00"))
    live = make_supervised(grid, 6, require_target=False)
    scored = make_supervised(grid, 6)
    assert live["origin"].max() == pd.Timestamp("2024-01-10 23:00")
    assert scored["origin"].max() == pd.Timestamp("2024-01-10 17:00")


def test_evaluate_on_unseen_data_beats_persistence_six_hours_ahead(bundle):
    rows = serve.evaluate(bundle, hourly_grid(readings("2024-04-01", "2024-04-30 23:00", seed=2)), last_days=14)
    six = next(r for r in rows if r["horizon_h"] == 6)
    assert six["skill_vs_persistence"] > 0.3
    assert 0.8 <= six["coverage90"] <= 1.0
    assert six["stations"] == 3
