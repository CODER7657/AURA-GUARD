import numpy as np
import pandas as pd
import pytest

from auraguard import backtest
from auraguard.features import hourly_grid, make_supervised


def daily_cycle(seed: int = 0) -> pd.DataFrame:
    """Three stations, January to April 2024, a strong daily cycle plus noise."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", "2024-04-30 23:00", freq="h")
    frames = []
    for i, s in enumerate(["A", "B", "C"]):
        cycle = 20 + 10 * i + 8 * np.sin(2 * np.pi * (t.hour - 7) / 24)
        frames.append(pd.DataFrame({"station": s, "time_gmt": t, "utc_offset_h": 0,
                                    "value": cycle + rng.normal(0, 2, len(t))}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def table12():
    return make_supervised(hourly_grid(daily_cycle()), 12)


def test_folds_never_fit_on_the_test_month():
    (fold,) = backtest.monthly_folds(["2024-04"], calibration_days=14)
    assert fold.fit_end == pd.Timestamp("2024-03-18")
    assert fold.test_start == pd.Timestamp("2024-04-01")
    assert fold.test_end == pd.Timestamp("2024-05-01")


def test_conformal_halfwidth_is_the_finite_sample_quantile():
    # 9 residuals at alpha = 0.1 -> ceil(10 * 0.9) = 9th smallest
    assert backtest.conformal_halfwidth(np.arange(1, 10), alpha=0.1) == 9.0


def test_persistence_has_zero_skill_against_itself(table12):
    rows = backtest.run(table12, 12, backtest.monthly_folds(["2024-03", "2024-04"]), ["persistence"])
    pooled = next(r for r in rows if r["fold"] == "all")
    assert pooled["skill_vs_persistence"] == pytest.approx(0.0)


def test_learned_models_beat_persistence_on_a_daily_cycle(table12):
    """12 hours ahead, 'same as now' is the wrong phase of the cycle; models that know the hour win."""
    rows = backtest.run(table12, 12, backtest.monthly_folds(["2024-04"]),
                        ["persistence", "climatology", "linear", "gradient_boosting"])
    pooled = {r["model"]: r for r in rows if r["fold"] == "all"}
    for name in ["climatology", "linear", "gradient_boosting"]:
        assert pooled[name]["skill_vs_persistence"] > 0.3


def test_intervals_cover_about_ninety_percent(table12):
    rows = backtest.run(table12, 12, backtest.monthly_folds(["2024-04"]), ["linear"])
    pooled = next(r for r in rows if r["fold"] == "all")
    assert 0.85 <= pooled["coverage"] <= 0.95
