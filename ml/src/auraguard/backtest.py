"""Walk-forward backtest with split-conformal prediction intervals.

For each test month M:

    fit          every row whose target hour is before (start of M - calibration days)
    calibrate    the last ``calibration_days`` before M: absolute errors set the interval
    test         every row whose target hour falls in M

The test month is never seen during fitting or calibration, and every origin and
target in it comes after them in time, so this is how the model would have performed
had it been run live through that month.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from auraguard.models import MODELS


@dataclass(frozen=True)
class Fold:
    month: str
    fit_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def monthly_folds(test_months: list[str], calibration_days: int = 14) -> list[Fold]:
    folds = []
    for m in test_months:
        start = pd.Timestamp(m + "-01")
        folds.append(Fold(m, start - pd.Timedelta(days=calibration_days), start, start + pd.offsets.MonthBegin(1)))
    return folds


def conformal_halfwidth(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample split-conformal quantile of absolute residuals."""
    r = np.sort(np.abs(residuals))
    n = len(r)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(r[k])


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
    }


def run(table: pd.DataFrame, horizon: int, folds: list[Fold], models: list[str], alpha: float = 0.1) -> list[dict]:
    """One result row per (model, fold), plus a pooled row per model."""
    rows, pooled = [], {m: {"y": [], "pred": [], "covered": [], "width": []} for m in models}
    for fold in folds:
        fit = table[table["target_time"] < fold.fit_end]
        cal = table[(table["target_time"] >= fold.fit_end) & (table["target_time"] < fold.test_start)]
        test = table[(table["target_time"] >= fold.test_start) & (table["target_time"] < fold.test_end)]
        if fit.empty or cal.empty or test.empty:
            continue
        for name in models:
            model = MODELS[name]().fit(fit)
            q = conformal_halfwidth(cal["y"].to_numpy() - model.predict(cal), alpha)
            t0 = time.perf_counter()
            pred = model.predict(test)
            per_row_ms = (time.perf_counter() - t0) / len(test) * 1000
            y = test["y"].to_numpy()
            covered = np.abs(y - pred) <= q
            row = {"horizon": horizon, "model": name, "fold": fold.month, **scores(y, pred)}
            row.update({"coverage": float(covered.mean()), "interval_halfwidth": q, "predict_ms_per_row": per_row_ms})
            rows.append(row)
            pooled[name]["y"].append(y)
            pooled[name]["pred"].append(pred)
            pooled[name]["covered"].append(covered)
            pooled[name]["width"].append(np.full(len(y), 2 * q))

    for name, p in pooled.items():
        if not p["y"]:
            continue
        y, pred = np.concatenate(p["y"]), np.concatenate(p["pred"])
        row = {"horizon": horizon, "model": name, "fold": "all", **scores(y, pred)}
        row.update(
            {
                "coverage": float(np.concatenate(p["covered"]).mean()),
                "interval_width_mean": float(np.concatenate(p["width"]).mean()),
            }
        )
        rows.append(row)

    base = next((r["mae"] for r in rows if r["model"] == "persistence" and r["fold"] == "all"), None)
    for r in rows:
        if r["fold"] == "all" and base:
            r["skill_vs_persistence"] = 1 - r["mae"] / base
    return rows
