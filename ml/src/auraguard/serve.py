"""Train the deployed forecaster and issue forecasts from the latest hour of data.

The deployed model is the backtest's best recipe, fitted once on all the training
data: one gradient-boosting model per horizon, fitted on every target hour before
the last ``calibration_days``, with those last days setting a split-conformal 90%
interval. This is the same split each backtest month used, moved to the end of the
data.

``forecast`` then builds features at each station's most recent reading, which can
come from a newer file than the one the model was trained on.
"""

from __future__ import annotations

import platform

import numpy as np
import pandas as pd
import sklearn

from auraguard.aqi import no2_aqi
from auraguard.backtest import conformal_halfwidth, scores
from auraguard.features import make_supervised
from auraguard.models import GradientBoosting, Persistence

HORIZONS = (1, 6, 12, 24)
ALPHA = 0.1


def train(grid: pd.DataFrame, horizons=HORIZONS, calibration_days: int = 14, alpha: float = ALPHA,
          seed: int = 0) -> dict:
    """Fit one model per horizon and its conformal half-width. Returns a picklable bundle."""
    codes = {s: i for i, s in enumerate(sorted(grid["station"].unique()))}
    data_end = grid["time_gmt"].max() + pd.Timedelta(hours=1)
    cal_start = data_end - pd.Timedelta(days=calibration_days)
    models, halfwidth, n_fit, n_cal = {}, {}, {}, {}
    for h in horizons:
        table = make_supervised(grid, h, codes)
        fit = table[table["target_time"] < cal_start]
        cal = table[table["target_time"] >= cal_start]
        model = GradientBoosting(seed).fit(fit)
        halfwidth[h] = conformal_halfwidth(cal["y"].to_numpy() - model.predict(cal), alpha)
        models[h], n_fit[h], n_cal[h] = model, len(fit), len(cal)
    return {
        "models": models,
        "halfwidth": halfwidth,
        "codes": codes,
        "alpha": alpha,
        "calibration_start": str(cal_start),
        "data_end": str(data_end),
        "rows_fit": n_fit,
        "rows_calibration": n_cal,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }


def latest_rows(grid: pd.DataFrame, horizon: int, codes: dict) -> pd.DataFrame:
    """Feature row at each station's latest reading, for a forecast ``horizon`` hours ahead."""
    table = make_supervised(grid, horizon, codes, require_target=False)
    return table.sort_values("origin").groupby("station", sort=True).tail(1).set_index("station")


def forecast(bundle: dict, grid: pd.DataFrame, stations: pd.DataFrame | None = None) -> list[dict]:
    """One record per station with a forecast and 90% interval at every horizon.

    ``stations`` (station, place, lat, lon) adds location details when given.
    """
    offsets = grid.groupby("station")["utc_offset_h"].first()
    file_end = grid["time_gmt"].max()
    meta = stations.set_index("station") if stations is not None else None
    out: dict[str, dict] = {}
    for h, model in sorted(bundle["models"].items()):
        rows = latest_rows(grid, h, bundle["codes"])
        if rows.empty:
            continue
        pred = model.predict(rows.reset_index())
        q = bundle["halfwidth"][h]
        for (station, row), p in zip(rows.iterrows(), pred):
            offset = int(offsets[station])
            origin = pd.Timestamp(row["origin"])
            target = origin + pd.Timedelta(hours=h)
            rec = out.setdefault(station, {
                "station": station,
                "issued_at_utc": origin.strftime("%Y-%m-%d %H:%M"),
                "utc_offset_h": offset,
                "last_reading_ppb": round(float(row["lag_0"]), 1),
                "hours_behind_file_end": int((file_end - origin) / pd.Timedelta(hours=1)),
                "forecasts": [],
            })
            if meta is not None and station in meta.index:
                m = meta.loc[station]
                rec.update({"place": m["place"], "lat": float(m["lat"]), "lon": float(m["lon"])})
            value = float(p)
            rec["forecasts"].append({
                "horizon_h": int(h),
                "target_time_utc": target.strftime("%Y-%m-%d %H:%M"),
                "target_time_local": (target + pd.Timedelta(hours=offset)).strftime("%Y-%m-%d %H:%M"),
                "no2_ppb": round(value, 1),
                "interval90_low_ppb": round(max(0.0, value - q), 1),
                "interval90_high_ppb": round(value + q, 1),
                "aqi_category": no2_aqi(value)["category"],
            })
    return [out[s] for s in sorted(out)]


def evaluate(bundle: dict, grid: pd.DataFrame, last_days: int | None = None) -> list[dict]:
    """Score the deployed models and persistence on ``grid``. One row per horizon.

    With ``last_days`` only target hours in each station's last ``last_days`` days of
    data are scored, so stations whose record ends earlier still count. Used by the
    monitor on data the model never saw.
    """
    rows = []
    for h, model in sorted(bundle["models"].items()):
        table = make_supervised(grid, h, bundle["codes"])
        if last_days:
            end = table.groupby("station")["target_time"].transform("max")
            table = table[table["target_time"] > end - pd.Timedelta(days=last_days)]
        if table.empty:
            continue
        y = table["y"].to_numpy()
        pred = model.predict(table)
        base = Persistence().predict(table)
        q = bundle["halfwidth"][h]
        s, b = scores(y, pred), scores(y, base)
        per_station = (
            pd.DataFrame({"station": table["station"], "err": np.abs(pred - y), "covered": np.abs(pred - y) <= q})
            .groupby("station").agg(mae=("err", "mean"), coverage=("covered", "mean"), n=("err", "size"))
        )
        rows.append({
            "horizon_h": int(h),
            "n": s["n"],
            "mae": s["mae"],
            "rmse": s["rmse"],
            "bias": s["bias"],
            "persistence_mae": b["mae"],
            "skill_vs_persistence": 1 - s["mae"] / b["mae"],
            "coverage90": float((np.abs(pred - y) <= q).mean()),
            "worst_station_coverage90": float(per_station["coverage"].min()),
            "worst_station_coverage_id": str(per_station["coverage"].idxmin()),
            "stations": int(len(per_station)),
        })
    return rows
