"""Turn hourly station series into a supervised table for one forecast horizon.

A row is a forecast made at an *origin* hour t for the *target* hour t + h. Every
feature is computed from values at or before t, except the calendar of the target
hour (hour of day and weekday are known in advance). Hours with no measurement stay
missing rather than being filled, so a model never sees invented data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = list(range(24))  # the last 24 hours, lag 0 = the origin hour itself


def hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex every station onto a complete hourly grid, leaving gaps as NaN."""
    frames = []
    for station, g in df.groupby("station", sort=True):
        s = g.set_index("time_gmt")["value"].sort_index()
        full = pd.date_range(s.index.min(), s.index.max(), freq="h")
        frames.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time_gmt": full,
                    "value": s.reindex(full).to_numpy(),
                    "utc_offset_h": int(g["utc_offset_h"].iloc[0]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_supervised(grid: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Features and target for forecasting ``horizon`` hours ahead (1 to 24)."""
    if not 1 <= horizon <= 24:
        raise ValueError("horizon must be between 1 and 24 hours")
    codes = {s: i for i, s in enumerate(sorted(grid["station"].unique()))}
    parts = []
    for station, g in grid.groupby("station", sort=True):
        y = g["value"].reset_index(drop=True)
        out = pd.DataFrame(
            {
                "station": station,
                "station_code": codes[station],
                "origin": g["time_gmt"].to_numpy(),
                "target_time": (g["time_gmt"] + pd.Timedelta(hours=horizon)).to_numpy(),
                "y": y.shift(-horizon).to_numpy(),
            }
        )
        for k in LAGS:
            out[f"lag_{k}"] = y.shift(k).to_numpy()
        # The target hour one day and one week earlier, both already observed at t.
        out["same_hour_yesterday"] = y.shift(24 - horizon).to_numpy()
        out["same_hour_last_week"] = y.shift(168 - horizon).to_numpy()
        out["mean_6h"] = y.rolling(6, min_periods=3).mean().to_numpy()
        out["mean_24h"] = y.rolling(24, min_periods=12).mean().to_numpy()
        out["std_24h"] = y.rolling(24, min_periods=12).std().to_numpy()

        local = pd.DatetimeIndex(out["target_time"]) + pd.Timedelta(hours=int(g["utc_offset_h"].iloc[0]))
        out["hour"] = local.hour
        out["weekday"] = local.dayofweek
        out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
        out["weekend"] = (local.dayofweek >= 5).astype(int)
        parts.append(out)

    table = pd.concat(parts, ignore_index=True)
    # A forecast needs the current reading and a real target to be scored against.
    return table.dropna(subset=["y", "lag_0"]).reset_index(drop=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    skip = {"station", "origin", "target_time", "y"}
    return [c for c in table.columns if c not in skip]
