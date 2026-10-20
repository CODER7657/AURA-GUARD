"""Forecasters compared in the backtest.

Three baselines a learned model has to beat, then two learned models. All share
``fit(train)`` and ``predict(table)`` on the supervised tables from ``features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from auraguard.features import feature_columns


class Persistence:
    """The next reading equals the current one."""

    def fit(self, train: pd.DataFrame) -> "Persistence":
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return table["lag_0"].to_numpy(dtype=float)


class SameHourYesterday:
    """The reading 24 hours before the target hour. Equals persistence at h = 24."""

    def fit(self, train: pd.DataFrame) -> "SameHourYesterday":
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        pred = table["same_hour_yesterday"].to_numpy(dtype=float)
        return np.where(np.isnan(pred), table["lag_0"].to_numpy(dtype=float), pred)


class Climatology:
    """Training-period mean for the station, local hour and weekday."""

    def fit(self, train: pd.DataFrame) -> "Climatology":
        self.table = train.groupby(["station", "hour", "weekday"])["y"].mean()
        self.station_mean = train.groupby("station")["y"].mean()
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(table[["station", "hour", "weekday"]])
        pred = self.table.reindex(keys).to_numpy()
        fallback = table["station"].map(self.station_mean).to_numpy()
        return np.where(np.isnan(pred), fallback, pred).astype(float)


class LinearLags:
    """Ridge regression on the lag, rolling and calendar features."""

    def fit(self, train: pd.DataFrame) -> "LinearLags":
        self.cols = [c for c in feature_columns(train) if c != "station_code"]
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        self.model.fit(train[self.cols], train["y"])
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.model.predict(table[self.cols])


class GradientBoosting:
    """Histogram gradient boosting with the station as a categorical feature.

    Missing lags are handled natively, so gaps in a station's record are not imputed.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "GradientBoosting":
        self.cols = feature_columns(train)
        self.model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=[self.cols.index("station_code")],
            early_stopping=False,
            random_state=self.seed,
        )
        self.model.fit(train[self.cols], train["y"])
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.model.predict(table[self.cols])


MODELS = {
    "persistence": Persistence,
    "same_hour_yesterday": SameHourYesterday,
    "climatology": Climatology,
    "linear": LinearLags,
    "gradient_boosting": GradientBoosting,
}
