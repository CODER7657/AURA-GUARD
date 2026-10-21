"""Typed tools the forecast analyst may call.

Every number the agent is allowed to say comes out of one of these functions. They
read only committed artifacts (results/forecasts.json, results/backtest_summary.json,
results/monitor_2025.json) and the EPA AQI table, and never call a model or the
network, so they are deterministic and cheap.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from auraguard.aqi import NO2_BREAKPOINTS, no2_aqi

RESULTS = Path(__file__).resolve().parents[3] / "results"

STATE_NAMES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
}


def _hour12(hhmm: str) -> str:
    h = int(hhmm[:2])
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


class ForecastTools:
    """The tool implementations, over one set of artifacts."""

    def __init__(self, results_dir: Path | str = RESULTS):
        d = Path(results_dir)
        self.forecasts = json.loads((d / "forecasts.json").read_text(encoding="utf-8"))
        summary = json.loads((d / "backtest_summary.json").read_text(encoding="utf-8"))
        self.pooled = {(r["horizon"], r["model"]): r for r in summary["pooled"]}
        oot = d / "monitor_2025.json"
        self.out_of_time = json.loads(oot.read_text(encoding="utf-8")) if oot.exists() else None
        self.meta = {s["station"]: s for s in summary["stations"]}
        self.by_station = {s["station"]: s for s in self.forecasts["stations"]}
        self.horizons = list(self.forecasts["horizons_h"])

    # -- helpers ---------------------------------------------------------------
    def station_ids(self) -> set[str]:
        return set(self.meta)

    def places(self) -> dict[str, str]:
        return {s: m["place"] for s, m in self.meta.items()}

    def _check(self, station: str, horizon_h) -> dict | None:
        if station not in self.meta:
            return {"error": "unknown station id; call list_stations for the valid ids"}
        if station not in self.by_station:
            return {"error": f"station {station} ({self.meta[station]['place']}) has no forecast: the newest "
                             f"EPA file ({self.forecasts['source']}) has no readings for it.",
                    "station": station, "place": self.meta[station]["place"]}
        try:
            h = int(horizon_h)
        except (TypeError, ValueError):
            h = None
        if h not in self.horizons:
            return {"error": f"no model for a {horizon_h} hour horizon. Available horizons: "
                             f"{', '.join(str(x) for x in self.horizons)} hours.",
                    "available_horizons_h": self.horizons}
        return None

    def _forecast(self, station: str, horizon_h) -> dict:
        return next(f for f in self.by_station[station]["forecasts"] if f["horizon_h"] == int(horizon_h))

    # -- tools -----------------------------------------------------------------
    def list_stations(self, state: str | None = None) -> dict:
        rows = []
        for sid, m in sorted(self.meta.items()):
            if state and state.lower() not in m["place"].lower():
                continue
            f = self.by_station.get(sid)
            rows.append({"station": sid, "place": m["place"], "lat": round(m["lat"], 3), "lon": round(m["lon"], 3),
                         "has_forecast": f is not None,
                         "issued_at_utc": f["issued_at_utc"] if f else None})
        return {"stations": rows, "count": len(rows),
                "note": "Places are county, state. Every station measures NO2 only."}

    def get_forecast(self, station: str, horizon_h: int) -> dict:
        err = self._check(station, horizon_h)
        if err:
            return err
        s, f = self.by_station[station], self._forecast(station, horizon_h)
        issued = datetime.strptime(s["issued_at_utc"], "%Y-%m-%d %H:%M") + timedelta(hours=s["utc_offset_h"])
        return {
            "station": station,
            "place": self.meta[station]["place"],
            "horizon_h": f["horizon_h"],
            "issued_at_utc": s["issued_at_utc"],
            "issued_at_local": issued.strftime("%Y-%m-%d %H:%M"),
            "issued_from": f"the last reading in the EPA file {self.forecasts['source']}; not a live forecast",
            "last_reading_ppb": s["last_reading_ppb"],
            "target_time_utc": f["target_time_utc"],
            "target_time_local": f["target_time_local"],
            "target_hour_local_12h": _hour12(f["target_time_local"][11:16]),
            "local_time_zone": f"local standard time, UTC{s['utc_offset_h']:+d}",
            "no2_ppb": f["no2_ppb"],
            "aqi_category": f["aqi_category"],
            "units": "ppb (parts per billion), hourly NO2",
        }

    def get_interval(self, station: str, horizon_h: int) -> dict:
        err = self._check(station, horizon_h)
        if err:
            return err
        f = self._forecast(station, horizon_h)
        h = f["horizon_h"]
        out = {
            "station": station,
            "horizon_h": h,
            "no2_ppb": f["no2_ppb"],
            "interval90_low_ppb": f["interval90_low_ppb"],
            "interval90_high_ppb": f["interval90_high_ppb"],
            "interval_level_pct": 90,
            "method": "split-conformal; the same half-width for every station at this horizon",
            "backtest_coverage_pct": round(100 * self.pooled[(h, "gradient_boosting")]["coverage"], 1),
            "aqi_category_low": no2_aqi(f["interval90_low_ppb"])["category"],
            "aqi_category_high": no2_aqi(f["interval90_high_ppb"])["category"],
        }
        oot = self._out_of_time(h)
        if oot:
            out["coverage_on_2025_pct"] = round(100 * oot["coverage90"], 1)
        return out

    def get_aqi_band(self, no2_ppb: float) -> dict:
        try:
            value = float(no2_ppb)
            out = no2_aqi(value)
        except (TypeError, ValueError):
            return {"error": "no2_ppb must be a number of parts per billion"}
        out["scale"] = "EPA AQI, NO2 1-hour breakpoints (ppb)"
        out["breakpoints_ppb"] = [{"category": n, "low": lo, "high": hi} for lo, hi, _, _, n in NO2_BREAKPOINTS]
        return out

    def get_backtest_skill(self, horizon_h: int) -> dict:
        try:
            h = int(horizon_h)
        except (TypeError, ValueError):
            h = None
        if (h, "gradient_boosting") not in self.pooled:
            return {"error": f"no backtest for a {horizon_h} hour horizon", "available_horizons_h": self.horizons}
        gb, base = self.pooled[(h, "gradient_boosting")], self.pooled[(h, "persistence")]
        out = {
            "horizon_h": h,
            "test_period": "July to December 2024, monthly walk-forward, 20 EPA monitors",
            "n_forecasts": gb["n"],
            "mae_ppb": round(gb["mae"], 2),
            "rmse_ppb": round(gb["rmse"], 2),
            "r2": round(gb["r2"], 3),
            "persistence_mae_ppb": round(base["mae"], 2),
            "improvement_vs_persistence_pct": round(100 * gb["skill_vs_persistence"], 1),
            "interval_level_pct": 90,
            "interval90_coverage_pct": round(100 * gb["coverage"], 1),
        }
        oot = self._out_of_time(h)
        if oot:
            out["out_of_time_2025"] = {
                "data": "all of 2025, model trained on 2024 only",
                "mae_ppb": round(oot["mae"], 2),
                "persistence_mae_ppb": round(oot["persistence_mae"], 2),
                "improvement_vs_persistence_pct": round(100 * oot["skill_vs_persistence"], 1),
                "interval90_coverage_pct": round(100 * oot["coverage90"], 1),
            }
        return out

    def _out_of_time(self, h: int) -> dict | None:
        if not self.out_of_time:
            return None
        return next((r for r in self.out_of_time["horizons"] if r["horizon_h"] == h), None)

    def call(self, name: str, args: dict) -> dict:
        fn = {"list_stations": self.list_stations, "get_forecast": self.get_forecast,
              "get_interval": self.get_interval, "get_aqi_band": self.get_aqi_band,
              "get_backtest_skill": self.get_backtest_skill}.get(name)
        if fn is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


_STATION = {"type": "string", "description": "station id from list_stations, e.g. 17-031-3103-1"}
_HORIZON = {"type": "integer", "enum": [1, 6, 12, 24], "description": "hours ahead of the issue time"}

TOOL_SCHEMAS = [
    _fn("list_stations", "List the NO2 monitoring stations (id, county and state, location, whether a "
        "forecast exists). Optionally filter by a state or county name.",
        {"state": {"type": "string", "description": "optional state or county name filter"}}, []),
    _fn("get_forecast", "Point forecast of hourly NO2 (ppb) at a station, horizon_h hours after the issue time, "
        "with the target time in UTC and local standard time and the AQI category.",
        {"station": _STATION, "horizon_h": _HORIZON}, ["station", "horizon_h"]),
    _fn("get_interval", "90% prediction interval for the forecast at a station and horizon, with its "
        "measured coverage.", {"station": _STATION, "horizon_h": _HORIZON}, ["station", "horizon_h"]),
    _fn("get_aqi_band", "EPA AQI value and category for a 1-hour NO2 concentration in ppb.",
        {"no2_ppb": {"type": "number"}}, ["no2_ppb"]),
    _fn("get_backtest_skill", "Measured accuracy at a horizon: backtest MAE, persistence MAE, improvement, "
        "interval coverage, and the 2025 out-of-time check.", {"horizon_h": _HORIZON}, ["horizon_h"]),
]
