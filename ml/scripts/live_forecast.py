"""Issue live NO2 forecasts from AirNow's latest hours with the deployed model.

    AIRNOW_API_KEY=... python scripts/live_forecast.py --out live/forecasts.json

Runs hourly in GitHub Actions. It fetches the last 72 hours for the 20 backtest
stations, applies the data contract (auraguard.airnow.validate), forecasts 1, 6, 12 and
24 hours ahead with the model in models/no2_gbm.joblib, and writes one JSON file the
static site reads. The model itself is never retrained here.

Exit codes: 0 forecasts written; 2 no usable data (the previous file is left in place,
so the site keeps showing the last good forecast, with its age).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

from auraguard import airnow, serve
from auraguard.features import hourly_grid

ROOT = Path(__file__).resolve().parents[1]
HISTORY_H = 72


def load_stations() -> pd.DataFrame:
    return pd.DataFrame(json.loads((ROOT / "models" / "stations.json").read_text(encoding="utf-8"))["stations"])


def build(records: list[dict], bundle: dict, stations: pd.DataFrame, now: pd.Timestamp) -> dict | None:
    """The published document, or None when no station passed the data contract."""
    hourly = airnow.to_hourly(records, stations)
    clean, report = airnow.validate(hourly, list(stations["station"]), now)
    if clean.empty:
        return None
    grid = hourly_grid(clean)
    forecasts = serve.forecast(bundle, grid, stations)
    return {
        "status": "live",
        "source": "EPA AirNow, hourly NO2 (preliminary, not yet quality-assured by EPA)",
        "model": bundle.get("description", "Gradient boosting, one model per horizon"),
        "model_versions": bundle.get("versions", {}),
        "issued_at_utc": now.strftime("%Y-%m-%d %H:%M"),
        "data_as_of_utc": report.data_as_of_utc,
        "interval": "90% split-conformal",
        "horizons_h": sorted(int(h) for h in bundle["models"]),
        "units": "ppb",
        "data_contract": report.as_dict(),
        "stations": forecasts,
    }


def write_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "live" / "forecasts.json")
    args = ap.parse_args()
    key = os.environ.get("AIRNOW_API_KEY", "").strip()
    if not key:
        print("AIRNOW_API_KEY is not set", file=sys.stderr)
        return 2
    now = pd.Timestamp.now(tz="UTC").tz_convert(None).floor("h")
    bundle = joblib.load(ROOT / "models" / "no2_gbm.joblib")
    records = airnow.fetch(key, now - pd.Timedelta(hours=HISTORY_H), now)
    doc = build(records, bundle, load_stations(), now)
    if doc is None:
        print(f"no station passed the data contract ({len(records)} AirNow records); keeping the last file", file=sys.stderr)
        return 2
    write_atomic(args.out, doc)
    c = doc["data_contract"]
    print(f"wrote {args.out}: {len(doc['stations'])} stations, data as of {doc['data_as_of_utc']} UTC; "
          f"missing {len(c['stations_missing'])}, stale {len(c['stations_stale'])}, sparse {len(c['stations_sparse'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
