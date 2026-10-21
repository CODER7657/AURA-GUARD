"""Train the deployed NO2 forecaster and write forecasts from the latest available hour.

    python scripts/deploy_forecaster.py --train data/hourly_42602_2024.zip \
        --latest data/hourly_42602_2026.zip

Trains on the 20 backtest stations in the training file (about a minute on a laptop),
saves the model bundle, then forecasts 1, 6, 12 and 24 hours ahead of each station's
last reading in the --latest file. Writes:

    models/no2_gbm.joblib       the four models, interval half-widths and versions
    results/forecasts.json      what the backend and the agent serve
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from auraguard import epa, serve
from auraguard.features import hourly_grid

ROOT = Path(__file__).resolve().parents[1]


def backtest_stations() -> pd.DataFrame:
    summary = json.loads((ROOT / "results" / "backtest_summary.json").read_text(encoding="utf-8"))
    return pd.DataFrame(summary["stations"])


def load_grid(path: str, stations: pd.DataFrame) -> pd.DataFrame:
    data = epa.read_hourly(path)
    return hourly_grid(data[data["station"].isin(stations["station"])])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="EPA AirData hourly NO2 file to train on")
    p.add_argument("--latest", help="newer EPA file to forecast from (default: the training file)")
    p.add_argument("--model", default=str(ROOT / "models" / "no2_gbm.joblib"))
    p.add_argument("--out", default=str(ROOT / "results" / "forecasts.json"))
    p.add_argument("--reuse-model", action="store_true", help="load --model instead of training")
    args = p.parse_args()

    stations = backtest_stations()
    model_path = Path(args.model)
    if args.reuse_model:
        bundle = joblib.load(model_path)
    else:
        t = time.time()
        bundle = serve.train(load_grid(args.train, stations))
        bundle["trained_on"] = Path(args.train).name
        print(f"trained {len(bundle['models'])} models in {time.time() - t:.0f}s", flush=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path, compress=3)
        print(f"saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    latest = args.latest or args.train
    grid = load_grid(latest, stations)
    records = serve.forecast(bundle, grid, stations)
    missing = sorted(set(stations["station"]) - {r["station"] for r in records})
    out = {
        "status": "real model forecasts, issued from the last hour in the EPA file, not live",
        "source": Path(latest).name,
        "model": "Gradient boosting, one model per horizon, trained on " + bundle.get("trained_on", "?"),
        "trained_through": bundle["data_end"],
        "interval": "split-conformal, nominal 90%, one half-width per horizon",
        "interval_halfwidth_ppb": {str(h): round(q, 2) for h, q in bundle["halfwidth"].items()},
        "horizons_h": sorted(int(h) for h in bundle["models"]),
        "units": "NO2, parts per billion; local times are local standard time, as EPA reports them",
        "stations_without_forecast": missing,
        "stations": records,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(records)} stations, issued at "
          f"{sorted({r['issued_at_utc'] for r in records})} UTC; no forecast for {missing or 'none'}")


if __name__ == "__main__":
    main()
