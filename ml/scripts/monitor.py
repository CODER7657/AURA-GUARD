"""Score the deployed forecaster on newer EPA data and fail loudly if it has degraded.

    python scripts/monitor.py --data data/hourly_42602_2026.zip --days 28
    python scripts/monitor.py --data data/hourly_42602_2025.zip --days 0 --out results/monitor_2025.json

Loads models/no2_gbm.joblib (trained on 2024 only), scores it on the given file and
compares it with persistence on exactly the same hours. Exits with status 1 if any
check fails. The thresholds were fixed before the first run and are not tuned:

    skill vs persistence  >= 0.00 at every horizon (the model must still beat "next = now")
    90% interval coverage >= 0.80 at every horizon
    MAE                   <= 2.0 x the backtest MAE at that horizon

Input drift (a station's mean NO2 in the window against its 2024 mean) is reported
as a warning only, since a seasonal window is expected to differ from a yearly mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from auraguard import epa, serve
from auraguard.features import hourly_grid

ROOT = Path(__file__).resolve().parents[1]
MIN_SKILL = 0.0
MIN_COVERAGE = 0.80
MAX_MAE_RATIO = 2.0
DRIFT_RATIO = (0.5, 2.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="EPA AirData hourly NO2 file the model was not trained on")
    p.add_argument("--days", type=int, default=28, help="score each station's last N days; 0 = the whole file")
    p.add_argument("--model", default=str(ROOT / "models" / "no2_gbm.joblib"))
    p.add_argument("--out", default=str(ROOT / "results" / "monitor_latest.json"))
    args = p.parse_args()

    bundle = joblib.load(args.model)
    if bundle["versions"]["scikit-learn"] != sklearn.__version__:
        print(f"WARNING: model saved with scikit-learn {bundle['versions']['scikit-learn']}, "
              f"running {sklearn.__version__}; install ml/requirements-monitor.txt", file=sys.stderr)
    summary = json.loads((ROOT / "results" / "backtest_summary.json").read_text(encoding="utf-8"))
    backtest_mae = {r["horizon"]: r["mae"] for r in summary["pooled"] if r["model"] == "gradient_boosting"}
    train_mean = {s["station"]: s["mean_ppb"] for s in summary["stations"]}

    data = epa.read_hourly(args.data)
    data = data[data["station"].isin(bundle["codes"])]
    grid = hourly_grid(data)
    rows = serve.evaluate(bundle, grid, last_days=args.days or None)

    failures = []
    for r in rows:
        h = r["horizon_h"]
        r["backtest_mae"] = round(backtest_mae[h], 4)
        r["mae_ratio_to_backtest"] = r["mae"] / backtest_mae[h]
        if r["skill_vs_persistence"] < MIN_SKILL:
            failures.append(f"h={h}: skill vs persistence {r['skill_vs_persistence']:+.3f} < {MIN_SKILL}")
        if r["coverage90"] < MIN_COVERAGE:
            failures.append(f"h={h}: 90% interval coverage {r['coverage90']:.3f} < {MIN_COVERAGE}")
        if r["mae_ratio_to_backtest"] > MAX_MAE_RATIO:
            failures.append(f"h={h}: MAE {r['mae']:.2f} is {r['mae_ratio_to_backtest']:.2f}x the backtest")
    if not rows:
        failures.append("no scoreable hours for the deployed stations in this file")

    # Input drift: the window each station was scored on, against its 2024 mean.
    if args.days:
        end = grid.groupby("station")["time_gmt"].transform("max")
        window = grid[grid["time_gmt"] > end - pd.Timedelta(days=args.days)]
    else:
        window = grid
    drift, warnings = {}, []
    for station, mean in window.groupby("station")["value"].mean().items():
        ratio = mean / train_mean[station] if train_mean.get(station) else None
        drift[station] = {"window_mean_ppb": round(float(mean), 2), "train_mean_ppb": train_mean.get(station),
                          "ratio": round(ratio, 2) if ratio else None}
        if ratio and not DRIFT_RATIO[0] <= ratio <= DRIFT_RATIO[1]:
            warnings.append(f"{station}: mean NO2 {mean:.1f} ppb vs {train_mean[station]} ppb in 2024")
    missing = sorted(set(bundle["codes"]) - set(grid["station"].unique()))
    if missing:
        warnings.append(f"no data for {len(missing)} deployed stations: {', '.join(missing)}")

    report = {
        "data": Path(args.data).name,
        "window": f"last {args.days} days of each station" if args.days else "whole file",
        "data_start_utc": str(grid["time_gmt"].min()),
        "data_end_utc": str(grid["time_gmt"].max()),
        "model_trained_on": bundle.get("trained_on"),
        "thresholds": {"min_skill_vs_persistence": MIN_SKILL, "min_coverage90": MIN_COVERAGE,
                       "max_mae_ratio_to_backtest": MAX_MAE_RATIO},
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "horizons": [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows],
        "input_drift": drift,
        "versions": {"scikit-learn": sklearn.__version__, "model_saved_with": bundle["versions"]},
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"{report['data']}, {report['window']}, {report['data_start_utc']} to {report['data_end_utc']} UTC")
    print("horizon  MAE   persistence  skill   coverage90  MAE/backtest")
    for r in rows:
        print(f"{r['horizon_h']:>4} h  {r['mae']:5.2f}  {r['persistence_mae']:9.2f}  {r['skill_vs_persistence']:+6.3f}"
              f"  {r['coverage90']:9.3f}  {r['mae_ratio_to_backtest']:9.2f}")
    for w in warnings:
        print("WARNING:", w)
    if failures:
        print("\nMONITOR FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("\nmonitor passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
