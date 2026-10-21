"""Write the model-performance file the backend serves, from the backtest summary.

    python scripts/export_app_metrics.py

Reads results/backtest_summary.json (and results/monitor_2025.json, the out-of-time
check, when present) and writes results/model_performance.json. The backend reads
only this file, so nothing it reports can drift from what was measured.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEST = "gradient_boosting"


def main() -> None:
    summary = json.loads((ROOT / "results" / "backtest_summary.json").read_text(encoding="utf-8"))
    pooled = {(r["horizon"], r["model"]): r for r in summary["pooled"]}
    horizons = sorted({h for h, _ in pooled})
    ms = [f["ms"] for f in summary["folds"] if f["model"] == BEST]

    oot_path = ROOT / "results" / "monitor_2025.json"
    oot = json.loads(oot_path.read_text(encoding="utf-8")) if oot_path.exists() else None
    out = {
        "status": "backtested; real NO2 forecasts served from results/forecasts.json",
        "live_forecasts": "NO2: real model forecasts issued from the last hour in the newest EPA file, not live. "
                          "AQI / PM2.5 forecasts beyond 1 hour: simulated",
        "model": "Gradient boosting (scikit-learn HistGradientBoostingRegressor), one model per horizon",
        "target": "Hourly NO2 at the monitor, parts per billion",
        "data": f"{summary['source']}: {len(summary['stations'])} US monitors, 2024",
        "evaluation": "Monthly walk-forward backtest, July to December 2024, trained only on earlier data",
        "horizons": {
            str(h): {
                "mae_ppb": round(pooled[(h, BEST)]["mae"], 2),
                "rmse_ppb": round(pooled[(h, BEST)]["rmse"], 2),
                "r2": round(pooled[(h, BEST)]["r2"], 3),
                "skill_vs_persistence": round(pooled[(h, BEST)]["skill_vs_persistence"], 3),
                "persistence_mae_ppb": round(pooled[(h, "persistence")]["mae"], 2),
                "interval90_coverage": round(pooled[(h, BEST)]["coverage"], 3),
            }
            for h in horizons
        },
        "predict_ms_per_row": round(sum(ms) / len(ms), 3),
    }
    if oot:
        out["out_of_time"] = {
            "data": f"{oot['data']}, whole year; model trained on 2024 only",
            "horizons": {
                str(r["horizon_h"]): {
                    "mae_ppb": round(r["mae"], 2),
                    "persistence_mae_ppb": round(r["persistence_mae"], 2),
                    "skill_vs_persistence": round(r["skill_vs_persistence"], 3),
                    "interval90_coverage": round(r["coverage90"], 3),
                }
                for r in oot["horizons"]
            },
        }
    path = ROOT / "results" / "model_performance.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
