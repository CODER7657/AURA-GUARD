"""Write the model-performance file the backend serves, from the backtest summary.

    python scripts/export_app_metrics.py

Reads results/backtest_summary.json and writes results/model_performance.json. The
backend reads only this file, so nothing it reports can drift from what was measured.
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

    out = {
        "status": "backtested; not yet serving live forecasts",
        "live_forecasts": "simulated until a trained model is deployed (see ml/README.md)",
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
    path = ROOT / "results" / "model_performance.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
