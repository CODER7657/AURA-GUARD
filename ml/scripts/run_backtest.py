"""Backtest every model at every horizon on an EPA hourly NO2 file.

Meant for Colab, not a laptop:

    python scripts/run_backtest.py --data hourly_42602_2024.zip --out results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from auraguard import backtest, epa
from auraguard.features import hourly_grid, make_supervised


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="EPA AirData hourly file (zip or csv)")
    p.add_argument("--out", default="results")
    p.add_argument("--stations", type=int, default=20)
    p.add_argument("--horizons", default="1,6,12,24")
    p.add_argument("--test-months", default="2024-07,2024-08,2024-09,2024-10,2024-11,2024-12")
    p.add_argument("--models", default="persistence,same_hour_yesterday,climatology,linear,gradient_boosting")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    data = epa.most_complete(epa.read_hourly(args.data), n=args.stations)
    print(f"{data['station'].nunique()} stations, {len(data):,} hourly readings ({time.time() - t:.0f}s)", flush=True)
    epa.stations(data).to_csv(out / "stations.csv", index=False)

    grid = hourly_grid(data)
    folds = backtest.monthly_folds(args.test_months.split(","))
    models = args.models.split(",")
    rows: list[dict] = []
    for h in [int(x) for x in args.horizons.split(",")]:
        t = time.time()
        table = make_supervised(grid, h)
        rows += backtest.run(table, h, folds, models)
        (out / "backtest.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        for r in rows:
            if r["horizon"] == h and r["fold"] == "all":
                print(
                    f"h={h:>2}  {r['model']:20s} MAE {r['mae']:.2f}  RMSE {r['rmse']:.2f}  R2 {r['r2']:.3f}"
                    f"  skill {r.get('skill_vs_persistence', 0):+.3f}  90% cover {r['coverage']:.3f}",
                    flush=True,
                )
        print(f"h={h} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
