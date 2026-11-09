"""Score the live forecasts in ml/live/history against the readings that followed.

    python scripts/score_live.py --history ../ml/live/history --out ../ml/live/score.json

Run by the hourly live-forecast job after it publishes, on the live-data work tree, so
the site always shows yesterday's and the last week's skill. Method: auraguard.live_score.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from auraguard.live_score import parse_utc, read_history, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", type=Path, required=True, help="directory of <date>.jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--now", help="UTC 'YYYY-MM-DD HH:MM' (tests); default: now")
    args = ap.parse_args(argv)
    now = parse_utc(args.now) if args.now else datetime.now(timezone.utc)
    runs = read_history(args.history.glob("*.jsonl"))
    doc = report(runs, now)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    y = doc["windows"]["yesterday"]["horizons"]
    summary = ", ".join(
        f"{h['horizon_h']}h n={h['n']} MAE {h['mae_ppb']} vs persistence {h['persistence_mae_ppb']}" for h in y
    )
    print(f"scored {len(runs)} runs; yesterday: {summary or 'nothing to score yet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
