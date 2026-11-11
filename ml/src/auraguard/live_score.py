"""Score the published live forecasts against the readings that arrived afterwards.

Every hourly run appends one line per run to ``ml/live/history/<date>.jsonl`` on the
``live-data`` branch. For each station a line records the time of its latest reading
(``issued_at_utc``), that reading (``last_reading_ppb``) and the forecasts made from it
(``target_time_utc``, ``no2_ppb``, 90% interval). That is enough to score without any
other data source:

* observed NO2 at hour T = ``last_reading_ppb`` of any line whose station was issued at T;
* the model's error = |forecast for T - observed at T|;
* persistence ("the next reading equals this one") = the issuing line's own reading.

A forecast whose target hour was never observed (a gap in AirNow, a station that failed
the data contract) is not scored; the count says how many were.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

FMT = "%Y-%m-%d %H:%M"


def parse_utc(text: str) -> datetime:
    """The history's "YYYY-MM-DD HH:MM" times are UTC."""
    return datetime.strptime(text, FMT).replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class HorizonScore:
    horizon_h: int
    n: int  # forecasts scored (their target hour was observed)
    unscored: int  # forecasts whose target hour has no reading
    mae_ppb: float | None
    persistence_mae_ppb: float | None
    skill_vs_persistence: float | None  # 1 - MAE / persistence MAE; > 0 beats persistence
    interval90_coverage: float | None  # share of readings inside the 90% interval


def read_history(files: Iterable[Path]) -> list[dict]:
    """All run lines from history files, oldest file first. Bad lines are skipped."""
    runs: list[dict] = []
    for path in sorted(files):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(run, dict):
                runs.append(run)
    return runs


def _stations(run: dict) -> Iterable[dict]:
    """Station entries of a run that can be scored. The history is append-only, so one
    malformed entry must be skipped, not raised on: it would fail every later run."""
    stations = run.get("stations")
    for s in stations if isinstance(stations, list) else []:
        if isinstance(s, dict) and isinstance(s.get("station"), str) and isinstance(s.get("issued_at_utc"), str):
            yield s


def _forecast(f: object) -> tuple[int, str, datetime, float, float, float] | None:
    """(horizon, target text, target time, forecast, low, high), or None if any part is unusable."""
    try:
        h, text = f["horizon_h"], f["target_time_utc"]  # type: ignore[index]
        values = (f["no2_ppb"], f["interval90_low_ppb"], f["interval90_high_ppb"])  # type: ignore[index]
        if not isinstance(h, int) or not all(isinstance(v, (int, float)) for v in values):
            return None
        return h, text, parse_utc(text), float(values[0]), float(values[1]), float(values[2])
    except (KeyError, TypeError, ValueError):
        return None


def observations(runs: Iterable[dict]) -> dict[tuple[str, str], float]:
    """(station, hour UTC) -> NO2 ppb, from the reading each run was issued from."""
    obs: dict[tuple[str, str], float] = {}
    for run in runs:
        for s in _stations(run):
            value = s.get("last_reading_ppb")
            if isinstance(value, (int, float)):
                obs[(s["station"], s["issued_at_utc"])] = float(value)
    return obs


def score(runs: list[dict], start: datetime, end: datetime) -> list[HorizonScore]:
    """Scores per horizon for forecasts whose target hour is in [start, end)."""
    obs = observations(runs)
    # A forecast can be published by several hourly runs from the same reading; count once.
    seen: set[tuple[str, str, int]] = set()
    rows: dict[int, list[tuple[float, float, float, float, float] | None]] = {}
    for run in runs:
        for s in _stations(run):
            station, persisted = s["station"], s.get("last_reading_ppb")
            forecasts = s.get("forecasts")
            for f in forecasts if isinstance(forecasts, list) else []:
                parsed = _forecast(f)
                if parsed is None:
                    continue
                h, target_text, target, value, low, high = parsed
                key = (station, s["issued_at_utc"], h)
                if not (start <= target < end) or key in seen:
                    continue
                seen.add(key)
                actual = obs.get((station, target_text))
                if actual is None or not isinstance(persisted, (int, float)):
                    rows.setdefault(h, []).append(None)
                    continue
                rows.setdefault(h, []).append((value, float(persisted), actual, low, high))
    out = []
    for h in sorted(rows):
        scored = [r for r in rows[h] if r is not None]
        n = len(scored)
        mae = sum(abs(p - a) for p, _, a, _, _ in scored) / n if n else None
        pmae = sum(abs(q - a) for _, q, a, _, _ in scored) / n if n else None
        cover = sum(lo <= a <= hi for _, _, a, lo, hi in scored) / n if n else None
        skill = 1 - mae / pmae if mae is not None and pmae else None
        out.append(
            HorizonScore(
                horizon_h=h,
                n=n,
                unscored=len(rows[h]) - n,
                mae_ppb=_r(mae),
                persistence_mae_ppb=_r(pmae),
                skill_vs_persistence=_r(skill, 3),
                interval90_coverage=_r(cover, 3),
            )
        )
    return out


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


def report(runs: list[dict], now: datetime) -> dict:
    """Yesterday (the last complete UTC day) and the last 7 complete days."""
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    windows = {"yesterday": (today - timedelta(days=1), today), "last_7_days": (today - timedelta(days=7), today)}
    return {
        "scored_at_utc": now.strftime(FMT),
        "method": "Each forecast is compared with the reading AirNow later reported for its target hour; "
        "persistence predicts the reading the forecast was issued from.",
        "windows": {
            name: {
                "from_utc": a.strftime(FMT),
                "to_utc": b.strftime(FMT),
                "horizons": [asdict(h) for h in score(runs, a, b)],
            }
            for name, (a, b) in windows.items()
        },
    }
