"""Fail when the published live forecast is older than a threshold.

    python scripts/check_forecast_age.py
    python scripts/check_forecast_age.py --max-age-hours 3 --url <forecasts.json URL>

Fetches the forecasts.json that the static site reads and compares its
`issued_at_utc` with the current time. Exit status 0 = fresh; 1 = stale, missing or
unreadable timestamp, or the fetch failed. The message says which. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

DEFAULT_URL = "https://raw.githubusercontent.com/CODER7657/AURA-GUARD/live-data/ml/live/forecasts.json"
DEFAULT_MAX_AGE = timedelta(hours=3)
ISSUED_FORMAT = "%Y-%m-%d %H:%M"  # what live_forecast.py writes, in UTC


class StaleForecast(Exception):
    """The forecast is too old, or its age cannot be established."""


def parse_issued_at(doc: object) -> datetime:
    """Return the document's `issued_at_utc` as an aware UTC datetime, or raise StaleForecast."""
    if not isinstance(doc, dict) or "issued_at_utc" not in doc:
        raise StaleForecast("forecasts.json has no issued_at_utc field")
    raw = doc["issued_at_utc"]
    if not isinstance(raw, str) or not raw.strip():
        raise StaleForecast(f"issued_at_utc is not a non-empty string: {raw!r}")
    try:
        return datetime.strptime(raw.strip(), ISSUED_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise StaleForecast(f"issued_at_utc is not in the form YYYY-MM-DD HH:MM: {raw!r}") from None


def check_age(doc: object, now: datetime, max_age: timedelta = DEFAULT_MAX_AGE) -> timedelta:
    """Return the forecast age; raise StaleForecast if it is older than `max_age`.

    Exactly `max_age` still passes; one minute more fails. A timestamp in the future is
    accepted (a few minutes of clock skew is normal) and reported as age zero.
    """
    issued = parse_issued_at(doc)
    age = max(now - issued, timedelta(0))
    if age > max_age:
        raise StaleForecast(f"forecast issued {issued:%Y-%m-%d %H:%M} UTC is {age} old; the limit is {max_age}")
    return age


def fetch(url: str, timeout: float = 20.0) -> object:
    """GET `url` and decode it as JSON; any network or decode failure raises StaleForecast."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise StaleForecast(f"could not fetch {url}: HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise StaleForecast(f"could not fetch {url}: {e}") from None
    except ValueError as e:  # includes UnicodeDecodeError and JSONDecodeError
        raise StaleForecast(f"{url} did not return valid JSON: {e}") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE.total_seconds() / 3600)
    args = p.parse_args(argv)
    limit = timedelta(hours=args.max_age_hours)
    try:
        age = check_age(fetch(args.url), datetime.now(timezone.utc), limit)
    except StaleForecast as e:
        print(f"STALE: {e}", file=sys.stderr)
        return 1
    print(f"fresh: forecast is {age} old (limit {limit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
