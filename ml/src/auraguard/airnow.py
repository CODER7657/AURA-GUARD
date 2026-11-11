"""Live hourly NO2 from EPA's AirNow API, shaped like :func:`auraguard.epa.read_hourly`.

AirNow reports each monitor by its AQS site code (``840`` + state + county + site), in
ppb, on UTC hours. The deployed model was trained on EPA AirData, where a station is
``state-county-site-POC``; a live reading is matched to the training station at the same
site. Nothing here is guessed: a site AirNow doesn't report simply has no rows.

Before any reading reaches the model it passes :func:`validate`, the data contract:
plausible values only, enough recent hours per station, and data that is actually fresh.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

API = "https://www.airnowapi.org/aq/data/"
US_BBOX = "-125.0,24.0,-66.0,50.0"
MIN_PPB, MAX_PPB = -5.0, 500.0  # AirNow can report small negative values near zero
MIN_HOURS_IN_LAST_24 = 12
MAX_STALENESS_H = 6


def site_key_of_aqs(code: str) -> str | None:
    """'840040133002' (or '040133002') -> '04-013-3002'; None for anything else."""
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) == 12 and digits.startswith("840"):
        digits = digits[3:]
    if len(digits) != 9:
        return None
    return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


RETRIES = 3
RETRY_WAIT_S = 5.0


def fetch_window(api_key: str, start: pd.Timestamp, end: pd.Timestamp, bbox: str = US_BBOX, timeout: float = 60, sleep=time.sleep) -> list[dict]:
    """Hourly NO2 concentrations from ``start`` to ``end`` (UTC hours, inclusive). The key never appears in errors."""
    params = {
        "startDate": start.strftime("%Y-%m-%dT%H"),
        "endDate": end.strftime("%Y-%m-%dT%H"),
        "parameters": "NO2",
        "BBOX": bbox,
        "dataType": "C",
        "format": "application/json",
        "verbose": "1",
        "monitorType": "0",
        "includerawconcentrations": "0",
        "API_KEY": api_key,
    }
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "aura-guard/1.0"})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https host
                body = r.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
            raise RuntimeError(_redact(f"AirNow HTTP {e.code} for {params['startDate']}..{params['endDate']}: {detail}", api_key)) from None
        except (TimeoutError, urllib.error.URLError) as e:
            # A slow or dropped connection is the usual reason an hourly run fails; try again.
            if attempt == RETRIES:
                reason = _redact(str(getattr(e, "reason", e)), api_key)
                raise RuntimeError(f"AirNow did not answer after {RETRIES} attempts: {reason}") from None
            sleep(RETRY_WAIT_S * attempt)
    data = json.loads(body)
    if not isinstance(data, list):
        raise ValueError(_redact(f"unexpected AirNow response: {str(data)[:200]}", api_key))
    return data


def fetch(api_key: str, start: pd.Timestamp, end: pd.Timestamp, window_h: int = 24, **kw) -> list[dict]:
    """Like fetch_window, in windows of ``window_h`` hours, so no single query is large."""
    out: list[dict] = []
    t = start
    while t <= end:
        stop = min(t + pd.Timedelta(hours=window_h - 1), end)
        out.extend(fetch_window(api_key, t, stop, **kw))
        t = stop + pd.Timedelta(hours=1)
    return out


def to_hourly(records: list[dict], stations: pd.DataFrame) -> pd.DataFrame:
    """AirNow records -> rows like epa.read_hourly, for the training stations only.

    ``stations`` needs columns station, utc_offset_h, lat, lon, place.
    """
    cols = ["station", "time_gmt", "utc_offset_h", "value", "lat", "lon", "place"]
    meta = stations.assign(site_key=stations["station"].str.rsplit("-", n=1).str[0]).set_index("site_key")
    rows = []
    for rec in records:
        if str(rec.get("Parameter", "")).upper() != "NO2":
            continue
        key = site_key_of_aqs(rec.get("FullAQSCode") or rec.get("IntlAQSCode") or "")
        if key is None or key not in meta.index:
            continue
        try:
            value = float(rec["Value"])
            time_gmt = pd.Timestamp(rec["UTC"])
            if time_gmt.tzinfo is not None:
                time_gmt = time_gmt.tz_convert(None)
        except (KeyError, TypeError, ValueError):
            continue
        m = meta.loc[key]
        rows.append((m["station"], time_gmt.floor("h"), int(m["utc_offset_h"]), value, m["lat"], m["lon"], m["place"]))
    out = pd.DataFrame(rows, columns=cols)
    return out.drop_duplicates(["station", "time_gmt"], keep="last").sort_values(["station", "time_gmt"]).reset_index(drop=True)


@dataclass
class Report:
    """What the data contract kept, dropped and why. Published with every forecast."""

    now_utc: str
    rows_in: int = 0
    rows_out_of_range: int = 0
    stations_ok: list[str] = field(default_factory=list)
    stations_sparse: list[str] = field(default_factory=list)
    stations_stale: list[str] = field(default_factory=list)
    stations_missing: list[str] = field(default_factory=list)
    data_as_of_utc: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def validate(df: pd.DataFrame, expected: list[str], now: pd.Timestamp) -> tuple[pd.DataFrame, Report]:
    """Keep only plausible readings from stations with enough fresh hours."""
    rep = Report(now_utc=now.strftime("%Y-%m-%d %H:%M"), rows_in=len(df))
    ok_range = df["value"].between(MIN_PPB, MAX_PPB)
    rep.rows_out_of_range = int((~ok_range).sum())
    df = df[ok_range].copy()
    df.loc[df["value"] < 0, "value"] = 0.0
    keep = []
    for st in expected:
        g = df[df["station"] == st]
        if g.empty:
            rep.stations_missing.append(st)
            continue
        last = g["time_gmt"].max()
        if (now - last) > pd.Timedelta(hours=MAX_STALENESS_H):
            rep.stations_stale.append(st)
            continue
        recent = g[g["time_gmt"] > last - pd.Timedelta(hours=24)]
        if len(recent) < MIN_HOURS_IN_LAST_24:
            rep.stations_sparse.append(st)
            continue
        keep.append(st)
    rep.stations_ok = keep
    out = df[df["station"].isin(keep)]
    if not out.empty:
        rep.data_as_of_utc = out["time_gmt"].max().strftime("%Y-%m-%d %H:%M")
    return out.reset_index(drop=True), rep
