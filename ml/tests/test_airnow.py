"""AirNow ingest: code mapping, the data contract, and an end-to-end build with the real model.

No network: records are built in the shape AirNow's /aq/data endpoint returns.
"""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from auraguard import airnow

ML = Path(__file__).resolve().parents[1]
STATIONS = pd.DataFrame(json.loads((ML / "models" / "stations.json").read_text(encoding="utf-8"))["stations"])
NOW = pd.Timestamp("2026-11-06 18:00")


def records(station: str, hours: int, end: pd.Timestamp = NOW, value=12.0, every: int = 1) -> list[dict]:
    state, county, site, _poc = station.split("-")
    code = f"840{state}{county}{site}"
    out = []
    for h in range(0, hours, every):
        t = end - pd.Timedelta(hours=h)
        v = value(h) if callable(value) else value
        out.append({"UTC": t.strftime("%Y-%m-%dT%H:%M"), "Parameter": "NO2", "Unit": "PPB", "Value": v,
                    "FullAQSCode": code, "IntlAQSCode": code, "SiteName": "x", "AgencyName": "y"})
    return out


# ---------- site codes ----------

@pytest.mark.parametrize("code, key", [
    ("840040133002", "04-013-3002"),
    ("040133002", "04-013-3002"),
    ("840-04-013-3002", "04-013-3002"),
    ("124000000001", None),   # Canada
    ("", None),
    ("12345", None),
])
def test_site_key_of_aqs(code, key):
    assert airnow.site_key_of_aqs(code) == key


# ---------- to_hourly ----------

def test_only_training_stations_and_no2_are_kept():
    st = STATIONS["station"].iloc[0]
    recs = records(st, 3) + [{"UTC": "2026-11-06T18:00", "Parameter": "OZONE", "Value": 40, "FullAQSCode": "840040133002"},
                             {"UTC": "2026-11-06T18:00", "Parameter": "NO2", "Value": 9, "FullAQSCode": "840999999999"}]
    df = airnow.to_hourly(recs, STATIONS)
    assert set(df["station"]) == {st}
    assert len(df) == 3
    assert list(df.columns) == ["station", "time_gmt", "utc_offset_h", "value", "lat", "lon", "place"]


def test_bad_records_are_skipped_not_fatal():
    st = STATIONS["station"].iloc[0]
    recs = records(st, 2) + [{"Parameter": "NO2", "FullAQSCode": "840040133002"},
                             {"UTC": "not a time", "Parameter": "NO2", "Value": 3, "FullAQSCode": "840040133002"},
                             {"UTC": "2026-11-06T10:00", "Parameter": "NO2", "Value": "n/a", "FullAQSCode": "840040133002"}]
    assert len(airnow.to_hourly(recs, STATIONS)) == 2


def test_duplicate_hours_keep_one_reading():
    st = STATIONS["station"].iloc[0]
    df = airnow.to_hourly(records(st, 2) + records(st, 2, value=30.0), STATIONS)
    assert len(df) == 2 and set(df["value"]) == {30.0}


def test_empty_input_gives_an_empty_frame():
    assert airnow.to_hourly([], STATIONS).empty


# ---------- data contract ----------

def test_contract_sorts_stations_into_ok_missing_stale_sparse():
    a, b, c = STATIONS["station"].iloc[:3]
    recs = records(a, 30) + records(b, 30, end=NOW - pd.Timedelta(hours=10)) + records(c, 30, every=4)
    clean, rep = airnow.validate(airnow.to_hourly(recs, STATIONS), list(STATIONS["station"]), NOW)
    assert rep.stations_ok == [a]
    assert b in rep.stations_stale
    assert c in rep.stations_sparse
    assert len(rep.stations_missing) == len(STATIONS) - 3
    assert set(clean["station"]) == {a}
    assert rep.data_as_of_utc == "2026-11-06 18:00"


def test_implausible_values_are_dropped_and_small_negatives_floored():
    st = STATIONS["station"].iloc[0]
    vals = {0: 900.0, 1: -50.0, 2: -2.0}
    recs = records(st, 30, value=lambda h: vals.get(h, 10.0))
    clean, rep = airnow.validate(airnow.to_hourly(recs, STATIONS), [st], NOW)
    assert rep.rows_out_of_range == 2
    assert clean["value"].min() == 0.0 and clean["value"].max() <= airnow.MAX_PPB


# ---------- end to end with the deployed model ----------

def load_script():
    spec = importlib.util.spec_from_file_location("live_forecast", ML / "scripts" / "live_forecast.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bundle():
    path = ML / "models" / "no2_gbm.joblib"
    if not path.exists():
        pytest.skip("model bundle not present")
    try:
        return joblib.load(path)
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        # The bundle is a pickle; it loads only with the scikit-learn it was saved with
        # (requirements-monitor.txt, used by the live-forecast workflow, which runs this path for real).
        pytest.skip(f"model bundle needs the pinned scikit-learn: {e}")


def test_build_forecasts_every_healthy_station(bundle):
    live = load_script()
    rng = np.random.default_rng(0)
    recs = []
    for st in STATIONS["station"]:
        recs += records(st, 72, value=lambda h: float(10 + 8 * np.sin(h / 4) + rng.normal()))
    doc = live.build(recs, bundle, STATIONS, NOW)
    assert doc["status"] == "live" and doc["data_as_of_utc"] == "2026-11-06 18:00"
    assert len(doc["stations"]) == len(STATIONS)
    first = doc["stations"][0]["forecasts"]
    assert [f["horizon_h"] for f in first] == doc["horizons_h"]
    for f in first:
        assert f["interval90_low_ppb"] <= f["no2_ppb"] <= f["interval90_high_ppb"]


def test_build_returns_none_when_nothing_passes(bundle):
    assert load_script().build([], bundle, STATIONS, NOW) is None


def test_write_is_atomic(tmp_path):
    live = load_script()
    out = tmp_path / "live" / "forecasts.json"
    live.write_atomic(out, {"a": 1})
    assert json.loads(out.read_text()) == {"a": 1}
    assert not list(out.parent.glob("*.tmp"))


# ---------- fetching ----------

def test_fetch_splits_into_windows_without_gaps(monkeypatch):
    calls = []
    monkeypatch.setattr(airnow, "fetch_window", lambda key, a, b, **kw: calls.append((a, b)) or [{"a": 1}])
    out = airnow.fetch("k", NOW - pd.Timedelta(hours=72), NOW)
    assert len(out) == len(calls) == 4
    assert calls[0][0] == NOW - pd.Timedelta(hours=72) and calls[-1][1] == NOW
    for (a1, b1), (a2, _) in zip(calls, calls[1:]):
        assert a2 == b1 + pd.Timedelta(hours=1)


def test_http_errors_carry_the_reason_but_never_the_key(monkeypatch):
    import io as _io
    import urllib.error

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, _io.BytesIO(b"Invalid API key SECRET123"))

    monkeypatch.setattr(airnow.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as e:
        airnow.fetch_window("SECRET123", NOW - pd.Timedelta(hours=1), NOW)
    assert "HTTP 400" in str(e.value) and "Invalid API key" in str(e.value)
    assert "SECRET123" not in str(e.value)


class _Reply:
    def __init__(self, body: str):
        self.body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def test_a_timeout_is_retried_and_then_succeeds(monkeypatch):
    calls, waits = [], []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("The read operation timed out")
        return _Reply("[]")

    monkeypatch.setattr(airnow.urllib.request, "urlopen", flaky)
    assert airnow.fetch_window("k", NOW - pd.Timedelta(hours=1), NOW, sleep=waits.append) == []
    assert len(calls) == 3 and waits == [airnow.RETRY_WAIT_S, 2 * airnow.RETRY_WAIT_S]


def test_retries_are_bounded_and_the_error_never_carries_the_key(monkeypatch):
    import urllib.error

    calls = []

    def down(req, timeout):
        calls.append(1)
        raise urllib.error.URLError(f"no route to {req.full_url}")

    monkeypatch.setattr(airnow.urllib.request, "urlopen", down)
    with pytest.raises(RuntimeError) as e:
        airnow.fetch_window("SECRET123", NOW - pd.Timedelta(hours=1), NOW, sleep=lambda s: None)
    assert len(calls) == airnow.RETRIES
    assert "did not answer" in str(e.value) and "SECRET123" not in str(e.value)


def test_an_http_error_is_not_retried(monkeypatch):
    import io as _io
    import urllib.error

    calls = []

    def boom(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, _io.BytesIO(b"no"))

    monkeypatch.setattr(airnow.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError):
        airnow.fetch_window("k", NOW - pd.Timedelta(hours=1), NOW, sleep=lambda s: pytest.fail("slept"))
    assert len(calls) == 1
