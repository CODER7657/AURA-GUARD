"""The stale-forecast alarm: age logic and failure messages. No network: urlopen is faked."""

import importlib.util
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ML = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_forecast_age", ML / "scripts" / "check_forecast_age.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
LIMIT = timedelta(hours=3)


def doc(issued: datetime) -> dict:
    return {"issued_at_utc": issued.strftime("%Y-%m-%d %H:%M")}


def test_fresh_document_passes():
    assert mod.check_age(doc(NOW - timedelta(minutes=20)), NOW) == timedelta(minutes=20)


def test_exactly_three_hours_passes():
    assert mod.check_age(doc(NOW - LIMIT), NOW) == LIMIT


def test_three_hours_and_one_minute_fails():
    with pytest.raises(mod.StaleForecast, match="limit is 3:00:00"):
        mod.check_age(doc(NOW - LIMIT - timedelta(minutes=1)), NOW)


def test_future_timestamp_counts_as_zero_age():
    assert mod.check_age(doc(NOW + timedelta(minutes=5)), NOW) == timedelta(0)


@pytest.mark.parametrize(
    "bad", [{}, [], None, "x", {"issued_at_utc": None}, {"issued_at_utc": ""}, {"issued_at_utc": 123}]
)
def test_missing_or_non_string_timestamp_fails_clearly(bad):
    with pytest.raises(mod.StaleForecast, match="issued_at_utc"):
        mod.check_age(bad, NOW)


@pytest.mark.parametrize("raw", ["yesterday", "2026-13-40 99:99", "2026-11-07", "2026-11-07T11:00:00Z"])
def test_garbled_timestamp_fails_clearly(raw):
    with pytest.raises(mod.StaleForecast, match="YYYY-MM-DD HH:MM"):
        mod.check_age({"issued_at_utc": raw}, NOW)


def test_custom_limit():
    with pytest.raises(mod.StaleForecast):
        mod.check_age(doc(NOW - timedelta(hours=2)), NOW, timedelta(hours=1))


class Resp:
    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def raiser(exc):
    def boom(url, timeout):
        raise exc

    return boom


def test_fetch_decodes_json(monkeypatch):
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda url, timeout: Resp(b'{"a": 1}'))
    assert mod.fetch("https://example.invalid/x") == {"a": 1}


def test_fetch_network_error_is_a_clear_failure(monkeypatch):
    monkeypatch.setattr(mod.urllib.request, "urlopen", raiser(urllib.error.URLError("name resolution failed")))
    with pytest.raises(mod.StaleForecast, match="could not fetch .*name resolution failed"):
        mod.fetch("https://example.invalid/x")


def test_fetch_http_error_is_a_clear_failure(monkeypatch):
    err = urllib.error.HTTPError("https://example.invalid/x", 404, "Not Found", None, None)
    monkeypatch.setattr(mod.urllib.request, "urlopen", raiser(err))
    with pytest.raises(mod.StaleForecast, match="HTTP 404"):
        mod.fetch("https://example.invalid/x")


def test_fetch_timeout_is_a_clear_failure(monkeypatch):
    monkeypatch.setattr(mod.urllib.request, "urlopen", raiser(TimeoutError("timed out")))
    with pytest.raises(mod.StaleForecast, match="could not fetch"):
        mod.fetch("https://example.invalid/x")


def test_fetch_non_json_is_a_clear_failure(monkeypatch):
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda url, timeout: Resp(b"<html>"))
    with pytest.raises(mod.StaleForecast, match="valid JSON"):
        mod.fetch("https://example.invalid/x")


def test_main_exit_codes(monkeypatch, capsys):
    fresh = doc(datetime.now(timezone.utc) - timedelta(minutes=5))
    monkeypatch.setattr(mod, "fetch", lambda url: fresh)
    assert mod.main([]) == 0
    assert "fresh" in capsys.readouterr().out

    monkeypatch.setattr(mod, "fetch", lambda url: {"issued_at_utc": "2020-01-01 00:00"})
    assert mod.main([]) == 1
    assert "STALE" in capsys.readouterr().err

    def down(url):
        raise mod.StaleForecast("could not fetch x: down")

    monkeypatch.setattr(mod, "fetch", down)
    assert mod.main([]) == 1
    assert "could not fetch" in capsys.readouterr().err
