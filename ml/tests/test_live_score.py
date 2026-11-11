"""Live scoring from the published history alone (auraguard.live_score)."""

import json
from datetime import datetime, timezone

import pytest
from auraguard import live_score as ls


def run(issued: str, reading: float, forecasts: list[tuple[int, str, float, float, float]], station="S1"):
    return {
        "issued_at_utc": issued,
        "data_as_of_utc": issued,
        "stations": [
            {
                "station": station,
                "issued_at_utc": issued,
                "last_reading_ppb": reading,
                "forecasts": [
                    {"horizon_h": h, "target_time_utc": t, "no2_ppb": p, "interval90_low_ppb": lo, "interval90_high_ppb": hi}
                    for h, t, p, lo, hi in forecasts
                ],
            }
        ],
    }


UTC = timezone.utc
D0, D1 = datetime(2026, 11, 8, tzinfo=UTC), datetime(2026, 11, 9, tzinfo=UTC)


def test_model_and_persistence_errors_against_the_later_reading():
    runs = [
        run("2026-11-08 10:00", 10.0, [(1, "2026-11-08 11:00", 12.0, 8.0, 16.0)]),
        run("2026-11-08 11:00", 13.0, []),  # the reading for 11:00 arrives with the next run
    ]
    [h1] = ls.score(runs, D0, D1)
    assert (h1.n, h1.unscored) == (1, 0)
    assert h1.mae_ppb == 1.0  # |12 - 13|
    assert h1.persistence_mae_ppb == 3.0  # |10 - 13|
    assert h1.skill_vs_persistence == pytest.approx(1 - 1 / 3, abs=1e-3)
    assert h1.interval90_coverage == 1.0


def test_a_target_hour_with_no_reading_is_counted_but_not_scored():
    runs = [run("2026-11-08 10:00", 10.0, [(1, "2026-11-08 11:00", 12.0, 8.0, 16.0)])]
    [h1] = ls.score(runs, D0, D1)
    assert (h1.n, h1.unscored, h1.mae_ppb, h1.skill_vs_persistence) == (0, 1, None, None)


def test_a_forecast_republished_from_the_same_reading_counts_once():
    f = [(1, "2026-11-08 11:00", 12.0, 8.0, 16.0)]
    runs = [run("2026-11-08 10:00", 10.0, f), run("2026-11-08 10:00", 10.0, f), run("2026-11-08 11:00", 13.0, [])]
    [h1] = ls.score(runs, D0, D1)
    assert h1.n == 1


def test_only_targets_inside_the_window_are_scored_and_stations_do_not_mix():
    runs = [
        run("2026-11-08 23:00", 10.0, [(1, "2026-11-09 00:00", 11.0, 5.0, 15.0)]),  # target is next day
        run("2026-11-09 00:00", 11.0, []),
        run("2026-11-08 10:00", 5.0, [(1, "2026-11-08 11:00", 5.0, 0.0, 9.0)], station="S2"),
        run("2026-11-08 11:00", 30.0, [], station="S1"),  # S1's reading must not score S2
    ]
    [h1] = ls.score(runs, D0, D1)
    assert (h1.n, h1.unscored) == (0, 1)


def test_interval_coverage_and_a_miss():
    runs = [
        run("2026-11-08 10:00", 10.0, [(6, "2026-11-08 16:00", 12.0, 9.0, 15.0)]),
        run("2026-11-08 16:00", 20.0, []),
    ]
    [h6] = ls.score(runs, D0, D1)
    assert h6.horizon_h == 6 and h6.interval90_coverage == 0.0 and h6.mae_ppb == 8.0


def test_report_windows_and_bad_lines(tmp_path):
    good = run("2026-11-08 10:00", 10.0, [(1, "2026-11-08 11:00", 12.0, 8.0, 16.0)])
    (tmp_path / "2026-11-08.jsonl").write_text(
        json.dumps(good) + "\n{not json\n\n" + json.dumps(run("2026-11-08 11:00", 13.0, [])) + "\n", encoding="utf-8"
    )
    runs = ls.read_history(tmp_path.glob("*.jsonl"))
    assert len(runs) == 2
    doc = ls.report(runs, datetime(2026, 11, 9, 5, 30, tzinfo=UTC))
    assert doc["windows"]["yesterday"]["from_utc"] == "2026-11-08 00:00"
    assert doc["windows"]["yesterday"]["horizons"][0]["n"] == 1
    assert doc["windows"]["last_7_days"]["from_utc"] == "2026-11-02 00:00"


def test_script_writes_the_report(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("score_live", Path(__file__).parents[1] / "scripts" / "score_live.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hist = tmp_path / "history"
    hist.mkdir()
    (hist / "2026-11-08.jsonl").write_text(
        json.dumps(run("2026-11-08 10:00", 10.0, [(1, "2026-11-08 11:00", 12.0, 8.0, 16.0)])) + "\n"
        + json.dumps(run("2026-11-08 11:00", 13.0, [])) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "score.json"
    assert mod.main(["--history", str(hist), "--out", str(out), "--now", "2026-11-09 01:00"]) == 0
    assert json.loads(out.read_text())["windows"]["yesterday"]["horizons"][0]["mae_ppb"] == 1.0


def test_malformed_history_is_skipped_not_fatal(tmp_path):
    good = run("2026-11-07 00:00", 10.0, [(1, "2026-11-07 01:00", 11.0, 8.0, 14.0)])
    later = run("2026-11-07 01:00", 12.0, [])
    broken = [
        "[1, 2]",
        '"text"',
        "not json",
        json.dumps({"stations": "x"}),
        json.dumps({"stations": [5, {"station": 1}, {"station": "S1"}]}),
        json.dumps({"stations": [{"station": "S1", "issued_at_utc": "2026-11-07 00:00", "forecasts": "x"}]}),
        json.dumps({"stations": [{"station": "S2", "issued_at_utc": "2026-11-07 00:00", "last_reading_ppb": 1.0,
                                  "forecasts": [3, {}, {"horizon_h": 1, "target_time_utc": "bad", "no2_ppb": 1,
                                                        "interval90_low_ppb": 0, "interval90_high_ppb": 2},
                                                {"horizon_h": 1, "target_time_utc": "2026-11-07 01:00",
                                                 "no2_ppb": None, "interval90_low_ppb": 0, "interval90_high_ppb": 2}]}]}),
    ]
    path = tmp_path / "2026-11-07.jsonl"
    path.write_text("\n".join([json.dumps(good), *broken, json.dumps(later)]), encoding="utf-8")
    runs = ls.read_history([path])
    assert all(isinstance(r, dict) for r in runs)
    start = datetime(2026, 11, 7, tzinfo=timezone.utc)
    end = datetime(2026, 11, 8, tzinfo=timezone.utc)
    (h1,) = ls.score(runs, start, end)
    assert (h1.horizon_h, h1.n, h1.mae_ppb) == (1, 1, 1.0)
