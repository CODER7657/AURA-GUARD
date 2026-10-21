"""The forecast analyst, offline: tools, scope rules, the grounding checker, and the
retry / fallback loop driven by a scripted model. No network, no API key."""

import pytest

from auraguard.agent import checker, scope
from auraguard.agent.analyst import Analyst
from auraguard.agent.llm import ScriptedClient, tool_call
from auraguard.agent.tools import STATE_NAMES, ForecastTools


@pytest.fixture(scope="module")
def tools():
    return ForecastTools()


@pytest.fixture(scope="module")
def station(tools):
    """A station that has a forecast, and its 6 h forecast and interval."""
    sid = sorted(tools.by_station)[0]
    return sid, tools.get_forecast(sid, 6), tools.get_interval(sid, 6)


def stations(tools):
    return [{"station": s, **{k: m[k] for k in ("place", "lat", "lon")}} for s, m in tools.meta.items()]


def run_check(tools, answer, question, outputs):
    sc = scope.classify(question, stations(tools), STATE_NAMES)
    return checker.check(answer, question, outputs, sc, tools.places(), STATE_NAMES)


# -- tools --------------------------------------------------------------------------
def test_tools_reject_unknown_stations_and_horizons(tools, station):
    assert "error" in tools.get_forecast("99-999-9999-9", 6)
    out = tools.get_forecast(station[0], 48)
    assert "error" in out and out["available_horizons_h"] == [1, 6, 12, 24]


def test_stations_without_new_data_say_so(tools):
    for sid in tools.forecasts["stations_without_forecast"]:
        assert "no forecast" in tools.get_forecast(sid, 6)["error"]


def test_backtest_skill_matches_the_summary(tools):
    out = tools.get_backtest_skill(6)
    assert out["mae_ppb"] == round(tools.pooled[(6, "gradient_boosting")]["mae"], 2)
    assert out["interval_level_pct"] == 90


# -- scope --------------------------------------------------------------------------
@pytest.mark.parametrize("question, reasons", [
    ("Will NO2 be high near Chicago tomorrow morning?", []),
    ("What about Kansas City at 6 hours?", []),
    ("Is PM2.5 bad in Chicago?", ["other_pollutant"]),
    ("How much ozone will Phoenix get?", ["other_pollutant"]),
    ("NO2 forecast for Houston?", ["no_station"]),
    ("NO2 in London tomorrow?", ["no_station"]),
    ("What is the NO2 forecast for Texas?", ["no_station"]),
    ("Should I use my inhaler if NO2 rises in Chicago?", ["medical"]),
])
def test_scope(tools, question, reasons):
    assert scope.classify(question, stations(tools), STATE_NAMES).reasons == reasons


# -- checker ------------------------------------------------------------------------
def test_numbers_must_come_from_tools(tools, station):
    sid, f, iv = station
    q = "What is the 6 hour forecast?"
    good = f"Station {sid}: {f['no2_ppb']} ppb, 90% interval {iv['interval90_low_ppb']} to {iv['interval90_high_ppb']} ppb."
    assert run_check(tools, good, q, [f, iv]).passed
    bad = run_check(tools, good + " That is 3.7 ppb above normal.", q, [f, iv])
    assert not bad.passed and "3.7" in bad.feedback()


def test_rounding_and_percentages_are_allowed(tools):
    out = {"mae": 3.6918, "skill": 0.3031}
    assert run_check(tools, "Error about 3.7 ppb, or 4 ppb, which is 30% better.", "How good?", [out]).passed
    assert not run_check(tools, "Error about 3.8 ppb.", "How good?", [out]).passed


def test_chemical_names_and_unicode_hyphens(tools, station):
    sid, f, _ = station
    fancy = sid.replace("-", "‑")
    assert run_check(tools, f"NO₂ (NO2) at {fancy}: {f['no2_ppb']} ppb.", "forecast?", [f]).passed


def test_invented_stations_and_states_fail(tools, station):
    _, f, _ = station
    assert not run_check(tools, "Station 48-201-0001-1 reads 1 ppb.", "forecast?", [{"x": 1}]).passed
    assert not run_check(tools, "Texas will be fine.", "forecast?", []).passed
    # naming a state the user asked about is fine
    assert run_check(tools, "I don't have a station in Texas.", "NO2 in Texas?", []).passed


def test_out_of_scope_needs_a_refusal_and_no_health_instruction(tools):
    assert not run_check(tools, "PM2.5 looks fine.", "Is PM2.5 bad in Chicago?", []).passed
    assert run_check(tools, "I only forecast NO2, so I can't tell you about PM2.5.", "Is PM2.5 bad in Chicago?",
                     []).passed
    q = "Is it safe for my son with asthma to play outside in Chicago?"
    assert not run_check(tools, "I can't say, but it is safe to play outside.", q, []).passed


# -- the loop -----------------------------------------------------------------------
def test_grounded_first_answer_is_returned(tools, station):
    sid, f, _ = station
    client = ScriptedClient([tool_call("get_forecast", {"station": sid, "horizon_h": 6}),
                             {"role": "assistant", "content": f"{sid}: {f['no2_ppb']} ppb, issued {f['issued_at_utc']}."}])
    out = Analyst(client, tools).ask("6 hour forecast?")
    assert out["source"] == "first" and out["grounded"]
    assert out["tool_calls"][0]["name"] == "get_forecast"


def test_failed_check_gets_one_retry_with_feedback(tools, station):
    sid, f, _ = station
    client = ScriptedClient([tool_call("get_forecast", {"station": sid, "horizon_h": 6}),
                             {"role": "assistant", "content": f"{sid}: 99.9 ppb."},
                             {"role": "assistant", "content": f"{sid}: {f['no2_ppb']} ppb."}])
    out = Analyst(client, tools).ask("6 hour forecast?")
    assert out["source"] == "retry" and out["grounded"]
    assert "99.9" in client.calls[-1][-1]["content"]  # the checker's feedback reached the model


def test_two_failures_fall_back_to_a_template_from_tool_outputs(tools, station):
    sid, f, iv = station
    client = ScriptedClient([tool_call("get_forecast", {"station": sid, "horizon_h": 6}),
                             tool_call("get_interval", {"station": sid, "horizon_h": 6}, "call_1"),
                             {"role": "assistant", "content": "About 12345 ppb."},
                             {"role": "assistant", "content": "Still 12345 ppb."}])
    out = Analyst(client, tools).ask("6 hour forecast?")
    assert out["source"] == "fallback" and out["grounded"]
    assert str(f["no2_ppb"]) in out["answer"] and str(iv["interval90_high_ppb"]) in out["answer"]


def test_out_of_scope_fallback_is_a_grounded_refusal(tools):
    client = ScriptedClient([{"role": "assistant", "content": "Ozone will be 41 ppb."},
                             {"role": "assistant", "content": "Ozone will be 41 ppb."}])
    out = Analyst(client, tools).ask("How much ozone will Phoenix get tomorrow?")
    assert out["source"] == "fallback" and out["grounded"]
    assert checker.is_refusal(out["answer"])


def test_api_failure_still_returns_a_checked_answer(tools):
    class Broken:
        def complete(self, messages, tools=None):
            raise RuntimeError("network down")

    out = Analyst(Broken(), tools).ask("NO2 in London tomorrow?")
    assert out["source"] == "fallback" and out["grounded"] and out["error"]


# -- grading ------------------------------------------------------------------------
def test_grading_rules():
    from auraguard.agent.grading import grade_answer, percentile, tool_calls_ok

    q = {"expect": "answer", "must_mention": ["17-031-3103-1", ["good", "moderate"]], "must_not_mention": ["live"]}
    assert grade_answer("Station 17‑031‑3103‑1 stays Good.", q)["correct"]
    assert not grade_answer("Station 17-031-3103-1 stays Good, live.", q)["correct"]
    r = {"expect": "refuse", "must_mention": []}
    assert grade_answer("I can't answer that; I only forecast NO2.", r)["correct"]
    assert not grade_answer("Ozone is 40 ppb.", r)["correct"]
    rec = {"tool_calls": [{"name": "get_forecast", "args": {"station": "A", "horizon_h": 12}},
                          {"name": "list_stations", "args": {}, "by": "fallback"}]}
    assert tool_calls_ok(rec, {"expected_calls": [{"name": "get_forecast", "args": {"horizon_h": [6, 12]}}]})
    assert not tool_calls_ok(rec, {"expected_calls": [{"name": "list_stations"}]})
    assert tool_calls_ok(rec, {"expected_calls": []}) is None
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 95) == 4


def test_rate_limit_reset_headers_parse():
    from auraguard.agent.llm import seconds

    assert seconds("929ms") == 0.929 and seconds("54.9s") == 54.9 and seconds("1m26.4s") == 86.4
    assert seconds(None) == 0.0
