"""The forecast analyst: an LLM that may only speak through typed tools, plus a checker.

    answer = Analyst(client).ask("Will NO2 be high near Chicago tomorrow morning?")

Flow for every question:

    1. the model calls tools and drafts an answer
    2. the deterministic checker (checker.check) tests the draft
    3. if it fails, the model gets the checker's feedback and one more try
    4. if that fails too, the answer is a template filled only from tool outputs

The model never sees the forecasting model, the data files or the network; it sees
the tool results and nothing else.
"""

from __future__ import annotations

import json
import time

from auraguard.agent import checker, scope
from auraguard.agent.tools import STATE_NAMES, TOOL_SCHEMAS, ForecastTools

SYSTEM_PROMPT = """You are the AURA-GUARD forecast analyst. You answer questions about hourly nitrogen \
dioxide (NO2) forecasts at 20 EPA monitoring stations in the United States, using only the tools.

Rules:
- Every number you write must be copied from a tool result in this conversation; you may round it. \
Do not calculate new numbers (no differences, averages, ratios or unit conversions) and do not use \
numbers you remember.
- Refer to stations by the id and place that list_stations returns. If the user names a city, call \
list_stations and use the station in that city's county. If no station is in or near that place, say so.
- Forecasts exist only 1, 6, 12 and 24 hours after each station's issue time. They are issued from the \
last reading in an EPA data file, not live, so always say when the forecast was issued. Local times are \
local standard time.
- For "will it be high" questions give the AQI category and the 90% interval (get_interval).
- For "how sure" questions use get_interval and get_backtest_skill.
- You only cover NO2. For other pollutants, places without a station, or medical or health questions, say \
plainly that you cannot answer that and say what you can answer. Never tell anyone what to do about their \
health.
- If the question contains a wrong premise, say so and correct it with tool results.
- At most 120 words, plain sentences, no tables."""

RETRY_PROMPT = """An automatic check rejected your answer: {issues}.
Write the answer again. Use only numbers that appear in tool results (call tools again if you need \
to), copy station ids exactly, and if the question is outside what you cover, say so plainly."""

MAX_TOOL_ROUNDS = 6


class Analyst:
    def __init__(self, client, tools: ForecastTools | None = None, max_tool_rounds: int = MAX_TOOL_ROUNDS):
        self.client = client
        self.tools = tools or ForecastTools()
        self.max_tool_rounds = max_tool_rounds
        self.stations = [{"station": s, **{k: m[k] for k in ("place", "lat", "lon")}}
                         for s, m in self.tools.meta.items()]

    # -- one model attempt: tool rounds until the model stops calling tools ----------
    def _run(self, messages: list[dict], trace: list[dict]) -> str:
        for _ in range(self.max_tool_rounds):
            msg = self.client.complete(messages, TOOL_SCHEMAS)
            calls = msg.get("tool_calls") or []
            if not calls:
                return (msg.get("content") or "").strip()
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args, out = {}, {"error": "arguments were not valid JSON"}
                else:
                    out = self.tools.call(name, args)
                trace.append({"name": name, "args": args, "output": out})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
        return ""

    def _check(self, answer: str, question: str, sc: scope.Scope, trace: list[dict]) -> checker.CheckResult:
        return checker.check(answer, question, [t["output"] for t in trace], sc, self.tools.places(), STATE_NAMES)

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        waited0 = getattr(self.client, "waited_s", 0.0)
        sc = scope.classify(question, self.stations, STATE_NAMES)
        trace: list[dict] = []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        attempts = []
        error = None
        try:
            first = self._run(messages, trace)
            c1 = self._check(first, question, sc, trace)
            attempts.append({"answer": first, "passed": c1.passed, "issues": c1.issues})
            if not c1.passed:
                messages += [{"role": "assistant", "content": first or "(no answer)"},
                             {"role": "user", "content": RETRY_PROMPT.format(issues=c1.feedback())}]
                second = self._run(messages, trace)
                c2 = self._check(second, question, sc, trace)
                attempts.append({"answer": second, "passed": c2.passed, "issues": c2.issues})
        except Exception as e:  # network or API failure: still answer, from the template
            error = f"{type(e).__name__}: {e}"

        if attempts and attempts[-1]["passed"]:
            answer, source = attempts[-1]["answer"], ("first" if len(attempts) == 1 else "retry")
            final_check = attempts[-1]
        else:
            answer = self.fallback(question, sc, trace)
            fc = self._check(answer, question, sc, trace)
            final_check = {"answer": answer, "passed": fc.passed, "issues": fc.issues}
            source = "fallback"

        total = time.perf_counter() - t0
        waited = getattr(self.client, "waited_s", 0.0) - waited0
        return {
            "question": question,
            "answer": answer,
            "source": source,
            "grounded": final_check["passed"],
            "final_issues": final_check["issues"],
            "attempts": attempts,
            "scope": {"in_scope": sc.in_scope, "reasons": sc.reasons, "places": sc.places},
            "tool_calls": trace,
            "latency_s": round(total, 3),
            "latency_excluding_rate_limit_s": round(total - waited, 3),
            "error": error,
        }

    # -- the template answer, built only from tool outputs ----------------------------
    def fallback(self, question: str, sc: scope.Scope, trace: list[dict]) -> str:
        if not sc.in_scope:
            parts = []
            if "other_pollutant" in sc.reasons:
                parts.append("I can't answer that: I only forecast NO2 (nitrogen dioxide), not other pollutants.")
            if "medical" in sc.reasons:
                parts.append("I can't give medical or health advice; please ask a health professional. "
                             "I can only report NO2 forecasts and their AQI category.")
            if "no_station" in sc.reasons:
                names = ", ".join(p["name"] for p in sc.places)
                listing = self._listed(trace)
                parts.append(f"I don't have a monitoring station near {names}. My NO2 forecasts cover: {listing}.")
            return " ".join(parts)

        lines = []
        intervals = {(t["args"].get("station"), t["output"].get("horizon_h")): t["output"]
                     for t in trace if t["name"] == "get_interval" and "error" not in t["output"]}
        seen = set()
        for t in trace:
            o = t["output"]
            if t["name"] != "get_forecast" or "error" in o or (o["station"], o["horizon_h"]) in seen:
                continue
            seen.add((o["station"], o["horizon_h"]))
            line = (f"Station {o['station']} ({o['place']}), forecast issued {o['issued_at_local']} local time "
                    f"from the last EPA reading, not live: {o['horizon_h']} hours ahead, at {o['target_time_local']} "
                    f"local time, NO2 {o['no2_ppb']} ppb ({o['aqi_category']}).")
            iv = intervals.get((o["station"], o["horizon_h"]))
            if iv:
                line += f" 90% interval {iv['interval90_low_ppb']} to {iv['interval90_high_ppb']} ppb."
            lines.append(line)
        for t in trace:
            o = t["output"]
            if t["name"] == "get_backtest_skill" and "error" not in o:
                lines.append(f"In the backtest at {o['horizon_h']} hours the mean absolute error was {o['mae_ppb']} "
                             f"ppb, against {o['persistence_mae_ppb']} ppb for persistence.")
        if lines:
            return " ".join(lines[:4])
        errors = [t["output"]["error"] for t in trace if "error" in t["output"]]
        if errors:
            return "I can't answer that from the forecast data: " + errors[-1]
        return ("I could not produce an answer I can check against the forecast data. I answer questions about "
                f"hourly NO2 forecasts at these stations: {self._listed(trace)}.")

    def _listed(self, trace: list[dict]) -> str:
        out = self.tools.list_stations()
        trace.append({"name": "list_stations", "args": {}, "output": out, "by": "fallback"})
        places = sorted({s["place"] for s in out["stations"]})
        return "; ".join(places)
