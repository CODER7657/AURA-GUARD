"""Grade analyst answers against pre-registered expectations. Deterministic.

A question in eval/agent_questions.json has:

    id, category        answerable | ambiguous | out_of_scope | wrong_premise
    question
    expect              answer | refuse | clarify_or_answer | correct_premise
    must_mention        strings the answer must contain; an inner list means "any of these"
    must_not_mention    strings the answer must not contain
    expected_calls      tool calls the model should make: {"name", "args"}; an args value
                        that is a list means "any of these values"

Grading is string matching, so it is strict about wording in some places and blind
to meaning in others; the report says so.
"""

from __future__ import annotations

import math

from auraguard.agent.checker import is_refusal, normalize


def _has(text: str, item) -> bool:
    options = item if isinstance(item, list) else [item]
    return any(normalize(str(o)).lower() in text for o in options)


def grade_answer(answer: str, q: dict) -> dict:
    text = normalize(answer or "").lower()
    refused = is_refusal(answer or "")
    mentions = all(_has(text, m) for m in q.get("must_mention", []))
    clean = not any(_has(text, m) for m in q.get("must_not_mention", []))
    expect = q["expect"]
    if expect == "refuse":
        correct = refused and mentions and clean
    elif expect == "clarify_or_answer":
        correct = clean and (mentions or "?" in (answer or ""))
    else:  # answer, correct_premise
        correct = mentions and clean
    return {"correct": bool(correct), "refused": refused, "mentions_ok": mentions, "forbidden_ok": clean}


def _matches(call: dict, want: dict) -> bool:
    if call["name"] != want["name"]:
        return False
    for k, v in want.get("args", {}).items():
        got = call.get("args", {}).get(k)
        options = v if isinstance(v, list) else [v]
        if not any(str(got) == str(o) for o in options):
            return False
    return True


def tool_calls_ok(record: dict, q: dict) -> bool | None:
    """None when the question pre-registered no expected calls."""
    want = q.get("expected_calls")
    if not want:
        return None
    made = [c for c in record.get("tool_calls", []) if c.get("by") != "fallback"]
    return all(any(_matches(c, w) for c in made) for w in want)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]
