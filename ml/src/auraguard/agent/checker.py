"""Deterministic grounding check for the analyst's final answer.

An answer passes only if:

1. every number in it equals, or is a rounding of, a number in a tool output from
   this conversation (or in the question itself); a fraction may be written as a
   percentage. Arithmetic the model did itself, such as a difference of two tool
   numbers, is rejected on purpose.
2. every station id in it is a real station, and every US state or "X County" it
   names belongs to a monitor or appears in the question.
3. an out-of-scope question (see ``scope``) gets a refusal, and a medical question
   gets no health instruction.

Nothing here calls a model. The same function checks the fallback answer, so the
fallback is held to the same rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from auraguard.agent.scope import Scope

STATION_ID = re.compile(r"\b\d{2}-\d{3}-\d{4}(?:-\d{1,2})?\b")
# Chemical names and similar tokens whose digits are not quantities.
NOT_NUMBERS = re.compile(r"\b(?:NO|NOx|PM|O|SO|CO|CO2|H2O)[0-9.₂₃]+\b|\bPM\s?2\.5\b|\bPM\s?10\b|\bnitrogen dioxide\b",
                         re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s?%| percent)?")
REFUSAL = re.compile(
    r"\b(can(?:no|')t|cannot|can ?not|unable|not able|(?:do|does)n'?t have|do not have|does not have|no forecast|"
    r"no (?:station|monitor|data|readings?|model)|not (?:available|covered|in scope)|only (?:forecast|cover|have|"
    r"provide|know)|outside (?:my|the|what)|isn'?t available|not something I|beyond (?:my|the) scope|"
    r"not a (?:doctor|medical))\b",
    re.IGNORECASE,
)
HEALTH_INSTRUCTION = re.compile(
    r"\b(you should (?:take|use|stay|avoid|go|exercise|run|wear|keep)|use your inhaler|take your|it(?:'s| is) "
    r"(?:perfectly )?safe (?:to|for)|(?:is|are) safe for|no need to worry)\b",
    re.IGNORECASE,
)


@dataclass
class CheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    numbers_checked: int = 0

    def feedback(self) -> str:
        return "; ".join(self.issues)


DASHES = str.maketrans({**{c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"},
                        "\u2019": "'", "\u00a0": " ", "\u202f": " "})


def normalize(text: str) -> str:
    """Models often write ids with non-breaking hyphens and apostrophes as curly quotes."""
    return (text or "").translate(DASHES)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.search(normalize(text)))


def _numbers_in(obj) -> list[float]:
    """Every number in a tool output, including numbers inside strings such as timestamps."""
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(m.group(1)) for m in NUMBER.finditer(STATION_ID.sub(" ", obj))]
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


def _grounded(token: str, is_pct: bool, sources: list[float]) -> bool:
    value = float(token)
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10 ** -decimals + 1e-9
    for t in sources:
        if abs(value - t) <= tol or abs(value - abs(t)) <= tol:
            return True
        if is_pct and abs(t) <= 1.5 and abs(value - 100 * abs(t)) <= tol:
            return True
    return False


def check(answer: str, question: str, tool_outputs: list, scope: Scope, station_places: dict[str, str],
          state_names: set[str]) -> CheckResult:
    """``station_places`` maps station id to "County, State"."""
    issues: list[str] = []
    text, question = normalize(answer), normalize(question)
    if not text.strip():
        return CheckResult(False, ["the answer is empty"])

    # 2. station ids, states and counties
    for sid in set(STATION_ID.findall(text)):
        if not any(s == sid or s.startswith(sid + "-") for s in station_places):
            issues.append(f"station id {sid} does not exist")
    known_states = {p.split(", ")[-1] for p in station_places.values()}
    known_counties = {p.split(", ")[0] for p in station_places.values()}
    for state in state_names:
        if re.search(rf"\b{re.escape(state)}\b", text) and state not in known_states and state not in question:
            # "Kansas" inside "Kansas City" or "New York" inside "New York City" is not a state claim
            if not re.search(rf"\b{re.escape(state)} City\b", text):
                issues.append(f"names {state}, where there is no monitor, though the question did not")
    for county in set(re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?) County\b", text)):
        if county not in known_counties and county not in question:
            issues.append(f"{county} County is not a monitor's county")

    # 1. numbers
    sources = _numbers_in(tool_outputs) + _numbers_in(question)
    body = NOT_NUMBERS.sub(" ", STATION_ID.sub(" ", text))
    found = list(NUMBER.finditer(body))
    bad = [m.group(0).strip() for m in found if not _grounded(m.group(1), bool(m.group(2)), sources)]
    if bad:
        issues.append("numbers not found in any tool output: " + ", ".join(sorted(set(bad))))

    # 3. scope
    if not scope.in_scope and not is_refusal(text):
        issues.append("the question is out of scope (" + ", ".join(scope.reasons) + ") but the answer does not "
                      "say so plainly")
    if "medical" in scope.reasons and HEALTH_INSTRUCTION.search(text):
        issues.append("the answer gives a health instruction; it must not give medical advice")

    return CheckResult(not issues, issues, len(found))
