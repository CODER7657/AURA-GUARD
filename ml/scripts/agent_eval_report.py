"""Tables from the agent eval's raw records. Reads numbers only; no API calls.

    python scripts/agent_eval_report.py

Reads eval/agent_questions.json and results/agent_eval_raw.json, writes
results/agent_eval_summary.json and results/agent_eval.md.

"Before the checker" grades the model's first draft as if it had been returned
directly; "after" grades what the analyst actually returned (first draft, retry or
template fallback).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from auraguard.agent.grading import grade_answer, percentile, tool_calls_ok

ROOT = Path(__file__).resolve().parents[1]


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def main() -> None:
    questions = {q["id"]: q for q in json.loads((ROOT / "eval" / "agent_questions.json").read_text(encoding="utf-8"))["questions"]}
    run = json.loads((ROOT / "results" / "agent_eval_raw.json").read_text(encoding="utf-8"))
    records = [r for r in run["records"] if r["id"] in questions]

    rows = []
    for r in records:
        q = questions[r["id"]]
        first = r["attempts"][0]["answer"] if r["attempts"] else ""
        rows.append({
            "id": r["id"],
            "category": q["category"],
            "expect": q["expect"],
            "source": r["source"],
            "grounded_first": bool(r["attempts"] and r["attempts"][0]["passed"]),
            "grounded_final": r["grounded"],
            "first": grade_answer(first, q),
            "final": grade_answer(r["answer"], q),
            "tool_calls_ok": tool_calls_ok(r, q),
            "calls": len([c for c in r["tool_calls"] if c.get("by") != "fallback"]),
            "latency_s": r["latency_s"],
            "latency_active_s": r["latency_excluding_rate_limit_s"],
            "error": r["error"],
            "first_issues": r["attempts"][0]["issues"] if r["attempts"] else ["no draft (API error)"],
        })

    n = len(rows)
    refuse = [x for x in rows if x["expect"] == "refuse"]
    answerable = [x for x in rows if x["expect"] in ("answer", "correct_premise", "clarify_or_answer")]
    with_calls = [x for x in rows if x["tool_calls_ok"] is not None]
    lat = [x["latency_s"] for x in rows]
    act = [x["latency_active_s"] for x in rows]
    sources = Counter(x["source"] for x in rows)

    summary = {
        "model": run["model"],
        "questions_commit": run["questions_commit"],
        "questions_sha256": run["questions_sha256"],
        "code_commit": run["code_commit"],
        "n_questions": n,
        "grounded_before_checker": sum(x["grounded_first"] for x in rows) / n,
        "grounded_after_checker": sum(x["grounded_final"] for x in rows) / n,
        "answer_source": dict(sources),
        "correct_behaviour_before": sum(x["first"]["correct"] for x in rows) / n,
        "correct_behaviour_after": sum(x["final"]["correct"] for x in rows) / n,
        "correct_refusal_before": sum(x["first"]["correct"] for x in refuse) / len(refuse) if refuse else None,
        "correct_refusal_after": sum(x["final"]["correct"] for x in refuse) / len(refuse) if refuse else None,
        "false_refusal_after": (sum(x["final"]["refused"] and not x["final"]["correct"] for x in answerable)
                                / len(answerable)) if answerable else None,
        "tool_call_accuracy": sum(x["tool_calls_ok"] for x in with_calls) / len(with_calls) if with_calls else None,
        "n_with_expected_calls": len(with_calls),
        "mean_tool_calls": sum(x["calls"] for x in rows) / n,
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "latency_excluding_rate_limit_p50_s": percentile(act, 50),
        "latency_excluding_rate_limit_p95_s": percentile(act, 95),
        "api_errors": sum(bool(x["error"]) for x in rows),
        "usage": run.get("usage"),
        "per_question": rows,
    }
    (ROOT / "results" / "agent_eval_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    cats = ["answerable", "ambiguous", "out_of_scope", "wrong_premise"]
    lines = [
        f"Model `{run['model']}` on {n} pre-registered questions "
        f"(questions commit `{run['questions_commit'][:7]}`, code commit `{run['code_commit'][:7]}`).",
        "",
        "| Metric | Before the checker (first draft) | After the checker (returned answer) |",
        "|---|---|---|",
        f"| Grounded: every number, station and place traceable to a tool output | "
        f"{pct(sum(x['grounded_first'] for x in rows), n)} | {pct(sum(x['grounded_final'] for x in rows), n)} |",
        f"| Expected behaviour (graded by string rules) | {pct(sum(x['first']['correct'] for x in rows), n)} | "
        f"{pct(sum(x['final']['correct'] for x in rows), n)} |",
        f"| Correct refusal, among questions that should be refused | "
        f"{pct(sum(x['first']['correct'] for x in refuse), len(refuse))} | "
        f"{pct(sum(x['final']['correct'] for x in refuse), len(refuse))} |",
        f"| Refused although answerable | {pct(sum(x['first']['refused'] and not x['first']['correct'] for x in answerable), len(answerable))} | "
        f"{pct(sum(x['final']['refused'] and not x['final']['correct'] for x in answerable), len(answerable))} |",
        "",
        f"Returned answers: {sources.get('first', 0)} first drafts, {sources.get('retry', 0)} retries after checker "
        f"feedback, {sources.get('fallback', 0)} template fallbacks. API errors: {summary['api_errors']}.",
        "",
        f"Tool-call accuracy (all pre-registered calls made, with the right station and horizon): "
        f"{pct(sum(x['tool_calls_ok'] for x in with_calls), len(with_calls))}. "
        f"Mean tool calls per question: {summary['mean_tool_calls']:.1f}.",
        "",
        f"Latency per question, end to end including retries: p50 {summary['latency_p50_s']:.1f} s, "
        f"p95 {summary['latency_p95_s']:.1f} s; excluding time spent waiting on API rate limits: "
        f"p50 {summary['latency_excluding_rate_limit_p50_s']:.1f} s, "
        f"p95 {summary['latency_excluding_rate_limit_p95_s']:.1f} s.",
        "",
        "| Category | Questions | Grounded before | Grounded after | Behaviour before | Behaviour after |",
        "|---|---|---|---|---|---|",
    ]
    for c in cats:
        sub = [x for x in rows if x["category"] == c]
        if not sub:
            continue
        lines.append(f"| {c.replace('_', ' ')} | {len(sub)} | {pct(sum(x['grounded_first'] for x in sub), len(sub))} | "
                     f"{pct(sum(x['grounded_final'] for x in sub), len(sub))} | "
                     f"{pct(sum(x['first']['correct'] for x in sub), len(sub))} | "
                     f"{pct(sum(x['final']['correct'] for x in sub), len(sub))} |")
    issues = Counter(i.split(":")[0] for x in rows if not x["grounded_first"] for i in x["first_issues"])
    lines += ["", "Why first drafts failed the checker (a draft can fail for several reasons):", ""]
    lines += [f"- {k}: {v}" for k, v in issues.most_common()]
    wrong = [x for x in rows if not x["final"]["correct"]]
    lines += ["", "Returned answers graded wrong: " + (", ".join(f"{x['id']} ({x['category']})" for x in wrong) or "none")]
    (ROOT / "results" / "agent_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
