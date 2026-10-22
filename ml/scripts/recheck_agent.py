"""Re-ask a few pre-registered questions after a checker change and grade them the same way.

    python scripts/recheck_agent.py B05 D01

A targeted re-check, not a new evaluation: it writes results/agent_fix_check.json with the
old record (from agent_eval_raw.json, which it never modifies) next to the new one.
Needs GROQ_API_KEY (environment or ml/.env).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from auraguard.agent.analyst import Analyst
from auraguard.agent.grading import grade_answer, tool_calls_ok
from auraguard.agent.llm import MODEL, ChatClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ids = sys.argv[1:]
    questions = {q["id"]: q for q in json.loads((ROOT / "eval" / "agent_questions.json").read_text("utf-8"))["questions"]}
    old = {r["id"]: r for r in json.loads((ROOT / "results" / "agent_eval_raw.json").read_text("utf-8"))["records"]}
    analyst = Analyst(ChatClient())
    out = {"note": "targeted re-check of the listed questions after the no-monitor-claim checker fix; not a new full "
                   "evaluation", "model": MODEL, "date": time.strftime("%Y-%m-%d"), "questions": []}
    for qid in ids:
        q = questions[qid]
        new = analyst.ask(q["question"])
        new["id"] = qid

        def summary(r):
            return {"answer": r["answer"], "source": r["source"], "grounded": r["grounded"],
                    "attempts": [{"passed": a["passed"], "issues": a["issues"], "answer": a["answer"]}
                                 for a in r["attempts"]],
                    "grade": grade_answer(r["answer"], q), "tool_calls_ok": tool_calls_ok(r, q)}

        out["questions"].append({"id": qid, "question": q["question"], "before": summary(old[qid]), "after": summary(new)})
        a = out["questions"][-1]["after"]
        print(qid, a["source"], "grounded", a["grounded"], "correct", a["grade"]["correct"], flush=True)
        time.sleep(2)
    (ROOT / "results" / "agent_fix_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                                           encoding="utf-8")


if __name__ == "__main__":
    main()
