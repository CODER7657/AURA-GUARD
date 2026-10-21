"""Run the forecast analyst on the pre-registered questions and save every raw record.

    python scripts/run_agent_eval.py                      # eval/agent_questions.json
    python scripts/agent_eval_report.py                   # tables from the raw records

Needs GROQ_API_KEY (environment or ml/.env). Writes results/agent_eval_raw.json after
every question, so an interrupted run resumes where it stopped. The questions file's
git commit and SHA-256 are recorded, so the report can show the questions were
committed before the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from auraguard.agent.analyst import SYSTEM_PROMPT, Analyst
from auraguard.agent.llm import MODEL, ChatClient

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", default=str(ROOT / "eval" / "agent_questions.json"))
    p.add_argument("--out", default=str(ROOT / "results" / "agent_eval_raw.json"))
    p.add_argument("--pause", type=float, default=2.0, help="seconds between questions (rate limits)")
    args = p.parse_args()

    qpath = Path(args.questions)
    questions = json.loads(qpath.read_text(encoding="utf-8"))["questions"]
    out_path = Path(args.out)
    if out_path.exists():
        run = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        rel = qpath.resolve().relative_to(ROOT.parent).as_posix()
        run = {
            "model": MODEL,
            "questions_file": rel,
            "questions_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
            "questions_commit": git("log", "-n", "1", "--format=%H", "--", str(qpath)),
            "code_commit": git("rev-parse", "HEAD"),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": [],
        }
    done = {r["id"] for r in run["records"]}
    client = ChatClient()
    analyst = Analyst(client)
    for q in questions:
        if q["id"] in done:
            continue
        record = analyst.ask(q["question"])
        if record["error"] and "HTTP 429" in record["error"]:
            # out of API quota: stop without saving, so the next run retries this question
            print(f"{q['id']}: rate limit not recovering, stopping; rerun to resume", flush=True)
            break
        record["id"] = q["id"]
        run["records"].append(record)
        run["usage"] = client.usage
        out_path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{q['id']:>4} {record['source']:8s} grounded={record['grounded']!s:5s} "
              f"{record['latency_excluding_rate_limit_s']:6.1f}s  {q['question'][:60]}", flush=True)
        time.sleep(args.pause)
    run["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
