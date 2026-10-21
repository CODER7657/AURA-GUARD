"""Ask the forecast analyst a question from the command line.

    python -m auraguard.agent "Will NO2 be high near Chicago tomorrow morning?"
    python -m auraguard.agent --json "How sure are you about the 6 hour forecast?"

Needs GROQ_API_KEY in the environment or in ml/.env. --json prints the full record:
tool calls, every draft, the checker's verdicts and latency.
"""

from __future__ import annotations

import argparse
import json
import sys

from auraguard.agent.analyst import Analyst
from auraguard.agent.llm import ChatClient, LLMError


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="+")
    p.add_argument("--json", action="store_true", help="print the full record as JSON")
    args = p.parse_args()
    question = " ".join(args.question)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        client = ChatClient()
    except LLMError as e:
        print(json.dumps({"error": str(e)}) if args.json else f"error: {e}", file=sys.stdout if args.json else sys.stderr)
        return 2
    result = Analyst(client).ask(question)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["answer"])
        print(f"\n[{result['source']} answer, grounded={result['grounded']}, "
              f"{len([t for t in result['tool_calls'] if t.get('by') != 'fallback'])} tool calls, "
              f"{result['latency_s']:.1f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
