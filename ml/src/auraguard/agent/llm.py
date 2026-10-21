"""A minimal OpenAI-compatible chat client (Groq by default), standard library only.

The key is read from the GROQ_API_KEY environment variable or from ml/.env, which
git ignores. It is never logged.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def api_key() -> str | None:
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


class LLMError(RuntimeError):
    pass


def seconds(value: str | None) -> float:
    """Groq reset headers look like '929ms', '54.9s' or '1m26.4s'."""
    if not value:
        return 0.0
    total, num = 0.0, ""
    i = 0
    while i < len(value):
        c = value[i]
        if c.isdigit() or c == ".":
            num += c
        elif value.startswith("ms", i):
            total, num, i = total + float(num or 0) / 1000, "", i + 1
        elif c in "hms":
            total, num = total + float(num or 0) * {"h": 3600, "m": 60, "s": 1}[c], ""
        i += 1
    return total


class ChatClient:
    """``complete(messages, tools)`` returns the assistant message as a dict.

    ``waited_s`` accumulates time spent sleeping on rate limits, so latency can be
    reported with and without it.
    """

    def __init__(self, model: str = MODEL, base_url: str = BASE_URL, key: str | None = None,
                 temperature: float = 0.0, reasoning_effort: str | None = "low", max_retries: int = 12,
                 timeout: float = 60.0, min_tokens_left: int = 4000):
        self.model, self.base_url, self.temperature = model, base_url.rstrip("/"), temperature
        self.reasoning_effort, self.max_retries, self.timeout = reasoning_effort, max_retries, timeout
        self.min_tokens_left = min_tokens_left
        self.key = key or api_key()
        if not self.key:
            raise LLMError("GROQ_API_KEY is not set (environment or ml/.env)")
        self.waited_s = 0.0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_completion_tokens": 1500}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        data = json.dumps(body).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data, method="POST",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                         "User-Agent": "auraguard-forecast-analyst/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode("utf-8"))
                    left = r.headers.get("x-ratelimit-remaining-tokens")
                    reset = seconds(r.headers.get("x-ratelimit-reset-tokens"))
                # Pace under a tokens-per-minute limit instead of running into it.
                if left is not None and int(left) < self.min_tokens_left and reset:
                    time.sleep(min(reset + 0.5, 65.0))
                    self.waited_s += min(reset + 0.5, 65.0)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                wait = max(float(e.headers.get("retry-after") or 0), seconds(e.headers.get("x-ratelimit-reset-tokens")))
                wait = (wait + 1.0) if wait else min(60.0, 4.0 * 2 ** attempt)
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries and wait <= 120:
                    time.sleep(wait)
                    self.waited_s += wait
                    continue
                # tool_use_failed: the model produced a malformed tool call; surface it as text
                if e.code == 400 and "tool_use_failed" in detail:
                    return {"role": "assistant", "content": "", "_error": "tool_use_failed"}
                raise LLMError(f"HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(4.0)
                    self.waited_s += 4.0
                    continue
                raise LLMError(str(e)) from None
        u = out.get("usage", {})
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["calls"] += 1
        msg = out["choices"][0]["message"]
        return {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}


class ScriptedClient:
    """Replays canned assistant messages; for tests and offline CI. No network."""

    def __init__(self, replies: list[dict]):
        self.replies, self.calls, self.waited_s = list(replies), [], 0.0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append([dict(m) for m in messages])
        self.usage["calls"] += 1
        if not self.replies:
            return {"role": "assistant", "content": ""}
        return self.replies.pop(0)


def tool_call(name: str, args: dict, call_id: str = "call_0") -> dict:
    """An assistant message asking for one tool call, in OpenAI format (for ScriptedClient)."""
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
