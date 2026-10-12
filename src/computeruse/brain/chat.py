"""
The model behind a Brain, reduced to one call: "here is the conversation and
the tools you may use -- which tool do you call next?"

Provider-neutral on purpose (ADR-0002): Groq's free tier for demos today, an
open model on a Kaggle GPU for experiments (no rate limits, fixed weights,
so a run can be repeated), anything else with an OpenAI-style
/chat/completions endpoint later. Swapping one for another is a base URL and
a model name, never a code change.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"


def read_key(name: str) -> Optional[str]:
    """An API key from the environment -- or, on Windows, from the user
    environment `setx` writes to, which a terminal opened before the setx
    (or a tool it started) never sees. The key is never printed."""
    value = os.environ.get(name)
    if value or os.name != "nt":
        return value
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0] or None
    except OSError:
        return None


@dataclass
class ToolCall:
    name: str  # "" when the model answered in prose instead of calling a tool
    arguments: dict[str, Any] = field(default_factory=dict)
    text: str = ""  # anything the model said alongside the call
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    latency_s: float = 0.0


class ChatModel(Protocol):
    name: str

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ToolCall: ...


class OpenAICompatibleChat:
    """Any server speaking OpenAI's /chat/completions with tools."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str],
        timeout: float = 60.0,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        post: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.name = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep
        if post is None:
            import requests

            post = requests.post
        self._post = post

    @classmethod
    def groq(cls, model: str = GROQ_DEFAULT_MODEL, **kwargs) -> "OpenAICompatibleChat":
        key = read_key("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Make a free key at console.groq.com, then in PowerShell run\n"
                '    setx GROQ_API_KEY "your-key"\n'
                "and open a NEW terminal. (Never paste the key into a chat or a file in the repo.)"
            )
        return cls(model, GROQ_BASE_URL, key, **kwargs)

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ToolCall:
        start = time.monotonic()
        payload = self._post_json({"model": self.name, "messages": messages, "tools": tools,
                                   "tool_choice": "auto", "temperature": 0})
        call = parse_completion(payload)
        call.latency_s = time.monotonic() - start
        return call

    def complete(self, messages: list[dict[str, Any]], **extra: Any) -> tuple[str, dict[str, int]]:
        """Plain text answer (no tools) and its token usage."""
        payload = self._post_json({"model": self.name, "messages": messages, "temperature": 0, **extra})
        usage = {k: int(v) for k, v in (payload.get("usage") or {}).items() if k in ("prompt_tokens", "completion_tokens")}
        return payload["choices"][0]["message"].get("content") or "", usage

    def _post_json(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        for attempt in range(self._max_retries + 1):
            response = self._post(self._url, headers=headers, json=body, timeout=self._timeout)
            if response.status_code == 429 and attempt < self._max_retries:
                # Free tiers limit tokens per minute; the server says how long to wait.
                wait = float(response.headers.get("retry-after", 2 ** attempt * 5))
                self._sleep(min(wait, 60.0))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"{self.name}: HTTP {response.status_code}: {response.text[:300]}")
            return response.json()
        raise RuntimeError(f"{self.name}: still rate-limited after {self._max_retries} retries")


def parse_completion(payload: dict[str, Any]) -> ToolCall:
    """The first tool call in an OpenAI-style completion. A model that
    writes the call as JSON in its text instead (small open models do) is
    read the same way; one that only talks yields name=""."""
    message = payload["choices"][0]["message"]
    usage = {k: int(v) for k, v in (payload.get("usage") or {}).items() if k in ("prompt_tokens", "completion_tokens")}
    text = message.get("content") or ""
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        return ToolCall(function.get("name", ""), arguments if isinstance(arguments, dict) else {}, text, usage)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and isinstance(data.get("tool") or data.get("name"), str):
                arguments = data.get("arguments") or data.get("args") or {}
                return ToolCall(data.get("tool") or data.get("name"), arguments, text, usage)
        except json.JSONDecodeError:
            pass
    return ToolCall("", {}, text, usage)


class ScriptedChat:
    """Plays back fixed tool calls -- for tests and for exercising the agent
    loop without spending any model budget."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.name = "scripted"
        self._calls = list(calls)
        self.seen: list[list[dict[str, Any]]] = []

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ToolCall:
        self.seen.append(messages)
        if not self._calls:
            return ToolCall("done", {"answer": "(script ran out)"})
        name, arguments = self._calls.pop(0)
        return ToolCall(name, arguments, usage={"prompt_tokens": sum(len(m["content"]) for m in messages) // 4})
