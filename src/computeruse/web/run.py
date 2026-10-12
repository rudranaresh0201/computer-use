"""
`computeruse web-run "<goal>"`: the agent does it, you watch.

Chrome opens (fresh profile), every step is printed as it happens, anything
risky waits for your yes, and the whole session is saved as a video plus
one line in runs/web_episodes.jsonl -- the same record the experiments use.
"""

from __future__ import annotations

import sys
from pathlib import Path

from computeruse.brain.chat import GROQ_DEFAULT_MODEL, OpenAICompatibleChat
from computeruse.web.agent import Step, WebAgent
from computeruse.web.browser import Browser

LOG = Path("runs/web_episodes.jsonl")
VIDEOS = Path("runs/videos")


def _print_step(step: Step) -> None:
    args = " ".join(f"{k}={v!r}" for k, v in step.arguments.items())
    status = "ok" if step.ok else "FAILED"
    changed = " (page changed)" if step.page_changed else ""
    tokens = step.prompt_tokens + step.completion_tokens
    print(f"{step.index:>3}. {step.action} {args}")
    print(f"     {status}: {step.detail}{changed}   [{step.model_seconds:.1f}s, {tokens / 1000:.1f}k tokens]")


def _ask(reason: str) -> bool:
    try:
        return input(f"     ⚠ the agent wants to do: {reason}. Allow? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def main(goal: str, start: str | None, model: str, max_steps: int, headless: bool, record: bool) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        chat = OpenAICompatibleChat.groq(model=model)
    except RuntimeError as e:
        print(e)
        return 2

    print(f"goal:  {goal}\nbrain: {chat.name} (Groq)   start: {start or '(blank page)'}\n")
    with Browser(headless=headless, record_video_dir=str(VIDEOS) if record else None) as web:
        agent = WebAgent(web, chat, max_steps=max_steps, confirm=_ask, on_step=_print_step, log_path=LOG)
        episode = agent.run(goal, start_url=start)
        video = web.video_path()
    print()
    if episode.status == "done":
        print(f"DONE: {episode.answer}")
    else:
        print(f"STOPPED ({episode.status}){': ' + episode.error if episode.error else ''}")
    print(f"{len(episode.steps)} steps, {episode.tokens / 1000:.1f}k tokens, {episode.seconds:.0f}s   -> {LOG}")
    if video:
        print(f"video: {Path(video).resolve()}")
    return 0 if episode.status == "done" else 1


DEFAULT_MODEL = GROQ_DEFAULT_MODEL
