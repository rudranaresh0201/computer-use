"""
The strong tree baseline: an LLM reads the accessibility tree and picks the
element to click -- how a real agent uses a tree, and the arm that keeps the
tree-vs-vision comparison honest. Name matching (compare.tree_answer) cannot
map "keep my work" to Save or use a panel heading to pick between two
Resets; an LLM can. What it still cannot do is click something the tree
does not contain, or tell nameless icons apart -- which is the question.

The LLM gets the tree as text (roles, names, headings, nesting, refs) and
the instruction, and answers with one ref or NONE. The click is the center
of that element's box.

Fixed before the test run (2026-10-12): the prompt below; model
openai/gpt-oss-120b on Groq, temperature 0, reasoning_effort "low";
the test subset = the first 50 test pages (seeds 1000-1049, 250 items,
50 per condition), chosen by seed order before any answer was seen --
Groq's free tier allows ~200K tokens a day, not 1,000 items.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from computeruse.eval.compare import center
from computeruse.eval.vlm_grounder import is_hit_normalized
from computeruse.web.browser import INTERACTIVE_ROLES, WebElement, parse_snapshot

MODEL = "openai/gpt-oss-120b"
EXTRA = {"reasoning_effort": "low", "max_completion_tokens": 400}

PROMPT = """You see the accessibility tree of an application window, the way a screen reader sees it:
one element per line as  role "name" [ref=...], indented by nesting. Headings and regions give context.
Buttons with no name have no text in the tree.

Instruction: {instruction}

Which ONE element should be clicked to follow the instruction? Answer with only its ref (for example e12).
If no element in the tree can be identified as the target, answer NONE.

Tree:
{tree}"""

_KEEP_ROLES = INTERACTIVE_ROLES | {"heading", "navigation", "region", "dialog", "banner", "main", "complementary", "toolbar", "group"}
_LINE = re.compile(r'^(\s*)-\s+([a-z]+)(.*)$')
_STRIP = re.compile(r"\s*\[(?:box=[^\]]*|cursor=pointer|active)\]")
_ANSWER = re.compile(r"\b((?:f\d+)?e\d+)\b|\bNONE\b")


def compact_tree(snapshot: str) -> str:
    """The tree a model reads: interactive elements and structure, no pixel
    boxes, no URLs, no bare layout containers."""
    lines = []
    for line in snapshot.splitlines():
        match = _LINE.match(line)
        if match is None or match.group(2) not in _KEEP_ROLES:
            continue
        lines.append(_STRIP.sub("", line).rstrip(":"))
    return "\n".join(lines)


def parse_answer(text: str, elements: list[WebElement]) -> Optional[WebElement]:
    """The element the model named, or None for NONE / an unknown ref / no answer."""
    refs = {e.ref: e for e in elements}
    for match in _ANSWER.finditer(text.strip()):
        if match.group(1):
            return refs.get(match.group(1))
        return None  # NONE came first
    return None


def ask(chat, instruction: str, snapshot: str) -> tuple[Optional[WebElement], str, dict[str, int]]:
    elements = parse_snapshot(snapshot)
    prompt = PROMPT.format(instruction=instruction, tree=compact_tree(snapshot))
    text, usage = chat.complete([{"role": "user", "content": prompt}], **EXTRA)
    return parse_answer(text, elements), text, usage


def run(
    items: list[dict[str, Any]],
    root: Path,
    chat,
    out: Path,
    viewport: tuple[int, int],
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Score `items`, appending one JSON line per item to `out`; items
    already in `out` are skipped, so an interrupted run resumes."""
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    for n, item in enumerate(items, 1):
        if item["id"] in done:
            continue
        snapshot = (root / item["tree"]).read_text(encoding="utf-8")
        element, text, usage = ask(chat, item["instruction"], snapshot)
        point = center(element, viewport) if element is not None else None
        row = {"id": item["id"], "condition": item["condition"], "answer": text.strip()[:80],
               "ref": element.ref if element else None, "name": element.name if element else None,
               "point": list(point) if point else None,
               "hit": is_hit_normalized(point, tuple(item["gt_box_norm"])), "usage": usage}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        done[item["id"]] = row
        log(f"[tree_llm] {n}/{len(items)} {item['condition']} {item['instruction'][:38]!r:<42} -> {row['name']!r} hit={row['hit']} "
            f"({sum(usage.values())} tok)")
    return [done[i["id"]] for i in items]
