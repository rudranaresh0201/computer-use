"""
`computeruse web [URL]`: drive Chrome by hand, through exactly the eyes and
hands an AI brain will get. You are the brain: `look` shows what the page
exposes, you pick a ref, the browser acts and says whether anything changed.
When the brain arrives it runs the same loop -- read the elements, choose
one action, check the result -- with a model in your seat.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .browser import ActResult, Browser, PageView

HELP = """\
  look                 what you can act on (refs like e7)
  tree                 the whole accessibility tree, the way a model will read it
  click e7             click an element
  type e6 some text    fill a text box   (add --enter to submit)
  press Enter          press a key (Enter, Escape, Tab, PageDown, ...)
  goto wikipedia.org   open a page
  scroll down|up       scroll the page
  back                 browser back
  shot [file.png]      save a screenshot
  help | quit"""


def _print_view(view: PageView, max_items: int = 60) -> None:
    items = view.interactive()
    print(f"\n{view.title or '(no title)'}  --  {view.url}")
    print(f"{len(items)} things you can act on" + (f" (first {max_items})" if len(items) > max_items else "") + ":")
    for element in items[:max_items]:
        print("  " + element.describe())


def _report(result: ActResult) -> None:
    print(("  ok: " if result.ok else "  FAILED: ") + result.detail)


def run_command(web: Browser, line: str) -> bool:
    """Execute one shell line. Returns False when the user quits."""
    try:
        words = shlex.split(line)
    except ValueError as e:
        print(f"  {e}")
        return True
    if not words:
        return True
    command, args = words[0].lower(), words[1:]

    if command in ("quit", "exit", "q"):
        return False
    if command == "help":
        print(HELP)
    elif command == "look":
        _print_view(web.observe())
    elif command == "tree":
        print(web.observe(boxes=False).snapshot)
    elif command == "click" and len(args) == 1:
        _report(web.click(args[0]))
    elif command == "type" and len(args) >= 2:
        submit = "--enter" in args
        text = " ".join(a for a in args[1:] if a != "--enter")
        _report(web.type(args[0], text, submit=submit))
    elif command == "press" and len(args) == 1:
        _report(web.press(args[0]))
    elif command == "goto" and len(args) == 1:
        _report(web.goto(args[0]))
    elif command == "scroll" and len(args) <= 1:
        _report(web.scroll(args[0] if args else "down"))
    elif command == "back":
        _report(web.back())
    elif command == "shot":
        path = Path(args[0] if args else "screenshot.png")
        path.write_bytes(web.screenshot())
        print(f"  saved {path.resolve()}")
    else:
        print(f"  didn't understand {line!r} -- type help")
    return True


def main(url: str | None, headless: bool = False) -> None:
    # Web pages are full of characters a Windows console code page can't
    # print; replace them rather than crash mid-session.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print("Opening Chrome (a fresh profile -- none of your logins or cookies). Type help for commands.")
    with Browser(headless=headless) as web:
        if url:
            _report(web.goto(url))
            _print_view(web.observe())
        while True:
            try:
                line = input("\nweb> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not run_command(web, line):
                break
