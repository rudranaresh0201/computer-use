"""
Windows UI Automation (UIA) perception layer — the differentiator called
out in docs/decisions/0001-architecture.md. Where a vision model has to
guess pixel coordinates for a button, UIA gives us its exact bounding box,
name, and role directly from the OS, when the app exposes a usable tree.

Not every app exposes a good tree (browsers rendering canvas content, games,
custom-drawn UIs) — this layer is expected to return a short or empty list
for those, and the vision/screenshot layer is the fallback. That's the
hybrid design, not a bug to chase here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pywinauto import Desktop


@dataclass
class UIElement:
    name: str
    control_type: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom), real screen px
    automation_id: Optional[str] = None

    @property
    def center(self) -> tuple[int, int]:
        left, top, right, bottom = self.rect
        return (left + right) // 2, (top + bottom) // 2


def get_foreground_window_tree(
    max_elements: int = 400,
    max_depth: int = 12,
) -> list[UIElement]:
    """Best-effort structured element list for the current foreground window.

    Returns [] rather than raising if there's no accessible window or the
    UIA backend fails — callers should treat an empty list as "fall back to
    vision", not as an error.

    max_depth was originally 6, tuned against classic Win32 apps. Packaged/
    WinUI3 apps (e.g. modern Paint's ribbon) nest their real interactive
    content substantially deeper -- confirmed by direct measurement, Paint's
    tool buttons and color swatches sit at depth 7-8, while chrome (Minimize/
    Save) sits at depth 2-4. The old cutoff silently dropped every ribbon
    control and collected chrome-only data for that app; see docs/journal.md,
    2026-07-16.
    """
    try:
        window = Desktop(backend="uia").window(active_only=True, visible_only=True)
        root = window.wrapper_object()
    except Exception:
        return []

    elements: list[UIElement] = []
    _walk(root, elements, max_elements, max_depth, depth=0)
    return elements


def _walk(node, elements: list[UIElement], max_elements: int, max_depth: int, depth: int) -> None:
    if len(elements) >= max_elements or depth > max_depth:
        return

    try:
        rect = node.rectangle()
        name = node.window_text() or ""
        control_type = node.element_info.control_type or ""
        automation_id = getattr(node.element_info, "automation_id", None)
        if rect.width() > 0 and rect.height() > 0:
            elements.append(
                UIElement(
                    name=name,
                    control_type=control_type,
                    rect=(rect.left, rect.top, rect.right, rect.bottom),
                    automation_id=automation_id or None,
                )
            )
    except Exception:
        pass  # a single bad node shouldn't abort the whole walk

    try:
        children = node.children()
    except Exception:
        return

    for child in children:
        if len(elements) >= max_elements:
            return
        _walk(child, elements, max_elements, max_depth, depth + 1)
