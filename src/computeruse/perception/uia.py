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
from pywinauto.uia_element_info import UIAElementInfo


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


@dataclass
class WindowTree:
    """The foreground window's own identity and rect alongside its element
    list, so a caller can *verify* which window it actually got.

    This exists because of a dataset-wide failure found 2026-07-19:
    `get_foreground_window_tree` returns whatever is active at that instant,
    and the dataset collector had no way to check that against the app it
    believed it was labeling. When focus was stolen mid-run (the user typing
    in another window, an app crashing after launch), the collector labeled
    the *editor's* elements as "file_explorer" / "audacity" and wrote them
    to labels.jsonl with no error. 196 of 542 rows -- 36% of the dataset,
    including every row of one held-out app -- were wrong this way. Returning
    the window's title/class/rect makes that verifiable instead of assumed.
    """

    title: str
    class_name: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom), real screen px
    elements: list["UIElement"]


def get_foreground_window_info(
    max_elements: int = 400,
    max_depth: int = 12,
) -> Optional[WindowTree]:
    """Like `get_foreground_window_tree`, but also reports which window was
    walked. Returns None (not an exception) when no window is accessible,
    matching this module's "empty means fall back to vision" contract."""
    try:
        window = Desktop(backend="uia").window(active_only=True, visible_only=True)
        root = window.wrapper_object()
        rect = root.rectangle()
        title = root.window_text() or ""
        try:
            class_name = root.class_name() or ""
        except Exception:
            class_name = ""
    except Exception:
        return None

    elements: list[UIElement] = []
    _walk(root, elements, max_elements, max_depth, depth=0)
    return WindowTree(
        title=title,
        class_name=class_name,
        rect=(rect.left, rect.top, rect.right, rect.bottom),
        elements=elements,
    )


def is_visible_at_center(rect: tuple[int, int, int, int], tolerance: int = 2) -> bool:
    """True if this rect's centre point actually hits this element on screen.

    UIA reports an element's rect whether or not anything is drawn on top of
    it. When a menu, flyout or dialog opens, every control it covers is
    still enumerated with a perfectly valid box -- so a naive collector
    labels targets the model physically cannot see. Observed live
    2026-07-19 in `calculator_0022`: with the navigation flyout open, the
    +, -, x and = buttons behind it were all labeled.

    The fix is a real hit-test. `UIAElementInfo.from_point` returns the
    topmost element at a screen point; if that element is ours, its rect is
    ours, and if it's a descendant of ours its rect sits inside ours. An
    occluder (the flyout's list item) is neither, so containment separates
    the two cases without needing runtime IDs.

    Fails *open* -- if the hit-test errors, keep the element. Silently
    dropping real training data on a flaky UIA call would be a worse bug
    than the one this fixes.
    """
    left, top, right, bottom = rect
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    try:
        topmost = UIAElementInfo.from_point(cx, cy)
        hit = topmost.rectangle
    except Exception:
        return True

    return (
        hit.left >= left - tolerance
        and hit.top >= top - tolerance
        and hit.right <= right + tolerance
        and hit.bottom <= bottom + tolerance
    )


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
