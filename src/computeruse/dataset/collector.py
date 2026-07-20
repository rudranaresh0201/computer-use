"""
Dataset collector — turns "an app window is open right now" into labeled
GUI-grounding training samples, with zero manual annotation (ADR-0003).

Reuses Phase 1's perception primitives directly: perception.screenshot.capture()
for the image + coordinate metadata, perception.uia.get_foreground_window_tree()
for the ground-truth element boxes. This module owns only the labeling step
(filter -> instruction -> sample) specified in
docs/research/gui-grounding-dataset-design.md — it does not know about the
14-app registry, train/held-out split assignment, or app-state walking. A
caller (a future collection-run script) is responsible for putting the target
app into the state it wants labeled, then invoking `collect_from_current_window`
once per state.
"""

from __future__ import annotations

import getpass
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from computeruse.perception.screenshot import capture
from computeruse.perception.uia import (
    UIElement,
    get_foreground_window_info,
    is_visible_at_center,
)


class WindowMismatchError(Exception):
    """Raised when the window that is actually in the foreground at label
    time isn't the app we believe we're collecting.

    Deliberately an exception and not a silent skip: the 2026-07-19 audit
    found 196 of 542 rows (36%, including 100% of one held-out app) were
    the *development editor's* elements labeled as the target app, because
    focus was stolen mid-run and nothing checked. Failing loudly costs one
    state; failing silently poisoned the whole dataset and cost several
    training runs before anyone looked at a picture.
    """

# Phrasing templates keyed by UIA control_type, per the design doc's
# instruction-generation section. Multiple templates per type so the model
# doesn't just learn one rigid phrasing; picked randomly per sample.
_INSTRUCTION_TEMPLATES: dict[str, list[str]] = {
    "Button": ["Click {name}", "Press the {name} button", "Select {name}"],
    "MenuItem": ["Click {name}", "Select {name} from the menu", "Open {name}"],
    "CheckBox": ["Check {name}", "Toggle the {name} option"],
    "RadioButton": ["Select the {name} option", "Choose {name}"],
    "Edit": ["Click the {name} field", "Select the {name} input box"],
    "ListItem": ["Click on {name}", "Select {name} from the list"],
    "TreeItem": ["Click on {name}", "Expand {name}"],
    "ComboBox": ["Open the {name} dropdown", "Click {name}"],
    "TabItem": ["Switch to the {name} tab", "Open the {name} tab"],
    "Hyperlink": ["Click the {name} link", "Open {name}"],
}
_FALLBACK_TEMPLATES = ["Click {name}"]

_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

# Filtering rule 3 (dedup): a container and its single child frequently
# report the same rect within a few px of rounding/border noise.
_DEDUP_TOLERANCE_PX = 4

# Filtering rule 4 (control-type coverage): cap how many elements of any one
# control_type survive per screenshot, so e.g. 40 toolbar buttons don't
# crowd out the one checkbox also on screen.
_DEFAULT_MAX_PER_CONTROL_TYPE = 8


@dataclass
class LabeledSample:
    """One (screenshot, element, instruction) triple — matches the JSON
    schema fixed in docs/research/gui-grounding-dataset-design.md field-for-field."""

    id: str
    screenshot_id: str
    screenshot_path: str
    real_size: tuple[int, int]
    scaled_size: tuple[int, int]
    scale_x: float
    scale_y: float
    element: dict[str, Any]  # {name, control_type, automation_id, bbox_real}
    instruction: str
    app: str
    app_richness: str
    split: str
    session_id: str
    source: str = "uia_auto_label"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _on_screen(rect: tuple[int, int, int, int], real_size: tuple[int, int]) -> bool:
    left, top, right, bottom = rect
    real_w, real_h = real_size
    return right > 0 and bottom > 0 and left < real_w and top < real_h


def window_matches(
    title: str,
    class_name: str,
    expect_title_contains: Optional[str],
    expect_class_contains: Optional[str],
) -> bool:
    """Same OR-semantics the orchestrator uses to *find* a window, reused
    here to confirm it at label time -- some apps have unreliable titles
    (Windows Terminal shows the shell's name), so a class match counts too.
    With neither expectation given, verification is skipped: that's the
    escape hatch for ad-hoc/manual collection, not the orchestrated path."""
    if expect_title_contains is None and expect_class_contains is None:
        return True
    if expect_title_contains and expect_title_contains.lower() in (title or "").lower():
        return True
    if expect_class_contains and expect_class_contains.lower() in (class_name or "").lower():
        return True
    return False


def _clip_to_window(
    elements: list[UIElement], window_rect: tuple[int, int, int, int]
) -> list[UIElement]:
    """Drop elements whose center falls outside the target window's rect.

    UIA tree walks can surface nodes belonging to overlapping or child
    windows (a dropdown hosted in its own top-level window, a scrollbar
    reported past the frame). Once the screenshot is cropped to the window,
    any such element would point outside the image entirely -- an
    unlearnable target. Centre-based rather than full-containment so a
    control that merely straddles the border by a pixel of rounding still
    counts.
    """
    left, top, right, bottom = window_rect
    kept = []
    for el in elements:
        cx, cy = el.center
        if left <= cx <= right and top <= cy <= bottom:
            kept.append(el)
    return kept


def drop_occluded(elements: list[UIElement], visible) -> list[UIElement]:
    """Keep only elements whose centre is actually the topmost thing on
    screen at that point.

    `visible` is injected (defaults to uia.is_visible_at_center at the call
    site) purely so this is testable without a live desktop -- the policy
    lives here, the OS call lives in the perception layer.

    Why it matters: when a menu or flyout is open, UIA still enumerates
    every control underneath it with a valid rect. Labeling those asks the
    model to point at something that isn't drawn, which is unlearnable and
    actively teaches it to ignore the image. This became a real problem the
    moment the registry grew a lot of `*_menu_open` states.
    """
    return [el for el in elements if visible(el.rect)]


def _to_window_space(
    rect: tuple[int, int, int, int], origin: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Translate a screen-absolute element rect into the cropped image's
    coordinate space. Must be applied to every stored bbox once the
    screenshot is a window crop rather than a full-desktop grab, or the
    labels describe a frame that no longer exists."""
    ox, oy = origin
    left, top, right, bottom = rect
    return (left - ox, top - oy, right - ox, bottom - oy)


def _rects_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    tol: int = _DEDUP_TOLERANCE_PX,
) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _dedup(elements: list[UIElement]) -> list[UIElement]:
    """Collapse near-identical rects to the deepest (most specific) node.

    `elements` arrives in tree-walk order (parent before children), so when a
    later element's rect is close to an already-kept one, it's a more
    specific descendant of that node — replace the coarser entry with it.
    """
    kept: list[UIElement] = []
    for el in elements:
        match_index = next(
            (i for i, k in enumerate(kept) if _rects_close(el.rect, k.rect)), None
        )
        if match_index is None:
            kept.append(el)
        else:
            kept[match_index] = el
    return kept


def _sample_by_control_type(
    elements: list[UIElement], max_per_type: int, rng: Optional[random.Random] = None
) -> list[UIElement]:
    """Keep up to max_per_type elements of each control_type.

    Elements arrive in tree-walk order, and window chrome (Minimize/
    Maximize/Close, nav buttons) sits near the top of that walk on every
    app -- taking the first N of a type systematically kept chrome and
    dropped the actual content (e.g. Calculator's digit buttons never
    survived the cap; every "rich" app's cap-limited types collapsed to
    the same handful of generic chrome buttons every other app also has).
    Sampling randomly instead means the cap no longer silently favors
    whatever the tree walker happened to visit first.
    """
    picker = rng or random
    by_type: dict[str, list[UIElement]] = {}
    for el in elements:
        by_type.setdefault(el.control_type, []).append(el)

    sampled: list[UIElement] = []
    for group in by_type.values():
        if len(group) > max_per_type:
            sampled.extend(picker.sample(group, max_per_type))
        else:
            sampled.extend(group)
    return sampled


def _pii_terms() -> list[str]:
    """The logged-in username always counts (catches account-name UI text
    and any file path under that user's home dir). Anything else that can't
    be derived generically -- a home Wi-Fi SSID, say -- goes in the
    COMPUTERUSE_DATASET_REDACT_TERMS env var (comma-separated) instead of
    being hardcoded here, so no personal string ends up committed to
    source. See docs/journal.md, 2026-07-16 incident: a real account name
    and Wi-Fi SSID were captured into labeled training data."""
    terms = [getpass.getuser()]
    extra = os.environ.get("COMPUTERUSE_DATASET_REDACT_TERMS", "")
    terms.extend(t.strip() for t in extra.split(",") if t.strip())
    return [t.lower() for t in terms if t]


def _contains_pii(name: str, pii_terms: list[str]) -> bool:
    low = name.lower()
    return any(term in low for term in pii_terms)


def filter_elements(
    elements: list[UIElement],
    real_size: tuple[int, int],
    max_per_control_type: int = _DEFAULT_MAX_PER_CONTROL_TYPE,
    rng: Optional[random.Random] = None,
) -> list[UIElement]:
    """Apply the dataset design doc's filtering rules 1, 3, 4, plus an
    instructable-control-type rule discovered from the v1 training run's
    spot-check: container/text control types (Window, Text, Pane, ...)
    have no real phrasing in _INSTRUCTION_TEMPLATES, so they fell back to
    the same generic "Click {name}" text. A window, its sub-panes, and a
    text label inside it can all share one UIA name (e.g. "Calculator"),
    which meant the identical instruction pointed at genuinely different
    boxes -- the model learned to ignore the image for that instruction
    instead of grounding it. Restricting samples to control types that have
    a real template removes the collision at the source.

    (Rule 2, foreground-window-only, is already enforced by
    `get_foreground_window_tree` scoping the walk to the active window.)
    """
    pii_terms = _pii_terms()
    named_and_visible = [
        el
        for el in elements
        if el.name.strip()
        and _on_screen(el.rect, real_size)
        and el.control_type in _INSTRUCTION_TEMPLATES
        and not _contains_pii(el.name, pii_terms)
    ]
    deduped = _dedup(named_and_visible)
    return _sample_by_control_type(deduped, max_per_control_type, rng)


def _disambiguated_names(elements: list[UIElement]) -> dict[int, str]:
    """Map id(element) -> the name to use in its instruction, adding a
    positional ordinal when multiple elements share the same
    (control_type, name).

    Two sibling elements can be genuinely distinct, on-screen, and
    correctly named the same thing -- e.g. two Windows Terminal tabs both
    named "Windows PowerShell". Without disambiguation both produce the
    identical instruction text ("Open the Windows PowerShell tab") pointing
    at two different pixels, which a model can't resolve no matter how it's
    trained: spot-checking the 2026-07-18 training run showed it predicting
    the midpoint of both tabs' positions for exactly this pair. This is a
    different failure than the Window/Pane/Text collision fixed in
    filter_elements (that one was wrong control types sharing a name;
    control-type filtering can't fix this one, since TabItem is exactly the
    type we want to keep). Ordinal assigned by left-to-right, top-to-bottom
    reading order, which matches how these duplicates are actually laid out
    in practice (tabs, list rows).
    """
    groups: dict[tuple[str, str], list[UIElement]] = {}
    for el in elements:
        groups.setdefault((el.control_type, el.name.strip()), []).append(el)

    names: dict[int, str] = {}
    for (_control_type, name), group in groups.items():
        if len(group) == 1:
            names[id(group[0])] = name
            continue
        ordered = sorted(group, key=lambda e: (e.rect[0], e.rect[1]))
        for idx, el in enumerate(ordered):
            ordinal = _ORDINALS[idx] if idx < len(_ORDINALS) else f"{idx + 1}th"
            names[id(el)] = f"{ordinal} {name}"
    return names


def generate_instruction(
    element: UIElement, rng: Optional[random.Random] = None, name: Optional[str] = None
) -> str:
    """Auto-generate a referring instruction from UIA metadata alone —
    the free label source (ADR-0003), no manual annotation.

    `name` overrides element.name.strip() when the caller has already
    resolved a disambiguated name (see _disambiguated_names) -- callers
    that don't care about duplicate-name collisions can omit it."""
    picker = rng or random
    templates = _INSTRUCTION_TEMPLATES.get(element.control_type, _FALLBACK_TEMPLATES)
    template = picker.choice(templates)
    return template.format(name=name if name is not None else element.name.strip())


def collect_from_current_window(
    app: str,
    app_richness: str,
    split: str,
    session_id: str,
    screenshot_seq: int,
    dataset_root: Path,
    max_per_control_type: int = _DEFAULT_MAX_PER_CONTROL_TYPE,
    rng: Optional[random.Random] = None,
    expect_title_contains: Optional[str] = None,
    expect_class_contains: Optional[str] = None,
    crop_to_window: bool = False,
    visibility_check=is_visible_at_center,
) -> list[LabeledSample]:
    """Capture the focused window and emit one LabeledSample per surviving
    UIA element.

    The caller is responsible for having already opened `app` and put it
    into the desired state (see module docstring) — this function only
    observes and labels, it doesn't drive the app.

    Two behaviours changed 2026-07-19 after the dataset audit (see
    WindowMismatchError and docs/journal.md):

    1. The foreground window is *verified* against
       `expect_title_contains`/`expect_class_contains` before anything is
       written. Mismatch raises rather than labeling the wrong app.
    2. `crop_to_window` (opt-in, default off) crops the screenshot to the
       target window's rect and translates element boxes into that crop's
       space, so the stored image is the app rather than the whole desktop
       with the app somewhere inside it. That makes targets far larger
       relative to the frame (median target is 0.16% of a full desktop) and
       removes the surrounding desktop, which is where all three PII
       incidents came from.

       It defaults to OFF because it changes the image format: a dataset
       must not mix cropped and full-desktop screenshots, or the model sees
       two different conventions for what an image even is. Flip it only
       when re-collecting *every* app in one pass, never for a partial
       top-up of an existing full-desktop dataset.
    """
    window = get_foreground_window_info()
    if window is None:
        raise WindowMismatchError(
            f"no accessible foreground window while collecting {app!r} -- "
            "nothing was written"
        )
    if not window_matches(
        window.title, window.class_name, expect_title_contains, expect_class_contains
    ):
        raise WindowMismatchError(
            f"foreground window is title={window.title!r} class={window.class_name!r}, "
            f"which does not match the expected app {app!r} "
            f"(title~{expect_title_contains!r} class~{expect_class_contains!r}). "
            "Focus was most likely stolen mid-run -- do not use the machine "
            "while collection is running. Nothing was written for this state."
        )

    screenshot_id = f"{app}_{screenshot_seq:04d}"
    image_dir = dataset_root / "images" / app
    image_path = image_dir / f"{screenshot_id}.png"

    screenshot = capture(
        save_path=image_path, region=window.rect if crop_to_window else None
    )
    if crop_to_window:
        # confine to the window, then rebase every rect onto the crop
        raw_elements = [
            UIElement(
                name=el.name,
                control_type=el.control_type,
                rect=_to_window_space(el.rect, screenshot.origin),
                automation_id=el.automation_id,
            )
            for el in _clip_to_window(window.elements, window.rect)
        ]
    else:
        # full-desktop frame: rects are already screen-absolute, which is
        # the same space the image covers, so no translation is needed
        raw_elements = window.elements
    elements = filter_elements(raw_elements, screenshot.real_size, max_per_control_type, rng)
    # Drop anything hidden behind an open menu/flyout/dialog. Done after
    # filtering (a hit-test per element is a real UIA round-trip, and only
    # ~10-40 elements survive filtering) but before disambiguation, so
    # ordinals like "second Windows PowerShell" are numbered over the
    # elements that are actually visible.
    elements = drop_occluded(elements, visibility_check)
    display_names = _disambiguated_names(elements)

    samples: list[LabeledSample] = []
    for i, element in enumerate(elements):
        instruction = generate_instruction(element, rng, name=display_names[id(element)])
        samples.append(
            LabeledSample(
                id=f"{screenshot_id}_e{i}",
                screenshot_id=screenshot_id,
                screenshot_path=image_path.relative_to(dataset_root).as_posix(),
                real_size=screenshot.real_size,
                scaled_size=screenshot.scaled_size,
                scale_x=screenshot.scale_x,
                scale_y=screenshot.scale_y,
                element={
                    "name": element.name.strip(),
                    "control_type": element.control_type,
                    "automation_id": element.automation_id,
                    "bbox_real": list(element.rect),
                },
                instruction=instruction,
                app=app,
                app_richness=app_richness,
                split=split,
                session_id=session_id,
            )
        )
    return samples


def write_samples(samples: list[LabeledSample], labels_path: Path) -> None:
    """Append samples to labels.jsonl — append, not overwrite, since
    collection happens across many separate runs (design doc)."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")
