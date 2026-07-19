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
from computeruse.perception.uia import UIElement, get_foreground_window_tree

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
) -> list[LabeledSample]:
    """Capture whatever window is currently focused and emit one
    LabeledSample per surviving UIA element.

    The caller is responsible for having already opened `app` and put it
    into the desired state (see module docstring) — this function only
    observes and labels, it doesn't drive the app.
    """
    screenshot_id = f"{app}_{screenshot_seq:04d}"
    image_dir = dataset_root / "images" / app
    image_path = image_dir / f"{screenshot_id}.png"

    screenshot = capture(save_path=image_path)
    raw_elements = get_foreground_window_tree()
    elements = filter_elements(raw_elements, screenshot.real_size, max_per_control_type, rng)
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
