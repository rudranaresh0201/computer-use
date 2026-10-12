"""
App-like pages with known click targets, for comparing the accessibility
tree against vision on equal terms.

The comparison only means something if the correct answer comes from
neither method (docs/research/gui-grounding-hypothesis.md, H4-H6). So the
answer comes from us: we generate the page, mark each target, and read its
true box from the DOM -- or, for a target drawn into a <canvas>, from the
coordinates we drew it at.

Every page is one generic app shell (header with icon buttons, sidebar,
panels, a canvas toolbar), randomized per seed -- theme, layout, labels,
icons -- and carries one target per condition:

    C1 named      a button whose visible label is its accessible name
    C2 unnamed    an icon button with no accessible name at all
    C3 reworded   a named button, asked for in other words ("keep my work" -> Save)
    C4 not in tree  a button drawn on a <canvas>: pixels only
    C5 ambiguous  a label that appears in several panels; the instruction says which

build.py renders each page in Chrome and checks, from the real tree, that
every target really is its condition before keeping it.
"""

from __future__ import annotations

import html
import random
from dataclasses import dataclass, field
from typing import Optional

VIEWPORT = (1568, 882)  # the training screenshots' size: the model sees what it was trained on

# Icon buttons: SVG only, no <title>, no aria-label -> an empty accessible name.
# Each comes with how a person would ask for it -- by purpose or by look.
ICONS: dict[str, tuple[str, list[str]]] = {
    "gear": ('<circle cx="12" cy="12" r="4" fill="none" stroke-width="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2" stroke-width="2"/>',
             ["open the settings", "click the gear icon"]),
    "trash": ('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" fill="none" stroke-width="2"/>',
              ["throw this away", "click the trash can"]),
    "bell": ('<path d="M6 16V11a6 6 0 0 1 12 0v5l2 2H4z" fill="none" stroke-width="2"/><path d="M10 20a2 2 0 0 0 4 0" stroke-width="2"/>',
             ["show my notifications", "click the bell"]),
    "search": ('<circle cx="10" cy="10" r="6" fill="none" stroke-width="2"/><path d="M15 15l6 6" stroke-width="2"/>',
               ["look something up", "click the magnifying glass"]),
    "close": ('<path d="M5 5l14 14M19 5L5 19" stroke-width="2.5"/>', ["dismiss this window", "click the X"]),
    "download": ('<path d="M12 3v12M7 10l5 5 5-5M4 20h16" fill="none" stroke-width="2"/>',
                 ["get a copy onto my computer", "click the down arrow"]),
    "plus": ('<path d="M12 4v16M4 12h16" stroke-width="2.5"/>', ["create a new one", "click the plus sign"]),
    "home": ('<path d="M3 11l9-8 9 8M5 10v10h14V10" fill="none" stroke-width="2"/>',
             ["go back to the start page", "click the house icon"]),
    "star": ('<path d="M12 3l3 6 6 1-4.5 4 1 6-5.5-3-5.5 3 1-6L3 10l6-1z" fill="none" stroke-width="2"/>',
             ["add this to favourites", "click the star"]),
    "heart": ('<path d="M12 20s-7-4.5-9-9a5 5 0 0 1 9-3 5 5 0 0 1 9 3c-2 4.5-9 9-9 9z" fill="none" stroke-width="2"/>',
              ["like this", "click the heart"]),
}

# C3: labels, and ways to ask for them that share no word with the label.
PARAPHRASES: dict[str, list[str]] = {
    "Save": ["keep my work", "store these changes"],
    "Cancel": ["never mind, back out", "abandon this"],
    "Submit": ["send the form off", "hand this in"],
    "Sign out": ["leave my account", "end my session"],
    "Undo": ["take back the last change", "revert what I just did"],
    "Refresh": ["load it again", "get the latest view"],
    "Rename": ["give it another title", "change what it's called"],
    "Duplicate": ["make a copy of it", "clone this item"],
    "Archive": ["put it away for later", "file this away"],
    "Upload": ["send a file up", "attach something from my computer"],
    "Continue": ["move on to the following step", "go ahead"],
    "Help": ["I need assistance", "show me the docs"],
}

PLAIN_LABELS = [
    "Preview", "Publish", "Filter", "Sort", "Invite", "Report", "Compare", "Merge", "Pin", "Mute",
    "Translate", "Schedule", "Approve", "Reject", "Assign", "Tag", "Move", "Lock", "Sync", "Print",
]
CANVAS_LABELS = ["Export", "Share", "Crop", "Rotate", "Zoom in", "Zoom out", "Draw", "Erase", "Fill", "Highlight"]
PANELS = ["Privacy", "Notifications", "Billing", "Storage", "Display", "Security", "Language", "Backups"]
DUPLICATED = ["Reset", "Edit", "Clear", "Manage", "Details", "Change"]
SIDEBAR = ["Overview", "Projects", "Calendar", "Messages", "Files", "Reports", "Team", "Activity"]
APP_NAMES = ["Nimbus", "Ledgerly", "Orbit Notes", "Pixelforge", "Taskwise", "Harbor CRM", "Quill", "Atlas Drive"]

THEMES = [
    {"bg": "#ffffff", "fg": "#1f2328", "panel": "#f6f8fa", "accent": "#0969da", "button": "#eaeef2", "canvas": "#dde3ea"},
    {"bg": "#0d1117", "fg": "#e6edf3", "panel": "#161b22", "accent": "#2f81f7", "button": "#21262d", "canvas": "#30363d"},
    {"bg": "#fffaf3", "fg": "#3b2f2f", "panel": "#f3e9dc", "accent": "#c2410c", "button": "#ead8c0", "canvas": "#e2cfb5"},
    {"bg": "#f4f7fb", "fg": "#102a43", "panel": "#ffffff", "accent": "#7c3aed", "button": "#e4e7eb", "canvas": "#d9e2ec"},
]


@dataclass(frozen=True)
class Target:
    target_id: str
    condition: str  # C1..C5
    instruction: str
    label: str  # what it is, for checking (icon name for C2)
    canvas_rect: Optional[tuple[int, int, int, int]] = None  # C4: x, y, w, h inside the canvas


@dataclass
class Page:
    page_id: str
    html: str
    targets: list[Target] = field(default_factory=list)


def _button(label: str, target_id: str = "", theme: dict | None = None) -> str:
    attr = f' data-target="{target_id}"' if target_id else ""
    return f'<button class="btn"{attr}>{html.escape(label)}</button>'


def _icon_button(icon: str, target_id: str = "") -> str:
    attr = f' data-target="{target_id}"' if target_id else ""
    return (f'<button class="icon"{attr}><svg viewBox="0 0 24 24" width="22" height="22" '
            f'stroke="currentColor" fill="currentColor">{ICONS[icon][0]}</svg></button>')


def generate(seed: int) -> Page:
    rng = random.Random(seed)
    page_id = f"p{seed:04d}"
    theme = rng.choice(THEMES)
    font = rng.choice([13, 14, 15, 16])
    sidebar_left = rng.random() < 0.7
    canvas_top = rng.random() < 0.5
    targets: list[Target] = []

    # -- pick each condition's target and its instruction --------------------
    icons = rng.sample(sorted(ICONS), 6)
    c2_icon = icons[0]
    targets.append(Target(f"{page_id}-c2", "C2", rng.choice(ICONS[c2_icon][1]), c2_icon))

    c3_label = rng.choice(sorted(PARAPHRASES))
    targets.append(Target(f"{page_id}-c3", "C3", rng.choice(PARAPHRASES[c3_label]), c3_label))

    plain = rng.sample(PLAIN_LABELS, 9)
    c1_label = plain[0]
    targets.append(Target(f"{page_id}-c1", "C1", rng.choice([f"click {c1_label}", f"press the {c1_label} button",
                                                             f"{c1_label}"]), c1_label))

    panels = rng.sample(PANELS, rng.choice([3, 4]))
    duplicated = rng.choice(DUPLICATED)
    c5_panel = rng.choice(panels)
    targets.append(Target(f"{page_id}-c5", "C5", rng.choice([f"click {duplicated} in the {c5_panel} panel",
                                                             f"{duplicated.lower()} the {c5_panel} settings"]), duplicated))

    canvas_labels = rng.sample(CANVAS_LABELS, 4)
    c4_index = rng.randrange(4)
    canvas_w, button_w, button_h, gap = 4 * 110 + 50, 100, 34, 10
    rects = [(10 + i * (button_w + gap), 8, button_w, button_h) for i in range(4)]
    targets.append(Target(f"{page_id}-c4", "C4", rng.choice([f"click {canvas_labels[c4_index]}",
                                                             f"{canvas_labels[c4_index]} the image"]),
                          canvas_labels[c4_index], rects[c4_index]))

    # -- the page --------------------------------------------------------------
    header_icons = [_icon_button(c2_icon, f"{page_id}-c2")] + [_icon_button(i) for i in icons[1:5]]
    rng.shuffle(header_icons)
    header_buttons = [_button(plain[1]), _button(plain[2])]
    sidebar = "".join(f'<a href="#{s.lower()}">{s}</a>' for s in rng.sample(SIDEBAR, 6))

    panel_html = []
    others = plain[3:]
    for name in panels:
        buttons = [_button(duplicated, f"{page_id}-c5" if name == c5_panel else "")]
        buttons.append(_button(others.pop()))
        rng.shuffle(buttons)
        panel_html.append(f'<section class="panel"><h2>{name}</h2><p>Settings for {name.lower()}.</p>{"".join(buttons)}</section>')
    actions = [_button(c1_label, f"{page_id}-c1"), _button(c3_label, f"{page_id}-c3"), _button(others.pop())]
    rng.shuffle(actions)

    draw = "".join(
        f"c.fillStyle='{theme['button']}';c.fillRect({x},{y},{w},{h});"
        f"c.fillStyle='{theme['fg']}';c.fillText({label!r},{x + w // 2},{y + h // 2});"
        for (x, y, w, h), label in zip(rects, canvas_labels)
    )
    canvas = (f'<canvas id="tools" width="{canvas_w}" height="50" style="width:{canvas_w}px;height:50px"></canvas>'
              # Wrapped in a function: set_content reuses the window, so a top-level
              # `const c` from the previous page would make this one throw and draw nothing.
              f"<script>(() => {{const c=document.getElementById('tools').getContext('2d');"
              f"c.fillStyle='{theme['canvas']}';c.fillRect(0,0,{canvas_w},50);"
              f"c.font='{font}px sans-serif';c.textAlign='center';c.textBaseline='middle';{draw}}})();</script>")

    main = [f'<div class="actions">{"".join(actions)}</div>', f'<div class="panels">{"".join(panel_html)}</div>']
    main.insert(0 if canvas_top else len(main), f'<div class="toolbar">{canvas}</div>')
    body = (f'<header><b>{rng.choice(APP_NAMES)}</b><span class="spacer"></span>{"".join(header_buttons)}'
            f'{"".join(header_icons)}</header>'
            f'<div class="layout">'
            + (f'<nav>{sidebar}</nav>' if sidebar_left else "")
            + f'<main>{"".join(main)}</main>'
            + ("" if sidebar_left else f'<nav>{sidebar}</nav>')
            + "</div>")
    css = f"""
    html,body{{margin:0;height:100%;overflow:hidden;background:{theme['bg']};color:{theme['fg']};
      font:{font}px system-ui,sans-serif}}
    header{{display:flex;align-items:center;gap:8px;padding:10px 18px;background:{theme['panel']};
      border-bottom:1px solid {theme['button']}}}
    .spacer{{flex:1}}
    .layout{{display:flex;height:calc(100% - 60px)}}
    nav{{width:{rng.choice([180, 200, 220])}px;padding:16px;display:flex;flex-direction:column;gap:10px;
      background:{theme['panel']}}}
    nav a{{color:{theme['accent']};text-decoration:none}}
    main{{flex:1;padding:18px 26px;display:flex;flex-direction:column;gap:18px}}
    .actions{{display:flex;gap:12px}}
    .panels{{display:grid;grid-template-columns:repeat({len(panels) if len(panels) < 4 else 2},1fr);gap:16px}}
    .panel{{background:{theme['panel']};padding:12px 16px;border-radius:8px}}
    .panel h2{{font-size:{font + 3}px;margin:0 0 6px}}
    .btn{{background:{theme['button']};color:{theme['fg']};border:1px solid {theme['accent']}55;border-radius:6px;
      padding:6px 14px;margin-right:8px;font:inherit;cursor:pointer}}
    .icon{{background:none;border:none;color:{theme['fg']};padding:6px;cursor:pointer}}
    """
    document = f"<!doctype html><html><head><title>{page_id}</title><style>{css}</style></head><body>{body}</body></html>"
    return Page(page_id, document, targets)
