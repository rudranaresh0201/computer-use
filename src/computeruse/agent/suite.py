"""
The sandbox task suite: small web apps where each task needs one of the paper's conditions, and
each page carries its own checker. The agent can only click and type; the checker reads the app's
state (`window.__verdict()`), so a task passes only if the app really changed -- never because the
agent said so.

    signup      C1/C3  named form fields + a named button; the goal words differ from the labels
    note        C2     the Save button is an icon with no name, and Ctrl+S is switched off
    mixer       C4     the volume buttons are drawn on a <canvas>: invisible to the tree
    archive     C5     three rows, three buttons all named "Archive"; archive the right one
    connect     drag   draw a line from the red dot to the blue dot on a canvas

Pages are generated per seed: positions and names of rows move, so repeated runs are not replays.
Run headless (no mouse taken, nothing on screen) -- free with a scripted brain, cheap with Groq.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

VIEWPORT = (1280, 800)


@dataclass
class SuiteTask:
    name: str
    condition: str
    goal: str
    html: str


def _page(title: str, body: str, script: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font: 15px system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1d2330; }}
 header {{ background: #2b3a55; color: #fff; padding: 12px 20px; font-weight: 600; }}
 main {{ padding: 24px 32px; }}
 .card {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; margin-bottom: 14px; max-width: 760px; }}
 button {{ font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid #9aa6bb; background: #fff; cursor: pointer; }}
 input, textarea {{ font: inherit; padding: 6px 8px; border: 1px solid #9aa6bb; border-radius: 6px; }}
 .toolbar button {{ width: 38px; height: 34px; padding: 4px; }}
 #status {{ margin-top: 10px; color: #3c6e47; min-height: 1.2em; }}
</style></head><body><header>{title}</header><main>{body}<div id="status"></div></main>
<script>{script}</script></body></html>"""


def signup(rng: random.Random) -> SuiteTask:
    name, email = rng.choice([("Ada Lovelace", "ada@example.com"), ("Alan Turing", "alan@example.org"),
                              ("Grace Hopper", "grace@example.net")])
    pad = rng.randint(0, 120)
    body = f"""<div class="card" style="margin-left:{pad}px">
 <h3>Join the newsletter</h3>
 <p><label>Full name <input id="n" aria-label="Full name"></label></p>
 <p><label>Email address <input id="e" aria-label="Email address"></label></p>
 <p><label><input type="checkbox" id="u"> Email me product updates</label></p>
 <button id="go">Sign up</button></div>"""
    script = f"""
window.__state = null;
document.getElementById('go').onclick = () => {{
  window.__state = {{n: n.value.trim(), e: e.value.trim(), u: u.checked}};
  document.getElementById('status').textContent = 'Thanks, you are signed up.';
}};
window.__verdict = () => {{
  const s = window.__state;
  if (!s) return [false, 'the form was never submitted'];
  const ok = s.n === {name!r} && s.e === {email!r} && s.u === true;
  return [ok, `submitted name=${{s.n}} email=${{s.e}} updates=${{s.u}}`];
}};"""
    goal = f"Subscribe {name} ({email}) to the newsletter, with product updates switched on."
    return SuiteTask("signup", "C1/C3", goal, _page("Newsletter", body, script))


SVG = {
    "bold": '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M7 4h6a4 4 0 0 1 0 8H7zM7 12h7a4 4 0 0 1 0 8H7z" fill="none" stroke="#1d2330" stroke-width="2.4"/></svg>',
    "italic": '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M10 4h8M6 20h8M14 4l-4 16" stroke="#1d2330" stroke-width="2.2"/></svg>',
    "save": '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M4 3h13l3 3v15H4z" fill="none" stroke="#1d2330" stroke-width="2"/><rect x="7" y="3" width="9" height="6" fill="#1d2330"/><rect x="7" y="13" width="10" height="7" fill="none" stroke="#1d2330" stroke-width="2"/></svg>',
    "trash": '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14" fill="none" stroke="#1d2330" stroke-width="2"/></svg>',
    "share": '<svg width="22" height="22" viewBox="0 0 24 24"><circle cx="6" cy="12" r="3" fill="#1d2330"/><circle cx="18" cy="5" r="3" fill="#1d2330"/><circle cx="18" cy="19" r="3" fill="#1d2330"/><path d="M8.5 10.5l7-4M8.5 13.5l7 4" stroke="#1d2330" stroke-width="2"/></svg>',
}


def note(rng: random.Random) -> SuiteTask:
    text = rng.choice(["Meeting at 5", "Buy oat milk", "Call the dentist"])
    icons = ["bold", "italic", "save", "trash", "share"]
    rng.shuffle(icons)
    buttons = "".join(f'<button data-k="{k}">{SVG[k]}</button>' for k in icons)  # no names, no titles
    body = f"""<div class="card"><div class="toolbar">{buttons}</div>
 <p><textarea id="t" aria-label="Note" rows="6" cols="60"></textarea></p></div>"""
    script = f"""
window.__saved = null; window.__trashed = false;
document.querySelectorAll('.toolbar button').forEach(b => b.onclick = () => {{
  const k = b.dataset.k;
  if (k === 'save') {{ window.__saved = t.value.trim(); document.getElementById('status').textContent = 'Saved.'; }}
  if (k === 'trash') {{ t.value = ''; window.__trashed = true; }}
}});
document.addEventListener('keydown', ev => {{ if (ev.ctrlKey && ev.key.toLowerCase() === 's') ev.preventDefault(); }});
window.__verdict = () => {{
  if (window.__saved === null) return [false, 'the note was never saved'];
  return [window.__saved === {text!r}, `saved ${{JSON.stringify(window.__saved)}}`];
}};"""
    return SuiteTask("note", "C2", f"Write the note \"{text}\" and save it.", _page("Notes", body, script))


def mixer(rng: random.Random) -> SuiteTask:
    channels = ["Music", "Voice", "Alerts"]
    rng.shuffle(channels)
    start = {c: rng.randint(2, 5) for c in channels}
    goal_ch = rng.choice(channels)
    want = start[goal_ch] + rng.choice([2, 3])
    script = f"""
const chans = {channels!r}; const vol = {start!r};
const cv = document.getElementById('cv'), g = cv.getContext('2d');
const rows = chans.map((c, i) => ({{c, y: 40 + i * 70}}));
function draw() {{
  g.fillStyle = '#fff'; g.fillRect(0, 0, cv.width, cv.height); g.font = '16px system-ui';
  for (const r of rows) {{
    g.fillStyle = '#1d2330'; g.fillText(r.c, 20, r.y + 22);
    g.strokeStyle = '#9aa6bb'; g.strokeRect(120, r.y, 40, 34); g.fillText('-', 136, r.y + 23);
    g.fillStyle = '#2b3a55'; g.fillRect(175, r.y + 12, vol[r.c] * 20, 10);
    g.fillStyle = '#1d2330'; g.fillText(String(vol[r.c]), 390, r.y + 22);
    g.strokeRect(420, r.y, 40, 34); g.fillText('+', 434, r.y + 23);
  }}
}}
cv.onclick = ev => {{
  const b = cv.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top;
  for (const r of rows) {{
    if (y < r.y || y > r.y + 34) continue;
    if (x >= 120 && x <= 160) vol[r.c] = Math.max(0, vol[r.c] - 1);
    if (x >= 420 && x <= 460) vol[r.c] = Math.min(10, vol[r.c] + 1);
  }}
  draw();
}};
draw();
window.__verdict = () => {{
  const others = chans.filter(c => c !== {goal_ch!r}).every(c => vol[c] === {start!r}[c]);
  return [vol[{goal_ch!r}] === {want} && others, `volumes ${{JSON.stringify(vol)}}`];
}};"""
    body = '<div class="card"><h3>Mixer</h3><canvas id="cv" width="500" height="240"></canvas></div>'
    goal = f"Set the {goal_ch} volume to {want}, leaving the other channels as they are."
    return SuiteTask("mixer", "C4", goal, _page("Mixer", body, script))


def archive(rng: random.Random) -> SuiteTask:
    people = ["Grace Hopper", "Katherine Johnson", "Margaret Hamilton", "Hedy Lamarr"]
    rng.shuffle(people)
    rows_people = people[:3]
    goal_person = rng.choice(rows_people)
    rows = "".join(f'<tr><td>Order {rng.randint(100, 999)}</td><td>{p}</td><td><button>Archive</button></td></tr>'
                   for p in rows_people)
    body = f'<div class="card"><h3>Open orders</h3><table cellpadding="8">{rows}</table></div>'
    script = """
window.__archived = [];
document.querySelectorAll('tr').forEach(tr => {
  tr.querySelector('button').onclick = () => { window.__archived.push(tr.children[1].textContent); tr.style.opacity = .35; };
});
window.__verdict = () => [JSON.stringify(window.__archived) === JSON.stringify([%r]),
                          `archived ${JSON.stringify(window.__archived)}`];""" % goal_person
    return SuiteTask("archive", "C5", f"Archive {goal_person}'s order, and only that one.",
                     _page("Orders", body, script))


def connect(rng: random.Random) -> SuiteTask:
    red = (rng.randint(60, 200), rng.randint(60, 300))
    blue = (rng.randint(320, 560), rng.randint(60, 300))
    script = f"""
const cv = document.getElementById('cv'), g = cv.getContext('2d');
const R = {list(red)}, B = {list(blue)}; let down = null; window.__strokes = [];
function dot(p, c) {{ g.fillStyle = c; g.beginPath(); g.arc(p[0], p[1], 9, 0, 7); g.fill(); }}
g.fillStyle = '#fff'; g.fillRect(0, 0, cv.width, cv.height); dot(R, '#d33'); dot(B, '#2a5bd7');
const at = ev => {{ const b = cv.getBoundingClientRect(); return [ev.clientX - b.left, ev.clientY - b.top]; }};
cv.onmousedown = ev => down = at(ev);
cv.onmouseup = ev => {{ if (!down) return; const up = at(ev); window.__strokes.push([down, up]);
  g.strokeStyle = '#1d2330'; g.lineWidth = 3; g.beginPath(); g.moveTo(...down); g.lineTo(...up); g.stroke(); down = null; }};
const near = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]) <= 18;
window.__verdict = () => {{
  const ok = window.__strokes.some(([a, b]) => (near(a, R) && near(b, B)) || (near(a, B) && near(b, R)));
  return [ok, `${{window.__strokes.length}} stroke(s)`];
}};"""
    body = '<div class="card"><h3>Sketch</h3><canvas id="cv" width="640" height="360"></canvas></div>'
    return SuiteTask("connect", "drag", "Draw a straight line from the red dot to the blue dot.",
                     _page("Sketch", body, script))


TASKS = {"signup": signup, "note": note, "mixer": mixer, "archive": archive, "connect": connect}


def build(name: str, seed: int, folder: Path) -> tuple[SuiteTask, str]:
    """Write the task's page for this seed; return it and its file:// URL."""
    task = TASKS[name](random.Random(f"{name}-{seed}"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-{seed}.html"
    path.write_text(task.html, encoding="utf-8")
    return task, path.resolve().as_uri()
