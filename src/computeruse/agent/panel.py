"""
The panel: a live page that shows the loop as it runs -- for watching, and for recording the demo.

    computeruse agent "<goal>" --site excalidraw --panel      -> http://127.0.0.1:8766

What it shows is what the research is about: each screenshot the brain decided on, a ring where the
action landed, and for every action WHICH CHANNEL placed it (TREE: a named element; EYES: our
grounder; VISION: the brain's own x, y), the router's reason, refusals, the brain's one-line
thought, and at the end the brain's claim next to the checker's verdict.

It is a sink for Session/Agent events (`events(name, payload)`), served as Server-Sent Events. A
panel opened late replays what it missed: every event so far, but only the latest screenshot.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

PORT = 8766


class Panel:
    def __init__(self, port: int = PORT, host: str = "127.0.0.1") -> None:
        self.history: list[tuple[str, dict]] = []
        self.screen: Optional[dict] = None
        self.clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_address[1]}"

    def start(self) -> "Panel":
        threading.Thread(target=self.server.serve_forever, daemon=True, name="panel").start()
        return self

    def stop(self) -> None:
        with self._lock:
            for q in self.clients:
                q.put(None)
        self.server.shutdown()

    def __call__(self, name: str, payload: dict) -> None:
        """The events sink: Session(events=panel) / Agent emits here."""
        with self._lock:
            if name == "screen":
                self.screen = payload
            else:
                self.history.append((name, payload))
            for q in self.clients:
                q.put((name, payload))

    def _subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for item in self.history:
                q.put(item)
            if self.screen:
                q.put(("screen", self.screen))
            self.clients.append(q)
        return q

    def _handler(self):
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/events":
                    return self._stream()
                if self.path in ("/", "/index.html"):
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = panel._subscribe()
                try:
                    while True:
                        try:
                            item = q.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                            continue
                        if item is None:
                            return
                        name, payload = item
                        self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    with panel._lock:
                        if q in panel.clients:
                            panel.clients.remove(q)

            def log_message(self, *args) -> None:
                pass

        return Handler


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Agent Loop</title>
<style>
:root { --bg:#0d1016; --card:#151a23; --line:#262d3b; --text:#e6e9ef; --dim:#8a93a6;
        --tree:#3ccf91; --eyes:#b58cff; --vision:#f2b84b; --bad:#ff6b6b; --brain:#6aa8ff; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 ui-monospace, "Cascadia Code", Consolas, monospace; }
header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:center; flex-wrap:wrap }
#goal { font-size:16px; font-weight:600; flex:1; min-width:260px }
.who { color:var(--dim); font-size:12px } .who b { color:var(--text); font-weight:500 }
.count { padding:3px 9px; border-radius:99px; font-size:12px; border:1px solid var(--line) }
main { display:grid; grid-template-columns: minmax(0,1.6fr) minmax(320px,1fr); gap:16px; padding:16px 20px }
@media (max-width: 900px) { main { grid-template-columns: 1fr } }
#stage { position:relative; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; align-self:start }
#shot { width:100%; display:block; min-height:200px }
.ring { position:absolute; width:46px; height:46px; margin:-23px 0 0 -23px; border-radius:50%; border:3px solid;
  pointer-events:none; animation:pop .9s ease-out }
.ring.look { border-style:dashed; opacity:.85 }
.tag { position:absolute; transform:translate(28px,-10px); font-size:12px; padding:2px 7px; border-radius:6px; color:#0d1016; white-space:nowrap }
@keyframes pop { from { transform:scale(2.2); opacity:0 } to { transform:scale(1); opacity:1 } }
#log { display:flex; flex-direction:column; gap:8px; max-height:calc(100vh - 120px); overflow:auto }
.turn { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px }
.thought { color:var(--brain) } .thought::before { content:"brain  "; color:var(--dim) }
.act { margin-top:6px; display:flex; gap:8px; align-items:baseline; flex-wrap:wrap }
.badge { font-size:11px; font-weight:700; padding:2px 7px; border-radius:5px; color:#0d1016; letter-spacing:.04em }
.b-tree { background:var(--tree) } .b-eyes { background:var(--eyes) } .b-vision { background:var(--vision) }
.b-refused { background:var(--bad) } .b-api { background:#5d6b84; color:#fff } .b-check { background:#9aa6bb }
.why { color:var(--dim); font-size:12px; margin-top:3px }
#verdict { display:none; margin:0 20px 20px; padding:16px 18px; border-radius:10px; border:2px solid; font-size:15px }
#verdict.pass { display:block; border-color:var(--tree) } #verdict.fail { display:block; border-color:var(--bad) }
#verdict .big { font-size:22px; font-weight:700 }
</style></head><body>
<header><div id="goal">waiting for the agent...</div>
 <div class="who"><b>brain</b> <span id="brain">-</span> (decides) · <b>loop, router, eyes, checker</b> ours</div>
 <span class="count" style="color:var(--tree)">TREE <b id="n-tree">0</b></span>
 <span class="count" style="color:var(--eyes)">EYES <b id="n-eyes">0</b></span>
 <span class="count" style="color:var(--vision)">VISION <b id="n-vision">0</b></span>
 <span class="count" style="color:var(--bad)">REFUSED <b id="n-refused">0</b></span></header>
<main><div id="stage"><img id="shot" alt=""></div><div id="log"></div></main>
<div id="verdict"></div>
<script>
const $ = id => document.getElementById(id), log = $("log"), stage = $("stage"), shot = $("shot");
const counts = {tree:0, eyes:0, vision:0, refused:0};
const color = {tree:"var(--tree)", eyes:"var(--eyes)", vision:"var(--vision)"};
let current = null, held = null, holdUntil = 0;
const bump = k => { counts[k]++; $("n-" + k).textContent = counts[k]; };
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function card() { current = document.createElement("div"); current.className = "turn"; log.appendChild(current);
  log.scrollTop = log.scrollHeight; return current; }
function ring(fx, fy, ch, label, look) {
  const r = document.createElement("div"); r.className = "ring" + (look ? " look" : "");
  r.style.left = (fx * 100) + "%"; r.style.top = (fy * 100) + "%"; r.style.borderColor = color[ch] || "#fff";
  const t = document.createElement("div"); t.className = "tag"; t.style.background = color[ch] || "#fff";
  t.textContent = label; r.appendChild(t); stage.appendChild(r);
}
function showScreen(src) { shot.src = src; stage.querySelectorAll(".ring").forEach(r => r.remove()); }
const es = new EventSource("/events");
es.addEventListener("started", e => { const p = JSON.parse(e.data); if (p.goal) $("goal").textContent = p.goal; });
es.addEventListener("run", e => { const p = JSON.parse(e.data); $("goal").textContent = p.goal; $("brain").textContent = p.brain + " · arm " + p.arm; });
es.addEventListener("screen", e => { const src = JSON.parse(e.data).src;  // keep the ring visible a moment first
  const wait = holdUntil - Date.now(); if (wait > 0) { held = src; setTimeout(() => { if (held) { showScreen(held); held = null; } }, wait); } else showScreen(src); });
es.addEventListener("thought", e => { card().innerHTML = `<div class="thought">${esc(JSON.parse(e.data).text)}</div>`; });
es.addEventListener("plan", e => { const p = JSON.parse(e.data); (current || card()).insertAdjacentHTML("beforeend",
  `<div class="act"><span class="badge b-check">PLAN</span>${p.steps.map((s, i) => esc((i + 1) + ". " + s)).join("<br>")}</div>`); });
es.addEventListener("route", e => { const p = JSON.parse(e.data); (current || card()).insertAdjacentHTML("beforeend",
  `<div class="act"><span class="badge b-${p.channel}">${p.channel.toUpperCase()}</span>${esc(p.intent)}</div><div class="why">${esc(p.why)}</div>`); });
es.addEventListener("refused", e => { const p = JSON.parse(e.data); bump("refused"); (current || card()).insertAdjacentHTML("beforeend",
  `<div class="act"><span class="badge b-refused">REFUSED</span>${esc(p.intent)}</div><div class="why">${esc(p.reason)}</div>`); });
es.addEventListener("locate", e => { const p = JSON.parse(e.data); ring(p.x, p.y, "eyes", "eyes: " + p.what + " (" + p.seconds + "s)", true); });
es.addEventListener("step", e => { const p = JSON.parse(e.data); const ch = p.channel; if (ch in color) bump(ch);  // api (keys, typing) is not a grounding channel
  if (p.x != null && shot.naturalWidth) ring(p.x / shot.naturalWidth, p.y / shot.naturalHeight, ch, p.label);
  holdUntil = Date.now() + 1100; });
es.addEventListener("receipt", e => { const p = JSON.parse(e.data); (current || card()).insertAdjacentHTML("beforeend",
  `<div class="act"><span class="badge b-check">CHECK ${p.step}</span>${esc(p.verdict)}: ${esc(p.summary)}</div>`); });
es.addEventListener("finished", e => { const p = JSON.parse(e.data), v = $("verdict");
  v.className = p.passed ? "pass" : "fail";
  v.innerHTML = `<div class="big">${p.passed ? "PASSED" : "NOT PASSED"}</div>
   <div>brain says: ${esc(p.answer || "(no claim)")}</div>
   <div>checker (reads the app, not the brain): ${esc(p.checker || "-")}</div>
   <div class="why">${p.actions} actions · ${p.seconds}s · status ${esc(p.status)}</div>`; });
</script></body></html>"""
