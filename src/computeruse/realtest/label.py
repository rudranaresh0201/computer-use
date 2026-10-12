"""
Tests C and D, step 2: `computeruse label` -- a local page for drawing the
ground truth by hand.

For each capture, a person boxes the things they could click and writes how
they would ask for each one ("pause the video"). Those boxes are the truth
for Test C -- from a person, not from UIA and not from a model -- and the
denominator for Test D: of everything a person can click, how much does
UIA expose?

Runs on 127.0.0.1 only; every change is saved at once to <capture>.labels.json.
"""

from __future__ import annotations

import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from computeruse.realtest.capture import OUT_DIR

_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Label clickables</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#f6f7f9;color:#1f2328;display:flex;height:100vh}
#left{flex:1;overflow:auto;padding:12px}#right{width:360px;padding:12px;border-left:1px solid #ddd;overflow:auto;background:#fff}
#wrap{position:relative;display:inline-block;cursor:crosshair}#img{display:block;max-width:none;user-select:none}
.box{position:absolute;border:2px solid #e11d48;background:#e11d4815}.box span{position:absolute;top:-18px;left:0;background:#e11d48;color:#fff;font-size:11px;padding:0 4px;border-radius:3px}
#draft{position:absolute;border:2px dashed #2563eb;display:none}
li{margin:6px 0}li button{margin-left:6px}input{width:100%;box-sizing:border-box;padding:6px;font:inherit}
h3{margin:4px 0 8px}.muted{color:#667085;font-size:12px}select{width:100%;padding:4px}
</style></head><body>
<div id="left"><div id="wrap"><img id="img"><div id="draft"></div></div></div>
<div id="right">
<h3>Label what a person can click</h3>
<div class="muted">Drag a box around anything clickable -- buttons, icons, tabs, sliders, menu items,
things drawn in custom controls. Then type how you'd ask someone to click it, in your own words
("pause the video", "the gear icon"). Aim for ~10 per screenshot, including the ones that look custom.</div>
<p><select id="pick"></select></p>
<p><button id="prev">&larr; prev</button> <button id="next">next &rarr;</button> <span id="count" class="muted"></span></p>
<div id="ask" style="display:none"><b>How would you ask for this?</b><input id="instr" placeholder="e.g. pause the video  (Enter to save, Esc to cancel)"></div>
<ol id="list"></ol>
</div>
<script>
let caps=[], cur=null, labels=[], pending=null, start=null;
const img=document.getElementById('img'), wrap=document.getElementById('wrap'), draft=document.getElementById('draft');
async function load(){caps=await (await fetch('/api/captures')).json();
  const pick=document.getElementById('pick');pick.innerHTML=caps.map(c=>`<option value="${c.id}">${c.id} (${c.labels} labels)</option>`).join('');
  pick.onchange=()=>open(pick.value); if(caps.length) open(caps[0].id); else document.getElementById('list').innerHTML='<li>No captures yet -- run: computeruse capture &lt;app&gt;</li>';}
async function open(id){cur=id;document.getElementById('pick').value=id;img.src='/img/'+id+'.png';
  labels=await (await fetch('/api/labels/'+id)).json();render();}
function scale(){return img.naturalWidth/img.clientWidth;}
function render(){wrap.querySelectorAll('.box').forEach(b=>b.remove());
  labels.forEach((l,i)=>{const s=scale(),[x1,y1,x2,y2]=l.box,d=document.createElement('div');d.className='box';
    Object.assign(d.style,{left:x1/s+'px',top:y1/s+'px',width:(x2-x1)/s+'px',height:(y2-y1)/s+'px'});d.innerHTML='<span>'+(i+1)+'</span>';wrap.appendChild(d);});
  document.getElementById('list').innerHTML=labels.map((l,i)=>`<li>${l.instruction.replace(/</g,'&lt;')}<button onclick="del(${i})">x</button></li>`).join('');
  document.getElementById('count').textContent=labels.length+' labels here';}
async function save(){await fetch('/api/labels/'+cur,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(labels)});
  const c=caps.find(c=>c.id===cur);if(c){c.labels=labels.length;document.querySelector(`option[value="${cur}"]`).textContent=`${cur} (${labels.length} labels)`;}}
function del(i){labels.splice(i,1);render();save();}
function pos(e){const r=img.getBoundingClientRect(),s=scale();return[Math.round((e.clientX-r.left)*s),Math.round((e.clientY-r.top)*s)];}
wrap.onmousedown=e=>{if(pending)return;start=pos(e);e.preventDefault();};
window.onmousemove=e=>{if(!start)return;const p=pos(e),s=scale();Object.assign(draft.style,{display:'block',
  left:Math.min(start[0],p[0])/s+'px',top:Math.min(start[1],p[1])/s+'px',width:Math.abs(p[0]-start[0])/s+'px',height:Math.abs(p[1]-start[1])/s+'px'});};
window.onmouseup=e=>{if(!start)return;const p=pos(e);const box=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];start=null;
  if(box[2]-box[0]<4||box[3]-box[1]<4){draft.style.display='none';return;}
  pending=box;document.getElementById('ask').style.display='block';const i=document.getElementById('instr');i.value='';i.focus();};
document.getElementById('instr').onkeydown=e=>{if(e.key==='Enter'&&e.target.value.trim()){labels.push({box:pending,instruction:e.target.value.trim()});done();save();render();}
  else if(e.key==='Escape')done();};
function done(){pending=null;draft.style.display='none';document.getElementById('ask').style.display='none';}
document.getElementById('prev').onclick=()=>{const i=caps.findIndex(c=>c.id===cur);if(i>0)open(caps[i-1].id);};
document.getElementById('next').onclick=()=>{const i=caps.findIndex(c=>c.id===cur);if(i<caps.length-1)open(caps[i+1].id);};
img.onload=render;window.onresize=render;load();
</script></body></html>"""


def make_handler(folder: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _capture_id(self, prefix: str, suffix: str = "") -> str | None:
            name = self.path[len(prefix):]
            name = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
            return name if _ID.match(name) and (folder / f"{name}.uia.json").exists() else None

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/captures":
                caps = []
                for meta in sorted(folder.glob("*.uia.json")):
                    cid = meta.name[: -len(".uia.json")]
                    labels = folder / f"{cid}.labels.json"
                    caps.append({"id": cid, "labels": len(json.loads(labels.read_text())) if labels.exists() else 0})
                self._send(200, json.dumps(caps).encode(), "application/json")
            elif self.path.startswith("/img/") and (cid := self._capture_id("/img/", ".png")):
                self._send(200, (folder / f"{cid}.png").read_bytes(), "image/png")
            elif self.path.startswith("/api/labels/") and (cid := self._capture_id("/api/labels/")):
                path = folder / f"{cid}.labels.json"
                self._send(200, path.read_bytes() if path.exists() else b"[]", "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            cid = self._capture_id("/api/labels/") if self.path.startswith("/api/labels/") else None
            if cid is None:
                self._send(404, b"not found", "text/plain")
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"[]")
            clean = [{"box": [int(v) for v in item["box"]], "instruction": str(item["instruction"])[:200]}
                     for item in body if isinstance(item, dict) and len(item.get("box", [])) == 4 and item.get("instruction")]
            (folder / f"{cid}.labels.json").write_text(json.dumps(clean, indent=1), encoding="utf-8")
            self._send(200, b"ok", "text/plain")

    return Handler


def serve(folder: Path = OUT_DIR, port: int = 8765, open_browser: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(folder))
    url = f"http://127.0.0.1:{port}/"
    print(f"labelling page: {url}   (captures in {folder.resolve()}; Ctrl+C to stop -- every label is already saved)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
