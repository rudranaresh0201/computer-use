"""
The eyes over the network: our grounder runs where there is a GPU (a free Kaggle T4, the setup that
scored 46.5) and the laptop asks it, one screenshot + one description at a time.

    on the GPU     computeruse eyes-serve --port 8000          (notebooks/eyes_server_kaggle.ipynb)
    on the laptop  setx CU_EYES_URL https://<tunnel>  ;  setx CU_EYES_TOKEN <the token it printed>
                   computeruse bench --brain groq --arm paper_blind --eyes remote

Why: on the laptop's CPU the model only fits squeezed to int8, and squeezed it hit 1 of 12 targets
and wrote "Window 4" instead of points (G11, runs/bench/p1_log.txt). Over the network the answer is
the real model's, at ~1-2 s on a T4 plus the round trip.

The wire is plain JSON over HTTP, so nothing on either side needs more than the standard library:

    POST /locate   {"image": <base64 JPEG/PNG>, "instruction": "the floppy-disk save icon"}
                -> {"point": [x, y] | null, "raw": "(412,96)", "seconds": 1.3}   (point in THAT image)
    GET  /health -> {"ok": true, "device": "cuda", "long_edge": 1568}

Every request carries `Authorization: Bearer <token>`: a tunnel URL is public, the token is what
keeps strangers off a GPU session.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

Locate = Callable[[bytes, str], tuple[Optional[tuple[int, int]], str, float]]  # image -> point, raw, seconds


def make_handler(locate: Locate, token: str, info: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self) -> bool:
            given = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if hmac.compare_digest(given.encode(), token.encode()):
                return True
            self._reply(401, {"error": "wrong or missing token"})
            return False

        def do_GET(self) -> None:  # noqa: N802 -- the http.server API
            if not self._allowed():
                return
            if self.path == "/health":
                self._reply(200, {"ok": True, **info})
            else:
                self._reply(404, {"error": "GET /health or POST /locate"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._allowed():
                return
            if self.path != "/locate":
                self._reply(404, {"error": "POST /locate"})
                return
            try:
                ask = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                image, instruction = base64.b64decode(ask["image"]), str(ask["instruction"])
            except (ValueError, KeyError, TypeError) as e:
                self._reply(400, {"error": f"bad request: {e}"})
                return
            try:
                point, raw, seconds = locate(image, instruction)
            except Exception as e:  # noqa: BLE001 -- one bad image must not take the server down
                self._reply(500, {"error": f"{e.__class__.__name__}: {str(e)[:200]}"})
                return
            self._reply(200, {"point": list(point) if point else None, "raw": raw, "seconds": round(seconds, 2)})

        def log_message(self, fmt: str, *args) -> None:  # one short line per request, not the default spam
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

    return Handler


def grounder_locate(grounder) -> Locate:
    def locate(image: bytes, instruction: str):
        found = grounder.locate(image, instruction)  # Grounder holds a lock: one generation at a time
        return found.point, found.raw, found.seconds
    return locate


def serve(locate: Locate, port: int = 8000, token: Optional[str] = None, info: Optional[dict] = None,
          host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Build the server (not started: call .serve_forever(), or run it in a thread)."""
    token = token or os.environ.get("CU_EYES_TOKEN") or secrets.token_urlsafe(18)
    server = ThreadingHTTPServer((host, port), make_handler(locate, token, info or {}))
    server.token = token  # type: ignore[attr-defined]
    return server


class RemoteEyes:
    """The Session's eyes, answered by a server: (jpeg, description) -> (point, seconds).
    `seconds` is the whole round trip -- what the agent actually waits."""

    def __init__(self, url: str, token: str, timeout: float = 60.0) -> None:
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        self.last_raw = ""

    @classmethod
    def from_env(cls) -> "RemoteEyes":
        url, token = os.environ.get("CU_EYES_URL"), os.environ.get("CU_EYES_TOKEN")
        if not url or not token:
            raise RuntimeError("Set CU_EYES_URL and CU_EYES_TOKEN (the eyes server prints both).")
        return cls(url, token)

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        req = urllib.request.Request(f"{self.url}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json",
                                              "User-Agent": "computeruse-eyes"})  # some tunnels drop bare urllib
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            raise RuntimeError(f"eyes server said HTTP {e.code}: {detail}") from e

    def health(self) -> dict:
        return self._call("GET", "/health")

    def locate(self, image: bytes, instruction: str) -> dict:
        return self._call("POST", "/locate", {"image": base64.b64encode(image).decode(), "instruction": instruction})

    def __call__(self, jpeg: bytes, description: str) -> tuple[Optional[tuple[int, int]], float]:
        start = time.perf_counter()
        try:
            answer = self.locate(jpeg, description)
        except (RuntimeError, OSError) as e:  # a dropped tunnel is "no answer", not a crashed episode
            self.last_raw = f"(eyes unreachable: {e})"
            return None, time.perf_counter() - start
        self.last_raw = answer.get("raw", "")
        point = answer.get("point")
        return (tuple(point) if point else None), time.perf_counter() - start
