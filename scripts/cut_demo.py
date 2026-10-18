"""
Cut recorded takes into one short video: segments, each at its own speed, with a caption bar, the
taskbar cropped off, scaled for posting.

    .venv\\Scripts\\python scripts\\cut_demo.py spec.json out.mp4

spec.json: {"inputs": ["take-a.mp4", "take-b.mp4"],
            "segments": [{"in": 0, "from": 16, "to": 27, "speed": 1, "caption": "..."}, ...]}

Times are seconds of the recording; speed 8 plays a minute in 7.5 s. Captions are drawn from text files,
so they need no escaping.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WIDTH, CROP_H = 1280, 1008  # the recording is 1920x1080; the bottom 72 px are the taskbar
FONT = Path(r"C:\Windows\Fonts\segoeuib.ttf")


def main(spec_path: str, out: str) -> None:
    import imageio_ffmpeg

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp(prefix="cut-"))
    shutil.copy(FONT, work / "font.ttf")  # relative paths keep the filter free of Windows drive colons
    parts = []
    for i, seg in enumerate(spec["segments"]):
        (work / f"c{i}.txt").write_text(seg.get("caption", ""), encoding="utf-8")
        speed = float(seg.get("speed", 1))
        chain = (f"[{seg['in']}:v]trim={seg['from']}:{seg['to']},setpts=(PTS-STARTPTS)/{speed},"
                 f"crop=1920:{CROP_H}:0:0,scale={WIDTH}:-2,fps=30")
        if seg.get("caption"):
            chain += (f",drawtext=fontfile=font.ttf:textfile=c{i}.txt:fontsize=30:fontcolor=white:"
                      "box=1:boxcolor=black@0.62:boxborderw=14:x=(w-tw)/2:y=h-th-34")
        parts.append(chain + f"[v{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(parts))) + \
        f"concat=n={len(parts)}:v=1:a=0[out]"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    for f in spec["inputs"]:
        cmd += ["-i", str(Path(f).resolve())]
    cmd += ["-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-preset", "slow", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(Path(out).resolve())]
    subprocess.run(cmd, check=True, cwd=work)
    print(out, f"{Path(out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
