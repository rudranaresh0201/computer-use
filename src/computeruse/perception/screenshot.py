"""
Screenshot capture + the resize/coordinate-scaling step that
docs/research/anthropic-computer-use.md flags as the easiest thing to get
subtly wrong: a Brain reasons over a (possibly resized) image and returns
coordinates in *that* image's space, not the real screen's. Every consumer
of a click coordinate must go through `to_real_coords` before acting on it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mss
from PIL import Image

# Conservative default: matches the smaller of Anthropic's documented model
# limits (1568px long edge for older models; newer models allow up to 2576px).
# Using the conservative value means we never have to know which model is on
# the other end of the Brain interface.
DEFAULT_MAX_LONG_EDGE = 1568


@dataclass
class Screenshot:
    real_size: tuple[int, int]  # actual screen resolution
    scaled_size: tuple[int, int]  # resolution of the image actually sent out
    scale_x: float  # multiply a scaled-space x by this to get real-space x
    scale_y: float
    png_bytes: bytes
    path: Optional[Path] = None

    def to_real_coords(self, x: int, y: int) -> tuple[int, int]:
        real_x = round(x * self.scale_x)
        real_y = round(y * self.scale_y)
        real_w, real_h = self.real_size
        real_x = max(0, min(real_x, real_w - 1))
        real_y = max(0, min(real_y, real_h - 1))
        return real_x, real_y


def capture(
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    save_path: Optional[Path] = None,
    monitor_index: int = 1,
) -> Screenshot:
    """Capture the real screen and return it resized to fit `max_long_edge`.

    `monitor_index=1` is mss's convention for "the primary monitor"
    (index 0 is a virtual bounding box of all monitors combined).
    """
    with mss.MSS() as sct:
        monitor = sct.monitors[monitor_index]
        raw = sct.grab(monitor)
    image = Image.frombytes("RGB", raw.size, raw.rgb)

    real_w, real_h = image.size
    long_edge = max(real_w, real_h)

    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        scaled_w = max(1, round(real_w * scale))
        scaled_h = max(1, round(real_h * scale))
        scaled_image = image.resize((scaled_w, scaled_h), Image.LANCZOS)
    else:
        scaled_w, scaled_h = real_w, real_h
        scaled_image = image

    buf = io.BytesIO()
    scaled_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(png_bytes)

    return Screenshot(
        real_size=(real_w, real_h),
        scaled_size=(scaled_w, scaled_h),
        scale_x=real_w / scaled_w,
        scale_y=real_h / scaled_h,
        png_bytes=png_bytes,
        path=save_path,
    )
