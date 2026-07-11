"""
Real screenshot capture (no mocking) — same safe, non-destructive operation
verified manually in the Phase 0 PoC. Only the resize/scale-back math is
what's actually under test here.
"""

from computeruse.perception.screenshot import Screenshot, capture


def test_capture_returns_real_and_scaled_sizes():
    shot = capture()
    assert shot.real_size[0] > 0 and shot.real_size[1] > 0
    assert shot.scaled_size[0] > 0 and shot.scaled_size[1] > 0
    assert len(shot.png_bytes) > 0


def test_capture_respects_max_long_edge():
    shot = capture(max_long_edge=800)
    assert max(shot.scaled_size) <= 800


def test_capture_does_not_upscale_small_targets():
    # A max_long_edge bigger than the real screen must never enlarge the image.
    shot = capture(max_long_edge=100_000)
    assert shot.scaled_size == shot.real_size
    assert shot.scale_x == 1.0
    assert shot.scale_y == 1.0


def test_to_real_coords_scales_and_clamps():
    shot = Screenshot(
        real_size=(1920, 1080),
        scaled_size=(960, 540),
        scale_x=2.0,
        scale_y=2.0,
        png_bytes=b"",
    )

    assert shot.to_real_coords(100, 50) == (200, 100)
    # out-of-bounds scaled coordinates must clamp into the real screen, not overflow it
    assert shot.to_real_coords(10_000, 10_000) == (1919, 1079)
    assert shot.to_real_coords(0, 0) == (0, 0)
