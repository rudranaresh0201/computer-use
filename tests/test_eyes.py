"""eyes.py's pure parts: shrinking the screenshot and mapping the grounder's answer back."""

from computeruse.eyes import fit, to_pixels


def test_fit_shrinks_to_the_long_edge_and_never_enlarges():
    assert fit((1568, 882), 1024) == (1024, 576)
    assert fit((882, 1568), 1024) == (576, 1024)
    assert fit((800, 600), 1024) == (800, 600)


def test_a_normalized_answer_lands_on_the_full_screenshot_whatever_size_the_model_saw():
    # the model answers in [0,1000) of whatever it was shown, so shrinking does not move the point
    assert to_pixels((500, 500), (1280, 720)) == (640, 360)
    assert to_pixels((250, 100), (1568, 882)) == (392, 88)
    assert to_pixels((1000, 1000), (1280, 720)) == (1279, 719)  # clamped inside
