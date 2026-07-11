"""
UIA is inherently environment-dependent (depends on whatever window happens
to be focused when tests run), so this suite only checks the contract we
promise callers: never raise, always return a list, and each element that
does come back has a well-formed rect. See docs/decisions/0001-architecture.md
for why an empty list is an expected, valid outcome (not every app exposes
a usable UIA tree).
"""

from computeruse.perception.uia import UIElement, get_foreground_window_tree


def test_get_foreground_window_tree_never_raises_and_returns_a_list():
    elements = get_foreground_window_tree()
    assert isinstance(elements, list)


def test_elements_have_well_formed_rects():
    elements = get_foreground_window_tree(max_elements=50)
    for el in elements:
        assert isinstance(el, UIElement)
        left, top, right, bottom = el.rect
        assert right >= left
        assert bottom >= top


def test_element_center_is_within_its_rect():
    el = UIElement(name="test", control_type="Button", rect=(10, 20, 110, 60))
    cx, cy = el.center
    assert 10 <= cx <= 110
    assert 20 <= cy <= 60
