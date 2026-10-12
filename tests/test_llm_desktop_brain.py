from computeruse.action.schema import ActionType
from computeruse.brain.chat import ScriptedChat
from computeruse.brain.llm_desktop import LLMDesktopBrain, render_elements
from computeruse.perception.uia import UIElement

CALC = [
    UIElement("Calculator", "Window", (0, 0, 400, 600)),
    UIElement("Display is 0", "Text", (10, 100, 390, 160)),
    UIElement("Seven", "Button", (10, 300, 100, 360)),
    UIElement("Plus", "Button", (300, 300, 390, 360)),
    UIElement("", "Button", (300, 400, 390, 460)),       # an icon with no name
    UIElement("", "Pane", (0, 0, 400, 600)),            # layout, not shown
]


def brain(calls, title="Calculator"):
    return LLMDesktopBrain(ScriptedChat(calls), "calculator", foreground_title=lambda: title)


def test_the_model_sees_buttons_named_text_and_nameless_icons_but_not_layout():
    text, ids = render_elements(CALC)
    assert '[u2] Button "Seven"' in text and '[u1] Text "Display is 0"' in text
    assert "[u4] Button (no name)" in text and "Pane" not in text and "Window" not in text
    assert ids["u2"].center == (55, 330)


def test_click_by_id_becomes_a_click_on_the_elements_center():
    action = brain([("click", {"id": "u2"})]).decide("type 7", None, CALC, [])
    assert action.type == ActionType.LEFT_CLICK and action.coordinate == (55, 330)


def test_an_id_not_on_screen_is_a_noop_the_model_is_told_about():
    b = brain([("click", {"id": "u99"})])
    action = b.decide("x", None, CALC, [])
    assert action.type == ActionType.WAIT and "not in the current list" in b.history[-1]


def test_type_key_and_done():
    b = brain([("type", {"text": "hello"}), ("key", {"keys": "Ctrl+S"}), ("done", {"answer": "42"})])
    assert b.decide("x", None, CALC, []).text == "hello"
    assert b.decide("x", None, CALC, []).text == "ctrl+s"
    final = b.decide("x", None, CALC, [])
    assert final.type == ActionType.DONE and final.text == "42"


def test_focus_leaving_the_app_ends_the_run_before_any_click():
    action = brain([("click", {"id": "u2"})], title="Windows PowerShell").decide("x", None, CALC, [])
    assert action.type == ActionType.DONE and action.text.startswith("STOPPED")
