"""The router: the paper's rules as code. One test per rule, each named after the result it encodes."""

import pytest

from computeruse.agent.intent import BadIntent, Intent, parse_intent
from computeruse.agent.router import Router, Unroutable, snap, tree_exact

BOXES = {
    "e1": ("Save", "button", (100, 10, 140, 30)),
    "e2": ("", "button", (150, 10, 170, 30)),  # an unnamed icon (Test A C2)
    "e3": ("Archive", "button", (100, 100, 160, 120)),
    "e4": ("Archive", "button", (100, 200, 160, 220)),  # same name twice (C5)
    "e5": ("Email", "textbox", (300, 50, 500, 70)),
}


def eyes_at(point, seconds=2.0):
    asked = []

    def eyes(description):
        asked.append(description)
        return point, seconds

    eyes.asked = asked
    return eyes


# --- intents ------------------------------------------------------------------------------------

def test_an_intent_is_read_from_json_even_inside_prose_or_a_code_fence():
    i = parse_intent('Sure.\n```json\n{"thought": "go", "action": "click", "target": "the save icon"}\n```')
    assert (i.action, i.target, i.thought) == ("click", "the save icon", "go")


@pytest.mark.parametrize("reply, why", [
    ("no json here", "no JSON"),
    ('{"action": "fly"}', "not one of"),
    ('{"action": "plan", "steps": []}', "plan needs steps"),
    ('{"action": "check", "step": 1}', "evidence.kind"),
    ('{"action": "type"}', "type needs text"),
])
def test_a_bad_reply_says_why_so_the_brain_can_fix_it(reply, why):
    with pytest.raises(BadIntent, match=why):
        parse_intent(reply)


# --- the paper arm ------------------------------------------------------------------------------

def test_a_named_pick_goes_through_the_tree_llm_named():
    g = Router("paper", eyes_at((1, 1))).route(Intent("click", element="e1"), BOXES, 1280)
    assert (g.channel, g.element) == ("tree", "e1")


def test_a_description_goes_to_our_grounder_the_vision_half_of_llm_named():
    eyes = eyes_at((160, 20))
    g = Router("paper", eyes).route(Intent("click", target="the floppy-disk save icon"), BOXES, 1280)
    assert (g.channel, g.x, g.y, g.eyes_seconds) == ("eyes", 160, 20, 2.0)
    assert eyes.asked == ["the floppy-disk save icon"]


def test_a_nameless_pick_is_never_clicked_through_the_tree_it_needs_a_description():
    with pytest.raises(Unroutable, match="has no name: describe it"):
        Router("paper", eyes_at((1, 1))).route(Intent("click", element="e2"), BOXES, 1280)
    g = Router("paper", eyes_at((160, 20))).route(Intent("click", element="e2", target="the gear icon"), BOXES, 1280)
    assert g.channel == "eyes" and g.element is None


def test_bare_coordinates_for_a_click_are_refused_say_what_not_where():
    with pytest.raises(Unroutable, match="say WHAT"):
        Router("paper", eyes_at((1, 1))).route(Intent("click", x=5, y=5), BOXES, 1280)


def test_when_the_eyes_have_no_answer_the_brains_own_point_is_the_last_resort():
    g = Router("paper", eyes_at(None)).route(Intent("click", target="the gear", x=7, y=8), BOXES, 1280)
    assert (g.channel, g.x, g.y) == ("vision", 7, 8)
    with pytest.raises(Unroutable, match="found nothing"):
        Router("paper_blind", eyes_at(None)).route(Intent("click", target="the gear"), BOXES, 1280)


def test_keys_and_typing_into_the_focused_field_need_no_grounding():
    r = Router("paper", None)
    assert r.route(Intent("key", keys="ctrl+s"), BOXES, 1280).channel == "api"
    assert r.route(Intent("type", text="hi"), BOXES, 1280).channel == "api"


def test_drag_can_be_grounded_at_both_ends_so_a_blind_brain_can_draw():
    eyes = iter([((10, 10), 1.0), ((90, 90), 1.5)])
    g = Router("paper_blind", lambda d: next(eyes)).route(
        Intent("drag", target="the red dot", to_target="the blue dot"), BOXES, 1280)
    assert (g.x, g.y, g.to_x, g.to_y, g.channel, g.eyes_seconds) == (10, 10, 90, 90, "eyes", 2.5)


def test_snap_is_off_by_default_because_dev_chose_llm_named_not_llm_snap():
    g = Router("paper", eyes_at((105, 15))).route(Intent("click", target="save"), BOXES, 1280)
    assert (g.x, g.y) == (105, 15)
    g = Router("paper", eyes_at((105, 15)), use_snap=True).route(Intent("click", target="save"), BOXES, 1280)
    assert (g.x, g.y) == (120, 20)


# --- the comparison arms ------------------------------------------------------------------------

def test_vision_arm_uses_only_the_brains_coordinates():
    g = Router("vision").route(Intent("click", x=3, y=4), BOXES, 1280)
    assert (g.channel, g.x, g.y) == ("vision", 3, 4)
    with pytest.raises(Unroutable, match="x, y only"):
        Router("vision").route(Intent("click", element="e1"), BOXES, 1280)


def test_tree_arm_takes_an_exact_unique_name_and_refuses_ambiguity():
    r = Router("tree")
    assert r.route(Intent("click", target="click Save"), BOXES, 1280).element == "e1"
    with pytest.raises(Unroutable):
        r.route(Intent("click", target="Archive"), BOXES, 1280)  # two of them: C5
    with pytest.raises(Unroutable):
        r.route(Intent("click", target="the gear icon"), BOXES, 1280)  # not in the tree: C2/C4


def test_both_arm_is_the_old_mcp_behaviour_named_element_or_the_brains_xy():
    r = Router("both")
    assert r.route(Intent("click", element="e1"), BOXES, 1280).channel == "tree"
    assert r.route(Intent("click", x=1, y=2), BOXES, 1280).channel == "vision"
    with pytest.raises(Unroutable):
        r.route(Intent("click", target="the gear icon"), BOXES, 1280)


def test_helpers_match_the_eval_rules():
    assert tree_exact("press the Save button", BOXES) == "e1"
    assert tree_exact("Archive", BOXES) is None
    assert snap((1000, 1000), BOXES, 1280) == (1000, 1000)  # nothing near: left alone
