"""The LLM-over-tree arm's text handling (eval/tree_llm.py); the model call itself is not tested here."""

from computeruse.eval import tree_llm
from computeruse.web.browser import parse_snapshot

SNAPSHOT = """- generic [ref=e1] [box=0,0,800,600]:
  - banner [ref=e2] [box=0,0,800,50]:
    - button [ref=e6] [cursor=pointer] [box=700,10,30,30]
  - main [ref=e7] [box=0,50,800,550]:
    - heading "Privacy" [level=2] [ref=e8] [box=10,60,100,20]
    - button "Reset" [ref=e9] [cursor=pointer] [box=10,90,60,30]
    - paragraph [ref=e10] [box=10,130,300,20]: Settings for privacy.
    - link "Docs" [ref=f1e11] [box=10,160,40,20]:
      - /url: "#docs\""""


def test_the_compact_tree_keeps_structure_and_targets_and_drops_noise():
    tree = tree_llm.compact_tree(SNAPSHOT)
    assert 'heading "Privacy" [level=2] [ref=e8]' in tree and 'button "Reset" [ref=e9]' in tree
    assert "button [ref=e6]" in tree  # the nameless icon is there, but says nothing about itself
    assert "box=" not in tree and "/url" not in tree and "paragraph" not in tree and "generic" not in tree


def test_answers_resolve_to_elements_or_to_none():
    elements = parse_snapshot(SNAPSHOT)
    assert tree_llm.parse_answer("e9", elements).name == "Reset"
    assert tree_llm.parse_answer("The answer is f1e11.", elements).name == "Docs"
    assert tree_llm.parse_answer("NONE", elements) is None
    assert tree_llm.parse_answer("e999", elements) is None  # a ref that isn't in the tree
    assert tree_llm.parse_answer("", elements) is None
