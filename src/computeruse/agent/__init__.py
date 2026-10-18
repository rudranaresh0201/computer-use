"""
Our agent: the loop, the router and the brains (docs/where-we-are.md, "the design flip").

    perceive -> decide -> ground -> act -> verify      (loop.py, LangGraph)
                  |         |
               brain     router                      (brains.py: Claude Code / Groq / scripted)
          says WHAT    decides WHERE                 (router.py: the paper's rules)

The brain never chooses the channel. It names an element or describes a target; the router
applies llm_named (Tests A/B) and our grounder; the Session acts and keeps the receipts.
"""
