"""
Our agent loop, in LangGraph. The brain decides; everything else is ours:

    perceive ──► decide ──► ground ──► act ──► perceive ...
                   │           │ (can't place it: the reason goes back to decide)
                   ├─ plan ────┴──► Session.set_plan ──► decide
                   ├─ check ──────► Session.check (a verifier needs evidence) ──► decide
                   └─ done ───────► every plan step verified? no: back to decide, told which
                                    yes: the task's own checker (if any) gives the verdict ──► END

    perceive  Session.observe(): the screenshot + the accessibility tree, numbered
    decide    the brain (brains.py) -> one JSON intent; a bad reply is a no-op it is told about
    ground    the router (router.py): the paper's rules pick tree / eyes / vision, or refuse
    act       Session.act(): approval gate, the action, did the screen change
    verify    Session.check() for plan steps; tasks.py-style checker for the whole task

Stops: the brain's verified done; MAX_TURNS brain calls; MAX_FAILURES failures in a row (refused,
failed, or no change); the person's stop. A claim of "done" with steps unverified is refused up to
MAX_EARLY_DONE times, then the episode ends as "claimed, not verified".

Every turn is written to the episode (agent/episode.py), so a run can be scored and replayed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .brains import Brain, View
from .intent import Intent
from .router import Grounded, Router, Unroutable

MAX_TURNS = 40
MAX_FAILURES = 4
MAX_EARLY_DONE = 2
Checker = Callable[[], tuple[bool, str]]  # the task's own verdict, read from the world, not from the agent


class LoopState(TypedDict, total=False):
    observation: str
    note: str
    intent: Optional[Intent]
    grounded: Optional[Grounded]
    turns: int
    failures: int
    early_done: int
    status: str  # "" while running; then done | unverified | gave_up | stopped | error


@dataclass
class Turn:
    n: int
    thought: str
    intent: str  # Intent.brief()
    channel: str = ""  # api | tree | eyes | vision | (empty: no action)
    why: str = ""
    result: str = ""
    ok: bool = True
    brain_seconds: float = 0.0
    eyes_seconds: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class Episode:
    goal: str
    target: str
    arm: str
    brain: str
    turns: list[Turn] = field(default_factory=list)
    status: str = ""
    claim: str = ""
    checker: Optional[tuple[bool, str]] = None
    seconds: float = 0.0
    steps_verified: int = 0
    plan: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """The task's own checker decides when there is one; otherwise every plan step must be verified."""
        if self.checker is not None:
            return bool(self.checker[0])
        return self.status == "done"

    @property
    def false_success(self) -> bool:
        """The agent said done, and the world disagrees: the number verimcp exists for."""
        return bool(self.claim) and self.status in ("done", "unverified") and not self.passed

    def channels(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.turns:
            if t.channel:
                out[t.channel] = out.get(t.channel, 0) + 1
        return out

    def tokens(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.turns:
            for k, v in t.tokens.items():
                out[k] = out.get(k, 0) + v
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "target": self.target, "arm": self.arm, "brain": self.brain, "status": self.status,
            "passed": self.passed, "false_success": self.false_success, "claim": self.claim,
            "checker": list(self.checker) if self.checker else None, "seconds": round(self.seconds, 1),
            "plan": self.plan, "steps_verified": self.steps_verified, "turns": len(self.turns),
            "actions": sum(1 for t in self.turns if t.channel), "channels": self.channels(), "tokens": self.tokens(),
            "trace": [t.__dict__ for t in self.turns],
        }


@dataclass
class Agent:
    session: Any  # computeruse.session.Session, with the target open
    brain: Brain
    router: Router
    goal: str
    checker: Optional[Checker] = None
    should_stop: Callable[[], bool] = lambda: False
    max_turns: int = MAX_TURNS
    episode: Optional[Episode] = None
    _screenshot: bytes = b""
    _size: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        self.episode = self.episode or Episode(self.goal, self.session.target or "", self.router.arm, self.brain.name)
        self.history: list[str] = []
        self.graph = self._build()

    # --- the graph ----------------------------------------------------------------------------

    def _build(self):
        g = StateGraph(LoopState)
        g.add_node("perceive", self.perceive)
        g.add_node("decide", self.decide)
        g.add_node("ground", self.ground)
        g.add_node("act", self.act)
        g.add_node("verify", self.verify)
        g.add_edge(START, "perceive")
        g.add_conditional_edges("perceive", lambda s: END if s.get("status") else "decide")
        g.add_conditional_edges("decide", self._after_decide,
                                {"ground": "ground", "verify": "verify", "decide": "decide", END: END})
        g.add_conditional_edges("ground", lambda s: "act" if s.get("grounded") else self._again(s),
                                {"act": "act", "decide": "decide", END: END})
        g.add_conditional_edges("act", lambda s: END if s.get("status") else "perceive",
                                {"perceive": "perceive", END: END})
        g.add_conditional_edges("verify", lambda s: END if s.get("status") else "decide",
                                {"decide": "decide", END: END})
        return g.compile()

    def run(self) -> Episode:
        start = time.time()
        final = self.graph.invoke({"turns": 0, "failures": 0, "early_done": 0, "note": "", "status": ""},
                                  {"recursion_limit": self.max_turns * 6 + 20})
        ep = self.episode
        ep.status = final.get("status") or "gave_up"
        ep.seconds = time.time() - start
        ep.plan = list(self.session.plan)
        ep.steps_verified = len(self.session.done)
        if self.checker is not None and ep.status != "error":
            try:
                ep.checker = self.checker()
            except Exception as e:  # noqa: BLE001 -- a checker that can't look is a failed check
                ep.checker = (False, f"the checker could not look ({e.__class__.__name__}: {str(e)[:120]})")
        self.session.events("finished", {"status": ep.status, "answer": ep.claim, "passed": ep.passed,
                                         "checker": ep.checker[1] if ep.checker else "", "brain": ep.brain,
                                         "arm": ep.arm, "actions": sum(1 for t in ep.turns if t.channel),
                                         "seconds": round(ep.seconds, 1)})
        return ep

    # --- nodes --------------------------------------------------------------------------------

    def perceive(self, state: LoopState) -> dict:
        if self.should_stop():
            return {"status": "stopped"}
        text, jpeg = self.session.observe()
        if not jpeg:  # the session refused: nothing open, the app lost the front, the browser went away
            return {"status": "error", "note": text}
        self._screenshot = jpeg
        self._size = self.session.shot.size
        return {"observation": text}

    def decide(self, state: LoopState) -> dict:
        turns = state.get("turns", 0) + 1
        if self.should_stop():
            return {"status": "stopped", "turns": turns}
        if turns > self.max_turns:
            return {"status": "gave_up", "turns": turns, "note": f"stopped after {self.max_turns} turns"}
        view = View(self.goal, self.router.arm, list(self.session.plan), set(self.session.done), self.history,
                    state.get("note", ""), state.get("observation", ""), self._screenshot, self._size)
        d = self.brain.decide(view)
        turn = Turn(len(self.episode.turns) + 1, d.intent.thought if d.intent else "",
                    d.intent.brief() if d.intent else "(no action)", brain_seconds=round(d.seconds, 1),
                    tokens=d.tokens)
        self.episode.turns.append(turn)
        if d.intent is None:
            turn.ok, turn.result = False, d.error
            self.history.append(f"{turn.n}. (reply not usable) -- {d.error[:120]}")
            failures = state.get("failures", 0) + 1
            if failures >= MAX_FAILURES:
                return {"status": "gave_up", "turns": turns, "failures": failures, "intent": None}
            return {"intent": None, "note": d.error, "turns": turns, "failures": failures}
        if turn.thought:
            self.session.events("thought", {"text": turn.thought})
        return {"intent": d.intent, "turns": turns, "grounded": None}

    def _after_decide(self, state: LoopState) -> str:
        if state.get("status"):
            return END
        intent = state.get("intent")
        if intent is None:
            return "decide"
        if intent.action in ("plan", "check", "done"):
            return "verify"
        return "ground"

    def ground(self, state: LoopState) -> dict:
        intent = state["intent"]
        turn = self.episode.turns[-1]
        try:
            g = self.router.route(intent, self.session.surface.boxes(), self._size[0])
        except Unroutable as e:
            turn.ok, turn.result = False, str(e)
            self.history.append(f"{turn.n}. {turn.intent} -- refused: {e}")
            return {"grounded": None, "note": f"Refused: {e}", "failures": state.get("failures", 0) + 1}
        turn.channel, turn.why, turn.eyes_seconds = g.channel, g.why, round(g.eyes_seconds, 1)
        if g.channel.startswith("eyes") and g.x is not None:
            self.session.located[(g.x, g.y)] = intent.target or g.why  # the Session reports it as "eyes"
        return {"grounded": g}

    def _again(self, state: LoopState) -> str:
        return END if state.get("failures", 0) >= MAX_FAILURES else "decide"

    def act(self, state: LoopState) -> dict:
        intent, g = state["intent"], state["grounded"]
        turn = self.episode.turns[-1]
        result = self.session.act(g.action, element=g.element, x=g.x, y=g.y, to_x=g.to_x, to_y=g.to_y,
                                  text=intent.text, keys=intent.keys, direction=intent.direction,
                                  amount=intent.amount, seconds=intent.seconds)
        bad = ("failed" in result or "did NOT change" in result or "did not approve" in result
               or "Observe first" in result or "Refused" in result or result.startswith(("No element", "Unknown")))
        turn.result, turn.ok = result, not bad
        self.history.append(f"{turn.n}. {turn.intent} via {g.channel} -> {result[:160]}")
        failures = state.get("failures", 0) + 1 if bad else 0
        status = "gave_up" if failures >= MAX_FAILURES else ""
        return {"note": result, "failures": failures, "status": status}

    def verify(self, state: LoopState) -> dict:
        intent = state["intent"]
        turn = self.episode.turns[-1]
        if intent.action == "plan":
            if not self.session.surface:
                note = "Nothing is open."
            else:
                note = self.session.set_plan(intent.steps)
            turn.result = note
            self.history.append(f"{turn.n}. made a plan of {len(intent.steps)} steps")
            return {"note": note}
        if intent.action == "check":
            note = self.session.check(intent.step or 0, intent.evidence.get("kind", ""), intent.evidence.get("value", ""))
            turn.result, turn.ok = note, note.startswith(f"Step {intent.step} verified")
            self.history.append(f"{turn.n}. {turn.intent} -> {note[:160]}")
            return {"note": note}
        # done
        open_steps = [i for i in range(1, len(self.session.plan) + 1) if i not in self.session.done]
        self.episode.claim = intent.claim
        if open_steps:
            early = state.get("early_done", 0) + 1
            note = (f"Not done: plan steps {open_steps} are not verified. Check each with evidence, "
                    "or say done explaining what could not be done.")
            turn.result, turn.ok = note, False
            self.history.append(f"{turn.n}. said done -- refused: steps {open_steps} unverified")
            if early > MAX_EARLY_DONE:
                return {"status": "unverified", "early_done": early}
            return {"note": note, "early_done": early}
        turn.result = "accepted: every plan step is verified"
        return {"status": "done"}
