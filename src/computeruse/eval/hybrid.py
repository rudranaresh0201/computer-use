"""
The hybrid ablation arm (docs/research/gui-grounding-hypothesis.md): "UIA
when available, else fine-tuned grounder" -- H2's actual claim, per the
hypothesis doc's own framing ("the other three [arms] exist partly to
justify it").

Deliberately just a combiner over two already-computed result sets, not a
third thing that re-runs predictions -- UIA-only and the grounder arm are
each scored independently (uia_only.run_arm, vlm_grounder.evaluate_arm),
and this module only decides, per example, whose hit/miss counts. That
keeps this pure and GPU-free: it's testable without a model or dataset,
using fixture UiaArmResults, well before a fine-tuned checkpoint exists to
supply the fallback side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from computeruse.eval.report import EvalRecord

if TYPE_CHECKING:
    # Import for typing only. uia_only -> dataset.collector -> perception.uia
    # -> pywinauto, which is Windows-only and fails to import on Linux --
    # and this module is deliberately pure (see the docstring above): it
    # combines two already-computed result sets and touches no OS API. A
    # runtime import here would make the combiner unimportable on the GPU
    # pod that runs the other three arms. `from __future__ import
    # annotations` (above) keeps the annotation below a string, so nothing
    # evaluates this at runtime.
    from computeruse.eval.uia_only import UiaArmResult


def combine(uia_results: list[UiaArmResult], grounder_hits: dict[str, bool]) -> list[EvalRecord]:
    """`grounder_hits` maps example_id -> hit, from whichever grounder arm
    is standing in as the fallback (fine-tuned once it exists; zero-shot
    works too for an early sanity check of the combining logic itself).

    Raises KeyError, does not silently skip or default to False, if an
    example UIA couldn't resolve is missing from grounder_hits -- both arms
    must have been run over the same example set, and a silent default
    would hide a real bug (e.g. the two arms scored different splits).
    """
    records = []
    for r in uia_results:
        hit = r.hit if r.available else grounder_hits[r.example_id]
        records.append(
            EvalRecord(
                example_id=r.example_id,
                arm="hybrid",
                app=r.app,
                app_richness=r.app_richness,
                split=r.split,
                hit=hit,
            )
        )
    return records
