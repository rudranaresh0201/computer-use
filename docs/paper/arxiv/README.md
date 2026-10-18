# The paper, in two versions

| file | what it is | length |
|---|---|---|
| `main.tex` / `main.pdf` | full version for arXiv: grounder, Tests A and B, gates, agent, failure log, appendices | 14 pages |
| `workshop.tex` / `workshop.pdf` | workshop version: the tree-first result, the decomposition, the gate | 3 pages + references (room for one more page) |

Both use the same `refs.bib` (every entry checked against its arXiv or venue page), the same figures
and the same numbers. Nothing is typed by hand: `scripts/paper_numbers.py` regenerates
`numbers.json` and `figures/` from the raw per-example results.

Red `[PENDING: ...]` marks experiments not yet run: the second seed's evaluation (E1), task-level
results for arms A to D, Test C (real targets the tree cannot see), and the demo trace.

Build (Tectonic, no TeX install needed):

```
tectonic -X compile main.tex
tectonic -X compile workshop.tex
```

Style rules for edits: no dashes as punctuation (write ranges as "6 to 10"), plain statements, every
claim tied to a number in `numbers.json` or a dated, commit-pinned entry of `docs/journal.md`.
