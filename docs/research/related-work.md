# Related work: what is already known, and what is ours

Checked 2026-10-13 by web search (abstracts, project pages, reviews -- not every full paper; items marked
*unverified* need the PDF read before the paper cites them). Rule: no "first" claim survives without this file.

## Already done by others -- we must cite, not claim

| work | what it did | what it means for us |
|---|---|---|
| **OS-Atlas** (Wu et al., 2024, arXiv 2410.23218) | large cross-platform grounding corpus; desktop data (~12K screenshots, mostly Windows) synthesised in simulated environments using the accessibility tree | "use accessibility-tree labels to train a vision grounder" is **not new**. Ours is the same idea at tiny, free scale (3,163 examples, one Kaggle T4). Also authors of ScreenSpot-v2 |
| **WinClick / WinSpot** (Hui et al., ACL 2025 short; arXiv 2503.04730) | Windows-specific grounding model (Phi-3-vision, continual pretraining, LLM-aligned data) and a 5,000-pair Windows benchmark | a Windows grounding benchmark exists; scoring on WinSpot would make our numbers comparable (*data access unverified*) |
| **GroundCUA / GroundNext** (Feizi et al., ICLR 2026; arXiv 2511.07332) | 56K human-annotated desktop screenshots, 87 apps, 3.56M elements | the strong-data end of desktop grounding; we are the cheap-data end |
| **UFO2** (Microsoft, arXiv 2504.14603) | Windows AgentOS: **hybrid control detection = UIA + OmniParser-v2 vision, merged by box IoU** | the closest system to our hybrid. It merges both sources; it does not (as far as checked) measure *when* the tree's answer should not be trusted, or show that a naive merge can be worse than vision |
| **OSWorld** (Xie et al., 2024) / **OS-Harm** (2025, arXiv 2506.14866) | task-level comparison of observation types: a11y tree, screenshot, both, set-of-marks; tree+screenshot best | the "tree vs vision vs both" question has been asked **at task level with frontier models**. Ours is at the **grounding** level, per condition, with truth independent of both |
| **ScaleCUA** (arXiv 2509.15221) | notes trees can be present but wrong: obscured elements, wrong coordinates right after launch | supports our "the tree can be confidently wrong" finding; cite it |
| **Aria-UI**, **UGround**, **SeeClick**, **ShowUI** | pure-vision grounders; SeeClick introduced ScreenSpot; ShowUI-2B shares our base model | context for our ScreenSpot number (61.9 vs ShowUI-2B 75.1 with ~80x more data, *cross-paper, unverified setup match*) |

## What looks unclaimed -- our narrow contributions (to re-check before writing)

1. **A controlled, per-condition grounding comparison** of tree vs vision vs hybrids, with ground truth from
   neither method (sandbox pages, 5 conditions) and the same question re-asked on real Windows trees damaged
   on purpose (Test B) and on genuinely blind real targets (Test C, pending).
2. **The naive hybrid can be worse than vision alone**: "tree first, else vision" lost 9.5-12 points to vision
   whenever the tree was incomplete, because name matching picks a wrong element confidently. We did not find
   this measured in UFO2 or the observation-ablation papers.
3. **Abstention as the gate**: an LLM reading the tree, allowed to say NONE, decides when to hand over to vision
   (`llm_guarded`, pre-registered): 83.2 vs vision 66.4 vs tree-LLM 55.6. We found no published decision rule
   for when to trust the tree.
4. **How blind UIA is in the wild** (Test D, pending): a per-app coverage table measured against boxes a person
   drew. Papers state trees have gaps; we found no per-app measurement.

Positioning sentence: *"Hybrids of accessibility trees and vision are common (UFO2, OSWorld setups); we measure
when each source should be trusted, show the naive merge fails, and give a cheap gate that works."*

## To read in full before submission
OS-Atlas section on desktop data; UFO2 hybrid-detection ablation (does it report fallback failures?); OSWorld's own
observation ablation; WinSpot data access; any 2026 work on GUI-agent abstention or grounding confidence.

Sources: arxiv.org/abs/2410.23218, arxiv.org/abs/2503.04730, aclanthology.org/2025.acl-short.85,
arxiv.org/abs/2511.07332, arxiv.org/abs/2504.14603, arxiv.org/abs/2404.07972, arxiv.org/abs/2506.14866,
arxiv.org/abs/2509.15221, arxiv.org/abs/2412.16256, arxiv.org/abs/2410.05243.
