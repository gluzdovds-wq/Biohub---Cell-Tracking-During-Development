# Hypotheses and experiment log

This file is the competition source of truth. Update it before launching an experiment and after every CV/LB result.

## Decision gates

- Primary CV: leave-one-embryo-out (LOEO); report each embryo/movie and pooled official metric.
- Promotion: positive paired delta on honest held-out embryos, no schema/scoring failures, acceptable runtime, and a plausible mechanism.
- Public LB is supporting evidence, not the promotion gate.
- A run with metric exploitation, hidden-test branching, train/test cache leakage, or unofficial node matching is rejected.

## Active hypotheses

| ID | Hypothesis | Controlled change | Evidence to collect | Status |
|---|---|---|---|---|
| H-001 | Learned temporal features materially outperform pure nearest-neighbor association. | EXP-003 vs EXP-001, with each pipeline otherwise intact. | LB score, node/edge counts, runtime. | queued |
| H-002 | Calibrating detection count near `estimated_number_of_nodes` improves adjusted edge Jaccard without hurting raw edge Jaccard. | Threshold sweep within one fixed model/linker. | raw/adjusted edge Jaccard, node ratio, paired movie deltas. | queued |
| H-003 | Independent-seed logit blending reduces detection/association variance and beats downstream-only repairs. | EXP-004 vs EXP-003. | honest LOEO delta and LB delta. | queued |
| H-004 | Association errors, especially wrong-partner choices after sudden motion, dominate missed detections. | Error decomposition on held-out movies. | wrong partner / missing link / missed endpoint counts. | queued |
| H-005 | Global graph constraints help only when candidate probabilities contain ambiguity not already resolved by assignment. | ILP on/off with identical detections and candidate graph. | paired official score, changed-edge audit. | queued |
| H-006 | Conservative division prediction is preferable because division prevalence and metric weight are low. | divisions off vs high-confidence-only. | division TP/FP/FN and total score delta. | queued |
| H-007 | Gap repair helps only in detection-recall-limited regimes and otherwise adds false structure. | gap repair off/on with independent image evidence. | recovered GT edges, added FP, node penalty. | queued |
| H-008 | Embryo-held-out CV is directionally reliable while crop-random CV is optimistically leaked. | LOEO vs random-crop split. | CV gap and rank correlation to LB variants. | queued |
| H-009 | Consensus pseudo-labels from diverse open trackers can replace missing dense edge supervision. | Cellpose/DoG detections + Ultrack/Trackastra/rule-based short-track agreement. | precision of consensus edges on sparse GT and LOEO gain. | queued |
| H-010 | A temporal affinity/motion field handles abrupt displacement better than centroid-distance extrapolation. | Predict local displacement/center-of-motion from adjacent 3D frames. | wrong-partner reduction, especially for correct targets 5–7 µm away. | queued |
| H-011 | Reverse-time edge evidence becomes useful when combined with dual-seed features before assignment. | Harmonic forward/reverse fusion at weight 0.30 on the fixed dual-seed pipeline. | EXP-005 vs EXP-004 LB and changed-edge/node audit. | artifact complete; LB pending |
| H-012 | Wider division geometry can add true branches safely when mutual-NN, future divergence, and DeepCenter vetoes all agree. | EXP-006 vs EXP-005; only division thresholds/guarded geometry differ materially. | division counts, graph diff and LB delta; source run is associated with `0.923`. | submitted; LB pending |
| H-013 | D4 TTA helps association when all views score the same physical detections, avoiding node-set misalignment. | Average node-transformer edge logits across eight inverse-aligned XY views; keep detection threshold, assignment and repair fixed. | Changed-edge audit, runtime multiplier, LOEO/LB delta against the calibrated graph. | EXP-007 running |
| H-014 | A three-model detector ensemble with preprocessing diversity provides a useful independent graph arm even if its standalone linker is weaker. | Bright, top-hat-v1 and top-hat-v2 3D U-Nets with flip TTA; appearance-aware physical Hungarian and conservative graph repair. | Artifact audit, exact-coordinate diversity vs EXP-006/007, source/LB evidence, pseudo-label agreement precision. | EXP-008 running |

## Runs

| Experiment | Date | Parent/source | Method / one controlled purpose | CV | Public LB | Runtime | State | Decision / notes |
|---|---|---|---|---:|---:|---:|---|---|
| EXP-001 | 2026-08-22 | `inversion/cell-tracking-getting-started-w-nearest-neighbor` | CPU sanity: percentile detection + Hungarian nearest-neighbor links. | — | **0.143**, submission `55686043` | — | complete v1 | Valid end-to-end floor; 7,067 nodes, 4,629 edges, 11,696 visible rows. |
| EXP-002 | 2026-08-22 | `isakatsuyoshi/biohub-rule-based-baseline` | Classical multi-scale blob detector + physical-distance Hungarian + 1-frame gap closure. | — | **0.826**, submission `55686045` | — | complete v1 | `+0.683` over NN floor, but `-0.095` below current rank-100; valuable as an orthogonal pseudo-label teacher. |
| EXP-003 | 2026-08-22 | `yusuketogashi/clean-approach-lightweight-local-cv-no-hack` | Clean single-seed TemporalUNet3D + node transformer + ILP + conservative repair. | mixed/in-sample diagnostics only; not promotion-grade | pending, submission `55686657` | ~35 min visible run + fixed-8 CV | submitted v1 | Visible output: 120,246 nodes, 115,957 edges, 236,203 rows; schema/dataset/id guards valid; public weights trained on all labelled movies. |
| EXP-004 | 2026-08-22 | `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` | Dual-seed aligned logit blend + transformer + ILP + DeepCenter-confirmed gaps. | public weights; not promotion-grade | pending, submission `55686487` | ~24 min visible run | submitted v1 | Visible output: 119,039 nodes, 114,863 edges, 233,902 rows; schema valid. |
| EXP-005 | 2026-08-22 | `yunusgmsoy/lb-0-920-biohub-cell-tracking-v17` | Dual-seed pipeline plus harmonic forward/reverse association, DeepCenter gap/division veto and strict guards. | public validator uses labelled movies; not promotion-grade | public parent `0.920`; our artifact not submitted | ~37 min | complete v1 | 122,032 nodes, 117,594 edges, 67 divisions; audit PASS. SHA-256 exactly matches the source `0.920` artifact: `9507eccb…6425`. |
| EXP-006 | 2026-08-22 | `yunusgmsoy/kimi-notebook-v17` | EXP-005 harmonic core with safe-division radii `12/15/10 µm`, mutual-NN, divergence and DeepCenter veto. | public validator uses labelled movies; not promotion-grade | pending, submission `55687578`; source run `0.923` | ~40 min | submitted v1 | 122,266 nodes, 118,156 edges, 455 divisions; audit PASS. Exact `BIOHUB_*` env diff is only three division radii; SHA-256 matches source artifact: `5c852379…1f4d`. |
| EXP-007 | 2026-08-22 | `ericwang03/biohub-daily-probe-lane-5` | Single TemporalUNet3D with D4 detection TTA and equal-weight eight-view shared-node association-logit TTA; calibrated motion and conservative divisions fixed. | public weights; no honest held-out score | no attributable source score; do not infer `0.934` from timestamp proximity | running | Kaggle v1 running | Clean source audit; exact public fork, private run, no submission slot spent. Intended evidence is artifact validity/diversity first, LB only after a controlled promotion decision. |
| EXP-008 | 2026-08-22 | `navazshfathi/best-score` | Three 3D U-Nets with bright/top-hat preprocessing diversity, flip-quartet TTA, appearance-aware Hungarian, line-fit smoothing, safe divisions and snap-only gaps. | public weights; no honest held-out score | source is ordered below EXP-006 by Kaggle `scoreDescending`, exact score not exposed | running | Kaggle v1 running | Independent detection family; clean audit found no train/GT access, fake nodes or out-of-volume exploit. Primary value is diversity/pseudo-label evidence unless standalone score is proven competitive. |

EXP-007 from `ericwang03/biohub-daily-probe-lane-5` tests H-013 with equal-weight eight-view D4 edge TTA on shared detections. The public source does not claim a measured LB gain; timestamp proximity to a leaderboard entry is explicitly treated as insufficient evidence.

## Cross-prediction diversity (visible four movies)

- EXP-003 vs EXP-004: exact centroid-set Jaccard `0.414665`; exact coordinate-edge Jaccard `0.317917`.
- EXP-002 vs EXP-003: node Jaccard `0.013752`; edge Jaccard `0.003088`.
- EXP-002 vs EXP-004: node Jaccard `0.014303`; edge Jaccard `0.003137`.
- EXP-004 vs EXP-005: node Jaccard `0.792003`; edge Jaccard `0.763670`; divisions `297 -> 67`.
- EXP-005 vs EXP-006: node Jaccard `0.975946`; edge Jaccard `0.966517`; divisions `67 -> 455`; only `1,726/2,288` edges are left/right-only.

Interpretation: dual-seed inference changes a material fraction of the learned graph, while the classical branch is nearly orthogonal. Do not union graphs directly because the node-count penalty would dominate; use agreement as high-precision pseudo-labels or train a selector/ranker.

## Run record template

Copy this block for every new run:

```text
Experiment:
Date / git commit:
Parent/control:
Hypothesis:
Only intended change:
Data split and leakage audit:
Metric implementation/version:
Config/checkpoint hashes:
Per-movie results:
Pooled official score:
Node ratio / edge TP-FP-FN / division TP-FP-FN:
Runtime and hardware:
Kaggle kernel/version/submission ID:
Public LB:
Unexpected observations:
Decision (promote / hold / kill):
Next experiment:
```
