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

## Runs

| Experiment | Date | Parent/source | Method / one controlled purpose | CV | Public LB | Runtime | State | Decision / notes |
|---|---|---|---|---:|---:|---:|---|---|
| EXP-001 | 2026-08-22 | `inversion/cell-tracking-getting-started-w-nearest-neighbor` | CPU sanity: percentile detection + Hungarian nearest-neighbor links. | — | **0.143**, submission `55686043` | — | complete v1 | Valid end-to-end floor; 7,067 nodes, 4,629 edges, 11,696 visible rows. |
| EXP-002 | 2026-08-22 | `isakatsuyoshi/biohub-rule-based-baseline` | Classical multi-scale blob detector + physical-distance Hungarian + 1-frame gap closure. | — | pending, submission `55686045` | — | submitted v1 | Visible output: 129,515 nodes, 121,799 edges, 251,314 rows; schema valid; dense graph may score slowly. |
| EXP-003 | 2026-08-22 | `yusuketogashi/clean-approach-lightweight-local-cv-no-hack` | Clean single-seed TemporalUNet3D + node transformer + ILP + conservative repair. | mixed/in-sample diagnostics only; not promotion-grade | pending, submission `55686657` | ~35 min visible run + fixed-8 CV | submitted v1 | Visible output: 120,246 nodes, 115,957 edges, 236,203 rows; schema/dataset/id guards valid; public weights trained on all labelled movies. |
| EXP-004 | 2026-08-22 | `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` | Dual-seed aligned logit blend + transformer + ILP + DeepCenter-confirmed gaps. | public weights; not promotion-grade | pending, submission `55686487` | ~24 min visible run | submitted v1 | Visible output: 119,039 nodes, 114,863 edges, 233,902 rows; schema valid. |

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
