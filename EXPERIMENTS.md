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

## Runs

| Experiment | Date | Parent/source | Method / one controlled purpose | CV | Public LB | Runtime | State | Decision / notes |
|---|---|---|---|---:|---:|---:|---|---|
| EXP-001 | 2026-08-22 | `inversion/cell-tracking-getting-started-w-nearest-neighbor` | CPU sanity: percentile detection + Hungarian nearest-neighbor links. | — | — | — | running v1 | Kaggle kernel `dmitriigluzdov/biohub-exp001-nearest-neighbor-sanity`. |
| EXP-002 | 2026-08-22 | `isakatsuyoshi/biohub-rule-based-baseline` | Classical multi-scale blob detector + physical-distance Hungarian + 1-frame gap closure. | — | public parent claimed ~0.857 | — | running v1 | Kaggle kernel `dmitriigluzdov/biohub-exp002-rule-based-classical`. |
| EXP-003 | 2026-08-22 | `yusuketogashi/clean-approach-lightweight-local-cv-no-hack` | Clean single-seed TemporalUNet3D + node transformer + ILP + conservative repair. | mixed/in-sample diagnostics only; not promotion-grade | public parent ~0.908 | — | running v1 | Kaggle kernel `dmitriigluzdov/biohub-exp003-clean-single-seed`; public weights trained on all labelled movies. |
| EXP-004 | 2026-08-22 | `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` | Dual-seed aligned logit blend + transformer + ILP + DeepCenter-confirmed gaps. | public weights; not promotion-grade | public family ~0.913 | — | waiting for GPU slot | Upload is ready; Kaggle batch-GPU concurrency limit is 2. |

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
