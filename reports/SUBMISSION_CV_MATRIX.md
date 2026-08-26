# Submission ↔ CV/OOF matrix

Updated 2026-08-26 after submissions `55781325–55781468` were registered.

## Bottom line

No submitted public-weight model currently has an **exact honest OOF score**. The dominant public TemporalUNet/dual-seed checkpoints were trained on all labelled competition movies. Evaluating those same checkpoints on train data is leakage, even when the notebook calls several volumes “validation”.

Our honest leave-one-embryo-out evidence belongs to separately retrained reciprocal checkpoints and therefore supports mechanisms, not the absolute `0.90–0.93` LB scale:

- registered Hungarian: `0.744130` on held-out `44b6`, `0.595767` on held-out `6bba`, pooled `0.615980`;
- weak learned tie-break minus registered: `+0.000734/+0.000771`;
- registered minus greedy: `+0.164190/+0.129587`;
- embryo gap: `0.148363` versus random five-fold movie-CV standard deviation `0.0271`;
- four-movie public-like bootstrap interval: `0.451683–0.786690` versus private-like 130-movie interval `0.585632–0.645486`.

Thus LB is useful for hidden-runtime validation and rejecting large regressions, but differences of `0.001–0.003` are not evidence of private ordering.

## Every account submission

| Experiment / submission | Public LB or state | Exact honest OOF | Closest honest evidence | Private-stability interpretation |
|---|---:|---|---|---|
| EXP001 / `55686043` | `0.143` | no | none | sanity only; reject |
| EXP002 / `55686045` | `0.826` | no untouched pre-tuning audit | none | classical floor; reject |
| EXP004 / `55686487` | `0.912` | no | related LOEO detector family only | correlated public baseline; hold |
| EXP003 / `55686657` | `0.908` | no | related LOEO detector family only | single-seed floor; reject |
| EXP006 / `55687578` | `0.919` | no | related LOEO family `0.744/0.596`; not this graph | correlated harmonic control; medium uncertainty |
| EXP014 / `55703183` | invalid format | no | none for coordinate change | wrapper invalid; do not infer quality |
| EXP019 / `55703198` | hidden runtime error | no | none for coordinate change | invalid; do not infer quality |
| EXP054 / `55705721` | `0.905` | no exact-model OOF | registered beats greedy `+0.164/+0.130`, but not harmonic | honest mechanism result did not justify wholesale topology replacement; reject |
| EXP055 / `55706857` | `0.893` | no exact-model OOF | same registered-linker evidence as EXP054 | coordinate addition worsened LB; reject |
| EXP005 / `55719392` | `0.920` | no | related LOEO family `0.744/0.596`; not this graph | clean conservative control; still domain-uncertain |
| EXP008 / `55732259` | `0.917` | no | no reciprocal retraining for its three U-Nets | detector-diverse hedge; lower LB but useful independence |
| EXP007 / `55732491` | `0.900` | no | no exact association-TTA OOF | large regression; reject |
| EXP039 / `55732718` | `0.906` | no | reject-only local delta `−0.00319`, not OOF | checkpoint unstable; reject |
| EXP028 / `55732720` | `0.919` | no | leaky proxy negative (`0.9131` vs `0.9235`) | sign agrees with LB; reject relaxed veto |
| EXP065 / `55761017` | `0.924` | no | related LOEO family only | strong, but source title `0.927` failed to reproduce; medium uncertainty |
| EXP066 / `55761018` | **`0.926`** | no | related LOEO family only | current leader; high public confidence, unknown unseen-embryo ordering |
| EXP067 / `55761031` | `0.919` | no | no exact continuation-guard OOF | orthogonal graph but lower LB; hedge only |
| EXP068 wrapper / `55761370` | invalid format | no | no min-cost-flow OOF | replaced by EXP068R |
| EXP069 wrapper / `55761371` | invalid format | no | none | replaced by EXP069R |
| EXP068R / `55781325` | pending | no | min-cost-flow arm awaits EXP063/064 caches | highly orthogonal, very uncertain; only 5 public divisions |
| EXP069R / `55781326` | pending | no | related LOEO family only | current Flex v11; exact duplicate of EXP070 |
| EXP070 / `55781466` | pending | no | related LOEO family only | duplicate of EXP069R, not an independent stability vote |
| EXP071 / `55781467` | pending | no | related LOEO family only | materially different bidirectional/harmonic hedge |
| EXP072 / `55781468` | pending | no | related LOEO family only | controlled reverse-weight hedge; moderately correlated with EXP066 |

## What must happen next

EXP063/064 are the prepared exact-detection linker comparison: registered, weak/heavy learned, two ILP policies, greedy/physical arms, and cached candidates for min-cost-flow, Kalman and particle-filter follow-ups. They are compile-clean but cannot run until the account GPU quota refreshes at `2026-08-29 00:00 UTC`; current usage exceeds the 6-hour weekly allowance.

For final selection, trust mechanisms with the same-sign paired embryo OOF first. Among models without exact OOF, retain several structurally different candidates rather than interpreting a `0.001` public lead as private certainty.
