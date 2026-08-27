# Submission ↔ CV/OOF matrix

Updated 2026-08-27 after the August 26 batch completed and EXP073–076 were registered. The August 27 addendum below corrects historical artifact-version provenance.

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
| EXP065 / `55761017` | `0.924` | no | related LOEO family only | v12 also scores 0.924 for the author; the title's best 0.927 belongs to v11, not runtime instability |
| EXP066 / `55761018` | **`0.926`** | no | related LOEO family only | current leader; high public confidence, unknown unseen-embryo ordering |
| EXP067 / `55761031` | `0.919` | no | no exact continuation-guard OOF | orthogonal graph but lower LB; hedge only |
| EXP068 wrapper / `55761370` | invalid format | no | no min-cost-flow OOF | replaced by EXP068R |
| EXP069 wrapper / `55761371` | invalid format | no | none | replaced by EXP069R |
| EXP068R / `55781325` | `0.884` | no | min-cost-flow arm awaits candidate caches | reject this pipeline; diversity alone did not help |
| EXP069R / `55781326` | `0.926` | no | related LOEO family only | correlated Flex v11; exact submitted-version artifact identity not verified |
| EXP070 / `55781466` | `0.926` | no | related LOEO family only | near-identical to EXP066; byte-identical to latest Flex v17, not independent evidence |
| EXP071 / `55781467` | `0.923` | no | related LOEO family only | chosen association-diverse validation candidate |
| EXP072 / `55781468` | `0.918` | no | related LOEO family only | reject standalone reverse-0.20 variant |

## August 27 additions

- EXP073 / `55808574`: pending; source SDW60 v1 score `0.927`; exact OOF **unavailable**. Detection-fusion change requires re-inference, not only CPU relinking.
- EXP074 / `55808576`: pending; source Anhad v21 score `0.927`; exact OOF **unavailable**. Different harmonic/division graph, but shared public-weight family.
- EXP075 / `55808638`: pending; historical source v11 score `0.927`; exact OOF **unavailable**. Version-correction probe; immutable local output audit unavailable. The CLI had silently downloaded v12 when passed `/11`.
- EXP076 / `55808636`: pending; source SEC25 v1 score `0.923`; exact OOF **unavailable**. Secondary edge-weight hypothesis, lower-priority exploratory test.

For all four: author scores are not account results or private estimates. See `submission_batch_20260827.json` for source/artifact hashes and the historical-version exception.

## What must happen next

EXP063/064 are the prepared exact-detection linker comparison: registered, weak/heavy learned, two ILP policies, greedy/physical arms, and cached candidates for min-cost-flow, Kalman and particle-filter follow-ups. They are compile-clean but cannot run until the account GPU quota refreshes at `2026-08-29 00:00 UTC`; current usage exceeds the 6-hour weekly allowance.

For final selection, trust mechanisms with the same-sign paired embryo OOF first. Among models without exact OOF, retain several structurally different candidates rather than interpreting a `0.001` public lead as private certainty.
