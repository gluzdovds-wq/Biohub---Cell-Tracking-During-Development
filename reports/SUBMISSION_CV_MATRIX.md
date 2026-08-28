# Submission ↔ CV/OOF matrix

Updated 2026-08-28: EXP073–076 scored, EXP077 CPU pilot complete, EXP078–082 registered. Historical artifact-version exceptions remain explicit.

## Bottom line

No submitted public-weight model currently has an **exact honest OOF score**. The dominant public TemporalUNet/dual-seed checkpoints were trained on all labelled competition movies. Evaluating those same checkpoints on train data is leakage, even when the notebook calls several volumes “validation”.

Our movie-held-out evidence belongs to separately retrained reciprocal checkpoints and therefore supports mechanisms, not the absolute `0.90–0.93` LB scale. Gradient training excludes the evaluated embryo, but checkpoint/threshold selection used separate movies of that embryo; this is not a fully independent unseen-embryo test:

- registered Hungarian: `0.744130` on held-out `44b6`, `0.595767` on held-out `6bba`, pooled `0.615980`;
- weak learned tie-break minus registered: `+0.000734/+0.000771`;
- registered minus greedy: `+0.164190/+0.129587`;
- embryo gap: `0.148363` versus random five-fold movie-CV standard deviation `0.0271`;
- four-movie public-like bootstrap interval: `0.451683–0.786690` versus private-like 130-movie interval `0.585632–0.645486`.

Thus LB is useful for hidden-runtime validation and rejecting large regressions, but differences of `0.001–0.003` are not evidence of private ordering.

EXP077 CPU pilot is **not an LB submission**. COMPLETE on four full audit movies in 20.30 minutes, with no failures. It compares our local-deformation linker with registered/weak and public-style ILP controls using no-TTA reciprocal weights. Local-flow minus weak control is `0.0` on each embryo and pooled; pooled scores are `0.624991 / 0.624970 / 0.624970 / 0.613500`. No GT divisions occur. These are not exact public-model OOF scores. Saved results contain each movie/embryo, sufficient statistics and paired deltas. Existing audit scores have already been inspected, so neither this pilot nor a later reuse is an untouched final holdout. Bootstrap intervals above are conditional resampling diagnostics, not intervals for the actual hidden leaderboard.

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
| EXP066 / `55761018` | `0.926` | no | related LOEO family only | former leader; unknown unseen-embryo ordering |
| EXP067 / `55761031` | `0.919` | no | no exact continuation-guard OOF | orthogonal graph but lower LB; hedge only |
| EXP068 wrapper / `55761370` | invalid format | no | no min-cost-flow OOF | replaced by EXP068R |
| EXP069 wrapper / `55761371` | invalid format | no | none | replaced by EXP069R |
| EXP068R / `55781325` | `0.884` | no | min-cost-flow arm awaits candidate caches | reject this pipeline; diversity alone did not help |
| EXP069R / `55781326` | `0.926` | no | related LOEO family only | correlated Flex v11; exact submitted-version artifact identity not verified |
| EXP070 / `55781466` | `0.926` | no | related LOEO family only | near-identical to EXP066; byte-identical to latest Flex v17, not independent evidence |
| EXP071 / `55781467` | `0.923` | no | related LOEO family only | chosen association-diverse validation candidate |
| EXP072 / `55781468` | `0.918` | no | related LOEO family only | reject standalone reverse-0.20 variant |

## August 27 additions

- EXP073 / `55808574`: COMPLETE, account **`0.927`**; SDW60 v1. Exact OOF **unavailable**. Detection-fusion change requires re-inference, not only CPU relinking.
- EXP074 / `55808576`: COMPLETE, account **`0.927`**; Anhad v21. Exact OOF **unavailable**. Different harmonic/division graph, but shared public-weight family.
- EXP075 / `55808638`: COMPLETE, account **`0.927`**; historical Evgen v11. Exact OOF **unavailable**. Version-correction probe; immutable local output audit unavailable. The CLI had silently downloaded v12 when passed `/11`.
- EXP076 / `55808636`: COMPLETE, account `0.923`; SEC25 v1. Exact OOF **unavailable**. Secondary edge-weight hypothesis, lower-priority exploratory test.

All four account scores now independently match the corresponding source-version scores, but are not private estimates. See `submission_batch_20260827.json` for receipts and the historical-version exception.

## August 28 additions

- EXP078 / `55836059`: PENDING; SDW70 v1, author `0.928`; exact OOF **unavailable**. Detection weight 0.70, same source family as EXP073.
- EXP079 / `55836064`: PENDING; Flex v22, author `0.928`; exact OOF **unavailable**. Epoch-2 DeepCenter and safe-division veto; not a new independent architecture.
- EXP080 / `55836067`: PENDING; SDW75 v1, no displayed author score; exact OOF **unavailable**. Detection weight 0.75, exploratory.
- EXP081 / `55836071`: PENDING; VEL10 v1, no displayed author score; exact OOF **unavailable**. Full velocity extrapolation on detection blend 0.475; exploratory.
- EXP082 / `55836074`: PENDING; MTL8 v1, author `0.923`; exact OOF **unavailable**. Stricter short-track filtering; sensitivity control, not evidence of superior private robustness.

All five are source-attributed full-code public reproductions, with frozen notebooks and unique audited output hashes. Author scores above are not account results. Daily quota: five used, zero available. Canonical manifest: `submission_batch_20260828.json`.

## What must happen next

EXP063/064 are the prepared shared-detection linker comparison: registered, weak/heavy learned, two ILP policies, greedy/physical arms, and cached candidates for min-cost-flow, Kalman and particle-filter follow-ups. Their GPU configuration must wait for quota refresh at `2026-08-29 00:00 UTC`. EXP077 now demonstrates a cheaper CPU/no-TTA route; bounded CPU shards need not wait, but change the inference configuration and are not exact submitted-model OOF. No extended run was launched on August 28.

For final selection, trust mechanisms with the same-sign paired embryo OOF first. Among models without exact OOF, retain several structurally different candidates rather than interpreting a `0.001` public lead as private certainty.
