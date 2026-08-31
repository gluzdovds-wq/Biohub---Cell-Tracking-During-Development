# Submission ↔ CV/OOF matrix

Updated 2026-08-31 after incident audit: EXP088–092 scored; EXP093 pending;
EXP094–097 failed invalid-format hidden reruns and have no LB evidence.
Historical artifact-version exceptions remain explicit.

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

## Honest CV evidence table

The large reciprocal audit was **not CPU inference**: EXP050/051 used the two
embryo-separated checkpoints on T4 x2, after which aggregation and bootstrap
analysis are cheap on CPU. The separate EXP077 run below is the Kaggle CPU
pilot the project log refers to.

### Reciprocal embryo-held-out audit: 183 complete movies

| Frozen policy | Held-out `44b6` (63) | Held-out `6bba` (120) | Pooled (183) | Paired interpretation |
|---|---:|---:|---:|---|
| greedy learned edges | `0.579940` | `0.466180` | `0.481198` | weak reference |
| registered Hungarian | `0.744130` | `0.595767` | `0.615980` | beats greedy by `+0.164190/+0.129587`, pooled `+0.134782` |
| registered + 10% learned tie-break | `0.744864` | `0.596538` | `0.616747` | beats registered by `+0.000734/+0.000771`, pooled `+0.000767` |

The paired bootstrap interval for weak tie-break minus registered is
`0.000186–0.001357` with `P(delta>0)=0.9948`. Registered minus greedy is much
larger and unambiguous: `0.114758–0.155750`. These are mechanism comparisons on
separately retrained single-seed checkpoints, not exact OOF scores for the
public dual-seed/EMA/D4 submissions.

### EXP077 Kaggle CPU pilot: four full movies, no TTA

| Frozen policy | `44b6` (2) | `6bba` (2) | Pooled (4) | Result |
|---|---:|---:|---:|---|
| registered Hungarian | `0.734515` | `0.590408` | `0.624991` | strongest pooled arm in this tiny pilot |
| registered + weak tie-break | `0.730861` | `0.591441` | `0.624970` | effectively tied; insufficient for a tiny-gain decision |
| local-flow + weak tie-break | `0.730861` | `0.591441` | `0.624970` | exact paired delta `0.0` versus weak on both embryos |
| public-style ILP | `0.725789` | `0.573284` | `0.613500` | lower pooled score in this configuration |

Runtime was `1218.073 s` (20.30 minutes) with zero failed movies. All annotated
and predicted division counts were zero, so the pilot says nothing about
division recall. It proves CPU feasibility and rejects a claimed local-flow
gain; it is not complete OOF and must not be assigned to an LB submission.

### Stability diagnostics

- Registered fold gap: `0.148363`, versus random movie five-fold score standard
  deviation `0.027072`. Random movie folds mix embryo identity and materially
  understate deployment shift.
- Registered pooled movie bootstrap: `0.591559–0.640048`, conditional on the
  two observed embryo domains.
- Four-movie public-like resampling: `0.451683–0.786690`; 130-movie
  private-like resampling: `0.585632–0.645486`. These intervals describe
  finite-movie noise for this retrained model, not the private score scale of
  EXP083–097.

### What can be transferred to final selection

- Trust registered physical motion over greedy linking; the sign is large and
  agrees on both embryos.
- The 10% learned ambiguity tie-break is a small but repeatable improvement in
  the full audit. Preserve it as a mechanism, not as a promised `+0.0008` LB
  gain.
- Do not promote local-flow from current evidence: its paired CPU delta is zero.
- Division ranking remains unresolved because the CPU pilot contains no GT
  divisions.
- EXP083–097 still have exact honest OOF = **unavailable**. Their public
  checkpoints saw labelled competition movies; ranking Stephen, EMA,
  threshold-0.40, division-heavy and fine-tuned-D4 requires fold-specific
  retraining or a genuinely independent labelled embryo.

## Current frontier submission mapping

| Experiment / receipt | Public LB | Exact submitted-model OOF | Closest local evidence | Current role |
|---|---:|---|---|---|
| EXP093 / `55907915` | pending; verified source `0.933` | no | same broad public-weight family; no exact fold retraining | provisional final A if account score reproduces |
| EXP094 / `55908273` | failed: invalid format | no | public-only artifact replay; no hidden inference | invalid; no evidence |
| EXP095 / `55908462` | failed: invalid format | no | public-only artifact replay; no hidden inference | invalid; no evidence |
| EXP096 / `55908629` | failed: invalid format | no | public-only artifact replay; no hidden inference | invalid; no evidence |
| EXP097 / `55908683` | failed: invalid format | no | public-only artifact replay; no hidden inference | invalid; no evidence |
| EXP083 / `55858606` | `0.931` | no | registered/weak mechanism evidence only; not its dual-seed graph | prior final-A control |
| EXP084 / `55858609` | `0.929` | no | same broad family; no exact fold retraining | correlated reserve |
| EXP085 / `55858612` | `0.928` | no | same broad family; no exact fold retraining | correlated reserve |
| EXP086 / `55858614` | `0.928` | no | same broad family; no exact fold retraining | correlated reserve |
| EXP087 / `55859147` | `0.926` | no | built-in `0.9294` is leaky and not CV | reject SDW90 direction |
| EXP088 / `55882197` | `0.926` | no | no exact EMA fold retraining | reject versus EXP083 |
| EXP089 / `55882683` | `0.924` | no | controlled public-LB interpolation only; `0.981219` edge overlap to EMA-1.0 | reject EMA interpolation |
| EXP090 / `55882198` | `0.928` | no | no exact threshold-0.40 fold run; `0.939815` edge overlap to EXP083 | best Aug30 diagnostic, still below frontier |
| EXP091 / `55882203` | `0.926` | no | CPU pilot has zero GT divisions, so it cannot validate this arm | reject division-heavy direction |
| EXP092 / `55882642` | `0.900` | no | no fold-specific fine-tuned-linker/D4 retraining; `0.744633` edge overlap to EXP083 | reject: diversity did not preserve quality |

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

- EXP078 / `55836059`: COMPLETE `0.928`; SDW70 v1; exact OOF **unavailable**. Detection weight 0.70, same source family as EXP073.
- EXP079 / `55836064`: COMPLETE `0.928`; Flex v22; exact OOF **unavailable**. Epoch-2 DeepCenter and safe-division veto; not a new independent architecture.
- EXP080 / `55836067`: COMPLETE `0.928`; SDW75 v1; exact OOF **unavailable**. Detection weight 0.75; its graph is nearly identical to SDW70.
- EXP081 / `55836071`: COMPLETE `0.926`; VEL10 v1; exact OOF **unavailable**. Full velocity extrapolation on detection blend 0.475.
- EXP082 / `55836074`: COMPLETE `0.923`; MTL8 v1; exact OOF **unavailable**. Stricter short-track filtering; sensitivity control.

All five are source-attributed full-code public reproductions, with frozen notebooks and unique audited output hashes. Equal LB scores are strongly correlated evidence: SDW70/SDW75 physical node/edge Jaccard is `0.970384/0.963701`, and SDW70/Flex v22 is `0.861317/0.829558`. Canonical manifest: `submission_batch_20260828.json`.

## August 29 additions

- EXP083 / `55858606`: COMPLETE `0.931`; clean Stephen v1; exact OOF **unavailable**.
- EXP084 / `55858609`: COMPLETE `0.929`; SDW85 v1; exact OOF **unavailable**.
- EXP085 / `55858612`: COMPLETE `0.928`; Evgen v15; exact OOF **unavailable**.
- EXP086 / `55858614`: COMPLETE `0.928`; Anvith v1; exact OOF **unavailable**.
- EXP087 / `55859147`: COMPLETE `0.926`; our controlled SDW90 fork changes only secondary detector weight `0.85→0.90`. Output audit PASS at 119,722 nodes / 115,437 edges / 251 divisions. Exact OOF **unavailable**. Its built-in four-train-movie proxy `0.9294` is leaky because the public weights saw competition train data and did not provide a reliable promotion signal.

EXP083–087 remain in the same public dual-seed/harmonic family. Stephen/Flex physical-node/edge Jaccard is `0.925603/0.910946`; SDW85/SDW75 is `0.951036/0.940384`; SDW90/SDW85 is `0.973106/0.967084`. These overlaps imply a narrow family search, not five independent estimates of private robustness. Daily quota: five used / zero available. Canonical manifest: `submission_batch_20260829.json`.

## August 30 additions

- EXP088 / `55882197`: COMPLETE `0.926`; full four-frame averaged-motion EMA. Exact OOF **unavailable**.
- EXP089 / `55882683`: COMPLETE `0.924`; our controlled half-weight EMA interpolation. Exact OOF **unavailable**; physical edge overlap `0.981219` to EMA-1.0.
- EXP090 / `55882198`: COMPLETE `0.928`; edge-candidate threshold 0.40. Exact OOF **unavailable**; physical edge overlap `0.939815` to EXP083 marks it as correlated sensitivity evidence.
- EXP091 / `55882203`: COMPLETE `0.926`; division-heavy graph with 384 predicted divisions. Exact OOF **unavailable**; physical division overlap `0.490486` to EXP083.
- EXP092 / `55882642`: COMPLETE `0.900`; exact offline fine-tuned-linker + D4 reproduction. Exact OOF **unavailable**; physical edge overlap `0.744633` to EXP083 and the artifact SHA exactly matches its reviewed public parent.

The diverse EXP092 hedge lost `0.031` to EXP083, while the best Aug30 arm remained `0.003` below. This falsifies the idea that graph diversity alone protects the private final.

## August 31 additions

- EXP093 / `55907915`: exact C33 v1 source-attributed anchor, verified source score `0.933`; account score pending.
- EXP094 / `55908273`: failed invalid format; public-only artifact replay.
- EXP095 / `55908462`: failed invalid format; public-only artifact replay.
- EXP096 / `55908629`: failed invalid format; public-only artifact replay.
- EXP097 / `55908683`: failed invalid format; public-only artifact replay.

Their downloaded public outputs keep the exact C33 graph and passed local
audits, but this establishes no LB evidence: the submitted code never consumed
the hidden runtime test set. Full API errors and recovery controls are recorded
in `RECOVERY_20260831.md`.

## What must happen next

EXP063/064 remain the prepared shared-detection linker comparison: registered,
weak/heavy learned, two ILP policies, greedy/physical arms, and cached candidates
for min-cost-flow, Kalman and particle-filter follow-ups. EXP077 demonstrates a
cheaper CPU/no-TTA route, but those runs change the inference configuration and
are not exact OOF for the submitted GPU/TTA artifacts. After EXP093–097 score,
the next compute should be whole-movie paired validation for a small set of
mechanism-distinct candidates rather than another same-family LB sweep.

For final selection, trust mechanisms with the same-sign paired embryo OOF first. Among models without exact OOF, retain several structurally different candidates rather than interpreting a `0.001` public lead as private certainty.
