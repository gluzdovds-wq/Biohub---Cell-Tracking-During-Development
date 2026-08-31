# Final-submission selection policy

Updated 2026-08-31. Kaggle permits two final submissions. This document is a
decision policy, not a claim that public-LB scores estimate the private rank.

## Current evidence

- The current scored account leader is EXP083 at `0.931`. EXP093 reproduces a
  source-version-card score of `0.933` and becomes provisional exploit slot A
  only after its account score is populated.
- EXP084–087 score `0.929/0.928/0.928/0.926`. They are variants of the same
  public dual-seed TemporalUNet + harmonic association family, so their similar
  scores are correlated evidence rather than independent robustness evidence.
- The public leaderboard currently scores only four visible test movies. The
  final code rerun uses hidden data, so a three-decimal lead inside this family
  is too small to establish private ordering.
- None of these public-weight submissions has honest exact-model OOF: the
  checkpoints were trained with the labelled competition movies. Train-movie
  validator scores from those notebooks remain leaky proxies.
- Competition discussion advice from the current first-place participant is
  consistent with our validation policy: score complete movies with the
  official metric and movie-level OOF splits; edge-random CV is misleading.

## August 30 diagnostic batch

- EXP088: full four-frame averaged-motion EMA. Strongest ready challenger, but
  still in the broad public family.
- EXP089: our controlled EMA weight `1.0 -> 0.5`. This is a paired
  interpolation, not a new architecture. Its physical edge Jaccard to EMA-1.0
  is `0.981219`.
- EXP090: association candidate threshold `0.48 -> 0.40`. Physical edge
  Jaccard to EXP083 is `0.939815`, so it is a sensitivity test, not a hedge.
- EXP091: division-heavy route with 384 predicted divisions. Physical edge and
  division Jaccard to EXP083 are `0.913738/0.490486`.
- EXP092: public fine-tuned linker weights plus D4 detection/association TTA,
  reproduced by us with Internet disabled. Its completed source artifact has
  physical edge Jaccard `0.744633` to EXP083 and is the only serious
  architecture/weight hedge in this batch. Our offline output is byte-identical
  to the reviewed public output; the original public version was not
  submission-eligible because Internet was enabled.

Submission receipts and final scores: EXP088 `55882197` / `0.926`, EXP089
`55882683` / `0.924`, EXP090 `55882198` / `0.928`, EXP091 `55882203` /
`0.926`, EXP092 `55882642` / `0.900`. The architecture-diverse EXP092 arm lost
`0.031` to EXP083, so it is no longer a final hedge.

Rejected before LB: the recent Detector3D notebook produced only 44 nodes from
one of four movies. Akihiro's apparent alternative is a near-duplicate of
EXP088 (`0.972646` physical edge Jaccard). The current Grafael harmonic and
Notoverkil base outputs are exact duplicates of already submitted EXP084 and
EXP083, respectively.

## August 31 localization batch

Incident correction: EXP094–097 all failed the hidden rerun with invalid-format
errors because their wrapper replayed frozen public submissions. They are not
selection candidates and provide no localization delta. EXP093 remains only a
pending anchor; retain EXP083 as provisional slot A until EXP093 receives a
valid account score.

EXP093 is the exact audited C33 v1 graph whose current Kaggle version card is
`0.933`. EXP094–097 are our coordinate-only variants of that same graph: C29
donor at `alpha=0.25/0.50`, C30 donor at `alpha=0.25`, and the more independent
Comb2 localization donor at `alpha=0.25`. Every arm retains exactly 115,046
edges and 291 divisions. Receipts: EXP093 `55907915`, EXP094 `55908273`, EXP095
`55908462`, EXP096 `55908629`, EXP097 `55908683`.

These four variants are useful paired localization tests, not slot-B graph
hedges. If one improves LB, prefer the smallest positive dose that reproduces
the gain; if displayed scores tie, do not infer private ordering. The rejected
edge-consensus arm changed four links, three with clearly worse length and
constant-velocity residual, and was not submitted.

## Frozen final-choice rule

1. Final slot A: the best clean full-inference candidate by a combination of
   public LB and same-sign whole-movie paired evidence. Until comparable OOF
   exists, EXP093 is provisional if it reproduces `0.933`; otherwise retain
   EXP083 (`0.931`).
2. Final slot B: prefer the strongest candidate with physical edge overlap
   below `0.85` to slot A if its public score is within `0.005` of the best, or
   if it later wins paired whole-movie OOF on both embryos.
3. If no diverse candidate passes that quality gate, choose the best
   mechanism-distinct clean candidate within the leading family. Do not spend a
   final slot on a near-identical SDW sweep merely because displayed scores tie.
4. A candidate more than `0.012` below the public best is not retained only for
   diversity without strong same-sign OOF evidence.
5. Metric hacks, hidden-test branching, public-CSV wrappers and incomplete
   four-movie outputs are never final candidates.

## Validation needed before September 29

The decisive offline test is whole-movie, official-metric, leave-one-embryo-out
evaluation with frozen policy. Report paired per-movie deltas, both embryo
means, pooled sufficient statistics and bootstrap uncertainty. For exact
public-weight models this requires fold-specific retraining; scoring the public
checkpoint on train is not OOF. A cheaper CPU/no-TTA run is useful for rejecting
mechanisms, but it must not be attached as the exact CV score of the submitted
GPU/TTA artifact.
