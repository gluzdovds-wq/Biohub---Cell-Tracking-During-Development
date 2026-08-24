# Frontier private-robustness decision

Generated 2026-08-24 before EXP008 received an account leaderboard score.

## Verdict

EXP005 at `0.920` is the strongest clean public control, but the score alone is not enough to call it private-robust. Its public weights used all labelled movies, so the exact model has no honest unseen-embryo OOF. The `+0.001` lead over EXP006 is far smaller than the observed embryo/domain and four-movie sampling uncertainty.

Use EXP005 as the primary final candidate, not as a private guarantee. Preserve one final slot for a structurally different detector family and promote new changes only when they preserve the clean graph or have paired embryo-disjoint evidence.

## Structural evidence

- EXP005 versus EXP006 at `2 µm`: node Jaccard `0.992594`, edge Jaccard `0.988545`; they are effectively the same detector/linker family. Their public scores `0.920/0.919` do not provide two independent estimates.
- EXP005 versus EXP008 at `2 µm`: node Jaccard `0.605883`, edge Jaccard `0.548440`. EXP008 is a materially different detector/linker hedge, despite its exact-source attribution of `0.917`.
- EXP005 has only 67 predicted divisions versus 455 in EXP006 and 352 in EXP008. This is conservative and reduces division-FP exposure, but private division prevalence is unknown.

## Honest stability evidence

The saved LOEO experiment evaluates mechanisms trained on the opposite embryo, not the exact public EXP005 weights. Registered motion scores `0.744130` on held-out `44b6` and `0.595767` on held-out `6bba`, an embryo gap of `0.148363`. A four-movie public-like resample has a very wide conditional interval (`0.451683–0.786690`), while a 130-movie private-like resample is narrower (`0.585632–0.645486`). See `reports/oof_stability.json` for every movie, fold aggregate and 10,000 bootstrap replicates.

These values do not map to the absolute `0.920` scale because the trained models differ. They do show that a one-thousandth public delta is not a reliable private ordering signal.

## Current submission policy

1. Keep EXP005 as the primary clean leader.
2. Evaluate EXP008 (`55732259`) as the detector-diverse hedge and EXP007 (`55732491`) as the higher-count D4 association-TTA probe; do not replace EXP005 solely on a tiny public delta.
3. Next production priority is EXP060/061: exact EXP005 topology with `0.50/0.25` detector-consensus coordinate doses. Both local artifacts pass full graph audit; hidden-compatible inference is built but weekly GPU quota currently prevents execution.
4. After the one EXP007 information probe, do not spend the three remaining slots on EXP039, registered-relink descendants, or public-artifact wrappers: they are reject-only negative, empirically worse, or invalid for hidden reruns.
