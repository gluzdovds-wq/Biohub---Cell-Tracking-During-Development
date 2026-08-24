# Leaderboard test queue

## Submitted on 2026-08-24

- EXP008 (`55732259`): detector-diverse three-UNet/flip-TTA hedge.
- EXP007 (`55732491`): shared-node D4 association-TTA probe.
- EXP039 (`55732718`): replace only the secondary checkpoint with an independently trained state.
- EXP028 (`55732720`): lower only the DeepCenter safe-division veto threshold `0.12→0.08`.

All are full-inference code submissions; one daily slot remains.

## Ready immediately after GPU quota resets

1. EXP060: exact `0.920` EXP005 topology plus detector-consensus coordinates, `alpha=0.50`.
2. EXP061: the same matches/topology with conservative `alpha=0.25`.

Both outputs come from one dual-inference notebook. Public artifacts and graph invariants already pass; only the hidden-compatible Kaggle run is missing.

## Next controlled LB hypotheses

1. Distance-adaptive EXP005/EXP008 coordinate blend: use a smaller dose near the `2 µm` match gate and `0.5` only under strong agreement. Compare against fixed EXP060/061, not directly tune an arbitrary curve on LB.
2. EXP005 division ablation ladder: exact harmonic graph with original 67 divisions, only physically weak divisions removed, and divisions disabled. Emit all arms from one inference run to isolate division prevalence on public test.
3. Detection-count calibration around EXP005: frozen neighboring thresholds around `0.96875`, with node-count/edge changes audited before score disclosure. Kill if both embryo folds disagree once comparable OOF exists.
4. Association-family comparison: harmonic forward/reverse versus weak learned registered and ILP only after EXP063/064 report both embryo signs.
5. Cached physical linkers: Kalman/constant-velocity, particle-filter and min-cost-flow policies promoted only when paired OOF is non-negative on both embryos.

The queue deliberately avoids metric hacks, public-CSV wrappers, registered-relink descendants already falsified at `0.905`, and intensity refinements falsified at `0.893`.
