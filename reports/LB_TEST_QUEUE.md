# Leaderboard test queue

## Results from 2026-08-24

- EXP008 (`55732259`): `0.917`, detector diversity is useful but below EXP005.
- EXP007 (`55732491`): `0.900`, reject association-logit D4 TTA.
- EXP039 (`55732718`): `0.906`, reject the independent secondary checkpoint.
- EXP028 (`55732720`): `0.919`, reject the relaxed DeepCenter veto versus EXP005 `0.920`.

## Submitted on 2026-08-25

- EXP065 (`55761017`, pending): exact clean source-attributed `0.927` frontier.
- EXP066 (`55761018`, pending): exact clean source-attributed `0.926` division-sub frontier.
- EXP067 (`55761031`, pending): General-V8 continuation guard / daughter completion, edge overlap `0.793` to EXP005.
- EXP068 (CPU wrapper v1 running): global min-cost-flow, edge overlap `0.602` to EXP005.
- EXP069 (CPU wrapper v1 running): harmonic-fusion high-score calibration with 470 divisions.

All five are code submissions or fail-closed code wrappers. Direct CSV attempts were rejected by the notebook-only competition gate and did not consume quota.

## Ready immediately after GPU quota resets

1. EXP060: exact `0.920` EXP005 topology plus detector-consensus coordinates, `alpha=0.50`.
2. EXP061: the same matches/topology with conservative `alpha=0.25`.

Both outputs come from one dual-inference notebook. Public artifacts and graph invariants already pass; only the hidden-compatible Kaggle run is missing.

## Next controlled LB hypotheses

1. Distance-adaptive EXP005/EXP008 coordinate blend: use a smaller dose near the `2 µm` match gate and `0.5` only under strong agreement. Compare against fixed EXP060/061, not directly tune an arbitrary curve on LB.
2. EXP005 division ablation ladder: exact harmonic graph with original 67 divisions, only physically weak divisions removed, and divisions disabled. Emit all arms from one inference run to isolate division prevalence on public test.
3. Detection-count calibration around EXP005: frozen neighboring thresholds around `0.96875`, with node-count/edge changes audited before score disclosure. Kill if both embryo folds disagree once comparable OOF exists.
4. Association-family comparison: harmonic forward/reverse versus weak learned registered and ILP only after EXP063/064 report both embryo signs.
5. Cached physical linkers: Kalman/constant-velocity and particle-filter policies promoted only when paired OOF is non-negative on both embryos. Min-cost-flow receives one explicit exploratory LB measurement as EXP068 because its edge overlap with EXP005 is only `0.602`.

The queue deliberately avoids metric hacks, public-CSV wrappers, registered-relink descendants already falsified at `0.905`, and intensity refinements falsified at `0.893`.
