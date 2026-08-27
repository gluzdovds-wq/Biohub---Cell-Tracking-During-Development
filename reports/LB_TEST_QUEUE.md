# Leaderboard test queue

## Current update: 2026-08-27

August 26 results are complete: EXP068R `0.884`, EXP069R `0.926`, EXP070 `0.926`, EXP071 `0.923`, EXP072 `0.918`. Best stays `0.926`.

Four new full-inference submissions are registered and pending: EXP073 SDW60 v1 (`55808574`), EXP074 Anhad v21 (`55808576`), EXP075 historical Evgen v11 (`55808638`), EXP076 SEC25 v1 (`55808636`). The first three have author-version score `0.927`; SEC25 has `0.923` and is exploratory. One daily slot remains and is not authorized by this four-submit request.

Version correction: Evgen v11, not v12, is the 0.927 version. The installed output CLI ignores `/version`; previous claims of exact historical output downloads are invalid unless independently verified. In particular, the latest Flex v17/Ahmet v1 duplicate does not itself prove submitted Flex v11 identity. The old dated sections below are historical records, not the current queue.

Next compute priority is the bounded paired-validation pilot, not another high-cost training batch. See `OOF_MODEL_COMPARISON.md`.

## Results from 2026-08-24

- EXP008 (`55732259`): `0.917`, detector diversity is useful but below EXP005.
- EXP007 (`55732491`): `0.900`, reject association-logit D4 TTA.
- EXP039 (`55732718`): `0.906`, reject the independent secondary checkpoint.
- EXP028 (`55732720`): `0.919`, reject the relaxed DeepCenter veto versus EXP005 `0.920`.

## Submitted on 2026-08-25

- EXP065 (`55761017`, pending): exact clean source-attributed `0.927` frontier.
- EXP066 (`55761018`, pending): exact clean source-attributed `0.926` division-sub frontier.
- EXP067 (`55761031`, pending): General-V8 continuation guard / daughter completion, edge overlap `0.793` to EXP005.
- EXP068 (`55761370`, invalid hidden format): public-output wrapper rejected. Retry exact full-inference `pawanmali/biohub-mcflow-v1` v2 after the UTC quota reset.
- EXP069 (`55761371`, invalid hidden format): public-output wrapper rejected. Retry exact full-inference `flexonafft/biohub-harmonic-fusion` v11 after the UTC quota reset.

The two wrapper errors did consume quota despite receiving no score. Exact immutable-version retries were attempted immediately and rejected with the explicit five-per-day limit. `scripts/resubmit_exp068_exp069.py` is duplicate-safe and ready for the next UTC reset.

## Submitted on 2026-08-26

- EXP068R (`55781325`, pending): corrected full-inference MCFlow v2.
- EXP069R (`55781326`, pending): corrected full-inference Flex v11.
- EXP070 (`55781466`, pending): Ahmet v1; subsequently proven byte-identical to EXP069R.
- EXP071 (`55781467`, pending): bidirectional/harmonic weight `0.40`, materially different graph.
- EXP072 (`55781468`, pending): controlled reverse-association weight `0.20`.

All five daily slots were registered as full code submissions. Version-specific output audit after registration found EXP069R and EXP070 share SHA `2dbb8d02…4fa7`; treat today as four unique graphs.

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
