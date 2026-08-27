# EXP077: CPU feasibility and our local-flow linker

## What is ours

The latest four LB submissions (EXP073–076) are attributed public reproductions. Our own weight/geometry experiments exist, but none has beaten the public-derived best. This pilot does not rename a fork as a new architecture: the detector and edge transformer are public; training checkpoints EXP009/010 are ours; the leave-self-out local deformation linker is newly implemented here.

Hypothesis H065: a single global displacement can miss coherent, spatially varying tissue motion. Estimate residual displacement from unambiguous mutual-nearest anchors within 20 microns, exclude the query's own anchor, require at least three neighbors with consistent residuals, and shrink the correction. Sparse/ambiguous neighborhoods fall back to global registration. Preserve detections and constrain assignment to the original physical gate. This is a testable physical association method, not a claim of scientific novelty or a new neural detector.

## Frozen experiment

- Kaggle CPU only, four full movies, interleaved embryos: `44b6_415c0a3a`, `6bba_96833384`, `44b6_abf82518`, `6bba_55c70843`.
- Selection: first two per embryo from `validation_budget_20260827.json`, chosen there by a fixed name hash, not by scores.
- Single-seed reciprocal checkpoints; model, model config, training contract and threshold-selection SHA checks fail closed.
- No flip TTA: one rather than four detector encode passes per window. This does **not** imply a measured 4x wall-time speedup; I/O, transformer and evaluation remain.
- Registered, registered + 10% learned tie-break, local-flow + same tie-break, and public-style ILP all use identical candidate caches.
- Six-hour total cap, 90-minute child-process limit per movie. A timeout remains a timeout, never a zero score or silently dropped successful fold.
- Save each movie's compressed candidates, physical scale, frozen graph snapshots, per-arm official metric components, elapsed times and inference provenance.
- No dataset download locally. No new training. No submission slots consumed.

## What the result can and cannot establish

Paired deltas use the same movies/detections and the supplied metric's aggregation, separately by embryo and pooled. Division TP/FP/FN are retained. The three Hungarian arms currently allow one child and cannot recover divisions; ILP is the division-aware reference. A local-flow win over registered alone is not evidence of superiority to ILP or the submitted harmonic ensembles.

Evaluated movies are excluded from training/checkpoint selection/calibration, and detector training is on the other embryo. However checkpoint selection and calibration used separate movies of the evaluated embryo. This is **movie-held-out** evaluation with an embryo-transfer limitation, not pristine unseen-embryo cross-validation. The audit movies have been examined in older experiments; they are not a new final holdout.

Four movies measure feasibility and can reject gross regressions. They cannot establish a tiny private improvement, a robust bootstrap interval, or the private score of the full dual-seed/TTA/DeepCenter model. Different checkpoint training lengths also confound a direct between-embryo score comparison.

## Continuation after measured timings

1. If feasible, evaluate the remaining 20 frozen pilot movies in bounded CPU shards using the same configuration, with no score-driven re-selection.
2. Preserve candidate caches as Kaggle notebook outputs. New physical linkers then consume only small caches, locally or on Kaggle CPU, without repeating neural inference.
3. Use the 24-movie paired signs, division counts and comparison to ILP to reject weak hypotheses. Do not use a four-movie winner as a production promotion gate.
4. Expand a frozen promising arm to all 183 audit movies only after extrapolating from measured per-movie/window times, with explicit failures and missingness. This remains a development audit, not an untouched final test.
5. For stronger unseen-embryo evidence, future training must choose checkpoints/thresholds using only the training embryo (nested split), then evaluate the other embryo once. Even then, two embryos do not bound private distribution shift.

Kaggle's documented CPU/GPU notebook session limit is 12 hours, so a two-day workload needs bounded sessions and saved outputs, not one uninterrupted session. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)

Status: [Kaggle v1 RUNNING](https://www.kaggle.com/code/dmitriigluzdov/biohub-exp077-cpu-held-out-local-flow-pilot), with all parent contracts verified. API source matches the local code exactly (SHA `5b79389f998de936d39d98c1aba49057372010ffaa1b7bc5db31658dbff5a36a`) and GPU is disabled. Ten local tests pass, including parity of both controls with EXP063.

First complete movie observed in live v1 logs: `44b6_415c0a3a.zarr`, 100 frames; neural inference `289.963 s`, total inference/four linkers/scoring `356.032 s`, process exit 0. Candidate cache contains 63,436 nodes / 47,669 edges. Registered/weak/local-flow/ILP scores are `0.677837 / 0.669063 / 0.669063 / 0.718948`; no divisions occur on this movie. The new local field has paired delta **0.0** against its weak control and does not outperform ILP. No promotion follows. `6bba_96833384.zarr` has started; full pilot/embryo results remain pending.

This observation supports CPU feasibility. A rough one-movie linear extrapolation is about 2.4 CPU wall hours for 24 movies or 18.1 hours for 183, excluding setup between shards. These are **not measured complete-run budgets**; movie density, solver time and failures may change them. Neither estimate applies to training from scratch or the exact public dual-seed/TTA pipeline. Use bounded shards below the 12-hour session limit.
