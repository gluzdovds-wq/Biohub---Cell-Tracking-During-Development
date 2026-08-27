# Honest OOF model comparison

## What is already honest

The current OOF detector is a TemporalUNet3D plus node transformer trained twice. The `44b6` checkpoint sees only `6bba` during training; the reciprocal checkpoint sees only `44b6`. Threshold/policy selection uses four checkpoint movies, confirmation uses four separate movies, and the reported audit covers 63/120 untouched movies.

Existing untouched results on identical detections:

- Holdout `44b6`: registered `0.744130`, weak learned tie-break `0.744864`, greedy `0.579940`, broad/strict physical prune `0.583013/0.582320`.
- Holdout `6bba`: registered `0.595767`, weak learned tie-break `0.596538`, greedy `0.466180`, broad/strict physical prune `0.473658/0.473254`.
- Weak learned minus registered is positive on both embryos (`+0.000734/+0.000771`). Registered minus greedy is much larger (`+0.164190/+0.129587`).

This is an honest comparison of linker/postprocess mechanisms. It is not an honest comparison of the exact EXP005/006/007/008 public weights: those weights used all labelled movies and therefore cannot produce unseen-embryo predictions without fold retraining.

## EXP063/064 comparison contract

The prepared reciprocal runs infer each untouched movie once, then compare these policies on exactly the same detections and learned edge candidates:

1. registered motion Hungarian;
2. registered plus 10% learned tie-break;
3. registered plus 70% learned score with learned-candidate requirement;
4. public ILP weights;
5. support-pack ILP weights;
6. greedy learned edges;
7. greedy plus broad physical division prune;
8. greedy plus strict physical division prune.

Every movie also exports a compressed cache of coordinates and edge candidates before scoring. Kalman, constant-velocity, particle-filter and min-cost-flow linkers can then be evaluated on CPU without repeating neural inference.

## CPU decision

CPU is appropriate for metric calculation, bootstraps, graph optimization and new linkers after candidate export. It is not appropriate for the full detector comparison under Kaggle's 12-hour notebook limit.

Corrected 2026-08-27 using actual EXP050/051 logs: those single-seed OOF evaluation jobs took 5,162.79 and 8,814.00 seconds, or **3.882 hours of combined wall time on T4 x2** (7.765 physical GPU-device hours). They evaluated 67/124 movies including confirmation. The earlier 28–38-hour extrapolation from the much heavier public dual-seed notebooks was not appropriate for these jobs. Kaggle quota units are not automatically equal to physical GPU-device hours.

One GPU inference/cache pass per fold is the economical starting point. Allow roughly 4–6 hours of combined T4 x2 wall time for comparable single-seed cache generation plus overhead; extra ILP arms should run on CPU afterwards. This is a planning estimate, not a benchmark of the expanded EXP063/064 code. The live quota on 2026-08-27 is 23,319.717 seconds used against 21,600 allowed; refresh is 2026-08-29 00:00 UTC (07:00 Novosibirsk).

## Critical scope correction

EXP063/064 do **not** implement exact EXP066/071/073 or EXP008. They contain neither dual-seed fusion nor reverse edge-logit export nor a fold-trained DeepCenter. Existing cache export saves only coordinates and forward candidate probabilities/distances. Bidirectional arms need a new reverse-logit inference/export path; changing detection fusion weight needs new detections, not just a CPU graph rewrite.

The existing public DeepCenter also needs an explicit training-provenance audit: reusing it inside otherwise LOEO inference is not automatically leakage-free. Disable it for an honest mechanism-only comparison or train an embryo-disjoint counterpart. Never label a shared single-seed approximation as the submitted model's OOF score.

## Affordable first step

`scripts/estimate_validation_budget.py --inventory` freezes 12 audit movie IDs per embryo using a deterministic hash independent of scores, reads only file metadata, and calculates cloud-download bytes and a runtime estimate. The 24-movie set is a screening pilot, not a new biological holdout. Model configurations must be frozen before evaluating it.

The initial inventory request was rate-limited before all file records were collected, so exact pilot download bytes are not claimed in this report. The script now checkpoints pagination and throttles metadata requests. The saved movie list/runtime estimate does not require the inventory; no image chunks were downloaded.

Use existing reciprocal checkpoints for that pilot. The old inference configuration scales to about half an hour on T4 x2; budget 1–2 hours for a newly instrumented forward/reverse pilot including setup and scoring, then revise the full-run estimate from measured timings. No expensive training is necessary for this mechanism-level step. It does not establish a 0.001 private gain or validate the exact public weights.

## Exact-model validation and platform choice

Exact comparisons need embryo-disjoint training for every learned component of each family. For the public dual-seed family this includes both seeds and the learned veto, with the same training protocol; EXP008 needs its distinct detector checkpoints. The old abbreviated reciprocal training jobs took about 7h10m and 8h38m on T4 x2, but cannot price full public-model reproduction or the other architectures. The previous blanket 40–70 GPU-hour estimate was unbenchmarked and is withdrawn as a reliable budget for these exact models.

Prefer Kaggle: the competition is mounted under `/kaggle/input`, no 80+ GB laptop download, existing checkpoints are already attached, and outputs can hold compressed per-movie caches and sufficient statistics. For this cache schema, raw array storage is approximately `16 * nodes + 24 * candidate_edges` bytes before compression and metadata; reverse logits add storage. Measure a pilot rather than promising a fixed cache size.

Colab is possible: download only the selected movie chunks, labels, code and checkpoints directly into the Colab VM with the user's Kaggle credentials stored as secrets. Do not put tokens into Git or notebook outputs. No laptop transfer is required. Process archives on VM-local disk and save completed per-movie results persistently. GPU type, availability and runtime limits vary; free runtimes are not guaranteed and can terminate. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html). This turn did not start Colab or purchase compute.

## What the score can establish

For each pair, save same-movie official sufficient statistics, aggregate with the official weighting/node adjustment, and report each embryo's delta, paired block bootstrap, lower bound and division TP/FP/FN. Group overlapping crops by source region/time when their mapping is available; movie-level resampling otherwise underestimates dependence. Report error correlation, not just graph overlap, once the paired predictions exist.

Two embryos cannot identify a precise population confidence interval for new private embryos. Treat the worst-fold result, domain sensitivity and same-sign paired deltas as stress tests, not a calibrated absolute private score. An independent external embryo with trustworthy compatible annotations would help without retraining, but the currently prepared Zebrahub Ultrack labels are pseudo-labels and only support an additional stress test.
