# Biohub — Cell Tracking During Development

Competition workspace for [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development): detect cell centroids and reconstruct lineage graphs in 3D+time zebrafish microscopy.

## Current objective

Reach the medal zone with a reproducible, leakage-safe pipeline. The current confirmed account best is EXP-066 at `0.926`; EXP-065 scored `0.924` despite its source title claiming `0.927`, so external notebook score labels are not treated as portable evidence. Five additional full-inference submissions were registered on 2026-08-26 and are pending. The final private leaderboard and eligibility still control actual medals, and exact public-weight OOF remains unavailable because those checkpoints used all labelled movies. The live API lists the competition deadline as 2026-09-29 23:59 UTC.

## Repository map

- `EXPERIMENTS.md` — the source of truth for hypotheses, controls, CV/LB results, and decisions.
- `docs/COMPETITION.md` — task, data, metric, validation and constraints.
- `docs/PUBLIC_SOLUTIONS.md` — audited public approaches and measured positive/negative results.
- `docs/ROGII_COMPARISON.md` — transferable lessons from our ROGII campaign.
- `reports/SUBMISSION_CV_MATRIX.md` — every account submission mapped to exact/related OOF evidence, LB state and private-stability interpretation.
- `kaggle_notebooks/` — self-contained Kaggle code-competition kernels.
- `scripts/` — repeatable Kaggle push/submit helpers.

Raw competition data, GEFF/Zarr stores, outputs, and submission CSVs are intentionally excluded from git.

Validate a downloaded visible-test artifact before submission:

```powershell
kaggle kernels output <owner/kernel> -p outputs/<experiment> --file-pattern '^submission\.csv$'
./scripts/audit_submission.ps1 -Path outputs/<experiment>/submission.csv -ExpectedDatasetCount 4
python ./scripts/compare_submissions.py outputs/<control>/submission.csv outputs/<candidate>/submission.csv
python ./scripts/analyze_topology_ablation.py outputs/<control>/submission.csv outputs/<candidate>/submission.csv --output outputs/<candidate>/topology_ablation.json
```

The comparator reports both exact-coordinate graph overlap and a default `2 µm` greedy same-frame physical match, so integer-centroid and subvoxel-refined detectors can be compared without conflating formatting with model diversity. The stricter topology-ablation analyzer requires identical node IDs, times and exact coordinates, then reports changed sources, division-parent overlap, physical edge/sister distances and constant-velocity residuals.

Inspect a bounded tail of a running kernel and validate a frozen submission candidate:

```powershell
python ./scripts/tail_kaggle_kernel.py <owner/kernel> --pattern 'Epoch|batches|Traceback' --lines 20
./scripts/submit_candidate.ps1 -Candidate EXP014       # validation-only dry run
./scripts/submit_candidate.ps1 -Candidate EXP014 -Submit
```

The submit helper checks kernel completion, the canonical artifact SHA, full schema/topology invariants and the live daily quota before spending a slot. It never retries the mutating submission request automatically.

The LOEO transition is also fail closed. A single watcher instance (guarded by a global mutex) polls the two parent kernels, verifies the completed split contract and checkpoint SHA through `verify_loeo_parent.ps1`, and only then pushes the corresponding untouched-audit kernel:

```powershell
./scripts/watch_loeo_and_launch.ps1 -Once              # status/logic smoke
./scripts/watch_loeo_and_launch.ps1 -PollSeconds 60    # bounded 8h watcher
```

A separate idempotent quota watcher keeps the pre-registered submission order `EXP014 → EXP019`. It checks the existing submission descriptions before every mutation, waits for a live slot, and delegates to the non-retrying SHA/audit helper for exactly one candidate at a time:

```powershell
./scripts/watch_quota_and_submit.ps1 -Once
./scripts/watch_quota_and_submit.ps1 -PollSeconds 60   # bounded 12h watcher
```

## Experiment discipline

1. Split by embryo, never by crop. Public checkpoints trained on all 199 labelled movies cannot provide honest local validation.
2. Change one mechanism at a time and compare paired per-movie deltas.
3. Always calculate the official node-count adjustment, not raw edge Jaccard alone.
4. Record failed ideas and broken diagnostics, not only wins.
5. Treat public LB as a noisy external check; select models with embryo-held-out CV and diversity.

## Initial submission ladder

- `EXP-001`: CPU nearest-neighbor sanity baseline.
- `EXP-002`: classical blob detection + Hungarian linking + conservative gap closure.
- `EXP-003`: clean single-seed TemporalUNet3D + node transformer + ILP/repair.
- `EXP-004`: dual-seed logit blend + node transformer + ILP + DeepCenter confirmation.

Each fork retains the original public notebook code and its attribution. Our changes in this first cycle are limited to private Kaggle kernel metadata and experiment tracking.
