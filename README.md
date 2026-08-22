# Biohub — Cell Tracking During Development

Competition workspace for [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development): detect cell centroids and reconstruct lineage graphs in 3D+time zebrafish microscopy.

## Current objective

Reach the medal zone with a reproducible, leakage-safe pipeline. As of 2026-08-22 the public leader is `0.958`, rank 10 is `0.943`, rank 50 is `0.929`, and rank 100 is `0.921`; the strongest audited clean open run is around `0.923`.

## Repository map

- `EXPERIMENTS.md` — the source of truth for hypotheses, controls, CV/LB results, and decisions.
- `docs/COMPETITION.md` — task, data, metric, validation and constraints.
- `docs/PUBLIC_SOLUTIONS.md` — audited public approaches and measured positive/negative results.
- `docs/ROGII_COMPARISON.md` — transferable lessons from our ROGII campaign.
- `kaggle_notebooks/` — self-contained Kaggle code-competition kernels.
- `scripts/` — repeatable Kaggle push/submit helpers.

Raw competition data, GEFF/Zarr stores, outputs, and submission CSVs are intentionally excluded from git.

Validate a downloaded visible-test artifact before submission:

```powershell
kaggle kernels output <owner/kernel> -p outputs/<experiment> --file-pattern '^submission\.csv$'
./scripts/audit_submission.ps1 -Path outputs/<experiment>/submission.csv -ExpectedDatasetCount 4
python ./scripts/compare_submissions.py outputs/<control>/submission.csv outputs/<candidate>/submission.csv
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
