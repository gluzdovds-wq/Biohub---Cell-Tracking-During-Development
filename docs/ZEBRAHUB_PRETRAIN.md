# Zebrahub external pretraining audit

Snapshot date: 2026-08-22.

## Authorization and source

The competition host explicitly states in [Is public Zebrahub data allowed as external training data?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330) that all Zebrahub data and resources, including tracking annotations, may be used and that there is no overlap with the competition test set. The [Zebrahub imaging atlas](https://zebrahub.sf.czbiohub.org/imaging) exposes seven light-sheet timelapses acquired on two instruments, with image volumes and Ultrack-derived cell tracks.

This authorization is competition-specific. It does not replace checking the source terms before redistributing raw data outside the allowed workflow.

## First real-data candidate: ZSNS001 tail

`ZSNS001_tail.ome.zarr` contains 791 frames. OME-Zarr level 0 has shape `(791,1,420,1217,1091)` and voxel scale `(1.24,0.439,0.439) µm`; levels 1 and 2 are spatially downsampled by 2 and 4. The 464-MiB `ZSNS001_tail_tracks.csv` columns are `track_id, NodeID, ParentTrackID, t, t_hier_id, z, y, x, area`, with z/y/x stored in physical micrometres.

A full streaming pass using `scripts/audit_zebrahub_tracks.py` found:

- 7,505,357 cell detections across frames 0–790;
- 101,676 tracks;
- 36,878 distinct parent tracks with divisions;
- 73,756 child tracks carrying `ParentTrackID`, exactly two per division parent;
- physical coordinate bounds z `28.52–509.64`, y `0.878–508.362`, x `1.317–465.34 µm`.

That single real timelapse supplies about 121 times the competition's roughly 304 labelled divisions and about 22% as many division events as the deliberately oversampled 165,267-event synthetic set, while also adding millions of ordinary real-domain association edges.

`scripts/extract_zebrahub_crop.py` is the verified image-side alignment primitive. It reads only the OME-Zarr chunks needed for one physical crop and samples an exact `64³` grid at the competition model's `1.625 µm` isotropic spacing. The first smoke crop is centred on the CSV's first annotated node `(440.2,111.945,184.819) µm` at `t=0`; level-1 source reading and interpolation preserve a bright local nucleus signal rather than merely producing a correctly shaped array.

## Pre-registered implementation path

1. EXP-025, internet enabled and CPU-only: select 256 train anchors from frames 10–629 and 64 checkpoint anchors from frames 650–789 with a deterministic per-frame hash rule; stream only their level-1 OME-Zarr chunks; reconstruct consecutive same-track and parent-to-child division edges; resample exact two-frame `64³ @ 1.625 µm` crops; write bounded `.npz` shards plus source/coordinate/hash receipt. The temporary track CSV is deleted before the kernel completes.
2. EXP-026: train the exact public TemporalUNet3D + node-transformer architecture for 12 epochs on the fixed time-disjoint shards. Every source shard hash is verified, and every annotated cell inside each crop is retained so true nuclei are not accidentally treated as negatives.
3. Real fine-tuning control: initialize from the external checkpoint and run exactly the same real-data epochs, optimizer and frozen reciprocal LOEO folds as a random-initialization control.
4. Promotion: require positive deltas on both untouched real audit folds, report node recall/count, conditional linking accuracy, edge TP/FP/FN and division TP/FP/FN. A synthetic/external validation score is never sufficient.

Main risks are instrument/appearance shift, imperfect Ultrack pseudo-labels, higher true cell density than the sparse competition annotations, and coordinate/resampling mistakes. Physical-coordinate conversion and source-timelapse separation therefore fail closed before training.

EXP-025 v2 completed on Kaggle and the downloaded output passed all 320 declared shard hashes. The bounded set contains 418,699 node annotations, 205,694 lineage edges and 2,029 division edges in 113.4 MB. A crop can contain up to 2,885 two-frame nodes, making annotation-density shift a real concern; EXP-026 therefore uses batch 2, and no external checkpoint is promoted without both reciprocal real-data audits.
