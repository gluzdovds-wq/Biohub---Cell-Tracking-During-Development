# Public frontier audit

Snapshot date: 2026-08-22. Public leaderboard top: `0.958`. Strong clean public kernels advertise about `0.908–0.920`; the current medal zone is roughly `0.94+` and will move.

## Main public solution families

### Classical detection and physical linking

- Percentile/Otsu/DoG/blob detection, local maxima or watershed.
- Physical-coordinate nearest-neighbour or Hungarian association.
- Constant-velocity extrapolation, motion gates, one-frame gap closure.
- Public examples range from simple sanity baselines to a claimed `0.857` rule-based pipeline.

Usefulness: cheap diversity, interpretable error decomposition, robust fallback. Limitation: wrong-partner errors in dense scenes and abrupt global/cell motion.

### Official learned baseline

- TemporalUNet3D predicts a detection map and voxel features.
- A cross-attention node transformer scores adjacent-frame pairs.
- ILP/global constraints enforce one parent, limited children, births/deaths and divisions.
- Sparse supervision trains only on known edges; unlabelled detections are not treated as negatives.

### Strong clean public frontier

- High detection threshold around `0.96875` to control the node-count penalty.
- Short-track filtering, conservative gap repair, motion relinking and carefully gated divisions.
- Independent-seed logit blending before peak extraction.
- Optional DeepCenter model only as independent evidence for marginal synthetic gaps.
- Current public examples: `0.908`, `0.913`, `0.916`, `0.917`, `0.920`.

## Measured positive results from an open ablation notebook

Source: `tomasa2/biohub-what-worked-and-what-didnt-for-me` (public LB 0.841, 11 paired changes).

- Learned edge prior: `+0.0048` paired over 16 movies; LB `0.826 -> 0.832`.
- Raising detection threshold `0.6 -> 0.85`: local `+0.0026`; LB `0.832 -> 0.838`.
- Raising `0.85 -> 0.96875`: predicted `+0.0075`; LB `0.838 -> 0.841`.
- Threshold gains came from the over-prediction penalty while raw edge Jaccard stayed flat.

## Measured null/negative results

- Naive division prediction: `-0.007`; divisions were extremely rare and false positives dominated.
- Gap repair: `-0.003`, better on 0/6 movies after fixing an inert evidence check.
- Per-movie adaptive motion gate: `+0.001`, non-significant (`p=0.85`).
- Default ILP replacing Hungarian on an already-resolved candidate assignment: `-0.001`, non-significant.
- Treating learned assignment hints as the edge set: worse on 6/10.
- Raw/velocity cost tuning: null or harmful.
- Harmonic fusion applied after exclusive assignment: mathematically inert.
- Reverse inference with the same single checkpoint: unstable, dominated by one movie; no robust gain.

The same audit attributed 57% of losses to wrong partners, 32% to a detected pair with no link, 11% to missed endpoints, and 3% to division semantics. Median correct-target distance for wrong-partner cases was 6.08 micrometres versus a typical 1.8 micrometre displacement, consistent with abrupt acceleration/global jumps.

## Risks and traps

- Public model checkpoints were trained on all 199 annotated movies; evaluating them on those movies is memorisation, not validation.
- Movie-to-movie score spread is large (reported roughly `0.46–0.984` in one LOEO setup), and public LB may be about 10% optimistic.
- Some consecutive frames are frozen and some sequences contain sudden global jumps.
- Dense graph operations can exceed runtime if implemented with cubic complexity.
- Node/edge schema mistakes can yield 0.0 or scoring errors despite good local graph scores.
- Metric-hack notebooks used fake forks outside the image; the host acknowledged the report. They are excluded from our plan.

## Research priorities

1. Honest LOEO training/validation with no public-weight leakage.
2. Association model for abrupt/global motion: registration-aware coordinates, forward/backward independent models, and calibrated multi-hypothesis candidates.
3. Detection-count calibration per embryo/morphology while preserving matched-node recall.
4. Model diversity: independent seeds/preprocessing, classical detector ensemble, possibly Trackastra/Ultrack candidate graphs.
5. Conservative topology: divisions and gap repair only with independent evidence.

## Public references audited

- https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor
- https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline
- https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend
- https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack
- https://www.kaggle.com/code/tomasa2/biohub-what-worked-and-what-didnt-for-me
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283
