# Public frontier audit

Snapshot date: 2026-08-22. Public leaderboard: rank 1 `0.958`, rank 10 `0.943`, rank 50 `0.929`, rank 100 `0.921`. Strong clean public kernels advertise about `0.908–0.920`; the practical first medal threshold is therefore just above the public plateau, although all cutoffs will move.

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
- Bidirectional association evidence can be fused before exclusive assignment; a harmonic forward/reverse dual-seed variant is the current clean open family around `0.920–0.923`.
- A newer unscored public probe applies eight-view XY D4 TTA to encoder features, inverse-aligns them, and averages node-transformer logits on one shared physical node set. This is technically sound enough to test, but it is not yet evidence of a gain and costs roughly eight encoder passes.
- Current public examples: `0.908`, `0.913`, `0.916`, `0.917`, `0.920`, `0.923`.

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

The apparent top of Kaggle's current `scoreDescending` kernel list is therefore misleading. Direct code inspection found explicit negative-time/out-of-volume hub and fork augmentation in `xiaoleilian/biohub-ct-mix-divaug`, three Kaiwalya variants, `amanatar/biohub-v6-ultra-best`, `muhammaddanyalmalik/cell-tracking`, and the newly indexed `boristown/dark-agi-biohub-cell-tracking-solution` (`0.952`). The latter's final cell connects up to 1,200 real components to a hub at `t=-1000, z=y=x=-10000` and appends five fake forks; its score is not evidence for the preceding D4-TTA tracker. The other leading notebooks advertise the same hack lineage. The highest clean artifact we can both attribute and reproduce remains the Yunus `0.923` run. Our EXP-007 and EXP-008 reproduce their respective clean public source CSVs exactly, but neither provides evidence above `0.923` yet.

An additional 2026-08-22 audit of `kunaldesale2408/biohub-cell-tracking` found a clean three-U-Net pipeline but no new artifact: its 19,102,928-byte output is byte-identical to EXP-008 (`SHA-256 d7ba9e6a…f2bb`, 126,419 nodes, 121,191 edges, maximum degrees 1/2). It is useful as an independent reproducibility check for that teacher arm, not as evidence of a higher clean score.

The same day's completed refresh of `navazshfathi/best-score` produced the clean EXP-006 artifact byte-for-byte (`SHA-256 5c852379…1f4d`, 122,266 nodes, 118,156 edges, maximum degrees 1/2). The new source is a harmonic/safe-division reimplementation. It independently confirms the reproducibility of the `0.923` base but does not move the clean frontier.

A later 11:24 UTC rerun of that Navaz notebook and the newly indexed `mtoshidesu/test-notebook-v17-d17ce0` both again produced EXP-006 byte-for-byte (same 12,514,365 bytes and SHA-256 `5c852379…1f4d`). Their adjacent positions in `scoreDescending` are therefore duplicate executions of one clean graph, not independent gains above `0.923`.

Two newly refreshed public notebooks were also downloaded and audited on 2026-08-22. `backtracking/biohub-general-v4` is clean and introduces boundary-track rescue plus a confidence/continuation division guard, but its own source describes the frozen parent as a verified `0.916` pipeline. Its artifact passes schema/topology checks with 119,563 nodes, 115,208 edges and 334 divisions (`SHA-256 eddb0ff9…a5f7`); at `2 µm` it overlaps EXP-006 by `0.833774/0.796725` node/edge Jaccard. This is an ablation source, not a new clean frontier.

`saitejabandaruin/biohub-masterpiece-tracker-version-16` is also clean, but reproduces its explicitly documented `0.913` artifact exactly (`SHA-256 8c1605b5…9e3f`; 120,797 nodes, 116,501 edges, 314 divisions; audit PASS). Its per-frame dual-seed candidate-retention guard is interesting for honest LOEO testing, but its measured leaderboard evidence is below EXP-006.

## Research priorities

1. Honest LOEO training/validation with no public-weight leakage.
2. Association model for abrupt/global motion: registration-aware coordinates, forward/backward independent models, and calibrated multi-hypothesis candidates.
3. Detection-count calibration per embryo/morphology while preserving matched-node recall.
4. Model diversity: independent seeds/preprocessing, classical detector ensemble, possibly Trackastra/Ultrack candidate graphs.
5. Conservative topology: divisions and gap repair only with independent evidence.

Two especially promising community directions are still under-exploited by the public plateau:

- train a temporal affinity/displacement field from adjacent 3D frames, so association is predicted as local motion rather than inferred only from centroid distance;
- create dense high-precision pseudo-labels from agreement among Cellpose/DoG detections and multiple open trackers (Ultrack, Trackastra, rules), then train the learned linker on far more than the <1% sparsely labelled edges. A public 18.5 GB synthetic dataset with 165k labelled divisions may add useful topology diversity, subject to domain-gap validation.

## Public references audited

- https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor
- https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline
- https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend
- https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack
- https://www.kaggle.com/code/yunusgmsoy/lb-0-920-biohub-cell-tracking-v17
- https://www.kaggle.com/code/yunusgmsoy/kimi-notebook-v17
- https://www.kaggle.com/code/ericwang03/biohub-daily-probe-lane-5
- https://www.kaggle.com/code/backtracking/biohub-general-v4
- https://www.kaggle.com/code/saitejabandaruin/biohub-masterpiece-tracker-version-16
- https://www.kaggle.com/code/mtoshidesu/test-notebook-v17-d17ce0
- https://www.kaggle.com/code/boristown/dark-agi-biohub-cell-tracking-solution
- https://www.kaggle.com/code/tomasa2/biohub-what-worked-and-what-didnt-for-me
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655
