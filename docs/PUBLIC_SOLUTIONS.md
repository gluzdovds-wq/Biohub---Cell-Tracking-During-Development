# Public frontier audit

Snapshot downloaded 2026-08-22 11:27 UTC (2,628 rows): rank 1 `0.958`, rank 10 `0.943`, rank 50 `0.929`, rank 87 `0.923`, rank 107 `0.921`, and rank 132 `0.920`. Scores of at least `0.917` extend through rank 354 because of ties, while `0.913` extends through rank 783. The current account score `0.826` is rank 1836. The clean reproduced `0.923` frontier is therefore already above the rank-263 line (10% of this snapshot), but official medal eligibility and final cutoffs are separate and will move as teams merge and scores improve.

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

The exact EXP-005→EXP-006 logs isolate the guarded division widening. On the four reused labelled movies, adjusted edge Jaccard fell `0.9152→0.9135`, while division Jaccard rose `0.0→0.1` (`TP/FP/FN 0/1/5→1/5/4`), moving the proxy `0.9152→0.9235`. The attributable public LB moved only `0.920→0.923`. Thus conservative division expansion was directionally useful, but this sparse in-sample proxy overstated the transferable gain by roughly 2–3×; it cannot justify further veto relaxation without independent evidence.

The completed one-line DeepCenter ablation confirms that warning. The refreshed Yunus run lowers only `DEEPCENTER_SAFE_DIV_THRESHOLD` from `0.12` to `0.08`, yet the capped proposal selection replaces 73 of 455 division parents and cascades into 93 removed / 35 added node IDs, 461 coordinate changes, and 143 removed / 100 added edges. The artifact is clean (122,208 nodes, 118,113 edges, SHA-256 `a15e99ba…c9f4`) and remains physically close to EXP-006 at `2 µm` (`0.997222/0.995869` node/edge Jaccard), but its reused-label diagnostic loses the only division TP (`1/5/4→0/5/5`) and drops the proxy `0.9235→0.9131`. EXP-028 is therefore rejected without spending a submission slot.

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

### Newly released synthetic supervision

The public CC0 builder `josefreitasalvesneto/biohub-synthetic-dataset` exposes 18.503 GB: 1,539 static native-resolution volumes and 2,174 six-frame pooled sequences with 4,056,226 nodes, 3,460,295 edges and 165,267 labelled divisions. Its motion constants are fitted to real training edges (median step `1.86 µm`, lag-1 persistence `0.30`, sister separation `7.24 µm`), but appearance matching is explicitly partial and divisions are deliberately inflated to `4.07%` of nodes versus roughly `0.26%` in the real annotations. It is suitable as pretraining supervision, not as a calibrated real-data model.

There is also a coordinate-contract trap in the released sequences: image tensors are already `(T,64,64,64)`, while node y/x values remain in the native 256-grid (confirmed on `seq_0000`, y/x up to `243.17`). EXP-024 therefore divides y/x by four before any crop or target construction and fails closed if the source contract changes. Any reported gain must survive a compute-matched real fine-tune and both untouched LOEO audits.

In the 2026-08-15 discussion [Possible big leaderboard shakeup](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352), the then rank-3 participant reported an undisclosed plug-in postprocessor adding roughly `0.03–0.05` across several public models and reaching `0.940` from one public model without tuning. No implementation, CV, artifact or graph diff was published, so this is not reproducible evidence and cannot justify a submission. It is, however, consistent with our decision to test registered motion/learned-probability graph reconstruction rather than spend the whole budget on small detector-threshold changes; the same thread's rank-7 participant independently emphasized reliable CV and synthetic division data.

The later discussion [What is the best model for this domain so far?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604) supplies a more actionable consensus from rank-7/rank-82 participants: the 3D U-Net + node transformer remains a strong backbone, but public checkpoints have plateaued and should be retrained; diagnose node recall/count, conditional linking accuracy, a GT-aware detection ceiling, and division candidate recall/ranking separately. The useful approximation `edge recall ≈ node recall² × conditional linking accuracy` explains why detector misses compound before linking. This directly supports EXP-009/010's real retraining and EXP-011/012's frozen-policy error decomposition.
- Dense graph operations can exceed runtime if implemented with cubic complexity.
- Node/edge schema mistakes can yield 0.0 or scoring errors despite good local graph scores.
- Metric-hack notebooks used fake forks outside the image; the host acknowledged the report. They are excluded from our plan.

The apparent top of Kaggle's current `scoreDescending` kernel list is therefore misleading. Direct code inspection found explicit negative-time/out-of-volume hub and fork augmentation in `xiaoleilian/biohub-ct-mix-divaug`, three Kaiwalya variants, `amanatar/biohub-v6-ultra-best`, `muhammaddanyalmalik/cell-tracking`, and the newly indexed `boristown/dark-agi-biohub-cell-tracking-solution` (`0.952`). The latter's final cell connects up to 1,200 real components to a hub at `t=-1000, z=y=x=-10000` and appends five fake forks; its score is not evidence for the preceding D4-TTA tracker. The other leading notebooks advertise the same hack lineage. The highest clean artifact we can both attribute and reproduce remains the Yunus `0.923` run. Our EXP-007 and EXP-008 reproduce their respective clean public source CSVs exactly, but neither provides evidence above `0.923` yet.

An additional 2026-08-22 audit of `kunaldesale2408/biohub-cell-tracking` found a clean three-U-Net pipeline but no new artifact: its 19,102,928-byte output is byte-identical to EXP-008 (`SHA-256 d7ba9e6a…f2bb`, 126,419 nodes, 121,191 edges, maximum degrees 1/2). It is useful as an independent reproducibility check for that teacher arm, not as evidence of a higher clean score.

The same day's completed refresh of `navazshfathi/best-score` produced the clean EXP-006 artifact byte-for-byte (`SHA-256 5c852379…1f4d`, 122,266 nodes, 118,156 edges, maximum degrees 1/2). The new source is a harmonic/safe-division reimplementation. It independently confirms the reproducibility of the `0.923` base but does not move the clean frontier.

A later 11:24 UTC rerun of that Navaz notebook and the newly indexed `mtoshidesu/test-notebook-v17-d17ce0` both again produced EXP-006 byte-for-byte (same 12,514,365 bytes and SHA-256 `5c852379…1f4d`). Their adjacent positions in `scoreDescending` are therefore duplicate executions of one clean graph, not independent gains above `0.923`.

The refreshed `anhadmahajan06/biohub-track-your-cells-development` run completed at 12:50 UTC and was audited from its actual output rather than its earlier source snapshot. Its 12,558,700-byte artifact is clean and schema/topology PASS: 122,910 nodes, 118,395 edges, 328 division parents, maximum in/out degree 1/2, SHA-256 `f326ea07…3445`. Against EXP-006, physical `2 µm` node/edge Jaccard is `0.928212/0.904797`, so this is a nearby dual-seed/retention/boundary-rescue ablation rather than an independent model family. Its own runtime receipt says `candidate_unverified`, has no validated promotion receipt, and explicitly forbids execute/push/submit until that gate exists; the completed kernel is also sorted below the known `0.917` entries. It therefore supplies no evidence above the clean `0.923` frontier and receives no submission slot.

The completed `reyhanksatria/graph-patches-for-cell-tracking-0-917-lb` notebook was also audited because it advertises explicit post-link gap/division patches. The log confirms those patches run, but the 19,102,928-byte output is EXP-008 byte-for-byte: 126,419 nodes, 121,191 edges, 352 divisions, SHA-256 `d7ba9e6a…f2bb`, full audit PASS. The explicit `0.917 LB` title and matching `scoreDescending` group therefore provide the first concrete score attribution for EXP-008. This is useful negative evidence (`-0.006` versus EXP-006), not an additional ensemble voter or frontier candidate.

`backtracking/biohub-medal-v1` is clean and distinct rather than a General-V4 duplicate: 122,921 nodes, 118,426 edges, 321 divisions, SHA-256 `130fbee1…27d1`, full audit PASS. It is nevertheless extremely close to EXP-006 at physical `2 µm` (`0.965096/0.955579` node/edge Jaccard), is sorted below exact-`0.917` EXP-008 and provides no explicit higher score. A strict Medal-V1 + EXP-008 unanimous edge vote over EXP-006 mapped 120,415/93,630 nodes but found zero alternative proposals. It is therefore neither a frontier candidate nor a useful additional teacher vote.

The explicit-`0.916` `hiranorm/new-lb-0-916-infer-ensemble-lf-exp002` artifact is genuinely distinct: 116,166 nodes, 111,762 edges, 80 divisions, SHA-256 `b1e741c8…b0cc`, and physical `2 µm` overlap with EXP-006 of `0.677704/0.622145`. It is not a metric hack, but six boundary nodes have a coordinate exactly `-1` (no negative time or remote hubs), so the raw CSV fails our strict in-volume submission gate. Used read-only with EXP-008 as a second diverse vote, it proposed three conflict-free EXP-006 edge replacements; all three more than doubled edge length and raised constant-velocity residual to `4.24–4.96 µm`. The branch is rejected as a submit/teacher candidate; its failure reinforces motion-vetoed consensus rather than raw vote counting.

Two newly refreshed public notebooks were also downloaded and audited on 2026-08-22. `backtracking/biohub-general-v4` is clean and introduces boundary-track rescue plus a confidence/continuation division guard, but its own source describes the frozen parent as a verified `0.916` pipeline. Its artifact passes schema/topology checks with 119,563 nodes, 115,208 edges and 334 divisions (`SHA-256 eddb0ff9…a5f7`); at `2 µm` it overlaps EXP-006 by `0.833774/0.796725` node/edge Jaccard. This is an ablation source, not a new clean frontier.

`saitejabandaruin/biohub-masterpiece-tracker-version-16` is also clean, but reproduces its explicitly documented `0.913` artifact exactly (`SHA-256 8c1605b5…9e3f`; 120,797 nodes, 116,501 edges, 314 divisions; audit PASS). Its per-frame dual-seed candidate-retention guard is interesting for honest LOEO testing, but its measured leaderboard evidence is below EXP-006.

The refreshed `xiaoleilian/biohub-m001-ens3-sm6-sim2` output is clean but not independent: its 126,419-node / 121,191-edge CSV has exact SHA-256 `d7ba9e6a…f2bb`, identical to EXP-008. Multiple rows in Kaggle's kernel ordering therefore represent the same prediction artifact and must not be counted as ensemble diversity.

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
