# Biohub OOF stability and leaderboard trust

Generated from the immutable EXP050/EXP051 leave-one-embryo-out receipts with 10,000 deterministic movie bootstraps (`seed=20260823`). Machine-readable results are in `reports/oof_stability.json`.

## Result

The public leaderboard is useful as a hidden-runtime and broad-family check, but it is too small and too domain-conditional to be our primary model-selection target. Promote changes using paired embryo-disjoint OOF deltas that agree in sign on both embryos.

- Registered Hungarian scores `0.744130` on the 63 untouched `44b6` movies and `0.595767` on the 120 untouched `6bba` movies. The between-embryo gap is `0.148363`.
- Within-embryo movie-bootstrap 95% intervals are `0.702443–0.785118` and `0.569055–0.622353`. The fold gap is substantially larger than ordinary movie-sampling uncertainty.
- Pooled registered OOF is `0.615980`; a stratified movie bootstrap gives `0.591559–0.640048`.
- A four-movie public-like resample gives a much wider `0.451683–0.786690` 95% interval. This is conditional on our two observed embryo domains and still excludes unseen-embryo domain shift.
- A 130-movie private-like resample gives `0.585632–0.645486`; it is much more stable against ordinary movie sampling, but its domain mixture is unknown.
- Random five-fold movie CV has fold-score standard deviation `0.0271`, while the actual leave-one-embryo gap is `0.1484`. Random folds mix embryo identity and substantially understate deployment shift.

## Paired method decisions

- Registered motion minus greedy is `+0.164190` on `44b6`, `+0.129587` on `6bba`, and `+0.134782` pooled. Stratified paired bootstrap 95% interval: `+0.114758–+0.155750`; all 10,000 replicates are positive. This is promotion-grade.
- Weak learned tie-break minus pure registered is `+0.000734`, `+0.000771`, and `+0.000767` pooled. Its paired bootstrap interval is `+0.000186–+0.001357`, with 99.48% positive replicates. It is supported, but the effect is too small to tune against a four-movie leaderboard.

## What the discussions add

- The host confirmed that train has only two embryo IDs, and that test has no embryo-ID overlap with train and is roughly similar in size. This makes embryo-disjoint validation the correct stress test: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793
- In the same thread, participants report poor correlation between deep-learning local CV and public LB and speculate that the public/private embryo composition may differ. That is community evidence, not an official split disclosure.
- A recent public notebook explicitly frames honest CV as 195 volumes versus only four public-LB volumes: https://www.kaggle.com/code/dariushafshar/honest-cv-on-195-volumes-public-lb-is-only-4
- The official metric micro-averages edge counts and weights adjusted edge Jaccard by sample size; node matching is capped at `7 µm`, and division contributes only `0.1`: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- The division metric was publicly shown to be gameable; the host acknowledged it and the leaderboard was subsequently rescored. Current official documentation now uses local directed topology rather than graph-wide reachability: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655

## Biohub versus ROGII shake-up

Biohub can have a large rank shake-up, but the failure mechanism differs from ROGII.

ROGII used row-pooled RMSE, so a small set of difficult wells or catastrophic path offsets could dominate through squared error. Its public split was about 26% and multiple final writeups show very large public/private movements. Biohub's Jaccard-based score does not have an unbounded squared-error tail: a bad edge changes TP/FP/FN counts, and the final score is dominated by micro-averaged adjusted edge Jaccard. That reduces the specific RMSE-outlier failure mode.

Biohub remains highly shake-up-prone because the public score appears to cover only four movies, train exposes only two embryo domains, test embryos are unseen, and large samples contribute more to the official micro-average. A systematic detector/count/linker mismatch on one private embryo can therefore move the final score substantially even without RMSE.

Our operating rule is consequently stricter than “trust CV” in the abstract:

1. Trust the sign and breadth of paired leave-one-embryo-out deltas.
2. Use public LB to confirm executable hidden inference and reject gross regressions.
3. Ignore tiny public deltas unless the same mechanism is positive on both embryo folds.
4. Choose final submissions from structurally different, CV-supported candidates rather than the two highest noisy public scores.
