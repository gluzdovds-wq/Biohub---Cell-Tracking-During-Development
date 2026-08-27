# Frontier private-robustness decision

Updated 2026-08-27 after the previous batch completed and four new submissions were registered.

## Verdict

EXP066 at `0.926` is the strongest confirmed control. EXP069R/070 also scored `0.926`, but do not add independent robustness evidence. Public weights used labelled competition movies, so no submitted exact pipeline has honest unseen-embryo OOF. A small public lead cannot certify private ordering.

The three **validation priorities**, not a claim of three available final-submission slots, are EXP066, EXP071 and EXP008. Reconsider the second priority if EXP073/074/075 produce a stronger confirmed result. These are structurally diverse, not proven error-uncorrelated.

## Structural evidence

- EXP066 (`0.926`): dual-seed/DeepCenter, forward association and guarded divisions; main accuracy control.
- EXP071 (`0.923`): bidirectional harmonic association at weight 0.40. Against EXP066, physical 2-micrometre node/edge/division Jaccard is `0.900011 / 0.873648 / 0.274162`.
- EXP008 (`0.917`): independent multi-U-Net detector/physical-linking family. Against EXP066 the corresponding overlap is `0.603450 / 0.545528 / 0.023644`. It sacrifices 0.009 public score for much stronger structural diversity.
- New EXP073 (pending, author `0.927`) versus EXP066: node/edge/division overlap `0.877350 / 0.847210 / 0.235294`.
- New EXP073 versus EXP074 (pending, author `0.927`): node/edge/division overlap `0.903837 / 0.877430 / 0.391727`.

These are label-free visible-test graph comparisons. Low overlap is not necessarily good and is not the correlation of metric errors; min-cost-flow EXP068R is a counterexample (`0.884`). Error correlation requires paired labelled predictions on a common split.

## Honest stability evidence

The saved LOEO experiment evaluates mechanisms trained on the opposite embryo, not the exact public weights. Registered motion scores `0.744130` on held-out `44b6` and `0.595767` on held-out `6bba`, an embryo gap of `0.148363`. The four- and 130-movie resamples in `reports/oof_stability.json` are conditional simulations; neither is a confidence interval for the actual hidden private score. The visible four movies are not proven to equal the scored public subset.

These values cannot map to the absolute `0.926` scale because the trained models differ. Two training embryos also limit any estimate of new-embryo uncertainty, regardless of the number of crops or bootstrap repetitions.

## Affordable validation policy

1. Keep data and checkpoints on Kaggle. No 80+ GB local download is needed. Use Colab only as an alternative cloud GPU with per-movie transfer/resume and no guarantees of free-GPU availability.
2. Freeze a 24-movie, two-embryo pilot using existing reciprocal checkpoints. It screens mechanisms, not exact submitted detector ensembles. See `reports/validation_budget_20260827.json` for movie IDs and measured-runtime extrapolation.
3. Extend cache export to preserve the forward/reverse evidence needed by the selected association policies; disable or retrain learned vetoes that saw held-out data. Existing EXP063/064 alone do not do this.
4. If the pilot is useful, run the larger fixed validation set once and reuse caches on CPU. Report official pooled score, each embryo, paired deltas/intervals, lower-tail performance and division errors. Any subsequently tuned variant requires a reserved confirmation set.
5. For exact EXP008 and dual-seed detector ranking, new fold-trained checkpoints or a genuinely independent annotated embryo are needed. No cheap same-training-data proxy can substitute for this.

Full reproducible cost/scope details are in `reports/OOF_MODEL_COMPARISON.md`. No training or Colab session was launched on 2026-08-27.
