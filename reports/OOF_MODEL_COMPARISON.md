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

Evidence: the two fold trainings already required roughly `7h10m` and `8h38m` on dual T4. Public full inference takes roughly 37–50 minutes for only four movies; the untouched OOF set has 183 movies. Even a linear T4 extrapolation is around 28–38 GPU-hours before a CPU slowdown. A small CPU smoke test is possible, but it would not be a stable OOF estimate.

The correct compute plan is one GPU inference/cache pass per embryo fold, followed by unlimited cheap CPU linker comparisons. EXP063/064 are built and compile-clean, but Kaggle currently rejects their launch because the account has reached its 30-hour weekly GPU quota.
