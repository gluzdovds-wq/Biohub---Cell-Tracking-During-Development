# Starting Kaggle kernels

All four kernels are private forks used to establish a reproducible score ladder. Original public code, markdown and attribution are retained.

- `exp001_nearest_neighbor` -> `dmitriigluzdov/biohub-exp001-nearest-neighbor-sanity`
- `exp002_rule_based` -> `dmitriigluzdov/biohub-exp002-rule-based-classical`
- `exp003_clean_single_seed` -> `dmitriigluzdov/biohub-exp003-clean-single-seed`
- `exp004_dual_seed_blend` -> `dmitriigluzdov/biohub-exp004-dual-seed-logit-blend`
- `exp005_harmonic_frontier` -> `dmitriigluzdov/biohub-exp005-harmonic-frontier`
- `exp006_kimi_division_frontier` -> `dmitriigluzdov/biohub-exp006-kimi-division-frontier`

The notebooks have internet disabled and attach the official competition data. GPU notebooks retain the public checkpoint/support datasets declared by their source metadata. EXP-005 is the clean public `0.920` family: dual-seed detection with harmonic forward/reverse association and DeepCenter-gated graph repair. EXP-006 adds widened division geometry protected by mutual-nearest-neighbor, divergence, and DeepCenter vetoes; its source run coincides with the author's public `0.923` result.
