# EXP038 own-seed checkpoint staging

Private, immutable Kaggle staging for one public kernel output needed by a
competition diagnostic.

- Original public kernel: `luffyh04/harshini-1`
- Original output filename: `own_seed_v1_edge_predictor_best.pth`
- Downloaded from the cancelled 2026-08-22 kernel version through the Kaggle CLI
- Bytes: `8,357,783`
- SHA-256: `b1507f6918192c0f5c15fd5091d97ff565b1fab14e67e01101446534abb6a7b7`

The binary is not committed to git. This staging copy does not change the
original attribution or license and is not itself validation or promotion
evidence.

Dataset v3 also includes a fail-closed artifact manifest and `weights.zip`
using the exact public model configuration. This lets the unchanged EXP006
artifact loader materialize the checkpoint without special inference code.
