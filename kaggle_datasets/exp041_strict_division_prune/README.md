# EXP041 strict division-prune staging

Private immutable staging for the H041 nested high-precision diagnostic.

- Inputs and mapping/singleton contract are exactly EXP040
- Output SHA-256: `21a42ffa33c8af7ef44b28f7edaea6a3d9666745139c9c51e132fed41a8fe114`
- Frozen gates: mutual mapping `2 µm`, rejected-child CV residual `>=7 µm`, residual margin `>=4 µm`
- Effect: 55 edge removals, 455 to 400 division parents; every removal is a strict subset of EXP040's 160 changes
- All 122,266 node rows and coordinates are unchanged

The binary CSV is not committed to git. The Kaggle dataset contains its full
receipt and topology audit. H041 was frozen while EXP040's reused-label audit
was still running; reused labels remain reject-only.
