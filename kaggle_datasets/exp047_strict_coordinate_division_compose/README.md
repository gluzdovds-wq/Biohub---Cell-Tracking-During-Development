# EXP047 strict coordinate/division composition

Private immutable staging for the pre-registered H047 disjoint composition.

- Node-row parent: EXP014 SHA `c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0`
- Edge-row parent: EXP041 SHA `21a42ffa33c8af7ef44b28f7edaea6a3d9666745139c9c51e132fed41a8fe114`
- Output SHA: `5dd662d8d12f91120425a11a7667059529ce53ad7eab4f756879e9477cf363f2`
- Exact effect: 122,266 nodes, 118,101 edges, 400 divisions and 93,630 coordinates changed relative to the edge parent
- Node IDs and times are identical across both parents; node rows are exact from EXP014 and edge rows are exact from EXP041

The binary CSV is not committed to git. Its build receipt keeps both promotion
gates false: EXP014 must first score at least EXP006 on the account leaderboard,
and the strict `7/4 µm` physical mechanism must be non-negative in total score,
adjusted-edge Jaccard and division TP on both untouched LOEO folds.
