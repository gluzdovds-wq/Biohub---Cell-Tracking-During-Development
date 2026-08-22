# EXP040 division-prune staging

Private immutable staging for the H040 reject-only diagnostic.

- Base EXP006 SHA-256: `5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d`
- Frozen EXP005 voter SHA-256: `9507eccb663635e1f761b8e3e2357952c8e208b24857ab4a16be60e6ecb66425`
- Frozen EXP008 voter SHA-256: `d7ba9e6af86a6bb0be8bd04a36d0c61564e857e03fbadf9a81508211a4a4f2bb`
- Output SHA-256: `9f0b0711b5ac0b078c5fb24332c2604c09118013116bc6fbe4d6f4e2eaa4a5e3`
- Frozen gates: mutual mapping `2 µm`, rejected-child CV residual `>=4 µm`, residual margin `>=2 µm`
- Effect: 160 edge removals, 455 to 295 division parents; all 122,266 node rows and coordinates are unchanged

The binary CSV is not committed to git. The Kaggle dataset also contains the
full build receipt and independent topology audit. Reused-label results may
reject this candidate but may not promote it.
