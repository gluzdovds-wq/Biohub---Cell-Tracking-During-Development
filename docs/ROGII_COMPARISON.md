# Biohub versus our ROGII campaign

The domains differ, but the modelling pattern is recognisably related.

## What transfers

- Both are structured inference problems with a sequential latent state constrained by physics/geometry.
- Both combine noisy observations with temporal continuity and rare discontinuities: geological path modes/jumps in ROGII, motion/division/gaps in Biohub.
- Both benefit from multiple candidate hypotheses followed by constrained decoding or global selection.
- Particle filtering, HMM/Viterbi-style decoding, ensembling and uncertainty gating are conceptually relevant to both.
- Validation grouping matters: by well in ROGII, by embryo in Biohub. Random row/crop splits leak identity and structure.
- Public LB can be optimistic or unrepresentative; paired grouped CV and immutable experiment logs remain essential.

## What does not transfer directly

- ROGII predicted one continuous TVT value per row and used pooled RMSE. Biohub outputs a variable-size graph and uses matched edge/division Jaccards plus a node-count adjustment.
- ROGII observation was mainly 1D gamma-ray/typewell alignment along a well. Biohub input is dense anisotropic 4D microscopy.
- A particle filter was a natural primary state estimator in ROGII. In Biohub, a per-cell PF alone cannot solve detection, identity competition, births/deaths and branching; it is better used as a motion prior or candidate generator inside a multi-object graph tracker.
- Biohub has permutation/assignment competition between thousands of similar cells. Global bipartite/min-cost-flow/ILP constraints are much more central.

## Evidence from our Kaggle history

Our ROGII kernels included `PF Baseline`, `rogii_hmm_smoother_grandmaster`, multi-seed likelihood PF, trajectory selectors, CatBoost, Viterbi/HMM and diverse ensembles. The strongest transferable habit is not any one model: it is separating observation likelihood from physical decoding, tracking uncertainty/modes, and logging every legal CV/LB delta.

## Practical translation

- ROGII PF state `(stratigraphic level, rate)` becomes a per-track motion state `(z,y,x,vz,vy,vx)` or a small mixture of such states.
- ROGII likelihood from GR/typewell alignment becomes appearance/transformer edge likelihood plus registration-corrected motion likelihood.
- ROGII path selector becomes min-cost-flow/ILP assignment across all cells, with birth/death/division constraints.
- ROGII uncertainty gating becomes conservative repair: only add a gap edge or division when independent models agree.

The common core is physics-guided sequential inference. The crucial Biohub addition is multi-object data association and lineage topology.

## Shake-up risk after grouped OOF

ROGII's public/private instability was amplified by row-pooled RMSE: rare wells with a coherent large path error receive a squared penalty and can dominate the aggregate. Biohub removes that particular unbounded tail, but it replaces it with severe domain-composition uncertainty. Only two embryo IDs are available for training, test embryo IDs do not overlap, and the public Biohub score is reported to cover only four movies.

Our registered-linker LOEO score changes from `0.744130` on held-out `44b6` to `0.595767` on held-out `6bba`, a `0.148363` domain gap. A four-movie bootstrap has a `0.451683–0.786690` 95% range even before accounting for unseen-embryo shift. Therefore Biohub can shake up as strongly in rank as ROGII, but due to embryo/domain mixture and tiny public sample rather than RMSE outliers.

The practical transfer from ROGII is exact: use grouped CV for selection and LB for execution sanity. In Biohub the grouping unit must be embryo, not movie or crop. Full receipts and rationale are in `reports/OOF_STABILITY.md` and `reports/oof_stability.json`.
