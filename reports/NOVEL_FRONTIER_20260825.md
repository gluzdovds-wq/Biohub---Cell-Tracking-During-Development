# Novel clean frontier scan — 2026-08-25

## Decision

The five-slot batch deliberately mixes three risk levels instead of spending all probes on correlated radius changes:

1. EXP065 and EXP066 freeze the newly visible clean `0.927/0.926` frontier.
2. EXP067 tests continuation guards and daughter completion with only `0.793` edge overlap to EXP005.
3. EXP068 targets a global min-cost-flow association family with only `0.602` edge overlap. Its first wrapper submission was invalid on the hidden rerun; immutable full-inference v2 is the corrected retry.
4. EXP069 keeps a high-score calibration arm while materially increasing the number of predicted divisions. Its first wrapper submission was invalid; immutable full-inference v11 is the corrected retry.

## Audited artifacts

- EXP065: `33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a`; 122,207 nodes, 117,919 edges, 263 divisions; PASS.
- EXP066: `5f4ec83d56b9fd0620473eed1e92a5a59932bf1c64e231111073ec008de2f7bc`; 120,748 nodes, 116,536 edges, 384 divisions; PASS.
- EXP067: `c7ea6f9893d7a57af8795dbd0928504259b66f7330d96871a1e1cbbf110b7d6b`; 120,236 nodes, 116,246 edges, 492 divisions; PASS.
- EXP068: `f7cf397733602d77ba7ec51b36472e89b6af7f7e379a6d3dcceaf18beab6e34c`; 116,860 nodes, 110,063 edges, 5 divisions; PASS.
- EXP069: `fd77de2afe9747dc873d57f3c46488d60d885d977fb43a7f99fcf7bb308974a2`; 120,866 nodes, 116,712 edges, 470 divisions; PASS.

All PASS artifacts have finite non-negative coordinates, valid adjacent-time edges, maximum indegree 1, maximum outdegree 2, and no fabricated hub structure.

## Rejected novel candidates

The learned-flow candidate was genuinely diverse (`0.638` physical edge overlap to EXP005) but contains three node rows with `x` or `y = -1`. It is withheld under the strict coordinate gate rather than consuming a slot with a known-invalid boundary representation. A Cellpose/Cyto2 candidate produced an empty 56-byte output and was rejected.

## Interpretation contract

EXP068 is exploratory, not a promotion based on an implied score. Its near-absence of divisions makes it especially useful for testing whether public LB rewards conservative global continuation, but a weak result will not falsify min-cost-flow generally without honest identical-detection OOF. EXP069 is intentionally correlated with EXP066 and isolates division prevalence more closely than the other three submissions.

Honest embryo-disjoint comparison remains EXP063/064. Both sources are frozen and compile, but Kaggle rejected their launches before execution because the account has exhausted its 30-hour weekly GPU quota. No public-weight validator is relabeled as OOF.

## Hidden-rerun correction

Submissions `55761370` and `55761371` used private CPU wrappers around saved public-test outputs. Kaggle reran those wrappers against hidden data, so both copied graphs for the wrong movies and completed with the generic incorrect-format error. The local CSV schema and graph audits were valid but could not establish hidden compatibility.

The corrected submissions are the original inference notebooks pinned to immutable versions: `pawanmali/biohub-mcflow-v1` v2 and `flexonafft/biohub-harmonic-fusion` v11. A direct retry was attempted after diagnosis, but Kaggle confirmed that the two invalid submissions had consumed the remaining daily slots. The duplicate-safe retry command is `python scripts/resubmit_exp068_exp069.py` after the next UTC quota reset.
