# Leaderboard test queue

## Current update: 2026-09-01

EXP093 is COMPLETE `0.933` and is the scored anchor. EXP098 is the sole
immediately authorized submission: clean full-inference C35 with corrected
fallback/provenance and bidirectional weight `0.15`. It passed exact source,
runtime-dataset, output SHA, schema and graph audits. C36, C37 and C38 are held
behind EXP098's account score; old EXP060/061 are not promoted automatically
because their `0.920` parent is now well behind the frontier.

After EXP098 is registered, no further slot may be used while it is pending,
scoreless or erroneous. Canonical receipt: `submission_batch_20260901.json`.

## Incident correction: 2026-08-31

EXP094–097 are failed invalid-format submissions, not pending score results.
Their code replayed frozen public parent submissions and did not infer on the
hidden runtime test set. They provide no LB evidence and are removed from the
active queue. EXP093 remains PENDING until the full Kaggle API object shows a
terminal state. No further candidate may be sent until the daily quota resets,
EXP093 is resolved, and the guarded hidden-compatibility protocol in
`AGENTS.md` passes. See `RECOVERY_20260831.md` for the canonical incident state.

## Current update: 2026-08-31

The August 30 batch is fully scored: EXP088 EMA `0.926`, EXP089 EMA-0.5
`0.924`, EXP090 edge-threshold-0.40 `0.928`, EXP091 division-heavy `0.926`,
and EXP092 fine-tuned linker + D4 `0.900`. The only materially graph-diverse
arm, EXP092, lost `0.031` to the current account leader and is rejected as a
final hedge. EXP090's small threshold change tied the previous frontier but did
not beat EXP083 (`0.931`).

Today's quota is five used / zero remaining. EXP093 is the exact audited C33 v1
public `0.933` anchor (`55907915`). EXP094–097 are our paired coordinate-only
experiments on the exact same 115,046-edge / 291-division graph:

- EXP094: C29 mutual-nearest localization donor, `alpha=0.25` (`55908273`).
- EXP095: C29 donor, `alpha=0.50` (`55908462`).
- EXP096: C30 donor, `alpha=0.25` (`55908629`).
- EXP097: lower-overlap Comb2 donor, `alpha=0.25` (`55908683`).

All five hidden-compatible code submissions completed local output/receipt/hash
audits. Their account scores were not yet populated when this record was
frozen. The four coordinate arms are controlled localization tests, not
independent graph hedges; a tie must not be interpreted as evidence of private
robustness. A proposed C33/C29/C30 edge-consensus arm was rejected before LB:
three of its four replacements worsened both link length and constant-velocity
residual.

Next priority is no longer another correlated public sweep. First record these
five scores, then run whole-movie official-metric OOF for a small set of
mechanism-distinct candidates. Final slot A is provisionally EXP093 (or the
smallest coordinate dose that clearly improves it); slot B requires either
physical edge overlap below `0.85` within `0.005` of the leader or same-sign
paired OOF on both embryos. Frozen receipts and hashes:
`submission_batch_20260831.json`; selection policy:
`FINAL_SELECTION_20260830.md`.

The sections below are retained as historical records.

## Current update: 2026-08-30

August 29 results are complete: EXP083 Stephen `0.931`, EXP084 SDW85 `0.929`,
EXP085 Evgen v15 `0.928`, EXP086 Anvith `0.928`, and controlled EXP087 SDW90
`0.926`. The SDW90 leaky train-movie proxy did not provide a reliable promotion
signal; exact OOF remains unavailable.

Today's five full-code submissions are registered and PENDING: EXP088 full EMA
`55882197`, EXP089 controlled EMA-0.5 `55882683`, EXP090 edge-threshold-0.40
`55882198`, EXP091 division-heavy `55882203`, and EXP092 offline fine-tuned
linker + D4 `55882642`. All outputs pass four-movie schema/topology audits.
EXP092 exactly reproduces its reviewed public-parent SHA with Internet disabled
and has physical edge overlap `0.744633` to EXP083; it is the serious hedge.
EXP089 has `0.981219` edge overlap to EMA-1.0 and is only a paired sensitivity
test. Daily quota is five used / zero remaining. Frozen receipts:
`submission_batch_20260830.json`; final policy: `FINAL_SELECTION_20260830.md`.

The paragraphs below retain the previous-day record for context.

August 28 results are complete: EXP078 SDW70, EXP079 Flex v22 and EXP080 SDW75 each scored **0.928**; EXP081 VEL10 scored `0.926`; EXP082 MTL8 scored `0.923`. Confirmed account best is **0.928**. The tie is correlated: SDW70/SDW75 physical node/edge overlap is `0.970384/0.963701`, while SDW70/Flex is `0.861317/0.829558`.

Five new full-code submissions are registered and PENDING: EXP083 Stephen v1 (`55858606`, verified clean author score `0.931`), EXP084 SDW85 v1 (`55858609`, actual `0.929` rather than the title's `0.938`), EXP085 Evgen v15 (`55858612`, `0.928`), EXP086 Anvith v1 (`55858614`, `0.928`) and EXP087 controlled SDW90 (`55859147`). The first four are attributed public reproductions. EXP087 is our controlled detector-mixture `0.85→0.90` fork; its Kaggle output passed full audit. Daily quota: five used / zero available. Frozen receipts: `submission_batch_20260829.json`.

The current `0.928–0.931` frontier is still a single broad dual-seed/harmonic family. Stephen/Flex physical node/edge overlap is `0.925603/0.910946`, SDW85/SDW75 is `0.951036/0.940384`, and SDW90/SDW85 is `0.973106/0.967084`. A higher public score within this cluster is worth testing but does not prove better private robustness. Exact submitted-model OOF remains unavailable; EXP087's built-in proxy `0.9294` is leaky and is not recorded as OOF.

Version correction: Evgen v11, not v12, is the 0.927 version. The installed output CLI ignores `/version`; previous claims of exact historical output downloads are invalid unless independently verified. In particular, the latest Flex v17/Ahmet v1 duplicate does not itself prove submitted Flex v11 identity. The old dated sections below are historical records, not the current queue.

EXP077's own local-flow CPU pilot is COMPLETE: four movies, 20.30 minutes, paired delta zero on both embryos, no annotated divisions. No promotion. The four candidate caches are now local (1.08 MB total), without an image download. Next compute priority is bounded CPU validation / new cached-linker hypotheses; neither the 24- nor 183-movie continuation has started. See `OOF_MODEL_COMPARISON.md`.

## Results from 2026-08-24

- EXP008 (`55732259`): `0.917`, detector diversity is useful but below EXP005.
- EXP007 (`55732491`): `0.900`, reject association-logit D4 TTA.
- EXP039 (`55732718`): `0.906`, reject the independent secondary checkpoint.
- EXP028 (`55732720`): `0.919`, reject the relaxed DeepCenter veto versus EXP005 `0.920`.

## Submitted on 2026-08-25

- EXP065 (`55761017`, pending): exact clean source-attributed `0.927` frontier.
- EXP066 (`55761018`, pending): exact clean source-attributed `0.926` division-sub frontier.
- EXP067 (`55761031`, pending): General-V8 continuation guard / daughter completion, edge overlap `0.793` to EXP005.
- EXP068 (`55761370`, invalid hidden format): public-output wrapper rejected. Retry exact full-inference `pawanmali/biohub-mcflow-v1` v2 after the UTC quota reset.
- EXP069 (`55761371`, invalid hidden format): public-output wrapper rejected. Retry exact full-inference `flexonafft/biohub-harmonic-fusion` v11 after the UTC quota reset.

The two wrapper errors did consume quota despite receiving no score. Exact immutable-version retries were attempted immediately and rejected with the explicit five-per-day limit. `scripts/resubmit_exp068_exp069.py` is duplicate-safe and ready for the next UTC reset.

## Submitted on 2026-08-26

- EXP068R (`55781325`, pending): corrected full-inference MCFlow v2.
- EXP069R (`55781326`, pending): corrected full-inference Flex v11.
- EXP070 (`55781466`, pending): Ahmet v1; subsequently proven byte-identical to EXP069R.
- EXP071 (`55781467`, pending): bidirectional/harmonic weight `0.40`, materially different graph.
- EXP072 (`55781468`, pending): controlled reverse-association weight `0.20`.

All five daily slots were registered as full code submissions. Version-specific output audit after registration found EXP069R and EXP070 share SHA `2dbb8d02…4fa7`; treat today as four unique graphs.

## Ready immediately after GPU quota resets

1. EXP060: exact `0.920` EXP005 topology plus detector-consensus coordinates, `alpha=0.50`.
2. EXP061: the same matches/topology with conservative `alpha=0.25`.

Both outputs come from one dual-inference notebook. Public artifacts and graph invariants already pass; only the hidden-compatible Kaggle run is missing.

## Next controlled LB hypotheses

1. Distance-adaptive EXP005/EXP008 coordinate blend: use a smaller dose near the `2 µm` match gate and `0.5` only under strong agreement. Compare against fixed EXP060/061, not directly tune an arbitrary curve on LB.
2. EXP005 division ablation ladder: exact harmonic graph with original 67 divisions, only physically weak divisions removed, and divisions disabled. Emit all arms from one inference run to isolate division prevalence on public test.
3. Detection-count calibration around EXP005: frozen neighboring thresholds around `0.96875`, with node-count/edge changes audited before score disclosure. Kill if both embryo folds disagree once comparable OOF exists.
4. Association-family comparison: harmonic forward/reverse versus weak learned registered and ILP only after EXP063/064 report both embryo signs.
5. Cached physical linkers: Kalman/constant-velocity and particle-filter policies promoted only when paired OOF is non-negative on both embryos. Min-cost-flow receives one explicit exploratory LB measurement as EXP068 because its edge overlap with EXP005 is only `0.602`.

The queue deliberately avoids metric hacks, public-CSV wrappers, registered-relink descendants already falsified at `0.905`, and intensity refinements falsified at `0.893`.
