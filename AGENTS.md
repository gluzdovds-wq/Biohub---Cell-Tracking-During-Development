# Biohub operating rules

Read `reports/RECOVERY_20260831.md`, then the newest section of
`EXPERIMENTS.md`, before doing any work. Treat repository receipts and live
Kaggle API fields as source of truth; chat memory and notebook titles are not.

## Submission safety (mandatory)

- Never use raw `kaggle competitions submit` or `competition_submit_code`.
  The only allowed mutation path is `scripts/submit_code_file_once.py` after
  its tests pass.
- Biohub is a code competition. A valid candidate must infer from the runtime
  competition test set. A wrapper that reads frozen public `submission.csv`
  files is never hidden-compatible, even when its public CSV passes schema,
  graph, hash, and four-dataset audits.
- Before a POST, record the hypothesis, parent, source/kernel version and SHA,
  hidden-test dataflow, expected output filename, local checks, and promotion
  gate in `EXPERIMENTS.md` or a committed receipt.
- Run the exact source in a clean Kaggle version with Internet off. Verify
  successful execution, one root `submission.csv`, exact columns/types,
  finite values, dynamic runtime dataset IDs, graph invariants, and output SHA.
- Submit each candidate through the guarded helper, then query its full API
  object, including `error_description`, `status`, `public_score`,
  `total_bytes`, ref, URL, and quota. A pending earlier submission does not
  block the next independently pre-registered and audited candidate.
- `COMPLETE` with an empty score is a failure until proven otherwise. Stop on
  the first anomaly, preserve the receipt, diagnose it, and do not spend more
  quota that day.
- Do not claim `PASS_FULL_INFERENCE` unless the code actually reads the runtime
  competition test data and performs inference or a hidden-compatible dynamic
  transform. Public artifact replay is not inference.

## Daily medal portfolio (mandatory)

- Split the five daily submission slots into two non-interchangeable tracks:
  slots 1-2 are `GOLD_PUBLIC`; slots 3-5 are `PRIVATE_ROBUST`.
- `GOLD_PUBLIC` searches for a path from the stable bronze frontier toward the
  current public gold-medal boundary. It may reproduce, repair, adapt, ensemble,
  or retune strong open notebooks, but every candidate must still pass the full
  hidden-compatible inference and receipt checks above.
- `PRIVATE_ROBUST` searches for reliable post-close performance. Promote only
  candidates supported by honest embryo-level holdout/OOF evidence, stability
  checks, or a credible mechanism-diversity argument. A tiny public-LB gain or
  an open-notebook title is not sufficient evidence for this track.
- Label every candidate and receipt with its track and daily slot number before
  POST. Compare public-track candidates with the current public frontier and
  private-track candidates with the current private-robust portfolio.
- Do not borrow unused slots between tracks merely to exhaust quota. Leave a
  slot unused when no candidate qualifies. Submission-safety stop conditions
  override the quota allocation.

## Experiment and context hygiene

- `EXPERIMENTS.md` is append-only evidence: correct false claims explicitly;
  never erase failed experiments or relabel leaky/proxy evidence as honest OOF.
- Keep current decisions in the newest dated section. Historical queues stay
  below and are not instructions.
- Do not submit correlated sweeps merely to fill quota. Prefer one diagnostic,
  observe it, then decide. Final selection requires mechanism diversity and
  honest paired evidence; a tiny public-LB delta is not private validation.
- Preserve unrelated user files such as `goals-log.txt`.
