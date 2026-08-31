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
- Submit one candidate only. Then query full API objects, including
  `error_description`, `status`, `public_score`, `total_bytes`, ref, URL, and
  quota. Do not submit another candidate while any earlier daily submission is
  pending, scoreless, or erroneous.
- A derived child may be submitted only after its parent is COMPLETE, has no
  error, and has a non-empty account score satisfying the pre-registered gate.
- `COMPLETE` with an empty score is a failure until proven otherwise. Stop on
  the first anomaly, preserve the receipt, diagnose it, and do not spend more
  quota that day.
- Do not claim `PASS_FULL_INFERENCE` unless the code actually reads the runtime
  competition test data and performs inference or a hidden-compatible dynamic
  transform. Public artifact replay is not inference.

## Experiment and context hygiene

- `EXPERIMENTS.md` is append-only evidence: correct false claims explicitly;
  never erase failed experiments or relabel leaky/proxy evidence as honest OOF.
- Keep current decisions in the newest dated section. Historical queues stay
  below and are not instructions.
- Do not submit correlated sweeps merely to fill quota. Prefer one diagnostic,
  observe it, then decide. Final selection requires mechanism diversity and
  honest paired evidence; a tiny public-LB delta is not private validation.
- Preserve unrelated user files such as `goals-log.txt`.
