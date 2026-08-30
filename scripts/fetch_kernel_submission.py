"""Download and audit only submission.csv from a completed Kaggle kernel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_submission_fast import audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-datasets", type=int, default=4)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    status = api.kernels_status(args.kernel)
    if str(status.status) != "KernelWorkerStatus.COMPLETE":
        raise RuntimeError(f"Kernel is not complete: {status.status}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files, _ = api.kernels_output(
        args.kernel,
        str(args.output_dir),
        file_pattern=r"^submission\.csv$",
        force=True,
        quiet=True,
        page_size=100,
    )
    submission = args.output_dir / "submission.csv"
    if submission not in {Path(path) for path in files} or not submission.is_file():
        raise FileNotFoundError(
            f"Completed kernel did not expose a root submission.csv: {files}"
        )
    receipt = audit(submission, expected_datasets=args.expected_datasets)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
