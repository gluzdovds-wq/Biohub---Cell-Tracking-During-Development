"""Submit one audited output from a completed Kaggle kernel exactly once."""
from __future__ import annotations

import argparse
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def read_with_retry(label, operation, attempts: int = 3):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last = exc
            if attempt == attempts:
                break
            print(f"READ_RETRY {label} attempt={attempt}/{attempts}", flush=True)
            time.sleep(attempt * 2)
    raise last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--file-name", default="submission.csv")
    parser.add_argument("--description", required=True)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    prior = read_with_retry(
        "competition_submissions",
        lambda: api.competition_submissions(args.competition),
    )
    existing = [item for item in prior if item.description == args.description]
    if existing:
        print(f"SKIP already registered ref={existing[0].ref}")
        return
    status = read_with_retry("kernels_status", lambda: api.kernels_status(args.kernel))
    if str(status.status) != "KernelWorkerStatus.COMPLETE":
        raise RuntimeError(f"Kernel is not complete: {status.status}")
    limits = read_with_retry(
        "submission_limits",
        lambda: api.competition_get_submission_limits(args.competition),
    )
    print(f"LIVE_LIMITS {limits}", flush=True)
    if limits.num_allowed_now < 1:
        raise RuntimeError("No submission quota remains")

    # No retry: after any transport ambiguity the account list must be checked.
    response = api.competition_submit_code(
        file_name=args.file_name,
        message=args.description,
        competition=args.competition,
        kernel=args.kernel,
        kernel_version=args.version,
    )
    print(f"SUBMITTED ref={response.ref}")


if __name__ == "__main__":
    main()
