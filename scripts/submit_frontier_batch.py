"""Submit only manually reviewed, frozen full-inference candidates.

Dry-run by default. Never retries a submission request after an ambiguous error.
The output CLI ignores /version, so artifacts must have an explicit provenance
receipt, not merely a version-looking download directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from kaggle.api.kaggle_api_extended import ApiGetKernelRequest, KaggleApi

from audit_submission_fast import audit

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "biohub-cell-tracking-during-development"


def read_with_retry(label, operation, attempts=3):
    """Retry only idempotent Kaggle reads; never wrap submission POSTs."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # Kaggle SDK wraps transport errors inconsistently.
            last = exc
            if attempt == attempts:
                break
            print(f"READ_RETRY {label} attempt={attempt}/{attempts}", flush=True)
            time.sleep(attempt * 2)
    raise last


def validate_batch(records):
    if not records or len(records) > 5:
        raise ValueError("Batch must contain one to five explicit candidates")
    for key in ("artifact_sha256", "description"):
        if len({r[key] for r in records}) != len(records):
            raise ValueError(f"Duplicate {key} in batch")


def validate_remote_source(current, record, source_path):
    if current.metadata.current_version_number != record["version"]:
        raise ValueError("Remote latest version changed; re-audit before submitting")
    remote_source = current.blob.source.replace("\r\n", "\n")
    local_source = source_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    mode = record.get("source_validation_mode", "exact_notebook_bytes_after_newline_normalization")
    if mode == "exact_notebook_bytes_after_newline_normalization":
        if remote_source != local_source:
            raise ValueError("Remote source differs from reviewed source")
        return
    if mode != "nonempty_notebook_code_cells_exact":
        raise ValueError(f"Unknown source validation mode: {mode}")

    def nonempty_code_cells(source):
        notebook = json.loads(source)
        return [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "".join(cell["source"]).strip()
        ]

    # Kaggle can rewrite notebook metadata and append empty cells when saving an
    # uploaded ipynb. Executable cells must remain byte-exact and ordered.
    if nonempty_code_cells(remote_source) != nonempty_code_cells(local_source):
        raise ValueError("Remote executable notebook code differs from reviewed source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--experiment", nargs="+")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["candidates"]
    if args.experiment:
        records = [r for r in records if r["experiment"] in args.experiment]
        if set(args.experiment) != {r["experiment"] for r in records}:
            raise ValueError("Unknown experiment requested")
    validate_batch(records)
    api = KaggleApi()
    api.authenticate()
    prior = {
        s.description for s in read_with_retry(
            "competition_submissions", lambda: api.competition_submissions(COMPETITION)
        )
    }
    for record in records:
        description = record["description"]
        if description in prior:
            print(f"SKIP already registered: {description}", flush=True)
            continue
        if record.get("source_review") != "PASS_FULL_INFERENCE_NO_METRIC_HACK":
            raise ValueError("Missing manual source review")
        if not record.get("artifact_version_verified"):
            raise ValueError("Artifact version not verified")
        source_path = ROOT / record["source_path"]
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_hash != record["source_sha256"]:
            raise ValueError("Source SHA drift")
        # Fail closed on latest-version drift: the installed output CLI silently
        # ignores /version. A local vNN-looking directory is not provenance.
        owner, slug = record["kernel"].split("/")
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        def fetch_current():
            with api.build_kaggle_client() as client:
                return client.kernels.kernels_api_client.get_kernel(request)

        current = read_with_retry("get_kernel", fetch_current)
        validate_remote_source(current, record, source_path)
        receipt = audit(ROOT / record["artifact_path"], expected_datasets=4)
        if receipt["sha256"] != record["artifact_sha256"]:
            raise ValueError("Artifact SHA drift")
        status = read_with_retry("kernels_status", lambda: api.kernels_status(record["kernel"]))
        if str(status.status) != "KernelWorkerStatus.COMPLETE":
            raise ValueError(f"Kernel not complete: {status}")
        print(json.dumps({"experiment": record["experiment"], "audit": receipt}), flush=True)
        if not args.submit:
            continue
        limits = read_with_retry(
            "submission_limits", lambda: api.competition_get_submission_limits(COMPETITION)
        )
        print(f"LIVE_LIMITS {limits}", flush=True)
        if limits.num_allowed_now < 1:
            raise RuntimeError("No submission quota remains")
        # Server quota enforcement remains authoritative. No automatic retries.
        response = api.competition_submit_code(
            file_name="submission.csv", message=description,
            competition=COMPETITION, kernel=record["kernel"],
            kernel_version=record["version"],
        )
        print(f"SUBMITTED {record['experiment']} ref={response.ref}", flush=True)
        prior.add(description)


if __name__ == "__main__":
    main()
