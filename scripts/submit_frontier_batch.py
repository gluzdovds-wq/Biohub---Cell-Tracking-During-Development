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

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_submission_fast import audit

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "biohub-cell-tracking-during-development"


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
    if not records or len(records) > 4:
        raise ValueError("Batch must contain one to four explicit candidates")
    hashes = [r["artifact_sha256"] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate artifacts in batch")
    api = KaggleApi()
    api.authenticate()
    prior = {s.description for s in api.competition_submissions(COMPETITION)}
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
        receipt = audit(ROOT / record["artifact_path"], expected_datasets=4)
        if receipt["sha256"] != record["artifact_sha256"]:
            raise ValueError("Artifact SHA drift")
        status = api.kernels_status(record["kernel"])
        if str(status.status) != "KernelWorkerStatus.COMPLETE":
            raise ValueError(f"Kernel not complete: {status}")
        print(json.dumps({"experiment": record["experiment"], "audit": receipt}), flush=True)
        if not args.submit:
            continue
        limits = api.competition_get_submission_limits(COMPETITION)
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
