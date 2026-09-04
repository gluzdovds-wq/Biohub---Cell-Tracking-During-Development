"""Submit one hidden-compatible Kaggle code candidate, fail closed, once."""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest


STATIC_PUBLIC_SUBMISSION_PATTERNS = (
    re.compile(r"rglob\(\s*['\"]submission\.csv['\"]\s*\)", re.IGNORECASE),
    re.compile(r"glob\(\s*['\"][^'\"]*submission\.csv['\"]\s*\)", re.IGNORECASE),
)

DATASET_INPUT_PATTERN = re.compile(
    r"/kaggle/input/datasets/([a-z0-9][a-z0-9_-]*)/([a-z0-9][a-z0-9_-]*)",
    re.IGNORECASE,
)

RETRYABLE_KAGGLE_SYSTEM_ERROR = (
    "A system error. Please try resubmitting to resolve the error and contact "
    "Kaggle Support if it persists."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_hidden_compatibility_source(
    source_path: Path,
    competition: str,
    expected_sha256: str,
) -> None:
    """Reject wrappers over frozen public predictions.

    A code-competition submission must discover the runtime test set and produce
    predictions for it. Reading parent ``submission.csv`` files from Kaggle
    inputs reproduces public dataset IDs during the hidden rerun and is not a
    hidden-compatible inference path.
    """
    if not source_path.is_file():
        raise RuntimeError(f"Source file is missing: {source_path}")
    observed_sha256 = file_sha256(source_path)
    if observed_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f"Source SHA mismatch: {observed_sha256} vs {expected_sha256.lower()}"
        )
    source = source_path.read_text(encoding="utf-8", errors="replace")
    if any(pattern.search(source) for pattern in STATIC_PUBLIC_SUBMISSION_PATTERNS):
        raise RuntimeError(
            "Hidden-compatibility gate failed: source discovers frozen parent "
            "submission.csv artifacts. A Biohub code submission must run on the "
            "runtime competition test set, not replay public predictions."
        )
    if competition not in source:
        raise RuntimeError(
            "Hidden-compatibility gate failed: the exact competition slug is not "
            "present in the submitted source, so runtime test discovery is not "
            "auditable."
        )


def validate_remote_kernel_identity(
    response,
    kernel: str,
    version: int,
    competition: str,
    expected_source_sha256: str,
) -> None:
    """Fail closed on version drift or unavailable explicit dataset inputs."""
    metadata = response.metadata
    current_version = int(metadata.current_version_number)
    if current_version != version:
        raise RuntimeError(
            f"Remote kernel version drift: requested v{version}, current v{current_version}"
        )

    remote_source = response.blob.source
    remote_sha256 = hashlib.sha256(remote_source.encode("utf-8")).hexdigest()
    if remote_sha256 != expected_source_sha256.lower():
        raise RuntimeError(
            f"Remote source SHA mismatch for {kernel}: {remote_sha256} vs "
            f"{expected_source_sha256.lower()}"
        )

    competitions = set(metadata.competition_data_sources or [])
    if competition not in competitions:
        raise RuntimeError(
            f"Remote kernel does not attach competition source {competition!r}"
        )

    attached_list = list(metadata.dataset_data_sources or [])
    if any(not str(item).strip() for item in attached_list):
        raise RuntimeError("Remote kernel metadata contains a blank dataset attachment")
    attached = {str(item).lower() for item in attached_list}
    referenced = {
        f"{match.group(1)}/{match.group(2)}".lower()
        for match in DATASET_INPUT_PATTERN.finditer(remote_source)
    }
    missing = sorted(referenced - attached)
    if missing:
        raise RuntimeError(
            "Remote kernel source references dataset inputs that are not attached: "
            + ", ".join(missing)
        )


def get_current_kernel(api: KaggleApi, kernel: str):
    parts = kernel.split("/")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError("Kernel must be owner/slug")
    request = ApiGetKernelRequest()
    request.user_name = parts[0]
    request.kernel_slug = parts[1]
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(request)


def _status_text(item) -> str:
    return str(getattr(item, "status", ""))


def _score(item):
    return getattr(item, "public_score", None)


def _error(item):
    return getattr(item, "error_description", None)


def validate_submission_history(
    submissions,
    now: datetime | None = None,
    retry_system_error_ref: int | None = None,
) -> None:
    """Allow pending experiments, but stop after a recorded scoring anomaly."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    todays = [item for item in submissions if getattr(item, "date", None).date() == today]
    failed = [item for item in todays if _error(item)]
    if retry_system_error_ref is not None:
        matches = [
            item
            for item in failed
            if int(getattr(item, "ref", -1)) == retry_system_error_ref
            and _error(item) == RETRYABLE_KAGGLE_SYSTEM_ERROR
            and _status_text(item) in {"ERROR", "SubmissionStatus.ERROR"}
            and not _score(item)
            and int(getattr(item, "total_bytes", 0) or 0) == 0
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "System-error retry gate failed: the authorized ref is not an "
                "exact zero-byte Kaggle infrastructure error"
            )
        failed = [item for item in failed if item is not matches[0]]
        print(
            f"AUTHORIZED_SYSTEM_ERROR_RETRY ref={retry_system_error_ref}",
            flush=True,
        )
    if failed:
        details = "; ".join(
            f"ref={item.ref} description={item.description!r} error={_error(item)}"
            for item in failed
        )
        raise RuntimeError(
            "Daily fail-closed gate: an earlier submission failed. Diagnose and "
            f"record it before any further submission. {details}"
        )
    completed_without_score = [
        item
        for item in todays
        if _status_text(item) == "SubmissionStatus.COMPLETE" and not _score(item)
    ]
    if completed_without_score:
        details = "; ".join(
            f"ref={item.ref} description={item.description!r}"
            for item in completed_without_score
        )
        raise RuntimeError(
            "Daily fail-closed gate: an earlier submission completed without a "
            f"score. Diagnose and record it before continuing. {details}"
        )


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
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--retry-system-error-ref", type=int)
    args = parser.parse_args()

    audit_hidden_compatibility_source(
        args.source_file,
        args.competition,
        args.source_sha256,
    )

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
    validate_submission_history(
        prior,
        retry_system_error_ref=args.retry_system_error_ref,
    )
    remote = read_with_retry(
        "get_current_kernel",
        lambda: get_current_kernel(api, args.kernel),
    )
    validate_remote_kernel_identity(
        remote,
        args.kernel,
        args.version,
        args.competition,
        args.source_sha256,
    )
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
