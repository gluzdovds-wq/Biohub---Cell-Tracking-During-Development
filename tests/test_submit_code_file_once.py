from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.submit_code_file_once import (
    audit_hidden_compatibility_source,
    file_sha256,
    validate_remote_kernel_identity,
    validate_submission_history,
)


NOW = datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)


def submission(**overrides):
    values = {
        "ref": 1,
        "date": NOW,
        "description": "candidate",
        "status": "SubmissionStatus.COMPLETE",
        "public_score": "0.933",
        "error_description": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_history_blocks_any_prior_error():
    rows = [submission(error_description="incorrect format", public_score=None)]
    with pytest.raises(RuntimeError, match="earlier submission failed"):
        validate_submission_history(rows, NOW)


def test_history_accepts_pending_submission():
    validate_submission_history(
        [submission(status="SubmissionStatus.PENDING", public_score=None)], NOW
    )


def test_history_blocks_completed_scoreless_submission_without_error():
    with pytest.raises(RuntimeError, match="completed without a score"):
        validate_submission_history([submission(public_score=None)], NOW)


def test_history_accepts_scored_completed_submission():
    validate_submission_history([submission()], NOW)


def test_source_gate_rejects_frozen_public_submission_wrapper(tmp_path: Path):
    source = tmp_path / "candidate.py"
    source.write_text(
        "COMP='biohub-cell-tracking-during-development'\n"
        "paths = Path('/kaggle/input').rglob('submission.csv')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="frozen parent"):
        audit_hidden_compatibility_source(
            source,
            "biohub-cell-tracking-during-development",
            file_sha256(source),
        )


def test_source_gate_requires_competition_runtime_path(tmp_path: Path):
    source = tmp_path / "candidate.py"
    source.write_text("print('no runtime test discovery')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="competition slug"):
        audit_hidden_compatibility_source(
            source,
            "biohub-cell-tracking-during-development",
            file_sha256(source),
        )


def test_source_gate_accepts_auditable_runtime_inference(tmp_path: Path):
    source = tmp_path / "candidate.py"
    source.write_text(
        "TEST_ROOT = '/kaggle/input/biohub-cell-tracking-during-development/test'\n",
        encoding="utf-8",
    )
    audit_hidden_compatibility_source(
        source,
        "biohub-cell-tracking-during-development",
        file_sha256(source),
    )


def remote_kernel(source: str, **metadata_overrides):
    metadata = {
        "current_version_number": 3,
        "competition_data_sources": ["biohub-cell-tracking-during-development"],
        "dataset_data_sources": ["owner/weights-v1"],
    }
    metadata.update(metadata_overrides)
    return SimpleNamespace(
        blob=SimpleNamespace(source=source),
        metadata=SimpleNamespace(**metadata),
    )


def test_remote_identity_accepts_exact_version_source_and_inputs():
    source = "WEIGHTS='/kaggle/input/datasets/owner/weights-v1/model.pt'\n"
    validate_remote_kernel_identity(
        remote_kernel(source),
        "owner/kernel",
        3,
        "biohub-cell-tracking-during-development",
        __import__("hashlib").sha256(source.encode()).hexdigest(),
    )


def test_remote_identity_rejects_version_drift():
    source = "print('runtime')\n"
    with pytest.raises(RuntimeError, match="version drift"):
        validate_remote_kernel_identity(
            remote_kernel(source, current_version_number=4),
            "owner/kernel",
            3,
            "biohub-cell-tracking-during-development",
            __import__("hashlib").sha256(source.encode()).hexdigest(),
        )


def test_remote_identity_rejects_unattached_explicit_dataset():
    source = "WEIGHTS='/kaggle/input/datasets/other/missing-v1/model.pt'\n"
    with pytest.raises(RuntimeError, match="not attached"):
        validate_remote_kernel_identity(
            remote_kernel(source),
            "owner/kernel",
            3,
            "biohub-cell-tracking-during-development",
            __import__("hashlib").sha256(source.encode()).hexdigest(),
        )


def test_remote_identity_rejects_blank_dataset_attachment():
    source = "print('runtime')\n"
    with pytest.raises(RuntimeError, match="blank dataset attachment"):
        validate_remote_kernel_identity(
            remote_kernel(source, dataset_data_sources=[""]),
            "owner/kernel",
            3,
            "biohub-cell-tracking-during-development",
            __import__("hashlib").sha256(source.encode()).hexdigest(),
        )
