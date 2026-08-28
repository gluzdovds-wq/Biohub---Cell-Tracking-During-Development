"""Submission guards, with no network calls or submission side effects."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from submit_frontier_batch import validate_batch, validate_remote_source


class SubmissionGuardTests(unittest.TestCase):
    def records(self, count):
        return [{"artifact_sha256": str(i), "description": f"EXP-{i}"} for i in range(count)]

    def test_five_allowed(self):
        validate_batch(self.records(5))

    def test_six_denied(self):
        with self.assertRaises(ValueError):
            validate_batch(self.records(6))

    def test_empty_denied(self):
        with self.assertRaises(ValueError):
            validate_batch([])

    def test_duplicate_hash_denied(self):
        rows = self.records(2)
        rows[1]["artifact_sha256"] = rows[0]["artifact_sha256"]
        with self.assertRaises(ValueError):
            validate_batch(rows)

    def test_duplicate_description_denied(self):
        rows = self.records(2)
        rows[1]["description"] = rows[0]["description"]
        with self.assertRaises(ValueError):
            validate_batch(rows)

    def response(self, version=1, source="x\n"):
        return SimpleNamespace(metadata=SimpleNamespace(current_version_number=version),
                               blob=SimpleNamespace(source=source))

    def test_version_drift_denied(self):
        with self.assertRaises(ValueError):
            validate_remote_source(self.response(2), {"version": 1}, Mock())

    def test_source_drift_denied(self):
        path = Mock()
        path.read_text.return_value = "different\n"
        with self.assertRaises(ValueError):
            validate_remote_source(self.response(), {"version": 1}, path)

    def test_only_newline_normalization_allowed(self):
        path = Mock()
        path.read_text.return_value = "x\n"
        validate_remote_source(self.response(source="x\r\n"), {"version": 1}, path)


if __name__ == "__main__":
    unittest.main()
