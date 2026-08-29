"""Submission guards, with no network calls or submission side effects."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from submit_frontier_batch import read_with_retry, validate_batch, validate_remote_source


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

    def notebook(self, code="x = 1\n", metadata=None, trailing_empty=False):
        cells = [{"cell_type": "code", "source": [code]}]
        if trailing_empty:
            cells.append({"cell_type": "code", "source": ""})
        return json.dumps({"metadata": metadata or {}, "cells": cells})

    def test_kaggle_notebook_metadata_normalization_allowed_explicitly(self):
        path = Mock()
        path.read_text.return_value = self.notebook(metadata={"local": True})
        current = self.response(source=self.notebook(
            metadata={"kaggle": "normalized"}, trailing_empty=True
        ))
        validate_remote_source(current, {
            "version": 1,
            "source_validation_mode": "nonempty_notebook_code_cells_exact",
        }, path)

    def test_executable_notebook_code_drift_denied(self):
        path = Mock()
        path.read_text.return_value = self.notebook(code="x = 1\n")
        current = self.response(source=self.notebook(code="x = 2\n"))
        with self.assertRaises(ValueError):
            validate_remote_source(current, {
                "version": 1,
                "source_validation_mode": "nonempty_notebook_code_cells_exact",
            }, path)

    def test_read_retry_recovers_but_does_not_repeat_success(self):
        calls = []

        def operation():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient read failure")
            return "ok"

        self.assertEqual(read_with_retry("test", operation, attempts=2), "ok")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
