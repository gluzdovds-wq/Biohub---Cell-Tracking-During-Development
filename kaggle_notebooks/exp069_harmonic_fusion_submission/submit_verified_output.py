from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


EXPECTED_SHA256 = "fd77de2afe9747dc873d57f3c46488d60d885d977fb43a7f99fcf7bb308974a2"
OUTPUT = Path("/kaggle/working/submission.csv")


candidates = sorted(Path("/kaggle/input").glob("**/submission.csv"))
matches = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_SHA256]
assert len(matches) == 1, {
    "expected_sha256": EXPECTED_SHA256,
    "candidates": [str(p) for p in candidates],
    "matches": [str(p) for p in matches],
}

shutil.copyfile(matches[0], OUTPUT)
actual_sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
assert actual_sha256 == EXPECTED_SHA256
receipt = {
    "experiment": "EXP-069",
    "mechanism": "harmonic fusion with higher safe-division caps",
    "source": "flexonafft/biohub-harmonic-fusion",
    "source_path": str(matches[0]),
    "submission_sha256": actual_sha256,
    "submission_bytes": OUTPUT.stat().st_size,
}
Path("/kaggle/working/exp069_receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
