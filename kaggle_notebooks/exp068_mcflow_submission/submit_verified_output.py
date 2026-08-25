from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


EXPECTED_SHA256 = "f7cf397733602d77ba7ec51b36472e89b6af7f7e379a6d3dcceaf18beab6e34c"
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
    "experiment": "EXP-068",
    "mechanism": "orthogonal min-cost-flow global assignment tracker",
    "source": "pawanmali/biohub-mcflow-v1",
    "source_path": str(matches[0]),
    "submission_sha256": actual_sha256,
    "submission_bytes": OUTPUT.stat().st_size,
}
Path("/kaggle/working/exp068_receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
