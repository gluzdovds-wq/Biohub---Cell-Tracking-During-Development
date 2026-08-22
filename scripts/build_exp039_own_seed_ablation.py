"""Build EXP039 as an exact EXP006 fork with only the secondary seed replaced."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_notebooks" / "exp006_kimi_division_frontier" / "kimi-notebook-v17.ipynb"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "exp039_own_seed_secondary_ablation"
OUTPUT = OUTPUT_DIR / "own_seed_secondary_ablation.ipynb"
RECEIPT = OUTPUT_DIR / "build_receipt.json"

SOURCE_SHA = "211421c2237f9f077a5e12b2faba26498190b4d300d513d21c9e57a10d5012af"
OLD_SECONDARY_SHA = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
NEW_SECONDARY_SHA = "b1507f6918192c0f5c15fd5091d97ff565b1fab14e67e01101446534abb6a7b7"
OLD_MANIFEST = "/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json"
NEW_MANIFEST = "/kaggle/input/datasets/dmitriigluzdov/biohub-exp038-own-seed-v1-checkpoint/ARTIFACT_MANIFEST.json"
OLD_SLUG = "biohub-temporal-unet3d-seed314159-v1"
NEW_SLUG = "biohub-exp038-own-seed-v1-checkpoint"
OLD_PRESET = "harmonic_mutual_support_association_fusion_v1"
NEW_PRESET = "exp039_own_seed_secondary_ablation_v1"
OLD_AXIS = "weighted harmonic forward/reverse association consensus on the fixed-90 dual-seed baseline"
NEW_AXIS = "replace only the EXP006 secondary checkpoint with the audited own_seed_v1 state"
OLD_TAG = "selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap"
NEW_TAG = "exp039_own_seed_secondary_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_recursive(value, old: str, new: str) -> tuple[object, int]:
    if isinstance(value, str):
        return value.replace(old, new), value.count(old)
    if isinstance(value, list):
        result = []
        count = 0
        for item in value:
            replaced, local_count = replace_recursive(item, old, new)
            result.append(replaced)
            count += local_count
        return result, count
    if isinstance(value, dict):
        result = {}
        count = 0
        for key, item in value.items():
            replaced, local_count = replace_recursive(item, old, new)
            result[key] = replaced
            count += local_count
        return result, count
    return value, 0


if sha256(SOURCE) != SOURCE_SHA:
    raise RuntimeError(f"EXP006 source drift: {sha256(SOURCE)} != {SOURCE_SHA}")
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
original_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]

replacements = [
    (OLD_SECONDARY_SHA, NEW_SECONDARY_SHA, 3),
    (OLD_MANIFEST, NEW_MANIFEST, 1),
    (f'_secondary_slug = "{OLD_SLUG}"', f'_secondary_slug = "{NEW_SLUG}"', 1),
    (OLD_PRESET, NEW_PRESET, 1),
    (OLD_AXIS, NEW_AXIS, 1),
    (OLD_TAG, NEW_TAG, 1),
]
replacement_counts = {}
for old, new, expected_count in replacements:
    notebook, count = replace_recursive(notebook, old, new)
    if count != expected_count:
        raise AssertionError({"old": old, "expected_count": expected_count, "observed_count": count})
    replacement_counts[old] = count

metadata = notebook.setdefault("metadata", {})
metadata.setdefault("kaggle", {})["title"] = "Biohub EXP039 Own Seed Secondary Ablation"
metadata["title"] = "Biohub EXP039 Own Seed Secondary Ablation"
for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

new_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
changed_cells = [index for index, (old, new) in enumerate(zip(original_cells, new_cells)) if old != new]
if changed_cells != [1, 3, 4]:
    raise AssertionError({"unexpected_changed_cells": changed_cells})

serialized = json.dumps(notebook, separators=(",", ":"), ensure_ascii=False) + "\n"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(serialized, encoding="utf-8")
output_sha = sha256(OUTPUT)
receipt = {
    "status": "PASS_EXACT_SECONDARY_CHECKPOINT_ABLATION_BUILD",
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": SOURCE_SHA,
    "output": str(OUTPUT.relative_to(ROOT)),
    "output_sha256": output_sha,
    "changed_cells": changed_cells,
    "replacement_counts": replacement_counts,
    "old_secondary_sha256": OLD_SECONDARY_SHA,
    "new_secondary_sha256": NEW_SECONDARY_SHA,
    "unchanged_contract": {
        "primary_checkpoint": True,
        "detection_weight": True,
        "edge_weight": True,
        "harmonic_bidirectional_fusion": True,
        "detection_threshold": True,
        "ilp": True,
        "gap_and_division_repairs": True,
        "deepcenter_veto": True,
    },
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
