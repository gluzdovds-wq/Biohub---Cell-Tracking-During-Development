"""Build the controlled SDW90 continuation from the reviewed public SDW85 v1.

The learned architecture, weights, association, repair and thresholds remain
unchanged. Only the secondary detector-logit mixture is moved 0.85 -> 0.90.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/research/frontier_20260829/rishabhr0y__biohub-938-sdw85__1/biohub-938-sdw85.ipynb"
OUTPUT = ROOT / "kaggle_notebooks/exp087_sdw90/sdw90.ipynb"
METADATA = OUTPUT.parent / "kernel-metadata.json"
EXPECTED_SOURCE_SHA256 = "6d0f4314afdb1f90021898edb3559841152e21c4e3b197dcae460d368a52aedd"
OLD = 'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.85"'
NEW = 'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.90"'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Reviewed SDW85 source drift")
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    matches = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            matches += source.count(OLD)
            source = source.replace(OLD, NEW)
            cell["source"] = source.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            # Notebook schema forbids execution fields on markdown/raw cells.
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    if matches != 1:
        raise RuntimeError(f"Expected one controlled detector-weight anchor, got {matches}")
    notebook["cells"].insert(0, {
        "cell_type": "markdown", "metadata": {},
        "source": [
            "# EXP087: controlled SDW90 continuation\n",
            "Source-attributed fork of Rishabh Roy's public SDW85 v1. "
            "Only secondary detection-logit weight changes 0.85 to 0.90.\n",
        ],
    })
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["isGpuEnabled"] = True
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    METADATA.write_text(json.dumps({
        "id": "dmitriigluzdov/biohub-exp087-controlled-sdw90",
        "title": "Biohub EXP087 Controlled SDW90",
        "code_file": OUTPUT.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }, indent=2) + "\n", encoding="utf-8")
    # Compile each cell independently, matching notebook execution semantics.
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"EXP087:cell{index}", "exec")
    print(json.dumps({
        "status": "PASS_EXP087_BUILD", "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256(OUTPUT), "controlled_change_count": matches,
        "from_detection_weight": 0.85, "to_detection_weight": 0.90,
    }, indent=2))


if __name__ == "__main__":
    main()
