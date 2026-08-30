"""Build EXP089: a controlled half-weight EMA motion continuation.

The reviewed public EMA v1 notebook is kept byte-for-byte equivalent except
for the motion-relink velocity weight, which changes from 1.0 to 0.5.  This
creates a clean interpolation test between the no-EMA baseline and full EMA.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs/research/frontier_20260830/grafael__biohub-ct-0940-ema__1"
    / "biohub-ct-0940-ema.ipynb"
)
OUTPUT = ROOT / "kaggle_notebooks/exp089_ema05/ema05.ipynb"
METADATA = OUTPUT.parent / "kernel-metadata.json"
EXPECTED_SOURCE_SHA256 = (
    "4f0fb3aff772f4aaa0b687477e255c05211412a458d9db085935a7f37f073f33"
)
OLD = 'os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "1.0"'
NEW = 'os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "0.5"'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Reviewed public EMA source drift")

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
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    if matches != 1:
        raise RuntimeError(f"Expected one EMA weight anchor, got {matches}")

    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP089: controlled half-weight EMA continuation\n",
                "Source-attributed fork of Grafael's public EMA v1. Only the "
                "four-frame averaged motion-relink velocity weight changes "
                "from 1.0 to 0.5.\n",
            ],
        },
    )
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})[
        "isGpuEnabled"
    ] = True

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    METADATA.write_text(
        json.dumps(
            {
                "id": "dmitriigluzdov/biohub-exp089-controlled-ema05",
                "title": "Biohub EXP089 Controlled EMA05",
                "code_file": OUTPUT.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [
                    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
                    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                    "pilkwang/biohub-tracking-support-pack-50ep-v1",
                ],
                "kernel_sources": [],
                "competition_sources": [
                    "biohub-cell-tracking-during-development"
                ],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"EXP089:cell{index}", "exec")

    print(
        json.dumps(
            {
                "status": "PASS_EXP089_BUILD",
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "output_sha256": sha256(OUTPUT),
                "controlled_change_count": matches,
                "from_velocity_weight": 1.0,
                "to_velocity_weight": 0.5,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
