"""Build an offline, competition-eligible reproduction of the FT-linker D4 run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs/research/frontier_20260830"
    / "muhanqiu__biohub-final-submission-our-weights__1"
    / "biohub-final-submission-our-weights.ipynb"
)
OUTPUT = ROOT / "kaggle_notebooks/exp092_finetuned_d4_offline/finetuned_d4.ipynb"
METADATA = OUTPUT.parent / "kernel-metadata.json"
EXPECTED_SOURCE_SHA256 = (
    "fae5d1b15bd1c3876d2334f50c2555403bf36904d10b70fb1a1b24f48078a596"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Reviewed fine-tuned D4 source drift")

    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)

    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP092: offline fine-tuned-linker D4 reproduction\n",
                "Source-attributed reproduction of Muhan Qiu's public notebook. "
                "Executable code and public weights are unchanged; Internet is "
                "disabled to make the version competition-eligible.\n",
            ],
        },
    )
    kaggle_meta = notebook.setdefault("metadata", {}).setdefault("kaggle", {})
    kaggle_meta["isGpuEnabled"] = True
    kaggle_meta["isInternetEnabled"] = False

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    METADATA.write_text(
        json.dumps(
            {
                "id": "dmitriigluzdov/biohub-exp092-finetuned-d4-offline",
                "title": "Biohub EXP092 Finetuned D4 Offline",
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
                    "muhanqiu/biohub-ft-weights-v1",
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
            compile("".join(cell["source"]), f"EXP092:cell{index}", "exec")

    print(
        json.dumps(
            {
                "status": "PASS_EXP092_BUILD",
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "output_sha256": sha256(OUTPUT),
                "executable_change_count": 0,
                "internet": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
