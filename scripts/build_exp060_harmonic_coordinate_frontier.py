"""Retarget EXP057 dual inference to the stronger EXP005 harmonic topology."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle_notebooks" / "exp057_coordinate_frontier_production"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "exp060_harmonic_coordinate_frontier"
SOURCE = SOURCE_DIR / "coordinate_frontier_production.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
OUTPUT = OUTPUT_DIR / "harmonic_coordinate_frontier.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"
RECEIPT = OUTPUT_DIR / "build_receipt.json"

SOURCE_SHA256 = "75744cd1409cb705f6d31f636801d8362f85b38006752de100eb9572f7914b85"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(SOURCE) != SOURCE_SHA256:
    raise RuntimeError("EXP057 production source drift")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
config_replacements = {
    'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "12.0"':
        'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "4.66"',
    'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "15.0"':
        'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.5"',
    'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"':
        'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "7.65"',
    'os.environ.get("BIOHUB_SAFE_DIV_DIVERGE_UM", "1.5")':
        'os.environ.get("BIOHUB_SAFE_DIV_DIVERGE_UM", "2.25")',
}
counts = {key: 0 for key in config_replacements}
for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    for old, new in config_replacements.items():
        counts[old] += source.count(old)
        source = source.replace(old, new)
    source = source.replace("EXP057", "EXP060")
    source = source.replace("EXP058", "EXP061")
    source = source.replace("EXP059", "EXP062")
    source = source.replace("exp057", "exp060")
    source = source.replace("exact EXP006", "exact EXP005")
    source = source.replace("EXP006 topology", "EXP005 topology")
    source = source.replace("EXP006 division", "EXP005 division")
    cell["source"] = source.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None
if any(count != 1 for count in counts.values()):
    raise AssertionError({"configuration_anchor_counts": counts})

title = "Biohub EXP060 Harmonic Coordinate Frontier"
notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = title
notebook["metadata"]["title"] = title

metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
metadata.update(
    {
        "id": "dmitriigluzdov/biohub-exp060-harmonic-coordinate-frontier",
        "title": title,
        "code_file": OUTPUT.name,
    }
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, separators=(",", ":"), ensure_ascii=False) + "\n",
    encoding="utf-8",
)
OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
receipt = {
    "status": "PASS_EXP060_BUILD",
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": SOURCE_SHA256,
    "output": str(OUTPUT.relative_to(ROOT)),
    "output_sha256": sha256(OUTPUT),
    "base_policy": "exact EXP005 harmonic inference settings",
    "donor_policy": "exact EXP008 three-UNet inference",
    "declared_candidates": {
        "EXP060": "alpha=0.50; exact EXP005 topology",
        "EXP061": "alpha=0.25; exact EXP005 topology",
        "EXP062": "diagnostic only: registered ordinary links plus EXP005 divisions",
    },
    "public_artifact_input": False,
    "configuration_anchor_counts": counts,
}
RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
