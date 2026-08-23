"""Build EXP055 by adding exact EXP019 coordinate refinement after EXP054 linking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle_notebooks" / "exp054_registered_production"
SOURCE = SOURCE_DIR / "registered_production.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "exp055_intensity_registered_production"
OUTPUT = OUTPUT_DIR / "intensity_registered_production.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"
RECEIPT = OUTPUT_DIR / "build_receipt.json"

SOURCE_SHA256 = "a6ad5103e5707873d6cf9644e226e6f5f1f577a4f23b81730924109a88092706"
OLD_TAG = "exp054_hidden_registered_hungarian_production_v1"
NEW_TAG = "exp055_hidden_intensity_registered_production_v1"
FUNCTION_ANCHOR = "\ndef registered_hungarian_edges("
CALL_ANCHOR = '''        if len({int(edge["target_id"]) for edge in edges}) != len(edges):
            raise AssertionError(f"{dataset}: registered linker created multiple incoming edges")

        for node_id in sorted(nodes_by_id):
'''
WRITER_ANCHOR = '''                "z": max(0, int(round(float(node["z"])))),
                "y": max(0, int(round(float(node["y"])))),
                "x": max(0, int(round(float(node["x"])))),
                "source_id": -1,
'''
WRITER_REPLACEMENT = '''                "z": float(node["z"]),
                "y": float(node["y"]),
                "x": float(node["x"]),
                "source_id": -1,
'''

INTENSITY_FUNCTIONS = r'''

def final_intensity_center(frame: np.ndarray, coordinate: np.ndarray) -> np.ndarray | None:
    """Exact EXP019 5x11x11 background-subtracted center of mass."""
    radius = np.asarray([2, 5, 5])
    center = np.rint(coordinate).astype(int)
    lower = np.maximum(center - radius, 0)
    upper = np.minimum(center + radius + 1, np.asarray(frame.shape))
    crop = frame[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ].astype(np.float32)
    if not crop.size:
        return None
    background = float(np.percentile(crop, 20.0))
    weights = np.clip(crop - background, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    zz, yy, xx = np.mgrid[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ]
    refined = np.asarray(
        [
            float((zz * weights).sum() / total),
            float((yy * weights).sum() / total),
            float((xx * weights).sum() / total),
        ]
    )
    return refined if np.isfinite(refined).all() else None


def refine_final_intensity_coordinates(
    dataset: str,
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    scale = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
    frame_cache: dict[int, np.ndarray] = {}
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(int(node_id))
    processed = 0
    accepted = 0
    rejected_large = 0
    rejected_empty = 0
    output_shift_milli_um_sum = 0
    for t in sorted(ids_by_t):
        frame = read_test_frame(dataset, t, frame_cache)
        for node_id in sorted(ids_by_t[t]):
            node = nodes_by_id[node_id]
            coordinate = np.asarray(
                [float(node["z"]), float(node["y"]), float(node["x"])],
                dtype=float,
            )
            processed += 1
            candidate = final_intensity_center(frame, coordinate)
            if candidate is None:
                rejected_empty += 1
                continue
            distance_um = float(np.linalg.norm((candidate - coordinate) * scale))
            if distance_um > 1.5:
                rejected_large += 1
                continue
            refined = 0.65 * coordinate + 0.35 * candidate
            node["z"], node["y"], node["x"] = map(float, refined)
            accepted += 1
            output_shift_milli_um_sum += int(round(0.35 * distance_um * 1000.0))
    stats["intensity_coordinate_processed"] = processed
    stats["intensity_coordinate_accepted"] = accepted
    stats["intensity_coordinate_rejected_large"] = rejected_large
    stats["intensity_coordinate_rejected_empty"] = rejected_empty
    stats["intensity_coordinate_output_shift_milli_um_sum"] = output_shift_milli_um_sum
    return nodes_by_id
'''

CALL_REPLACEMENT = '''        if len({int(edge["target_id"]) for edge in edges}) != len(edges):
            raise AssertionError(f"{dataset}: registered linker created multiple incoming edges")

        # Freeze topology first; H055 changes coordinates only after H052 edges exist.
        nodes_by_id = refine_final_intensity_coordinates(dataset, nodes_by_id, filter_stats)

        for node_id in sorted(nodes_by_id):
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


observed_source_sha = sha256(SOURCE)
if observed_source_sha != SOURCE_SHA256:
    raise RuntimeError(f"EXP054 source drift: {observed_source_sha} != {SOURCE_SHA256}")
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
original_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]

tag_replacements = 0
for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    count = source.count(OLD_TAG)
    if count:
        source = source.replace(OLD_TAG, NEW_TAG)
        cell["source"] = source.splitlines(keepends=True)
        tag_replacements += count
if tag_replacements != 1:
    raise AssertionError({"tag_replacements": tag_replacements})

cell = notebook["cells"][6]
cell_source = "".join(cell["source"])
counts = {
    "function_anchor": cell_source.count(FUNCTION_ANCHOR),
    "call_anchor": cell_source.count(CALL_ANCHOR),
    "writer_anchor": cell_source.count(WRITER_ANCHOR),
}
if counts != {"function_anchor": 1, "call_anchor": 1, "writer_anchor": 1}:
    raise AssertionError(counts)
cell_source = cell_source.replace(FUNCTION_ANCHOR, INTENSITY_FUNCTIONS + FUNCTION_ANCHOR, 1)
cell_source = cell_source.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)
cell_source = cell_source.replace(WRITER_ANCHOR, WRITER_REPLACEMENT, 1)
cell["source"] = cell_source.splitlines(keepends=True)

metadata = notebook.setdefault("metadata", {})
metadata.setdefault("kaggle", {})["title"] = "Biohub EXP055 Intensity Registered Production"
metadata["title"] = "Biohub EXP055 Intensity Registered Production"
for notebook_cell in notebook["cells"]:
    if notebook_cell.get("cell_type") == "code":
        notebook_cell["outputs"] = []
        notebook_cell["execution_count"] = None
new_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
changed_cells = [index for index, (old, new) in enumerate(zip(original_cells, new_cells)) if old != new]
if changed_cells != [3, 6]:
    raise AssertionError({"unexpected_changed_cells": changed_cells})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, separators=(",", ":"), ensure_ascii=False) + "\n",
    encoding="utf-8",
)
kernel_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
kernel_metadata["id"] = "dmitriigluzdov/biohub-exp055-intensity-registered-production"
kernel_metadata["title"] = "Biohub EXP055 Intensity Registered Production"
kernel_metadata["code_file"] = OUTPUT.name
OUTPUT_METADATA.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")

receipt = {
    "status": "PASS_EXP055_HIDDEN_COMPATIBLE_BUILD",
    "hypothesis": "H055",
    "source_sha256": SOURCE_SHA256,
    "output_sha256": sha256(OUTPUT),
    "changed_cells": changed_cells,
    "tag_replacements": tag_replacements,
    "anchor_replacements": counts,
    "hidden_test_dynamic": True,
    "topology_frozen_before_coordinate_refinement": True,
    "public_artifact_input": False,
    "intensity_policy": {
        "radius_zyx": [2, 5, 5],
        "background_percentile": 20.0,
        "max_raw_shift_um": 1.5,
        "alpha": 0.35,
    },
}
RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
