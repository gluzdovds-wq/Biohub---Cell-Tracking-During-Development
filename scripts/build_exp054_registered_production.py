"""Build the hidden-compatible EXP054 production fork from exact EXP006."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle_notebooks" / "exp006_kimi_division_frontier"
SOURCE = SOURCE_DIR / "kimi-notebook-v17.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "exp054_registered_production"
OUTPUT = OUTPUT_DIR / "registered_production.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"
RECEIPT = OUTPUT_DIR / "build_receipt.json"

SOURCE_SHA256 = "211421c2237f9f077a5e12b2faba26498190b4d300d513d21c9e57a10d5012af"
OLD_TAG = "selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap"
NEW_TAG = "exp054_hidden_registered_hungarian_production_v1"

FUNCTION_ANCHOR = "\n\ndef motion_relink_edges("
CALL_ANCHOR = '''        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        for node_id in sorted(nodes_by_id):
'''

REGISTERED_FUNCTION = r'''

def registered_hungarian_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
) -> list[dict[str, object]]:
    """Apply the exact H052 linker to final, submission-rounded node positions."""
    registered_gate_um = 7.0
    registered_motion_scale_um = 3.0
    registered_inlier_um = 4.0
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(int(node_id))
    for node_ids in ids_by_t.values():
        node_ids.sort()
    times = sorted(ids_by_t)
    if times and times != list(range(times[0], times[-1] + 1)):
        raise AssertionError({"reason": "noncontiguous node frames", "times": times})

    links: list[dict[str, object]] = []
    frame_count = 0
    initial_inliers = 0
    accepted_residuals: list[float] = []
    minimum_score = float(np.exp(-registered_gate_um / registered_motion_scale_um))
    for t in times[:-1]:
        source_ids = ids_by_t[t]
        target_ids = ids_by_t[t + 1]
        if not source_ids or not target_ids:
            continue
        source_points = np.stack([_position_um(nodes_by_id[node_id]) for node_id in source_ids])
        target_points = np.stack([_position_um(nodes_by_id[node_id]) for node_id in target_ids])
        tree = cKDTree(target_points)
        _, nearest = tree.query(source_points, k=1)
        displacement = target_points[np.asarray(nearest, dtype=int)] - source_points
        shift = np.median(displacement, axis=0)
        residual_to_shift = np.linalg.norm(displacement - shift, axis=1)
        inliers = residual_to_shift <= registered_inlier_um
        initial_inliers += int(inliers.sum())
        if int(inliers.sum()) >= 3:
            shift = np.median(displacement[inliers], axis=0)

        residual = np.linalg.norm(
            source_points[:, None, :] + shift[None, None, :] - target_points[None, :, :],
            axis=2,
        )
        valid = residual < registered_gate_um
        score = np.exp(-residual / registered_motion_scale_um)
        cost = np.where(valid, 1.0 - score, 1e6)
        augmented = np.concatenate(
            [
                cost,
                np.full(
                    (len(source_ids), len(source_ids)),
                    1.0 - minimum_score,
                    dtype=float,
                ),
            ],
            axis=1,
        )
        rows, columns = linear_sum_assignment(augmented)
        for row, column in zip(rows, columns):
            if (
                column < len(target_ids)
                and valid[row, column]
                and score[row, column] >= minimum_score
            ):
                accepted_residual = float(residual[row, column])
                links.append(
                    {
                        "source_id": int(source_ids[row]),
                        "target_id": int(target_ids[column]),
                        "edge_prob": float(score[row, column]),
                        "distance_um": accepted_residual,
                    }
                )
                accepted_residuals.append(accepted_residual)
        frame_count += 1

    links.sort(key=lambda edge: (int(edge["source_id"]), int(edge["target_id"])))
    stats["registered_hungarian_edges"] = len(links)
    stats["registered_hungarian_frames"] = frame_count
    stats["registered_hungarian_initial_inliers"] = initial_inliers
    stats["registered_hungarian_residual_median_milli_um"] = (
        int(round(float(np.median(accepted_residuals)) * 1000.0)) if accepted_residuals else 0
    )
    stats["registered_hungarian_residual_max_milli_um"] = (
        int(round(float(np.max(accepted_residuals)) * 1000.0)) if accepted_residuals else 0
    )
    return links
'''

REGISTERED_CALL = '''        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        # H052 was defined on the exact integer coordinates emitted by EXP006.
        # Freeze those final node positions before relinking so the public-test
        # implementation is semantically identical and hidden-test adaptive.
        nodes_by_id = {
            node_id: {
                **node,
                "z": max(0, int(round(float(node["z"])))),
                "y": max(0, int(round(float(node["y"])))),
                "x": max(0, int(round(float(node["x"])))),
            }
            for node_id, node in nodes_by_id.items()
        }
        filter_stats["pre_registered_hungarian_edges"] = len(edges)
        edges = registered_hungarian_edges(nodes_by_id, filter_stats)
        if len({int(edge["source_id"]) for edge in edges}) != len(edges):
            raise AssertionError(f"{dataset}: registered linker created multiple outgoing edges")
        if len({int(edge["target_id"]) for edge in edges}) != len(edges):
            raise AssertionError(f"{dataset}: registered linker created multiple incoming edges")

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
    raise RuntimeError(f"EXP006 source drift: {observed_source_sha} != {SOURCE_SHA256}")

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
if cell_source.count(FUNCTION_ANCHOR) != 1:
    raise AssertionError({"function_anchor_count": cell_source.count(FUNCTION_ANCHOR)})
if cell_source.count(CALL_ANCHOR) != 1:
    raise AssertionError({"call_anchor_count": cell_source.count(CALL_ANCHOR)})
cell_source = cell_source.replace(FUNCTION_ANCHOR, REGISTERED_FUNCTION + FUNCTION_ANCHOR, 1)
cell_source = cell_source.replace(CALL_ANCHOR, REGISTERED_CALL, 1)
cell["source"] = cell_source.splitlines(keepends=True)

metadata = notebook.setdefault("metadata", {})
metadata.setdefault("kaggle", {})["title"] = "Biohub EXP054 Registered Production"
metadata["title"] = "Biohub EXP054 Registered Production"
for notebook_cell in notebook["cells"]:
    if notebook_cell.get("cell_type") == "code":
        notebook_cell["outputs"] = []
        notebook_cell["execution_count"] = None

new_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
changed_cells = [index for index, (old, new) in enumerate(zip(original_cells, new_cells)) if old != new]
if changed_cells != [3, 6]:
    raise AssertionError({"unexpected_changed_cells": changed_cells})

serialized = json.dumps(notebook, separators=(",", ":"), ensure_ascii=False) + "\n"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(serialized, encoding="utf-8")

kernel_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
kernel_metadata["id"] = "dmitriigluzdov/biohub-exp054-registered-production"
kernel_metadata["title"] = "Biohub EXP054 Registered Production"
kernel_metadata["code_file"] = OUTPUT.name
OUTPUT_METADATA.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")

receipt = {
    "status": "PASS_EXP054_HIDDEN_COMPATIBLE_BUILD",
    "hypothesis": "H054",
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": SOURCE_SHA256,
    "output": str(OUTPUT.relative_to(ROOT)),
    "output_sha256": sha256(OUTPUT),
    "changed_cells": changed_cells,
    "tag_replacements": tag_replacements,
    "function_anchor_replacements": 1,
    "call_anchor_replacements": 1,
    "hidden_test_dynamic": {
        "competition_source_preserved": kernel_metadata["competition_sources"],
        "public_submission_artifact_input": False,
        "test_stem_discovery_preserved": True,
        "full_model_inference_preserved": True,
    },
    "registered_policy": {
        "coordinates": "final EXP006 submission-rounded node positions",
        "scale_um": [1.625, 0.40625, 0.40625],
        "shift": "median nearest displacement; recompute on <=4um inliers when >=3",
        "gate_um_strict": 7.0,
        "motion_scale_um": 3.0,
        "assignment": "Hungarian with per-source unmatched dummy",
        "maximum_in_degree": 1,
        "maximum_out_degree": 1,
    },
}
RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
