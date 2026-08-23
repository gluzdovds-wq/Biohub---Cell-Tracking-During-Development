"""Build one hidden-compatible run that emits three controlled frontier candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "kaggle_notebooks" / "exp006_kimi_division_frontier"
DONOR_PHASE_DIR = ROOT / "kaggle_notebooks" / "exp056_detector_registered_production"
DONOR_METADATA = ROOT / "kaggle_notebooks" / "exp008_three_unet_flip_tta" / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "exp057_coordinate_frontier_production"
BASE = BASE_DIR / "kimi-notebook-v17.ipynb"
BASE_METADATA = BASE_DIR / "kernel-metadata.json"
DONOR_PHASE = DONOR_PHASE_DIR / "detector_registered_production.ipynb"
OUTPUT = OUTPUT_DIR / "coordinate_frontier_production.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"
RECEIPT = OUTPUT_DIR / "build_receipt.json"

BASE_SHA256 = "211421c2237f9f077a5e12b2faba26498190b4d300d513d21c9e57a10d5012af"
DONOR_PHASE_SHA256 = "c9cfcc8f7db4b360d91517bc03952b03922411fc8e8288b465dd3656542bdf77"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path, expected in ((BASE, BASE_SHA256), (DONOR_PHASE, DONOR_PHASE_SHA256)):
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"source drift: {path}: {observed} != {expected}")

notebook = json.loads(BASE.read_text(encoding="utf-8"))
phase_notebook = json.loads(DONOR_PHASE.read_text(encoding="utf-8"))
if len(phase_notebook["cells"]) < 3:
    raise AssertionError("EXP056 phase notebook contract changed")
cleanup_source = "".join(phase_notebook["cells"][-3]["source"]).replace("EXP056", "EXP057")
donor_source = "".join(phase_notebook["cells"][-2]["source"])
if donor_source.count("exp056_donor_submission.csv") != 1:
    raise AssertionError("EXP056 donor output anchor drift")
donor_source = donor_source.replace("exp056_donor_submission.csv", "exp057_donor_submission.csv")

frontier_source = r'''# EXP057/058/059: one hidden run, three predeclared controlled outputs.
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

BASE_PATH = Path("/kaggle/working/submission.csv")
DONOR_PATH = Path("/kaggle/working/exp057_donor_submission.csv")
PRIMARY_PATH = Path("/kaggle/working/submission.csv")
CONSERVATIVE_PATH = Path("/kaggle/working/submission_alpha025.csv")
REGDIV_PATH = Path("/kaggle/working/submission_alpha050_regdiv.csv")
RECEIPT_PATH = Path("/kaggle/working/exp057_frontier_receipt.json")
SCALE_FRONTIER = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
SPATIAL_FRONTIER = ["z", "y", "x"]
GATE_UM_FRONTIER = 2.0
EXPECTED_COLUMNS_FRONTIER = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]


def _frontier_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frontier_mutual_matches(base_points, donor_points):
    if not len(base_points) or not len(donor_points):
        return []
    base_physical = base_points * SCALE_FRONTIER
    donor_physical = donor_points * SCALE_FRONTIER
    donor_tree = cKDTree(donor_physical)
    base_tree = cKDTree(base_physical)
    distances, base_to_donor = donor_tree.query(base_physical, k=1)
    _, donor_to_base = base_tree.query(donor_physical, k=1)
    return [
        (base_index, int(donor_index), float(distance))
        for base_index, (distance, donor_index) in enumerate(zip(distances, base_to_donor))
        if distance <= GATE_UM_FRONTIER and int(donor_to_base[int(donor_index)]) == base_index
    ]


def _frontier_registered_edges(nodes):
    ids_by_t = {}
    points = {}
    for row in nodes.itertuples():
        node_id = int(row.node_id)
        timepoint = int(row.t)
        ids_by_t.setdefault(timepoint, []).append(node_id)
        points[node_id] = np.array([float(row.z), float(row.y), float(row.x)]) * SCALE_FRONTIER
    for node_ids in ids_by_t.values():
        node_ids.sort()
    times = sorted(ids_by_t)
    edges = []
    minimum_score = float(np.exp(-7.0 / 3.0))
    for timepoint in times[:-1]:
        source_ids = ids_by_t[timepoint]
        target_ids = ids_by_t.get(timepoint + 1, [])
        if not source_ids or not target_ids:
            continue
        source_points = np.stack([points[node_id] for node_id in source_ids])
        target_points = np.stack([points[node_id] for node_id in target_ids])
        tree = cKDTree(target_points)
        _, nearest = tree.query(source_points, k=1)
        displacement = target_points[np.asarray(nearest, dtype=int)] - source_points
        shift = np.median(displacement, axis=0)
        inliers = np.linalg.norm(displacement - shift, axis=1) <= 4.0
        if int(inliers.sum()) >= 3:
            shift = np.median(displacement[inliers], axis=0)
        residual = np.linalg.norm(
            source_points[:, None, :] + shift[None, None, :] - target_points[None, :, :], axis=2
        )
        valid = residual < 7.0
        score = np.exp(-residual / 3.0)
        cost = np.where(valid, 1.0 - score, 1e6)
        augmented = np.concatenate(
            [cost, np.full((len(source_ids), len(source_ids)), 1.0 - minimum_score)], axis=1
        )
        rows, columns = linear_sum_assignment(augmented)
        edges.extend(
            (int(source_ids[row]), int(target_ids[column]))
            for row, column in zip(rows, columns)
            if column < len(target_ids) and valid[row, column] and score[row, column] >= minimum_score
        )
    return edges


base = pd.read_csv(BASE_PATH, index_col=0)
donor = pd.read_csv(DONOR_PATH, index_col=0)
if list(base.columns) != EXPECTED_COLUMNS_FRONTIER or list(donor.columns) != EXPECTED_COLUMNS_FRONTIER:
    raise AssertionError({"base_columns": list(base.columns), "donor_columns": list(donor.columns)})
base_nodes = base[base["row_type"] == "node"]
base_edges = base[base["row_type"] == "edge"]
donor_nodes = donor[donor["row_type"] == "node"]
donor_frames = {key: frame for key, frame in donor_nodes.groupby(["dataset", "t"], sort=False)}
matched = []
for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
    donor_frame = donor_frames.get((dataset, timepoint))
    if donor_frame is None:
        continue
    base_coordinates = base_frame[SPATIAL_FRONTIER].to_numpy(dtype=float)
    donor_coordinates = donor_frame[SPATIAL_FRONTIER].to_numpy(dtype=float)
    for base_offset, donor_offset, distance in _frontier_mutual_matches(base_coordinates, donor_coordinates):
        matched.append(
            (
                int(base_frame.index[base_offset]),
                donor_coordinates[donor_offset],
                float(distance),
                str(dataset),
            )
        )
if not matched:
    raise AssertionError("detector consensus produced zero mutual matches")


def _frontier_alpha(alpha, path):
    result = base.copy(deep=True)
    result[SPATIAL_FRONTIER] = result[SPATIAL_FRONTIER].astype(float)
    for row_index, donor_coordinate, _distance, _dataset in matched:
        base_coordinate = base.loc[row_index, SPATIAL_FRONTIER].to_numpy(dtype=float)
        result.loc[row_index, SPATIAL_FRONTIER] = (1.0 - alpha) * base_coordinate + alpha * donor_coordinate
    if not base_edges[["dataset", "source_id", "target_id"]].reset_index(drop=True).equals(
        result[result["row_type"] == "edge"][["dataset", "source_id", "target_id"]].reset_index(drop=True)
    ):
        raise AssertionError("coordinate candidate changed topology")
    result.index.name = "id"
    result.to_csv(path, float_format="%.17g", lineterminator="\n", na_rep="")
    return result


alpha050 = _frontier_alpha(0.50, PRIMARY_PATH)
alpha025 = _frontier_alpha(0.25, CONSERVATIVE_PATH)

regdiv_parts = []
regdiv_diagnostics = {}
for dataset, node_rows in alpha050[alpha050["row_type"] == "node"].groupby("dataset", sort=True):
    # Preserve the exact EXP006 division decisions, but replace all ordinary
    # one-to-one links with registered motion. Registration uses base integer
    # nodes exactly as H052; the emitted coordinates remain detector consensus.
    base_dataset_nodes = base_nodes[base_nodes["dataset"] == dataset]
    base_dataset_edges = base_edges[base_edges["dataset"] == dataset]
    registered = set(_frontier_registered_edges(base_dataset_nodes))
    outgoing_counts = base_dataset_edges.groupby("source_id").size()
    division_sources = {int(source) for source, count in outgoing_counts.items() if int(count) >= 2}
    division_edges = {
        (int(row.source_id), int(row.target_id))
        for row in base_dataset_edges.itertuples()
        if int(row.source_id) in division_sources
    }
    division_targets = {target for _source, target in division_edges}
    registered = {
        (source, target)
        for source, target in registered
        if source not in division_sources and target not in division_targets
    }
    hybrid_edges = sorted(registered | division_edges)
    indegree = {}
    outdegree = {}
    for source, target in hybrid_edges:
        outdegree[source] = outdegree.get(source, 0) + 1
        indegree[target] = indegree.get(target, 0) + 1
    if max(indegree.values(), default=0) > 1 or max(outdegree.values(), default=0) > 2:
        raise AssertionError({"dataset": dataset, "max_in": max(indegree.values()), "max_out": max(outdegree.values())})
    edge_frame = pd.DataFrame(
        [
            (dataset, "edge", -1, -1, -1.0, -1.0, -1.0, source, target)
            for source, target in hybrid_edges
        ],
        columns=EXPECTED_COLUMNS_FRONTIER,
    )
    regdiv_parts.extend([node_rows.copy(), edge_frame])
    regdiv_diagnostics[str(dataset)] = {
        "registered_edges_after_conflict_removal": len(registered),
        "preserved_division_edges": len(division_edges),
        "division_sources": len(division_sources),
        "hybrid_edges": len(hybrid_edges),
    }
regdiv = pd.concat(regdiv_parts, ignore_index=True)
regdiv.index.name = "id"
regdiv.to_csv(REGDIV_PATH, float_format="%.17g", lineterminator="\n", na_rep="")

distance_array = np.asarray([item[2] for item in matched], dtype=float)
receipt = {
    "status": "PASS_EXP057_FRONTIER_PRODUCTION",
    "public_artifact_input": False,
    "matched_nodes": len(matched),
    "matched_fraction": len(matched) / len(base_nodes),
    "match_distance_um_mean": float(distance_array.mean()),
    "match_distance_um_p95": float(np.quantile(distance_array, 0.95)),
    "outputs": {
        "EXP057": {"path": str(PRIMARY_PATH), "sha256": _frontier_sha(PRIMARY_PATH), "alpha": 0.50, "topology": "exact EXP006"},
        "EXP058": {"path": str(CONSERVATIVE_PATH), "sha256": _frontier_sha(CONSERVATIVE_PATH), "alpha": 0.25, "topology": "exact EXP006"},
        "EXP059": {"path": str(REGDIV_PATH), "sha256": _frontier_sha(REGDIV_PATH), "alpha": 0.50, "topology": "registered ordinary links plus exact EXP006 division edges"},
    },
    "regdiv_diagnostics": regdiv_diagnostics,
}
RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
'''

for source in (cleanup_source, donor_source, frontier_source):
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    )
for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
metadata = notebook.setdefault("metadata", {})
metadata.setdefault("kaggle", {})["title"] = "Biohub EXP057 Coordinate Frontier Production"
metadata["title"] = "Biohub EXP057 Coordinate Frontier Production"

base_kernel = json.loads(BASE_METADATA.read_text(encoding="utf-8"))
donor_kernel = json.loads(DONOR_METADATA.read_text(encoding="utf-8"))
dataset_sources = list(dict.fromkeys(base_kernel["dataset_sources"] + donor_kernel["dataset_sources"]))
kernel_metadata = dict(base_kernel)
kernel_metadata.update(
    {
        "id": "dmitriigluzdov/biohub-exp057-coordinate-frontier-production",
        "title": "Biohub EXP057 Coordinate Frontier Production",
        "code_file": OUTPUT.name,
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
    }
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
OUTPUT_METADATA.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
receipt = {
    "status": "PASS_EXP057_BUILD",
    "base": str(BASE.relative_to(ROOT)),
    "base_sha256": BASE_SHA256,
    "donor_phase": str(DONOR_PHASE.relative_to(ROOT)),
    "donor_phase_sha256": DONOR_PHASE_SHA256,
    "output": str(OUTPUT.relative_to(ROOT)),
    "output_sha256": sha256(OUTPUT),
    "appended_cells": 3,
    "declared_candidates": ["EXP057", "EXP058", "EXP059"],
    "dataset_sources": dataset_sources,
    "hidden_test_dynamic": {
        "public_submission_artifact_input": False,
        "base_full_inference": True,
        "donor_full_inference": True,
        "test_stem_discovery_preserved": True,
    },
}
RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
