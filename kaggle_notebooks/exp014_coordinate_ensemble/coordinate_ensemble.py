"""Build the pre-registered EXP014 coordinate-only ensemble from frozen parent outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT = WORK / "submission.csv"
RECEIPT = WORK / "exp014_receipt.json"

BASE_SHA256 = "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d"
DONOR_SHA256 = "d7ba9e6af86a6bb0be8bd04a36d0c61564e857e03fbadf9a81508211a4a4f2bb"
EXPECTED_OUTPUT_SHA256 = "6fcc8d2298144ad84dc5f151e589989829a540169ed3e9fa1aea762333b42109"
EXPECTED_NODES = 122_266
EXPECTED_EDGES = 118_156
EXPECTED_MATCHES = 93_630
ALPHA = 0.5
GATE_UM = 2.0
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
SPATIAL = ["z", "y", "x"]
KEYS = ["dataset", "node_id"]
EXPECTED_COLUMNS = [
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_parent(expected_sha256: str) -> Path:
    matches = []
    observed = []
    for path in INPUT.rglob("submission.csv"):
        digest = sha256(path)
        observed.append({"path": str(path), "sha256": digest})
        if digest == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            {
                "expected_sha256": expected_sha256,
                "matches": list(map(str, matches)),
                "observed_submission_csvs": observed,
            }
        )
    return matches[0]


def mutual_matches(base: np.ndarray, donor: np.ndarray) -> list[tuple[int, int, float]]:
    if not len(base) or not len(donor):
        return []
    base_physical = base * SCALE
    donor_physical = donor * SCALE
    donor_tree = cKDTree(donor_physical)
    base_tree = cKDTree(base_physical)
    distances, base_to_donor = donor_tree.query(base_physical, k=1)
    _, donor_to_base = base_tree.query(donor_physical, k=1)
    return [
        (base_index, int(donor_index), float(distance))
        for base_index, (distance, donor_index) in enumerate(zip(distances, base_to_donor))
        if distance <= GATE_UM and int(donor_to_base[int(donor_index)]) == base_index
    ]


base_path = locate_parent(BASE_SHA256)
donor_path = locate_parent(DONOR_SHA256)
base = pd.read_csv(base_path, index_col=0)
donor = pd.read_csv(donor_path, index_col=0)
if list(base.columns) != EXPECTED_COLUMNS or list(donor.columns) != EXPECTED_COLUMNS:
    raise ValueError({"base_columns": list(base.columns), "donor_columns": list(donor.columns)})

result = base.copy(deep=True)
result[SPATIAL] = result[SPATIAL].astype(float)
base_nodes = base[base["row_type"] == "node"]
donor_nodes = donor[donor["row_type"] == "node"]
donor_frames = {key: frame for key, frame in donor_nodes.groupby(["dataset", "t"], sort=False)}
distances = []
match_count_by_dataset: dict[str, int] = {}

for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
    donor_frame = donor_frames.get((dataset, timepoint))
    if donor_frame is None:
        match_count_by_dataset.setdefault(dataset, 0)
        continue
    base_coordinates = base_frame[SPATIAL].to_numpy(dtype=float)
    donor_coordinates = donor_frame[SPATIAL].to_numpy(dtype=float)
    matches = mutual_matches(base_coordinates, donor_coordinates)
    if matches:
        base_offsets = np.fromiter((item[0] for item in matches), dtype=int)
        donor_offsets = np.fromiter((item[1] for item in matches), dtype=int)
        row_indices = base_frame.index.to_numpy()[base_offsets]
        result.loc[row_indices, SPATIAL] = (
            (1.0 - ALPHA) * base_coordinates[base_offsets]
            + ALPHA * donor_coordinates[donor_offsets]
        )
        distances.extend(item[2] for item in matches)
    match_count_by_dataset[dataset] = match_count_by_dataset.get(dataset, 0) + len(matches)

base_edges = base[base["row_type"] == "edge"][["dataset", "source_id", "target_id"]].reset_index(drop=True)
result_edges = result[result["row_type"] == "edge"][["dataset", "source_id", "target_id"]].reset_index(drop=True)
result_nodes = result[result["row_type"] == "node"]
if not base_edges.equals(result_edges):
    raise AssertionError("Edge topology changed")
if not base_nodes[KEYS + ["t"]].equals(result_nodes[KEYS + ["t"]]):
    raise AssertionError("Node identity or time changed")
if len(result_nodes) != EXPECTED_NODES or len(result_edges) != EXPECTED_EDGES:
    raise AssertionError({"nodes": len(result_nodes), "edges": len(result_edges)})
if len(distances) != EXPECTED_MATCHES:
    raise AssertionError({"matched_nodes": len(distances), "expected": EXPECTED_MATCHES})

result.index.name = "id"
result.to_csv(OUTPUT)
output_sha256 = sha256(OUTPUT)
if output_sha256 != EXPECTED_OUTPUT_SHA256:
    raise AssertionError(
        {"output_sha256": output_sha256, "expected_output_sha256": EXPECTED_OUTPUT_SHA256}
    )

distance_array = np.asarray(distances)
receipt = {
    "status": "PASS",
    "method": "mutual-nearest coordinate-only convex ensemble",
    "base": str(base_path),
    "base_sha256": sha256(base_path),
    "donor": str(donor_path),
    "donor_sha256": sha256(donor_path),
    "output": str(OUTPUT),
    "output_sha256": output_sha256,
    "alpha": ALPHA,
    "gate_um": GATE_UM,
    "scale_zyx_um": SCALE.tolist(),
    "nodes": len(result_nodes),
    "edges": len(result_edges),
    "matched_nodes": len(distances),
    "matched_fraction": len(distances) / len(result_nodes),
    "match_count_by_dataset": match_count_by_dataset,
    "match_distance_um_mean": float(distance_array.mean()),
    "match_distance_um_p95": float(np.quantile(distance_array, 0.95)),
    "topology_unchanged": True,
    "node_ids_and_times_unchanged": True,
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
