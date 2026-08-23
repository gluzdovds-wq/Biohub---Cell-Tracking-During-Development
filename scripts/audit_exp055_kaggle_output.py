"""Audit EXP055 Kaggle output against exact EXP019 nodes and EXP052 edges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "outputs" / "exp055_kaggle_v1" / "submission.csv"
NODE_PARENT = ROOT / "outputs" / "exp019_intensity_coordinate_refine" / "submission.csv"
EDGE_PARENT = ROOT / "outputs" / "exp052_kaggle_v1" / "submission.csv"
RECEIPT = ROOT / "kaggle_notebooks" / "exp055_intensity_registered_production" / "kaggle_v1_output_receipt.json"
EXPECTED_NODE_PARENT_SHA256 = "7487ecb7de8c110caffd35bd043902b484ee4634ec58d020caebabfad9296c6d"
EXPECTED_EDGE_PARENT_SHA256 = "3791f74f9247be99d3a9e673cd2ff9fd942764f1ad0b1d0a597d150b7a7c9fab"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if sha256(NODE_PARENT) != EXPECTED_NODE_PARENT_SHA256:
    raise AssertionError({"node_parent_sha256": sha256(NODE_PARENT)})
if sha256(EDGE_PARENT) != EXPECTED_EDGE_PARENT_SHA256:
    raise AssertionError({"edge_parent_sha256": sha256(EDGE_PARENT)})

candidate = pd.read_csv(CANDIDATE, index_col=0)
node_parent = pd.read_csv(NODE_PARENT, index_col=0)
edge_parent = pd.read_csv(EDGE_PARENT, index_col=0)
if list(candidate.columns) != list(node_parent.columns):
    raise AssertionError({"candidate_columns": list(candidate.columns)})

candidate_nodes = candidate[candidate["row_type"] == "node"].reset_index(drop=True)
parent_nodes = node_parent[node_parent["row_type"] == "node"].reset_index(drop=True)
identity_columns = ["dataset", "row_type", "node_id", "t"]
if not candidate_nodes[identity_columns].equals(parent_nodes[identity_columns]):
    raise AssertionError("EXP055 node identity/order differs from EXP019")
candidate_coordinates = candidate_nodes[["z", "y", "x"]].to_numpy(dtype=np.float64)
parent_coordinates = parent_nodes[["z", "y", "x"]].to_numpy(dtype=np.float64)
coordinate_delta = np.abs(candidate_coordinates - parent_coordinates)
maximum_coordinate_delta = float(coordinate_delta.max())
different_coordinate_values = int(np.count_nonzero(coordinate_delta))
if not np.array_equal(candidate_coordinates, parent_coordinates):
    raise AssertionError(
        {
            "reason": "EXP055 coordinates differ from EXP019",
            "maximum_coordinate_delta": maximum_coordinate_delta,
            "different_coordinate_values": different_coordinate_values,
        }
    )

candidate_edges = candidate[candidate["row_type"] == "edge"]
parent_edges = edge_parent[edge_parent["row_type"] == "edge"]
candidate_edge_set = set(
    zip(
        candidate_edges["dataset"].astype(str),
        candidate_edges["source_id"].astype(int),
        candidate_edges["target_id"].astype(int),
    )
)
parent_edge_set = set(
    zip(
        parent_edges["dataset"].astype(str),
        parent_edges["source_id"].astype(int),
        parent_edges["target_id"].astype(int),
    )
)
missing = parent_edge_set - candidate_edge_set
extra = candidate_edge_set - parent_edge_set
if missing or extra:
    raise AssertionError(
        {
            "missing_edges": len(missing),
            "extra_edges": len(extra),
            "missing_sample": sorted(missing)[:10],
            "extra_sample": sorted(extra)[:10],
        }
    )

maximum_in = int(candidate_edges.groupby(["dataset", "target_id"]).size().max())
maximum_out = int(candidate_edges.groupby(["dataset", "source_id"]).size().max())
if maximum_in != 1 or maximum_out != 1:
    raise AssertionError({"maximum_in": maximum_in, "maximum_out": maximum_out})

receipt = {
    "status": "PASS_EXP055_KAGGLE_V1_EXACT_PARENT_SEMANTICS",
    "candidate_sha256": sha256(CANDIDATE),
    "node_parent_sha256": EXPECTED_NODE_PARENT_SHA256,
    "edge_parent_sha256": EXPECTED_EDGE_PARENT_SHA256,
    "nodes": len(candidate_nodes),
    "edges": len(candidate_edges),
    "node_identity_and_coordinates_exact": True,
    "maximum_coordinate_delta": maximum_coordinate_delta,
    "different_coordinate_values": different_coordinate_values,
    "edge_set_exact": True,
    "missing_edges": 0,
    "extra_edges": 0,
    "divisions": 0,
    "maximum_in_degree": maximum_in,
    "maximum_out_degree": maximum_out,
    "datasets": sorted(candidate["dataset"].astype(str).unique()),
}
RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
