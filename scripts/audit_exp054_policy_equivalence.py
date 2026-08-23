"""Prove the embedded EXP054 linker reproduces EXP052 on exact EXP006 nodes."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle_notebooks" / "exp054_registered_production" / "registered_production.ipynb"
BASE = ROOT / "outputs" / "exp006" / "submission.csv"
REFERENCE = ROOT / "outputs" / "exp052_kaggle_v1" / "submission.csv"
RECEIPT = ROOT / "kaggle_notebooks" / "exp054_registered_production" / "policy_equivalence_receipt.json"
EXPECTED_BASE_SHA256 = "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d"
EXPECTED_REFERENCE_SHA256 = "3791f74f9247be99d3a9e673cd2ff9fd942764f1ad0b1d0a597d150b7a7c9fab"
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if sha256(BASE) != EXPECTED_BASE_SHA256:
    raise AssertionError({"base_sha256": sha256(BASE)})
if sha256(REFERENCE) != EXPECTED_REFERENCE_SHA256:
    raise AssertionError({"reference_sha256": sha256(REFERENCE)})

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cell_source = "".join(notebook["cells"][6]["source"])
tree = ast.parse(cell_source)
selected = [
    node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in {"_position_um", "registered_hungarian_edges"}
]
if [node.name for node in selected] != ["_position_um", "registered_hungarian_edges"]:
    raise AssertionError({"embedded_functions": [node.name for node in selected]})
namespace = {
    "np": np,
    "cKDTree": cKDTree,
    "linear_sum_assignment": linear_sum_assignment,
    "VOXEL_SCALE_UM": VOXEL_SCALE_UM,
}
module = ast.Module(body=selected, type_ignores=[])
exec(compile(module, str(NOTEBOOK), "exec"), namespace)
linker = namespace["registered_hungarian_edges"]

base = pd.read_csv(BASE, index_col=0)
reference = pd.read_csv(REFERENCE, index_col=0)
nodes = base[base["row_type"] == "node"]
reference_edges = reference[reference["row_type"] == "edge"]
observed_edges: set[tuple[str, int, int]] = set()
telemetry: dict[str, dict[str, int]] = {}
for dataset, frame in nodes.groupby("dataset", sort=True):
    nodes_by_id = {
        int(row.node_id): {
            "node_id": int(row.node_id),
            "t": int(row.t),
            "z": max(0, int(round(float(row.z)))),
            "y": max(0, int(round(float(row.y)))),
            "x": max(0, int(round(float(row.x)))),
        }
        for row in frame.itertuples()
    }
    stats: dict[str, int] = {}
    links = linker(nodes_by_id, stats)
    telemetry[str(dataset)] = stats
    observed_edges.update(
        (str(dataset), int(edge["source_id"]), int(edge["target_id"])) for edge in links
    )

expected_edges = set(
    zip(
        reference_edges["dataset"].astype(str),
        reference_edges["source_id"].astype(int),
        reference_edges["target_id"].astype(int),
    )
)
missing = expected_edges - observed_edges
extra = observed_edges - expected_edges
if missing or extra:
    raise AssertionError(
        {
            "expected_edges": len(expected_edges),
            "observed_edges": len(observed_edges),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_sample": sorted(missing)[:10],
            "extra_sample": sorted(extra)[:10],
        }
    )

receipt = {
    "status": "PASS_EXACT_EXP052_EDGE_EQUIVALENCE",
    "notebook_sha256": sha256(NOTEBOOK),
    "base_sha256": EXPECTED_BASE_SHA256,
    "reference_sha256": EXPECTED_REFERENCE_SHA256,
    "nodes": int(len(nodes)),
    "expected_edges": len(expected_edges),
    "observed_edges": len(observed_edges),
    "missing_edges": 0,
    "extra_edges": 0,
    "datasets": sorted(telemetry),
    "telemetry": telemetry,
}
RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in receipt.items() if key != "telemetry"}, indent=2))
