"""Compose EXP014 and EXP019 coordinates without changing EXP006 topology."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT = WORK / "submission.csv"
RECEIPT = WORK / "exp022_receipt.json"

LEFT_SHA256 = "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0"
RIGHT_SHA256 = "7487ecb7de8c110caffd35bd043902b484ee4634ec58d020caebabfad9296c6d"
EXPECTED_FINGERPRINT = "7989529a98de7c86e97ff5d23f2d3d2bc29dee68e8ea1e21b7c1366d00500e53"
EXPECTED_NODES = 122_266
EXPECTED_EDGES = 118_156
LEFT_WEIGHT = 0.6
SPATIAL = ["z", "y", "x"]
NODE_IDENTITY = ["dataset", "node_id", "t"]
EDGE_IDENTITY = ["dataset", "source_id", "target_id"]
EXPECTED_COLUMNS = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(expected: str) -> Path:
    matches = [path for path in INPUT.rglob("submission.csv") if sha256(path) == expected]
    if len(matches) != 1:
        observed = {str(path): sha256(path) for path in INPUT.rglob("submission.csv")}
        raise RuntimeError({"expected": expected, "matches": list(map(str, matches)), "observed": observed})
    return matches[0]


def canonical(frame: pd.DataFrame, row_type: str, columns: list[str]) -> pd.DataFrame:
    return (
        frame[frame["row_type"] == row_type][columns]
        .sort_values(columns)
        .rename_axis("_row_index")
        .reset_index()
    )


def coordinate_fingerprint(frame: pd.DataFrame) -> str:
    coords = frame.loc[frame["row_type"] == "node", SPATIAL].to_numpy(np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite node coordinate")
    quantized = np.rint(coords * 100_000_000).astype("<i8", copy=False)
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


left_path = locate(LEFT_SHA256)
right_path = locate(RIGHT_SHA256)
left = pd.read_csv(left_path, index_col=0)
right = pd.read_csv(right_path, index_col=0)
if list(left.columns) != EXPECTED_COLUMNS or list(right.columns) != EXPECTED_COLUMNS:
    raise ValueError({"left_columns": list(left.columns), "right_columns": list(right.columns)})
if len(left) != len(right):
    raise ValueError({"left_rows": len(left), "right_rows": len(right)})

left_nodes = canonical(left, "node", NODE_IDENTITY)
right_nodes = canonical(right, "node", NODE_IDENTITY)
left_edges = canonical(left, "edge", EDGE_IDENTITY)
right_edges = canonical(right, "edge", EDGE_IDENTITY)
if not left_nodes[NODE_IDENTITY].equals(right_nodes[NODE_IDENTITY]):
    raise AssertionError("Parent node identities or times differ")
if not left_edges[EDGE_IDENTITY].equals(right_edges[EDGE_IDENTITY]):
    raise AssertionError("Parent edge topology differs")
if len(left_nodes) != EXPECTED_NODES or len(left_edges) != EXPECTED_EDGES:
    raise AssertionError({"nodes": len(left_nodes), "edges": len(left_edges)})

left_coords = left.loc[left_nodes["_row_index"], SPATIAL].to_numpy(np.float64)
right_coords = right.loc[right_nodes["_row_index"], SPATIAL].to_numpy(np.float64)
blended = LEFT_WEIGHT * left_coords + (1.0 - LEFT_WEIGHT) * right_coords
result = left.copy(deep=True)
result[SPATIAL] = result[SPATIAL].astype(float)
result.loc[left_nodes["_row_index"], SPATIAL] = blended
result.index.name = "id"
result.to_csv(OUTPUT, float_format="%.17g", lineterminator="\n", na_rep="")

serialized = pd.read_csv(OUTPUT, index_col=0)
fingerprint = coordinate_fingerprint(serialized)
if fingerprint != EXPECTED_FINGERPRINT:
    raise AssertionError({"fingerprint": fingerprint, "expected": EXPECTED_FINGERPRINT})

scale = np.asarray([1.625, 0.40625, 0.40625])
left_shift = np.linalg.norm((blended - left_coords) * scale, axis=1)
right_shift = np.linalg.norm((blended - right_coords) * scale, axis=1)
receipt = {
    "status": "PASS",
    "method": "identity-aligned 0.6 EXP014 + 0.4 EXP019 coordinate blend",
    "left": str(left_path),
    "left_sha256": sha256(left_path),
    "right": str(right_path),
    "right_sha256": sha256(right_path),
    "output_sha256": sha256(OUTPUT),
    "coordinate_fingerprint_precision_pixels": 1e-8,
    "output_coordinate_fingerprint": fingerprint,
    "nodes": len(left_nodes),
    "edges": len(left_edges),
    "left_weight": LEFT_WEIGHT,
    "topology_unchanged": True,
    "node_ids_and_times_unchanged": True,
    "mean_shift_from_left_um": float(left_shift.mean()),
    "mean_shift_from_right_um": float(right_shift.mean()),
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
