"""Apply EXP014/019 coordinates only where both corrections agree in direction."""

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
RECEIPT = WORK / "exp023_receipt.json"

BASE_SHA256 = "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d"
LEFT_SHA256 = "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0"
RIGHT_SHA256 = "7487ecb7de8c110caffd35bd043902b484ee4634ec58d020caebabfad9296c6d"
EXPECTED_FINGERPRINT = "370f75250c312d908eab15ffd68a86fc831660fb41f100b9b50b8aaf1a130457"
EXPECTED_NODES = 122_266
EXPECTED_EDGES = 118_156
EXPECTED_BOTH_CHANGED = 91_869
EXPECTED_ELIGIBLE = 55_338
LEFT_WEIGHT = 0.6
MIN_COSINE = 0.5
SCALE = np.asarray([1.625, 0.40625, 0.40625])
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


base_path = locate(BASE_SHA256)
left_path = locate(LEFT_SHA256)
right_path = locate(RIGHT_SHA256)
base = pd.read_csv(base_path, index_col=0)
left = pd.read_csv(left_path, index_col=0)
right = pd.read_csv(right_path, index_col=0)
for name, frame in (("base", base), ("left", left), ("right", right)):
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError({f"{name}_columns": list(frame.columns)})
if len({len(base), len(left), len(right)}) != 1:
    raise ValueError({"base_rows": len(base), "left_rows": len(left), "right_rows": len(right)})

base_nodes = canonical(base, "node", NODE_IDENTITY)
left_nodes = canonical(left, "node", NODE_IDENTITY)
right_nodes = canonical(right, "node", NODE_IDENTITY)
base_edges = canonical(base, "edge", EDGE_IDENTITY)
left_edges = canonical(left, "edge", EDGE_IDENTITY)
right_edges = canonical(right, "edge", EDGE_IDENTITY)
for candidate in (left_nodes, right_nodes):
    if not base_nodes[NODE_IDENTITY].equals(candidate[NODE_IDENTITY]):
        raise AssertionError("Parent node identities or times differ")
for candidate in (left_edges, right_edges):
    if not base_edges[EDGE_IDENTITY].equals(candidate[EDGE_IDENTITY]):
        raise AssertionError("Parent edge topology differs")
if len(base_nodes) != EXPECTED_NODES or len(base_edges) != EXPECTED_EDGES:
    raise AssertionError({"nodes": len(base_nodes), "edges": len(base_edges)})

base_coords = base.loc[base_nodes["_row_index"], SPATIAL].to_numpy(np.float64)
left_coords = left.loc[left_nodes["_row_index"], SPATIAL].to_numpy(np.float64)
right_coords = right.loc[right_nodes["_row_index"], SPATIAL].to_numpy(np.float64)
left_delta = (left_coords - base_coords) * SCALE
right_delta = (right_coords - base_coords) * SCALE
left_norm = np.linalg.norm(left_delta, axis=1)
right_norm = np.linalg.norm(right_delta, axis=1)
both_changed = (left_norm > 1e-9) & (right_norm > 1e-9)
cosine = np.full(len(base_coords), np.nan)
cosine[both_changed] = np.sum(
    left_delta[both_changed] * right_delta[both_changed], axis=1
) / (left_norm[both_changed] * right_norm[both_changed])
eligible = both_changed & (cosine >= MIN_COSINE)
if int(both_changed.sum()) != EXPECTED_BOTH_CHANGED or int(eligible.sum()) != EXPECTED_ELIGIBLE:
    raise AssertionError(
        {
            "both_changed": int(both_changed.sum()),
            "expected_both_changed": EXPECTED_BOTH_CHANGED,
            "eligible": int(eligible.sum()),
            "expected_eligible": EXPECTED_ELIGIBLE,
        }
    )

proposed = LEFT_WEIGHT * left_coords + (1.0 - LEFT_WEIGHT) * right_coords
result_coords = base_coords.copy()
result_coords[eligible] = proposed[eligible]
result = base.copy(deep=True)
result[SPATIAL] = result[SPATIAL].astype(float)
result.loc[base_nodes["_row_index"], SPATIAL] = result_coords
result.index.name = "id"
result.to_csv(OUTPUT, float_format="%.17g", lineterminator="\n", na_rep="")

serialized = pd.read_csv(OUTPUT, index_col=0)
fingerprint = coordinate_fingerprint(serialized)
if fingerprint != EXPECTED_FINGERPRINT:
    raise AssertionError({"fingerprint": fingerprint, "expected": EXPECTED_FINGERPRINT})

receipt = {
    "status": "PASS",
    "method": "direction-gated 0.6 EXP014 + 0.4 EXP019 coordinate blend",
    "base": str(base_path),
    "base_sha256": sha256(base_path),
    "left": str(left_path),
    "left_sha256": sha256(left_path),
    "right": str(right_path),
    "right_sha256": sha256(right_path),
    "output_sha256": sha256(OUTPUT),
    "coordinate_fingerprint_precision_pixels": 1e-8,
    "output_coordinate_fingerprint": fingerprint,
    "nodes": len(base_nodes),
    "edges": len(base_edges),
    "both_changed": int(both_changed.sum()),
    "eligible_nodes": int(eligible.sum()),
    "eligible_fraction": float(eligible.mean()),
    "cosine_median_on_both_changed": float(np.nanmedian(cosine)),
    "left_weight": LEFT_WEIGHT,
    "min_cosine": MIN_COSINE,
    "topology_unchanged": True,
    "node_ids_and_times_unchanged": True,
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
