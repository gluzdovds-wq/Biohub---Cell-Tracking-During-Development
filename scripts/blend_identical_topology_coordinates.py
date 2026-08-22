"""Blend coordinates of two submissions with identical graph identity/topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
SPATIAL_COLUMNS = ["z", "y", "x"]
IDENTITY_COLUMNS = ["dataset", "node_id", "t"]
EDGE_COLUMNS = ["dataset", "source_id", "target_id"]
SCALE_ZYX_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(frame: pd.DataFrame, row_type: str, columns: list[str]) -> pd.DataFrame:
    return (
        frame[frame["row_type"] == row_type][columns]
        .sort_values(columns)
        .rename_axis("_row_index")
        .reset_index()
    )


def build(left_path: Path, right_path: Path, output_path: Path, left_weight: float) -> dict:
    if not 0.0 <= left_weight <= 1.0:
        raise ValueError(f"left_weight must be in [0, 1], got {left_weight}")
    right_weight = 1.0 - left_weight
    left = pd.read_csv(left_path, index_col=0)
    right = pd.read_csv(right_path, index_col=0)
    if list(left.columns) != EXPECTED_COLUMNS or list(right.columns) != EXPECTED_COLUMNS:
        raise ValueError({"left_columns": list(left.columns), "right_columns": list(right.columns)})
    if len(left) != len(right):
        raise ValueError({"left_rows": len(left), "right_rows": len(right)})

    left_nodes = canonical(left, "node", IDENTITY_COLUMNS)
    right_nodes = canonical(right, "node", IDENTITY_COLUMNS)
    if not left_nodes[IDENTITY_COLUMNS].equals(right_nodes[IDENTITY_COLUMNS]):
        raise AssertionError("node IDs or times differ")
    left_edges = canonical(left, "edge", EDGE_COLUMNS)
    right_edges = canonical(right, "edge", EDGE_COLUMNS)
    if not left_edges[EDGE_COLUMNS].equals(right_edges[EDGE_COLUMNS]):
        raise AssertionError("edge topology differs")

    left_coords = left.loc[left_nodes["_row_index"], SPATIAL_COLUMNS].to_numpy(dtype=float)
    right_coords = right.loc[right_nodes["_row_index"], SPATIAL_COLUMNS].to_numpy(dtype=float)
    blended_coords = left_weight * left_coords + right_weight * right_coords
    result = left.copy(deep=True)
    result.loc[left_nodes["_row_index"], SPATIAL_COLUMNS] = blended_coords
    result.index.name = "id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)

    left_shift = np.linalg.norm((blended_coords - left_coords) * SCALE_ZYX_UM, axis=1)
    right_shift = np.linalg.norm((blended_coords - right_coords) * SCALE_ZYX_UM, axis=1)
    receipt = {
        "status": "PASS",
        "method": "identity-aligned convex coordinate blend",
        "left": str(left_path),
        "left_sha256": sha256(left_path),
        "right": str(right_path),
        "right_sha256": sha256(right_path),
        "left_weight": left_weight,
        "right_weight": right_weight,
        "output_sha256": sha256(output_path),
        "nodes": len(left_nodes),
        "edges": len(left_edges),
        "node_ids_and_times_unchanged": True,
        "topology_unchanged": True,
        "displacement_from_left_um": {
            "mean": float(left_shift.mean()),
            "p95": float(np.quantile(left_shift, 0.95)),
            "max": float(left_shift.max()),
        },
        "displacement_from_right_um": {
            "mean": float(right_shift.mean()),
            "p95": float(np.quantile(right_shift, 0.95)),
            "max": float(right_shift.max()),
        },
    }
    output_path.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-weight", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.left, args.right, args.output, args.left_weight),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
