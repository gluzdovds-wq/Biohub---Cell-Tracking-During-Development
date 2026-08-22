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


def build(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    left_weight: float,
    base_path: Path | None = None,
    min_cosine: float | None = None,
) -> dict:
    if not 0.0 <= left_weight <= 1.0:
        raise ValueError(f"left_weight must be in [0, 1], got {left_weight}")
    if (base_path is None) != (min_cosine is None):
        raise ValueError("base_path and min_cosine must be provided together")
    if min_cosine is not None and not -1.0 <= min_cosine <= 1.0:
        raise ValueError(f"min_cosine must be in [-1, 1], got {min_cosine}")
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
    proposed_coords = left_weight * left_coords + right_weight * right_coords
    eligibility = None
    direction_receipt = None
    if base_path is None:
        result = left.copy(deep=True)
        result_nodes = left_nodes
        blended_coords = proposed_coords
    else:
        base = pd.read_csv(base_path, index_col=0)
        if list(base.columns) != EXPECTED_COLUMNS or len(base) != len(left):
            raise ValueError({"base_columns": list(base.columns), "base_rows": len(base)})
        base_nodes = canonical(base, "node", IDENTITY_COLUMNS)
        base_edges = canonical(base, "edge", EDGE_COLUMNS)
        if not base_nodes[IDENTITY_COLUMNS].equals(left_nodes[IDENTITY_COLUMNS]):
            raise AssertionError("base node IDs or times differ")
        if not base_edges[EDGE_COLUMNS].equals(left_edges[EDGE_COLUMNS]):
            raise AssertionError("base edge topology differs")
        base_coords = base.loc[base_nodes["_row_index"], SPATIAL_COLUMNS].to_numpy(dtype=float)
        left_delta = (left_coords - base_coords) * SCALE_ZYX_UM
        right_delta = (right_coords - base_coords) * SCALE_ZYX_UM
        left_norm = np.linalg.norm(left_delta, axis=1)
        right_norm = np.linalg.norm(right_delta, axis=1)
        both_changed = (left_norm > 1e-9) & (right_norm > 1e-9)
        cosine = np.full(len(base_coords), np.nan, dtype=float)
        cosine[both_changed] = np.sum(
            left_delta[both_changed] * right_delta[both_changed], axis=1
        ) / (left_norm[both_changed] * right_norm[both_changed])
        eligibility = both_changed & (cosine >= float(min_cosine))
        blended_coords = base_coords.copy()
        blended_coords[eligibility] = proposed_coords[eligibility]
        result = base.copy(deep=True)
        result_nodes = base_nodes
        direction_receipt = {
            "base": str(base_path),
            "base_sha256": sha256(base_path),
            "min_cosine": min_cosine,
            "both_changed": int(both_changed.sum()),
            "eligible_nodes": int(eligibility.sum()),
            "eligible_fraction": float(eligibility.mean()),
            "cosine_mean_on_both_changed": float(np.nanmean(cosine)),
            "cosine_median_on_both_changed": float(np.nanmedian(cosine)),
        }
    result[SPATIAL_COLUMNS] = result[SPATIAL_COLUMNS].astype(float)
    result.loc[result_nodes["_row_index"], SPATIAL_COLUMNS] = blended_coords
    result.index.name = "id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)

    left_shift = np.linalg.norm((blended_coords - left_coords) * SCALE_ZYX_UM, axis=1)
    right_shift = np.linalg.norm((blended_coords - right_coords) * SCALE_ZYX_UM, axis=1)
    receipt = {
        "status": "PASS",
        "method": (
            "direction-gated identity-aligned convex coordinate blend"
            if base_path is not None
            else "identity-aligned convex coordinate blend"
        ),
        "left": str(left_path),
        "left_sha256": sha256(left_path),
        "right": str(right_path),
        "right_sha256": sha256(right_path),
        "left_weight": left_weight,
        "right_weight": right_weight,
        "direction_gate": direction_receipt,
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
    parser.add_argument("--base", type=Path)
    parser.add_argument("--min-cosine", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.left,
                args.right,
                args.output,
                args.left_weight,
                args.base,
                args.min_cosine,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
