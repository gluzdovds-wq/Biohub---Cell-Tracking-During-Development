"""Refine one submission's node coordinates without changing its graph topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SPATIAL_COLUMNS = ["z", "y", "x"]
KEY_COLUMNS = ["dataset", "node_id"]
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


def mutual_matches(
    base_coordinates: np.ndarray,
    donor_coordinates: np.ndarray,
    scale: np.ndarray,
    gate_um: float,
) -> list[tuple[int, int, float]]:
    if not len(base_coordinates) or not len(donor_coordinates):
        return []
    base_physical = base_coordinates * scale[None, :]
    donor_physical = donor_coordinates * scale[None, :]
    donor_tree = cKDTree(donor_physical)
    base_tree = cKDTree(base_physical)
    base_to_donor_distance, base_to_donor = donor_tree.query(base_physical, k=1)
    _, donor_to_base = base_tree.query(donor_physical, k=1)
    matches = []
    for base_index, (distance, donor_index) in enumerate(
        zip(base_to_donor_distance, base_to_donor)
    ):
        donor_index = int(donor_index)
        if distance <= gate_um and int(donor_to_base[donor_index]) == base_index:
            matches.append((base_index, donor_index, float(distance)))
    return matches


def build(
    base_path: Path,
    donor_path: Path,
    output_path: Path,
    alpha: float,
    gate_um: float,
    scale: np.ndarray,
) -> dict:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    base = pd.read_csv(base_path, index_col=0)
    donor = pd.read_csv(donor_path, index_col=0)
    if list(base.columns) != EXPECTED_COLUMNS or list(donor.columns) != EXPECTED_COLUMNS:
        raise ValueError({"base_columns": list(base.columns), "donor_columns": list(donor.columns)})

    result = base.copy(deep=True)
    result[SPATIAL_COLUMNS] = result[SPATIAL_COLUMNS].astype(float)
    base_nodes = base[base["row_type"] == "node"]
    donor_nodes = donor[donor["row_type"] == "node"]
    donor_frames = {
        key: frame for key, frame in donor_nodes.groupby(["dataset", "t"], sort=False)
    }
    distances = []
    match_count_by_dataset = {}

    for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
        donor_frame = donor_frames.get((dataset, timepoint))
        if donor_frame is None:
            match_count_by_dataset.setdefault(dataset, 0)
            continue
        base_coordinates = base_frame[SPATIAL_COLUMNS].to_numpy(dtype=float)
        donor_coordinates = donor_frame[SPATIAL_COLUMNS].to_numpy(dtype=float)
        matches = mutual_matches(base_coordinates, donor_coordinates, scale, gate_um)
        if matches:
            base_offsets = np.fromiter((match[0] for match in matches), dtype=int)
            donor_offsets = np.fromiter((match[1] for match in matches), dtype=int)
            matched_row_indices = base_frame.index.to_numpy()[base_offsets]
            refined = (
                (1.0 - alpha) * base_coordinates[base_offsets]
                + alpha * donor_coordinates[donor_offsets]
            )
            result.loc[matched_row_indices, SPATIAL_COLUMNS] = refined
            distances.extend(match[2] for match in matches)
        match_count_by_dataset[dataset] = match_count_by_dataset.get(dataset, 0) + len(matches)

    base_edges = base[base["row_type"] == "edge"][
        ["dataset", "source_id", "target_id"]
    ].reset_index(drop=True)
    result_edges = result[result["row_type"] == "edge"][
        ["dataset", "source_id", "target_id"]
    ].reset_index(drop=True)
    if not base_edges.equals(result_edges):
        raise AssertionError("edge topology changed")
    if not base_nodes[KEY_COLUMNS + ["t"]].equals(
        result[result["row_type"] == "node"][KEY_COLUMNS + ["t"]]
    ):
        raise AssertionError("node identity or time changed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.index.name = "id"
    result.to_csv(output_path)
    distance_array = np.asarray(distances, dtype=float)
    receipt = {
        "status": "PASS",
        "method": "mutual-nearest coordinate-only convex ensemble",
        "base": str(base_path),
        "base_sha256": sha256(base_path),
        "donor": str(donor_path),
        "donor_sha256": sha256(donor_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "alpha": alpha,
        "gate_um": gate_um,
        "scale_zyx_um": scale.tolist(),
        "nodes": int((result["row_type"] == "node").sum()),
        "edges": int((result["row_type"] == "edge").sum()),
        "matched_nodes": len(distances),
        "matched_fraction": len(distances) / max(1, len(base_nodes)),
        "match_count_by_dataset": match_count_by_dataset,
        "match_distance_um": {
            "mean": float(distance_array.mean()) if len(distance_array) else None,
            "p50": float(np.quantile(distance_array, 0.50)) if len(distance_array) else None,
            "p95": float(np.quantile(distance_array, 0.95)) if len(distance_array) else None,
            "max": float(distance_array.max()) if len(distance_array) else None,
        },
        "base_coordinate_displacement_um": {
            "mean": float(alpha * distance_array.mean()) if len(distance_array) else None,
            "p95": float(alpha * np.quantile(distance_array, 0.95)) if len(distance_array) else None,
            "max": float(alpha * distance_array.max()) if len(distance_array) else None,
        },
        "topology_unchanged": True,
        "node_ids_and_times_unchanged": True,
    }
    receipt_path = output_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gate-um", type=float, default=2.0)
    parser.add_argument("--scale-zyx", type=float, nargs=3, default=(1.625, 0.40625, 0.40625))
    args = parser.parse_args()
    receipt = build(
        args.base,
        args.donor,
        args.output,
        args.alpha,
        args.gate_um,
        np.asarray(args.scale_zyx, dtype=float),
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
