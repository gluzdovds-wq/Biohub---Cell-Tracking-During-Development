"""Build a topology-preserving coordinate consensus from two detector donors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EXPECTED_COLUMNS = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]
SPATIAL = ["z", "y", "x"]
SCALE_ZYX_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError({"path": str(path), "columns": list(frame.columns)})
    return frame


def mutual_map(base: np.ndarray, donor: np.ndarray, gate_um: float) -> dict[int, int]:
    if not len(base) or not len(donor):
        return {}
    base_physical = base * SCALE_ZYX_UM
    donor_physical = donor * SCALE_ZYX_UM
    donor_tree = cKDTree(donor_physical)
    base_tree = cKDTree(base_physical)
    distances, base_to_donor = donor_tree.query(base_physical, k=1)
    _, donor_to_base = base_tree.query(donor_physical, k=1)
    return {
        index: int(donor_index)
        for index, (distance, donor_index) in enumerate(zip(distances, base_to_donor))
        if distance <= gate_um and int(donor_to_base[int(donor_index)]) == index
    }


def coordinate_fingerprint(frame: pd.DataFrame) -> str:
    coords = frame.loc[frame["row_type"] == "node", SPATIAL].to_numpy(np.float64)
    quantized = np.rint(coords * 100_000_000).astype("<i8", copy=False)
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


def build(
    base_path: Path,
    donor_a_path: Path,
    donor_b_path: Path,
    output_path: Path,
    gate_um: float,
    min_cosine: float,
    alpha: float,
) -> dict:
    if gate_um <= 0 or not -1.0 <= min_cosine <= 1.0 or not 0.0 <= alpha <= 1.0:
        raise ValueError({"gate_um": gate_um, "min_cosine": min_cosine, "alpha": alpha})
    base = load(base_path)
    donor_a = load(donor_a_path)
    donor_b = load(donor_b_path)
    base_nodes = base[base["row_type"] == "node"]
    donor_nodes = {
        "a": donor_a[donor_a["row_type"] == "node"],
        "b": donor_b[donor_b["row_type"] == "node"],
    }
    donor_frames = {
        name: {key: frame for key, frame in nodes.groupby(["dataset", "t"], sort=False)}
        for name, nodes in donor_nodes.items()
    }

    result = base.copy(deep=True)
    result[SPATIAL] = result[SPATIAL].astype(float)
    cosine_values: list[float] = []
    eligible_cosines: list[float] = []
    applied_shifts: list[float] = []
    common_matches = 0
    both_nonzero = 0
    rejected_out_of_volume = 0
    eligible_by_dataset: dict[str, int] = {}
    eligible_row_indices: list[int] = []
    eligible_coordinates: list[np.ndarray] = []

    for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
        frame_a = donor_frames["a"].get((dataset, timepoint))
        frame_b = donor_frames["b"].get((dataset, timepoint))
        if frame_a is None or frame_b is None:
            continue
        base_coords = base_frame[SPATIAL].to_numpy(np.float64)
        coords_a = frame_a[SPATIAL].to_numpy(np.float64)
        coords_b = frame_b[SPATIAL].to_numpy(np.float64)
        map_a = mutual_map(base_coords, coords_a, gate_um)
        map_b = mutual_map(base_coords, coords_b, gate_um)
        common = sorted(set(map_a) & set(map_b))
        common_matches += len(common)
        for base_offset in common:
            delta_a = (coords_a[map_a[base_offset]] - base_coords[base_offset]) * SCALE_ZYX_UM
            delta_b = (coords_b[map_b[base_offset]] - base_coords[base_offset]) * SCALE_ZYX_UM
            norm_a = float(np.linalg.norm(delta_a))
            norm_b = float(np.linalg.norm(delta_b))
            if norm_a <= 1e-9 or norm_b <= 1e-9:
                continue
            both_nonzero += 1
            cosine = float(np.dot(delta_a, delta_b) / (norm_a * norm_b))
            cosine_values.append(cosine)
            if cosine < min_cosine:
                continue
            donor_mean = 0.5 * (coords_a[map_a[base_offset]] + coords_b[map_b[base_offset]])
            proposed = (1.0 - alpha) * base_coords[base_offset] + alpha * donor_mean
            if not np.isfinite(proposed).all() or np.any(proposed < 0.0):
                rejected_out_of_volume += 1
                continue
            row_index = base_frame.index[base_offset]
            eligible_row_indices.append(int(row_index))
            eligible_coordinates.append(proposed)
            eligible_cosines.append(cosine)
            applied_shifts.append(
                float(np.linalg.norm((proposed - base_coords[base_offset]) * SCALE_ZYX_UM))
            )
            eligible_by_dataset[dataset] = eligible_by_dataset.get(dataset, 0) + 1

    if eligible_row_indices:
        result.loc[eligible_row_indices, SPATIAL] = np.asarray(eligible_coordinates)

    base_edges = base[base["row_type"] == "edge"][["dataset", "source_id", "target_id"]]
    result_edges = result[result["row_type"] == "edge"][["dataset", "source_id", "target_id"]]
    if not base_edges.equals(result_edges):
        raise AssertionError("Edge topology changed")
    if not base_nodes[["dataset", "node_id", "t"]].equals(
        result[result["row_type"] == "node"][["dataset", "node_id", "t"]]
    ):
        raise AssertionError("Node identity or time changed")

    result.index.name = "id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    shifts = np.asarray(applied_shifts)
    cosines = np.asarray(cosine_values)
    eligible = np.asarray(eligible_cosines)
    receipt = {
        "status": "PASS",
        "method": "two-donor direction-gated coordinate consensus",
        "base": str(base_path),
        "base_sha256": sha256(base_path),
        "donor_a": str(donor_a_path),
        "donor_a_sha256": sha256(donor_a_path),
        "donor_b": str(donor_b_path),
        "donor_b_sha256": sha256(donor_b_path),
        "gate_um": gate_um,
        "min_cosine": min_cosine,
        "alpha_toward_donor_mean": alpha,
        "common_mutual_matches": common_matches,
        "both_nonzero": both_nonzero,
        "cosine_mean": float(cosines.mean()) if len(cosines) else None,
        "cosine_median": float(np.median(cosines)) if len(cosines) else None,
        "eligible_nodes": len(shifts),
        "eligible_by_dataset": eligible_by_dataset,
        "eligible_cosine_mean": float(eligible.mean()) if len(eligible) else None,
        "rejected_out_of_volume": rejected_out_of_volume,
        "applied_shift_um_mean": float(shifts.mean()) if len(shifts) else None,
        "applied_shift_um_p95": float(np.quantile(shifts, 0.95)) if len(shifts) else None,
        "applied_shift_um_max": float(shifts.max()) if len(shifts) else None,
        "nodes": int((result["row_type"] == "node").sum()),
        "edges": int((result["row_type"] == "edge").sum()),
        "node_ids_and_times_unchanged": True,
        "topology_unchanged": True,
        "output_sha256": sha256(output_path),
        "output_coordinate_fingerprint": coordinate_fingerprint(result),
    }
    output_path.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor-a", type=Path, required=True)
    parser.add_argument("--donor-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-um", type=float, default=2.0)
    parser.add_argument("--min-cosine", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    receipt = build(
        args.base,
        args.donor_a,
        args.donor_b,
        args.output,
        args.gate_um,
        args.min_cosine,
        args.alpha,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
