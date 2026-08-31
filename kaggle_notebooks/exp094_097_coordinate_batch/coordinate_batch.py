"""Reproduce EXP094-097 from immutable public kernel output artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


INPUT_ROOT = Path(os.environ.get("BIOHUB_COORD_BATCH_INPUT_ROOT", "/kaggle/input"))
OUTPUT_ROOT = Path(os.environ.get("BIOHUB_COORD_BATCH_OUTPUT_ROOT", "/kaggle/working"))
SPATIAL_COLUMNS = ["z", "y", "x"]
EXPECTED_COLUMNS = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]
SCALE_ZYX_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
GATE_UM = 2.0

PARENT_HASHES = {
    "c33": "bd53a723431b0c9b136d6a217ecb470bdb8cf0c44193e5e467542c5e23d30d0f",
    "c29": "d9a02c10f39045833f7cc0a993c49a4a72924adb35ff0b7d3124d07571f1fcf7",
    "c30": "c2a05c177d320259a332769e11b2ea5f849ab0802324ebc3e34f59bf3d25c7ad",
    "comb2": "656caa7274a9c2845862e6ba34ddada7a55eb206f466e232958d8aac6f7794a2",
}
SELECTED_EXPERIMENT = "EXP097"
ARM_CONFIG = {
    "EXP094": {"donor": "c29", "alpha": 0.25, "sha256": "5544a65ae725d033ef2ec397b7a9d0f6be0752b60baabae2606b9e950ec1d07b"},
    "EXP095": {"donor": "c29", "alpha": 0.50, "sha256": "07a7c80599e66d93af09dedbb0affdc26b3fa3bb37d2ac91dcb8f70b1d915182"},
    "EXP096": {"donor": "c30", "alpha": 0.25, "sha256": "b09330498efd4454ec665cd5c9bcd70a868553c6170a1410c8c5883fe1cb9f5e"},
    "EXP097": {"donor": "comb2", "alpha": 0.25, "sha256": "3a12d83c2dda4390a17a5b1e15203a3980c9caacfe407a3e05d76574f1f5769f"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_parents(required_keys: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    wanted = {PARENT_HASHES[key]: key for key in required_keys}
    for path in INPUT_ROOT.rglob("submission.csv"):
        digest = sha256(path)
        key = wanted.get(digest)
        if key is not None:
            if key in found:
                raise RuntimeError(f"duplicate parent hash for {key}: {found[key]} and {path}")
            found[key] = path
            if set(found) == required_keys:
                break
    missing = sorted(required_keys - set(found))
    if missing:
        raise FileNotFoundError({"missing_parent_artifacts": missing})
    return found


def mutual_matches(base_xyz: np.ndarray, donor_xyz: np.ndarray) -> list[tuple[int, int, float]]:
    if not len(base_xyz) or not len(donor_xyz):
        return []
    base_physical = base_xyz * SCALE_ZYX_UM[None, :]
    donor_physical = donor_xyz * SCALE_ZYX_UM[None, :]
    donor_tree = cKDTree(donor_physical)
    base_tree = cKDTree(base_physical)
    base_to_donor_distance, base_to_donor = donor_tree.query(base_physical, k=1)
    _, donor_to_base = base_tree.query(donor_physical, k=1)
    matches = []
    for base_index, (distance, donor_index) in enumerate(
        zip(base_to_donor_distance, base_to_donor)
    ):
        donor_index = int(donor_index)
        if distance <= GATE_UM and int(donor_to_base[donor_index]) == base_index:
            matches.append((base_index, donor_index, float(distance)))
    return matches


def build(
    base: pd.DataFrame,
    donor: pd.DataFrame,
    alpha: float,
    output_name: str,
    expected_sha256: str,
) -> dict:
    result = base.copy(deep=True)
    result[SPATIAL_COLUMNS] = result[SPATIAL_COLUMNS].astype(float)
    base_nodes = base[base["row_type"] == "node"]
    donor_nodes = donor[donor["row_type"] == "node"]
    donor_frames = {
        key: frame for key, frame in donor_nodes.groupby(["dataset", "t"], sort=False)
    }
    distances: list[float] = []

    for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
        donor_frame = donor_frames.get((dataset, timepoint))
        if donor_frame is None:
            continue
        base_xyz = base_frame[SPATIAL_COLUMNS].to_numpy(dtype=float)
        donor_xyz = donor_frame[SPATIAL_COLUMNS].to_numpy(dtype=float)
        matches = mutual_matches(base_xyz, donor_xyz)
        if not matches:
            continue
        base_offsets = np.fromiter((match[0] for match in matches), dtype=int)
        donor_offsets = np.fromiter((match[1] for match in matches), dtype=int)
        row_indices = base_frame.index.to_numpy()[base_offsets]
        result.loc[row_indices, SPATIAL_COLUMNS] = (
            (1.0 - alpha) * base_xyz[base_offsets] + alpha * donor_xyz[donor_offsets]
        )
        distances.extend(match[2] for match in matches)

    base_edges = base[base["row_type"] == "edge"][["dataset", "source_id", "target_id"]]
    result_edges = result[result["row_type"] == "edge"][["dataset", "source_id", "target_id"]]
    if not base_edges.reset_index(drop=True).equals(result_edges.reset_index(drop=True)):
        raise AssertionError("topology changed")

    output_path = OUTPUT_ROOT / output_name
    result.index.name = "id"
    result.to_csv(output_path, lineterminator="\n")
    actual_hash = sha256(output_path)
    if actual_hash != expected_sha256:
        raise AssertionError({"output": output_name, "expected": expected_sha256, "actual": actual_hash})
    return {
        "status": "PASS",
        "output": output_name,
        "sha256": actual_hash,
        "alpha": alpha,
        "nodes": int((result["row_type"] == "node").sum()),
        "edges": int((result["row_type"] == "edge").sum()),
        "matched_nodes": len(distances),
        "mean_match_distance_um": float(np.mean(distances)),
        "topology_unchanged": True,
    }


selected = ARM_CONFIG[SELECTED_EXPERIMENT]
required_parent_keys = {"c33", selected["donor"]}
parents = locate_parents(required_parent_keys)
frames = {key: pd.read_csv(path, index_col=0) for key, path in parents.items()}
for key, frame in frames.items():
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError({key: list(frame.columns)})

receipt = build(
    frames["c33"],
    frames[selected["donor"]],
    selected["alpha"],
    "submission.csv",
    selected["sha256"],
)
(OUTPUT_ROOT / "coordinate_batch_receipt.json").write_text(
    json.dumps({"status": "PASS", "selected_experiment": SELECTED_EXPERIMENT, "parents": {key: str(path) for key, path in parents.items()}, "output": receipt}, indent=2),
    encoding="utf-8",
)
print(json.dumps(receipt, indent=2))
