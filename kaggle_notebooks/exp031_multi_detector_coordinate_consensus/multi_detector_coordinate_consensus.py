"""Rebuild gated EXP031 from three frozen prediction artifacts."""

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
OUTPUT = WORK / "submission.csv"
RECEIPT = WORK / "exp031_receipt.json"
WORK.mkdir(parents=True, exist_ok=True)

BASE_SHA = "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d"
DONOR_A_SHA = "d7ba9e6af86a6bb0be8bd04a36d0c61564e857e03fbadf9a81508211a4a4f2bb"
DONOR_B_SHA = "b1e741c81bfcd435c4e789711a8dd7523031c9517e3687b46e0cf413380db0cc"
# This is computed after writing and re-reading CSV.  The original local build
# receipt hashed the in-memory float array and was therefore not a portable
# serialization contract even though coordinates differed by at most 2.9e-14.
EXPECTED_FINGERPRINT = "26f3847e3fb6f90fa40b200c68b3ad9427cec6a194f4080d290aa318ce8173b0"
EXPECTED_NODES = 122_266
EXPECTED_EDGES = 118_156
EXPECTED_COMMON = 82_517
EXPECTED_BOTH_NONZERO = 78_933
EXPECTED_ELIGIBLE = 50_909
EXPECTED_ELIGIBLE_BY_DATASET = {
    "44b6_0113de3b": 13_497,
    "44b6_0b24845f": 3_648,
    "6bba_05b6850b": 2_561,
    "6bba_05db0fb1": 31_203,
}
EXPECTED_COLUMNS = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]
SPATIAL = ["z", "y", "x"]
SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)
GATE_UM = 2.0
MIN_COSINE = 0.5
ALPHA = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(expected_sha: str) -> Path:
    observed = []
    matches = []
    for path in INPUT.rglob("submission.csv"):
        digest = sha256(path)
        observed.append({"path": str(path), "sha256": digest})
        if digest == expected_sha:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError({"expected_sha": expected_sha, "matches": list(map(str, matches)), "observed": observed})
    return matches[0]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError({"path": str(path), "columns": list(frame.columns)})
    return frame


def mutual_map(base: np.ndarray, donor: np.ndarray) -> dict[int, int]:
    if not len(base) or not len(donor):
        return {}
    base_physical = base * SCALE
    donor_physical = donor * SCALE
    donor_tree = cKDTree(donor_physical)
    base_tree = cKDTree(base_physical)
    distances, base_to_donor = donor_tree.query(base_physical, k=1)
    _, donor_to_base = base_tree.query(donor_physical, k=1)
    return {
        index: int(donor_index)
        for index, (distance, donor_index) in enumerate(zip(distances, base_to_donor))
        if distance <= GATE_UM and int(donor_to_base[int(donor_index)]) == index
    }


def coordinate_fingerprint(frame: pd.DataFrame) -> str:
    coords = frame.loc[frame["row_type"] == "node", SPATIAL].to_numpy(np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite output coordinates")
    quantized = np.rint(coords * 100_000_000).astype("<i8", copy=False)
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


base_path = locate(BASE_SHA)
donor_a_path = locate(DONOR_A_SHA)
donor_b_path = locate(DONOR_B_SHA)
base = load(base_path)
donor_a = load(donor_a_path)
donor_b = load(donor_b_path)
base_nodes = base[base["row_type"] == "node"]
donor_frames = {
    "a": {key: frame for key, frame in donor_a[donor_a["row_type"] == "node"].groupby(["dataset", "t"], sort=False)},
    "b": {key: frame for key, frame in donor_b[donor_b["row_type"] == "node"].groupby(["dataset", "t"], sort=False)},
}

result = base.copy(deep=True)
result[SPATIAL] = result[SPATIAL].astype(float)
common_count = 0
both_nonzero = 0
eligible_rows = []
eligible_coordinates = []
eligible_cosines = []
shifts = []
eligible_by_dataset: dict[str, int] = {}
rejected_out_of_volume = 0

for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
    frame_a = donor_frames["a"].get((dataset, timepoint))
    frame_b = donor_frames["b"].get((dataset, timepoint))
    if frame_a is None or frame_b is None:
        continue
    base_coords = base_frame[SPATIAL].to_numpy(np.float64)
    coords_a = frame_a[SPATIAL].to_numpy(np.float64)
    coords_b = frame_b[SPATIAL].to_numpy(np.float64)
    map_a = mutual_map(base_coords, coords_a)
    map_b = mutual_map(base_coords, coords_b)
    common = sorted(set(map_a) & set(map_b))
    common_count += len(common)
    for offset in common:
        delta_a = (coords_a[map_a[offset]] - base_coords[offset]) * SCALE
        delta_b = (coords_b[map_b[offset]] - base_coords[offset]) * SCALE
        norm_a = float(np.linalg.norm(delta_a))
        norm_b = float(np.linalg.norm(delta_b))
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            continue
        both_nonzero += 1
        cosine = float(np.dot(delta_a, delta_b) / (norm_a * norm_b))
        if cosine < MIN_COSINE:
            continue
        donor_mean = 0.5 * (coords_a[map_a[offset]] + coords_b[map_b[offset]])
        proposed = (1.0 - ALPHA) * base_coords[offset] + ALPHA * donor_mean
        if not np.isfinite(proposed).all() or np.any(proposed < 0.0):
            rejected_out_of_volume += 1
            continue
        eligible_rows.append(int(base_frame.index[offset]))
        eligible_coordinates.append(proposed)
        eligible_cosines.append(cosine)
        shifts.append(float(np.linalg.norm((proposed - base_coords[offset]) * SCALE)))
        eligible_by_dataset[dataset] = eligible_by_dataset.get(dataset, 0) + 1

result.loc[eligible_rows, SPATIAL] = np.asarray(eligible_coordinates)
base_edges = base[base["row_type"] == "edge"][["dataset", "source_id", "target_id"]]
result_edges = result[result["row_type"] == "edge"][["dataset", "source_id", "target_id"]]
result_nodes = result[result["row_type"] == "node"]
if not base_edges.equals(result_edges):
    raise AssertionError("Edge topology changed")
if not base_nodes[["dataset", "node_id", "t"]].equals(result_nodes[["dataset", "node_id", "t"]]):
    raise AssertionError("Node identity or time changed")
observed_contract = {
    "nodes": len(result_nodes),
    "edges": len(result_edges),
    "common": common_count,
    "both_nonzero": both_nonzero,
    "eligible": len(eligible_rows),
    "eligible_by_dataset": eligible_by_dataset,
    "rejected_out_of_volume": rejected_out_of_volume,
}
expected_contract = {
    "nodes": EXPECTED_NODES,
    "edges": EXPECTED_EDGES,
    "common": EXPECTED_COMMON,
    "both_nonzero": EXPECTED_BOTH_NONZERO,
    "eligible": EXPECTED_ELIGIBLE,
    "eligible_by_dataset": EXPECTED_ELIGIBLE_BY_DATASET,
    "rejected_out_of_volume": 0,
}
if observed_contract != expected_contract:
    raise AssertionError({"observed": observed_contract, "expected": expected_contract})

result.index.name = "id"
result.to_csv(OUTPUT, float_format="%.17g", lineterminator="\n", na_rep="")
serialized = pd.read_csv(OUTPUT, index_col=0)
fingerprint = coordinate_fingerprint(serialized)
if fingerprint != EXPECTED_FINGERPRINT:
    raise AssertionError({"fingerprint": fingerprint, "expected": EXPECTED_FINGERPRINT})

shift_array = np.asarray(shifts)
cosine_array = np.asarray(eligible_cosines)
receipt = {
    "status": "PASS",
    "method": "two-donor direction-gated coordinate consensus",
    "parents": {
        "base": {"path": str(base_path), "sha256": sha256(base_path)},
        "donor_a": {"path": str(donor_a_path), "sha256": sha256(donor_a_path)},
        "donor_b": {"path": str(donor_b_path), "sha256": sha256(donor_b_path)},
    },
    **observed_contract,
    "gate_um": GATE_UM,
    "min_cosine": MIN_COSINE,
    "alpha_toward_donor_mean": ALPHA,
    "eligible_cosine_mean": float(cosine_array.mean()),
    "applied_shift_um_mean": float(shift_array.mean()),
    "applied_shift_um_p95": float(np.quantile(shift_array, 0.95)),
    "applied_shift_um_max": float(shift_array.max()),
    "topology_unchanged": True,
    "node_ids_and_times_unchanged": True,
    "output_sha256": sha256(OUTPUT),
    "output_coordinate_fingerprint": fingerprint,
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
