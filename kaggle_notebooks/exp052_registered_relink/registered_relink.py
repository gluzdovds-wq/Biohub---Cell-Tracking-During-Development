"""Immutable Kaggle wrapper for the promotion-gated EXP052 registered relink."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

BASE_SHA256 = "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d"
EXPECTED_OUTPUT_SHA256 = "3791f74f9247be99d3a9e673cd2ff9fd942764f1ad0b1d0a597d150b7a7c9fab"
EXPECTED_NODES = 122_266
EXPECTED_EDGES = 117_708
SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
GATE_UM = 7.0
MOTION_SCALE_UM = 3.0
INLIER_UM = 4.0
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


def link_frames(source: pd.DataFrame, target: pd.DataFrame) -> tuple[list[tuple[int, int]], dict]:
    if source.empty or target.empty:
        return [], {"shift_um": [0.0, 0.0, 0.0], "links": 0, "inliers": 0}
    source_points = source[["z", "y", "x"]].to_numpy(dtype=float) * SCALE
    target_points = target[["z", "y", "x"]].to_numpy(dtype=float) * SCALE
    tree = cKDTree(target_points)
    _, nearest = tree.query(source_points, k=1)
    displacement = target_points[np.asarray(nearest, dtype=int)] - source_points
    shift = np.median(displacement, axis=0)
    residual_to_shift = np.linalg.norm(displacement - shift, axis=1)
    inliers = residual_to_shift <= INLIER_UM
    if int(inliers.sum()) >= 3:
        shift = np.median(displacement[inliers], axis=0)

    residual = np.linalg.norm(
        source_points[:, None, :] + shift[None, None, :] - target_points[None, :, :],
        axis=2,
    )
    valid = residual < GATE_UM
    score = np.exp(-residual / MOTION_SCALE_UM)
    minimum_score = float(np.exp(-GATE_UM / MOTION_SCALE_UM))
    cost = np.where(valid, 1.0 - score, 1e6)
    augmented = np.concatenate(
        [cost, np.full((len(source), len(source)), 1.0 - minimum_score, dtype=float)],
        axis=1,
    )
    rows, columns = linear_sum_assignment(augmented)
    source_ids = source["node_id"].to_numpy(dtype=np.int64)
    target_ids = target["node_id"].to_numpy(dtype=np.int64)
    links = [
        (int(source_ids[row]), int(target_ids[column]))
        for row, column in zip(rows, columns)
        if column < len(target) and valid[row, column] and score[row, column] >= minimum_score
    ]
    accepted_residuals = [
        float(residual[row, column])
        for row, column in zip(rows, columns)
        if column < len(target) and valid[row, column] and score[row, column] >= minimum_score
    ]
    telemetry = {
        "shift_um": list(map(float, shift)),
        "initial_nearest_inliers": int(inliers.sum()),
        "sources": int(len(source)),
        "targets": int(len(target)),
        "links": int(len(links)),
        "accepted_residual_um_median": float(np.median(accepted_residuals)) if accepted_residuals else None,
        "accepted_residual_um_max": float(np.max(accepted_residuals)) if accepted_residuals else None,
    }
    return links, telemetry


def build(base_path: Path, output_path: Path) -> dict:
    observed_sha = sha256(base_path)
    if observed_sha != BASE_SHA256:
        raise ValueError({"expected_base_sha256": BASE_SHA256, "actual": observed_sha})
    base = pd.read_csv(base_path, index_col=0)
    if list(base.columns) != EXPECTED_COLUMNS:
        raise ValueError({"columns": list(base.columns)})
    nodes = base[base["row_type"] == "node"].copy()
    old_edges = base[base["row_type"] == "edge"].copy()
    if nodes.duplicated(["dataset", "node_id"]).any():
        raise AssertionError("duplicate node identity")
    if not np.isfinite(nodes[["z", "y", "x"]].to_numpy(dtype=float)).all():
        raise AssertionError("non-finite node coordinates")

    links_by_dataset: dict[str, list[tuple[int, int]]] = {}
    telemetry = []
    for dataset, dataset_nodes in nodes.groupby("dataset", sort=True):
        frames = {int(t): frame.sort_values("node_id") for t, frame in dataset_nodes.groupby("t", sort=True)}
        if sorted(frames) != list(range(min(frames), max(frames) + 1)):
            raise AssertionError(f"{dataset}: noncontiguous frames")
        links = []
        for timepoint in range(min(frames), max(frames)):
            frame_links, frame_telemetry = link_frames(frames[timepoint], frames[timepoint + 1])
            links.extend(frame_links)
            telemetry.append({"dataset": str(dataset), "t": timepoint, **frame_telemetry})
        links_by_dataset[str(dataset)] = links

    edge_rows = []
    for dataset in sorted(links_by_dataset):
        for source, target in sorted(links_by_dataset[dataset]):
            edge_rows.append(
                {
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": source,
                    "target_id": target,
                }
            )
    edges = pd.DataFrame(edge_rows, columns=EXPECTED_COLUMNS)

    node_keys = set(zip(nodes["dataset"].astype(str), nodes["node_id"].astype(int)))
    source_keys = set(zip(edges["dataset"].astype(str), edges["source_id"].astype(int)))
    target_keys = set(zip(edges["dataset"].astype(str), edges["target_id"].astype(int)))
    if not source_keys <= node_keys or not target_keys <= node_keys:
        raise AssertionError("dangling edge endpoint")
    times = {(str(row.dataset), int(row.node_id)): int(row.t) for row in nodes.itertuples()}
    if any(times[(str(row.dataset), int(row.target_id))] != times[(str(row.dataset), int(row.source_id))] + 1 for row in edges.itertuples()):
        raise AssertionError("nonconsecutive edge")
    maximum_in = int(edges.groupby(["dataset", "target_id"]).size().max())
    maximum_out = int(edges.groupby(["dataset", "source_id"]).size().max())
    if maximum_in > 1 or maximum_out > 1:
        raise AssertionError({"maximum_in": maximum_in, "maximum_out": maximum_out})

    old_edge_keys = set(zip(old_edges["dataset"].astype(str), old_edges["source_id"].astype(int), old_edges["target_id"].astype(int)))
    new_edge_keys = set(zip(edges["dataset"].astype(str), edges["source_id"].astype(int), edges["target_id"].astype(int)))
    result = pd.concat([nodes, edges], ignore_index=True)
    result.index.name = "id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, float_format="%.17g", lineterminator="\n")
    roundtrip = pd.read_csv(output_path, index_col=0)
    roundtrip_nodes = roundtrip[roundtrip["row_type"] == "node"].reset_index(drop=True)
    expected_nodes = nodes.reset_index(drop=True)
    if not expected_nodes[["dataset", "row_type", "node_id", "t"]].equals(
        roundtrip_nodes[["dataset", "row_type", "node_id", "t"]]
    ) or not np.array_equal(
        expected_nodes[["z", "y", "x"]].to_numpy(dtype=np.float64),
        roundtrip_nodes[["z", "y", "x"]].to_numpy(dtype=np.float64),
    ):
        raise AssertionError("node rows changed during serialization")

    receipt = {
        "status": "PASS_EXP052_REGISTERED_RELINK_BUILD",
        "hypothesis": "H052",
        "base_sha256": observed_sha,
        "output_sha256": sha256(output_path),
        "nodes": int(len(nodes)),
        "old_edges": int(len(old_edges)),
        "edges": int(len(edges)),
        "retained_old_edges": int(len(old_edge_keys & new_edge_keys)),
        "removed_old_edges": int(len(old_edge_keys - new_edge_keys)),
        "added_edges": int(len(new_edge_keys - old_edge_keys)),
        "divisions": 0,
        "maximum_in_degree": maximum_in,
        "maximum_out_degree": maximum_out,
        "scale_um": SCALE.tolist(),
        "registered_residual_gate_um": GATE_UM,
        "registered_motion_scale_um": MOTION_SCALE_UM,
        "registered_shift_inlier_um": INLIER_UM,
        "node_rows_semantically_exact_from_exp006": True,
        "frame_telemetry": telemetry,
        "promotion_gate": {
            "registered_nonnegative_vs_greedy_on_each_untouched_fold": False,
            "registered_positive_vs_greedy_pooled": False,
            "submission_allowed": False,
        },
    }
    output_path.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    input_root = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
    work_root = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
    observed = []
    matches = []
    for path in input_root.rglob("submission.csv"):
        digest = sha256(path)
        observed.append({"path": str(path), "sha256": digest})
        if digest == BASE_SHA256:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError({"expected_base_sha256": BASE_SHA256, "matches": list(map(str, matches)), "observed": observed})
    output_path = work_root / "submission.csv"
    receipt = build(matches[0], output_path)
    if (
        receipt["output_sha256"] != EXPECTED_OUTPUT_SHA256
        or receipt["nodes"] != EXPECTED_NODES
        or receipt["edges"] != EXPECTED_EDGES
        or receipt["divisions"] != 0
        or receipt["maximum_in_degree"] != 1
        or receipt["maximum_out_degree"] != 1
        or receipt["promotion_gate"]["submission_allowed"] is not False
    ):
        raise AssertionError({"reason": "EXP052 immutable wrapper drift", "receipt": receipt})
    summary = {key: value for key, value in receipt.items() if key != "frame_telemetry"}
    summary["frame_telemetry_records"] = len(receipt["frame_telemetry"])
    summary["wrapper_status"] = "PASS_IMMUTABLE_EXP052_ARTIFACT"
    summary["submission_allowed_by_this_receipt"] = False
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
