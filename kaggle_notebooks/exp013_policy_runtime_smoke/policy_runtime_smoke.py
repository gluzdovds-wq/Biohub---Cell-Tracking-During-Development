"""Kaggle-runtime smoke test for graph construction, ILP policies, and GEFF I/O."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    raise FileNotFoundError(SUPPORT)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--find-links",
        str(SUPPORT / "wheels"),
        "-r",
        str(SUPPORT / "requirements-unet-ilp-kaggle-predownload.txt"),
    ]
)

REPO = WORK / "tracking_repo"
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(SUPPORT / "repo", REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import tracksdata as td
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from biohub_tracking.io import save_graph
from predict_unet_transformer import build_graph

coords = [
    (0, 0.0, 0.0, 0.0),
    (0, 0.0, 5.0, 0.0),
    (0, 0.0, 10.0, 0.0),
    (1, 0.0, 0.0, 5.0),
    (1, 0.0, 5.0, 5.0),
    (1, 0.0, 10.0, 5.0),
]
candidate_edges = [
    (0, 3, 0.70, 5.0),
    (0, 4, 0.95, 7.1),
    (1, 4, 0.70, 5.0),
    (1, 3, 0.95, 7.1),
    (2, 5, 0.70, 5.0),
    (2, 4, 0.95, 7.1),
]


def registered_probability_links(coords, edges, scale):
    coords_array = np.asarray(coords, dtype=float)
    scale = np.asarray(scale, dtype=float)
    linked = []
    times = sorted(set(coords_array[:, 0].astype(int))) if len(coords_array) else []
    for timepoint in times[:-1]:
        source_ids = np.flatnonzero(coords_array[:, 0].astype(int) == timepoint)
        target_ids = np.flatnonzero(coords_array[:, 0].astype(int) == timepoint + 1)
        source = coords_array[source_ids, 1:4] * scale
        target = coords_array[target_ids, 1:4] * scale
        tree = cKDTree(target)
        _, nearest = tree.query(source, k=1)
        displacement = target[np.asarray(nearest, dtype=int)] - source
        shift = np.median(displacement, axis=0)
        residual_to_shift = np.linalg.norm(displacement - shift, axis=1)
        inliers = residual_to_shift <= 4.0
        if int(inliers.sum()) >= 3:
            shift = np.median(displacement[inliers], axis=0)

        residual = np.linalg.norm(
            source[:, None, :] + shift[None, None, :] - target[None, :, :], axis=2
        )
        learned_probability = np.zeros_like(residual)
        source_rows = {int(node_id): row for row, node_id in enumerate(source_ids)}
        target_columns = {int(node_id): column for column, node_id in enumerate(target_ids)}
        for edge_source, edge_target, edge_probability, _ in edges:
            row = source_rows.get(int(edge_source))
            column = target_columns.get(int(edge_target))
            if row is not None and column is not None:
                learned_probability[row, column] = max(
                    learned_probability[row, column], float(edge_probability)
                )
        score = 0.7 * learned_probability + 0.3 * np.exp(-residual / 3.0)
        valid = (residual < 7.0) & (learned_probability > 0.0)
        cost = np.where(valid, 1.0 - score, 1e6)
        augmented = np.concatenate(
            [cost, np.full((len(source), len(source)), 1.0 - 0.55, dtype=float)], axis=1
        )
        rows, columns = linear_sum_assignment(augmented)
        for row, column in zip(rows, columns):
            if column >= len(target) or not valid[row, column] or score[row, column] < 0.55:
                continue
            raw_distance = float(np.linalg.norm(source[row] - target[column]))
            linked.append(
                (int(source_ids[row]), int(target_ids[column]), float(score[row, column]), raw_distance)
            )
    return linked

results = {}
for policy, appearance_weight, disappearance_weight in (
    ("ilp_public", 0.0, 1.4),
    ("ilp_support", 0.1, 0.1),
):
    graph = build_graph(coords, candidate_edges)
    solver = td.solvers.ILPSolver(
        edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
        division_weight=1.0,
    )
    solved = solver.solve(graph)
    node_rows = list(solved.node_attrs().iter_rows(named=True))
    edge_rows = list(solved.edge_attrs().iter_rows(named=True))
    incoming = {}
    outgoing = {}
    for edge in edge_rows:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        outgoing[source] = outgoing.get(source, 0) + 1
        incoming[target] = incoming.get(target, 0) + 1
    assert max(incoming.values(), default=0) <= 1
    assert max(outgoing.values(), default=0) <= 2

    path = WORK / f"{policy}.geff"
    save_graph(solved, path)
    restored = td.graph.IndexedRXGraph.from_geff(path)
    restored_graph = restored[0] if isinstance(restored, tuple) else restored
    assert restored_graph.num_nodes() == solved.num_nodes()
    assert restored_graph.num_edges() == solved.num_edges()
    results[policy] = {
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "max_in_degree": max(incoming.values(), default=0),
        "max_out_degree": max(outgoing.values(), default=0),
        "geff_roundtrip": True,
    }

hybrid_edges = registered_probability_links(coords, candidate_edges, (1.0, 1.0, 1.0))
assert [(source, target) for source, target, _, _ in hybrid_edges] == [(0, 3), (1, 4), (2, 5)]
hybrid_graph = build_graph(coords, hybrid_edges)
hybrid_path = WORK / "registered_prob_hungarian.geff"
save_graph(hybrid_graph, hybrid_path)
hybrid_restored = td.graph.IndexedRXGraph.from_geff(hybrid_path)
hybrid_restored_graph = hybrid_restored[0] if isinstance(hybrid_restored, tuple) else hybrid_restored
assert hybrid_restored_graph.num_nodes() == hybrid_graph.num_nodes()
assert hybrid_restored_graph.num_edges() == hybrid_graph.num_edges()
results["registered_prob_hungarian"] = {
    "nodes": hybrid_graph.num_nodes(),
    "edges": hybrid_graph.num_edges(),
    "selected_edges": [[source, target] for source, target, _, _ in hybrid_edges],
    "rejects_higher_probability_motion_inconsistent_edges": True,
    "geff_roundtrip": True,
}

result_path = WORK / "policy_runtime_smoke.json"
result_path.write_text(json.dumps({"status": "PASS", "policies": results}, indent=2), encoding="utf-8")
print(result_path.read_text())
