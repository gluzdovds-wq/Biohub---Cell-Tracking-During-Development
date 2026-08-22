"""Audit an edge-only Biohub submission ablation against its frozen control."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]
SCALE_ZYX_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> tuple[pd.DataFrame, dict[tuple[str, int], tuple[int, np.ndarray]], set[tuple[str, int, int]]]:
    frame = pd.read_csv(path, index_col=0)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError({"path": str(path), "columns": list(frame.columns)})
    nodes: dict[tuple[str, int], tuple[int, np.ndarray]] = {}
    for row in frame[frame["row_type"] == "node"].itertuples():
        key = (str(row.dataset), int(row.node_id))
        if key in nodes:
            raise ValueError(f"Duplicate node: {key}")
        nodes[key] = (
            int(row.t),
            np.asarray([row.z, row.y, row.x], dtype=np.float64),
        )
    edges = {
        (str(row.dataset), int(row.source_id), int(row.target_id))
        for row in frame[frame["row_type"] == "edge"].itertuples()
    }
    if len(edges) != int((frame["row_type"] == "edge").sum()):
        raise ValueError(f"Duplicate edge in {path}")
    return frame, nodes, edges


def outgoing(edges: set[tuple[str, int, int]]) -> dict[tuple[str, int], set[int]]:
    result: dict[tuple[str, int], set[int]] = defaultdict(set)
    for dataset, source, target in edges:
        result[(dataset, source)].add(target)
    return result


def incoming(edges: set[tuple[str, int, int]]) -> dict[tuple[str, int], set[int]]:
    result: dict[tuple[str, int], set[int]] = defaultdict(set)
    for dataset, source, target in edges:
        result[(dataset, target)].add(source)
    return result


def edge_length(nodes: dict[tuple[str, int], tuple[int, np.ndarray]], dataset: str, source: int, target: int) -> float:
    source_t, source_point = nodes[(dataset, source)]
    target_t, target_point = nodes[(dataset, target)]
    if target_t != source_t + 1:
        raise AssertionError(f"Non-consecutive edge: {(dataset, source, target, source_t, target_t)}")
    return float(np.linalg.norm((target_point - source_point) * SCALE_ZYX_UM))


def sister_distance(
    nodes: dict[tuple[str, int], tuple[int, np.ndarray]], dataset: str, targets: list[int]
) -> float | None:
    if len(targets) != 2:
        return None
    left_t, left_point = nodes[(dataset, targets[0])]
    right_t, right_point = nodes[(dataset, targets[1])]
    if left_t != right_t:
        raise AssertionError(f"Division daughters are not contemporaneous: {(dataset, targets)}")
    return float(np.linalg.norm((right_point - left_point) * SCALE_ZYX_UM))


def constant_velocity_residual(
    nodes: dict[tuple[str, int], tuple[int, np.ndarray]],
    incoming_edges: dict[tuple[str, int], set[int]],
    dataset: str,
    source: int,
    target: int,
) -> float | None:
    predecessors = incoming_edges.get((dataset, source), set())
    if len(predecessors) != 1:
        return None
    predecessor = next(iter(predecessors))
    predecessor_t, predecessor_point = nodes[(dataset, predecessor)]
    source_t, source_point = nodes[(dataset, source)]
    target_t, target_point = nodes[(dataset, target)]
    if predecessor_t != source_t - 1 or target_t != source_t + 1:
        return None
    predicted = 2.0 * source_point - predecessor_point
    return float(np.linalg.norm((target_point - predicted) * SCALE_ZYX_UM))


def analyse(control_path: Path, candidate_path: Path) -> dict:
    _, control_nodes, control_edges = load(control_path)
    _, candidate_nodes, candidate_edges = load(candidate_path)
    if control_nodes.keys() != candidate_nodes.keys():
        raise AssertionError(
            {
                "control_only_nodes": len(control_nodes.keys() - candidate_nodes.keys()),
                "candidate_only_nodes": len(candidate_nodes.keys() - control_nodes.keys()),
            }
        )
    coordinate_mismatches = []
    for key, (control_t, control_point) in control_nodes.items():
        candidate_t, candidate_point = candidate_nodes[key]
        if control_t != candidate_t or not np.array_equal(control_point, candidate_point):
            coordinate_mismatches.append(key)
    if coordinate_mismatches:
        raise AssertionError({"node_or_coordinate_mismatches": len(coordinate_mismatches)})

    control_out = outgoing(control_edges)
    candidate_out = outgoing(candidate_edges)
    control_in = incoming(control_edges)
    candidate_in = incoming(candidate_edges)
    control_divisions = {key for key, targets in control_out.items() if len(targets) == 2}
    candidate_divisions = {key for key, targets in candidate_out.items() if len(targets) == 2}
    if any(len(targets) > 2 for targets in control_out.values()) or any(
        len(targets) > 2 for targets in candidate_out.values()
    ):
        raise AssertionError("Out-degree above two")

    control_only_edges = control_edges - candidate_edges
    candidate_only_edges = candidate_edges - control_edges
    changed_sources = sorted(
        {(dataset, source) for dataset, source, _ in control_only_edges | candidate_only_edges}
    )
    changed = []
    for dataset, source in changed_sources:
        old_targets = sorted(control_out.get((dataset, source), set()))
        new_targets = sorted(candidate_out.get((dataset, source), set()))
        changed.append(
            {
                "dataset": dataset,
                "source_id": source,
                "control_targets": old_targets,
                "candidate_targets": new_targets,
                "control_edge_lengths_um": [
                    edge_length(control_nodes, dataset, source, target) for target in old_targets
                ],
                "candidate_edge_lengths_um": [
                    edge_length(candidate_nodes, dataset, source, target) for target in new_targets
                ],
                "control_constant_velocity_residuals_um": [
                    constant_velocity_residual(
                        control_nodes, control_in, dataset, source, target
                    )
                    for target in old_targets
                ],
                "candidate_constant_velocity_residuals_um": [
                    constant_velocity_residual(
                        candidate_nodes, candidate_in, dataset, source, target
                    )
                    for target in new_targets
                ],
                "control_sister_distance_um": sister_distance(
                    control_nodes, dataset, old_targets
                ),
                "candidate_sister_distance_um": sister_distance(
                    candidate_nodes, dataset, new_targets
                ),
                "control_is_division": len(old_targets) == 2,
                "candidate_is_division": len(new_targets) == 2,
            }
        )

    all_datasets = sorted({dataset for dataset, _ in control_nodes})
    per_dataset = {}
    for dataset in all_datasets:
        c_edges = {edge for edge in control_edges if edge[0] == dataset}
        n_edges = {edge for edge in candidate_edges if edge[0] == dataset}
        c_div = {key for key in control_divisions if key[0] == dataset}
        n_div = {key for key in candidate_divisions if key[0] == dataset}
        per_dataset[dataset] = {
            "control_edges": len(c_edges),
            "candidate_edges": len(n_edges),
            "control_only_edges": len(c_edges - n_edges),
            "candidate_only_edges": len(n_edges - c_edges),
            "control_divisions": len(c_div),
            "candidate_divisions": len(n_div),
            "changed_sources": sum(item["dataset"] == dataset for item in changed),
        }

    edge_union = control_edges | candidate_edges
    division_union = control_divisions | candidate_divisions
    return {
        "status": "PASS",
        "contract": "identical node IDs, times and exact float64 coordinates; edge-only ablation",
        "control": str(control_path),
        "control_sha256": sha256(control_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "nodes": len(control_nodes),
        "control_edges": len(control_edges),
        "candidate_edges": len(candidate_edges),
        "edge_jaccard": len(control_edges & candidate_edges) / len(edge_union) if edge_union else 1.0,
        "control_only_edges": len(control_only_edges),
        "candidate_only_edges": len(candidate_only_edges),
        "control_divisions": len(control_divisions),
        "candidate_divisions": len(candidate_divisions),
        "division_parent_jaccard": (
            len(control_divisions & candidate_divisions) / len(division_union)
            if division_union else 1.0
        ),
        "removed_division_parents": len(control_divisions - candidate_divisions),
        "added_division_parents": len(candidate_divisions - control_divisions),
        "changed_sources": len(changed),
        "per_dataset": per_dataset,
        "changes": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = analyse(args.control, args.candidate)
    rendered = json.dumps(receipt, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
