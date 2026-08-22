#!/usr/bin/env python3
"""Compare two Biohub submission graphs using physical node coordinates."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


Node = tuple[str, int, float, float, float]
Edge = tuple[str, Node, Node]
VOXEL_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


@dataclass
class Graph:
    nodes: set[Node]
    edges: set[Edge]
    divisions: set[Node]


def read_graph(path: Path) -> Graph:
    nodes: set[Node] = set()
    raw_edges: list[tuple[str, int, int]] = []
    node_by_id: dict[tuple[str, int], Node] = {}

    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            if row["row_type"] == "node":
                node = (
                    dataset,
                    int(row["t"]),
                    float(row["z"]),
                    float(row["y"]),
                    float(row["x"]),
                )
                nodes.add(node)
                node_by_id[(dataset, int(row["node_id"]))] = node
            elif row["row_type"] == "edge":
                raw_edges.append(
                    (dataset, int(row["source_id"]), int(row["target_id"]))
                )

    edges: set[Edge] = set()
    out_degree: Counter[Node] = Counter()
    for dataset, source_id, target_id in raw_edges:
        source = node_by_id[(dataset, source_id)]
        target = node_by_id[(dataset, target_id)]
        edges.add((dataset, source, target))
        out_degree[source] += 1

    return Graph(
        nodes=nodes,
        edges=edges,
        divisions={node for node, degree in out_degree.items() if degree == 2},
    )


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def summarize(label: str, graph: Graph) -> str:
    return (
        f"{label}: nodes={len(graph.nodes):,} edges={len(graph.edges):,} "
        f"divisions={len(graph.divisions):,}"
    )


def by_dataset(values: set, dataset_index: int = 0) -> dict[str, set]:
    grouped: dict[str, set] = defaultdict(set)
    for value in values:
        grouped[value[dataset_index]].add(value)
    return grouped


def match_nodes_physical(
    left: set[Node], right: set[Node], radius_um: float
) -> dict[Node, Node]:
    """Greedily match same-frame nodes by ascending physical distance."""
    left_frames: dict[tuple[str, int], list[Node]] = defaultdict(list)
    right_frames: dict[tuple[str, int], list[Node]] = defaultdict(list)
    for node in left:
        left_frames[(node[0], node[1])].append(node)
    for node in right:
        right_frames[(node[0], node[1])].append(node)

    matches: dict[Node, Node] = {}
    for frame_key in set(left_frames) & set(right_frames):
        left_nodes = left_frames[frame_key]
        right_nodes = right_frames[frame_key]
        left_xyz = np.asarray([node[2:] for node in left_nodes]) * VOXEL_SCALE_UM
        right_xyz = np.asarray([node[2:] for node in right_nodes]) * VOXEL_SCALE_UM
        tree = cKDTree(right_xyz)
        candidates: list[tuple[float, int, int]] = []
        for left_index, neighbors in enumerate(
            tree.query_ball_point(left_xyz, r=radius_um)
        ):
            for right_index in neighbors:
                distance = float(
                    np.linalg.norm(left_xyz[left_index] - right_xyz[right_index])
                )
                candidates.append((distance, left_index, right_index))

        used_left: set[int] = set()
        used_right: set[int] = set()
        for _, left_index, right_index in sorted(candidates):
            if left_index in used_left or right_index in used_right:
                continue
            used_left.add(left_index)
            used_right.add(right_index)
            matches[left_nodes[left_index]] = right_nodes[right_index]
    return matches


def fuzzy_summary(left: Graph, right: Graph, radius_um: float) -> str:
    node_map = match_nodes_physical(left.nodes, right.nodes, radius_um)
    node_matches = len(node_map)
    node_union = len(left.nodes) + len(right.nodes) - node_matches

    mapped_edges = {
        (dataset, node_map[source], node_map[target])
        for dataset, source, target in left.edges
        if source in node_map and target in node_map
    }
    edge_matches = len(mapped_edges & right.edges)
    edge_union = len(left.edges) + len(right.edges) - edge_matches

    mapped_divisions = {node_map[node] for node in left.divisions if node in node_map}
    division_matches = len(mapped_divisions & right.divisions)
    division_union = len(left.divisions) + len(right.divisions) - division_matches
    return (
        f"physical@{radius_um:g}um: "
        f"node_jaccard={node_matches / node_union if node_union else 1.0:.6f} "
        f"edge_jaccard={edge_matches / edge_union if edge_union else 1.0:.6f} "
        f"division_jaccard={division_matches / division_union if division_union else 1.0:.6f} "
        f"node_matches={node_matches:,} edge_matches={edge_matches:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--match-radius-um",
        type=float,
        default=2.0,
        help="Radius for approximate same-frame physical matching (default: 2.0)",
    )
    args = parser.parse_args()

    left = read_graph(args.left)
    right = read_graph(args.right)

    print(summarize(str(args.left), left))
    print(summarize(str(args.right), right))
    print(
        "all: "
        f"node_jaccard={jaccard(left.nodes, right.nodes):.6f} "
        f"edge_jaccard={jaccard(left.edges, right.edges):.6f} "
        f"division_jaccard={jaccard(left.divisions, right.divisions):.6f} "
        f"nodes_left_only={len(left.nodes - right.nodes):,} "
        f"nodes_right_only={len(right.nodes - left.nodes):,} "
        f"edges_left_only={len(left.edges - right.edges):,} "
        f"edges_right_only={len(right.edges - left.edges):,}"
    )
    print(fuzzy_summary(left, right, args.match_radius_um))

    left_nodes = by_dataset(left.nodes)
    right_nodes = by_dataset(right.nodes)
    left_edges = by_dataset(left.edges)
    right_edges = by_dataset(right.edges)
    left_divisions = by_dataset(left.divisions)
    right_divisions = by_dataset(right.divisions)
    for dataset in sorted(set(left_nodes) | set(right_nodes)):
        ln, rn = left_nodes[dataset], right_nodes[dataset]
        le, re = left_edges[dataset], right_edges[dataset]
        ld, rd = left_divisions[dataset], right_divisions[dataset]
        print(
            f"{dataset}: node_jaccard={jaccard(ln, rn):.6f} "
            f"edge_jaccard={jaccard(le, re):.6f} "
            f"divisions={len(ld):,}->{len(rd):,} "
            f"edges_delta={len(re) - len(le):+,}"
        )


if __name__ == "__main__":
    main()
