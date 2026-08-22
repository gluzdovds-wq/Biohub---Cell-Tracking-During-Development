#!/usr/bin/env python3
"""Compare two Biohub submission graphs using physical node coordinates."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


Node = tuple[str, int, int, int, int]
Edge = tuple[str, Node, Node]


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
                    int(row["z"]),
                    int(row["y"]),
                    int(row["x"]),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
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
