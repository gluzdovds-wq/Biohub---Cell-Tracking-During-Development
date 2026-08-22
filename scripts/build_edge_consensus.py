"""Apply only unanimous alternative edge proposals from two independent trackers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_coordinate_ensemble import EXPECTED_COLUMNS, SPATIAL_COLUMNS, mutual_matches


Edge = tuple[str, int, int]
Node = tuple[str, int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def node_mapping(
    base_nodes: pd.DataFrame,
    tracker_nodes: pd.DataFrame,
    scale: np.ndarray,
    gate_um: float,
) -> tuple[dict[Node, int], dict[str, int]]:
    tracker_frames = {
        key: frame for key, frame in tracker_nodes.groupby(["dataset", "t"], sort=False)
    }
    mapping: dict[Node, int] = {}
    counts: dict[str, int] = defaultdict(int)
    for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
        tracker_frame = tracker_frames.get((dataset, timepoint))
        if tracker_frame is None:
            continue
        matches = mutual_matches(
            base_frame[SPATIAL_COLUMNS].to_numpy(dtype=float),
            tracker_frame[SPATIAL_COLUMNS].to_numpy(dtype=float),
            scale,
            gate_um,
        )
        base_ids = base_frame["node_id"].to_numpy(dtype=int)
        tracker_ids = tracker_frame["node_id"].to_numpy(dtype=int)
        for base_offset, tracker_offset, _ in matches:
            mapping[(dataset, int(tracker_ids[tracker_offset]))] = int(base_ids[base_offset])
            counts[dataset] += 1
    return mapping, dict(counts)


def translate_edges(
    tracker: pd.DataFrame,
    mapping: dict[Node, int],
) -> tuple[set[Edge], set[Node], dict[Node, set[int]]]:
    translated: set[Edge] = set()
    original_out: dict[Node, list[int]] = defaultdict(list)
    translated_out: dict[Node, set[int]] = defaultdict(set)
    for row in tracker[tracker["row_type"] == "edge"].itertuples():
        dataset = str(row.dataset)
        source = int(row.source_id)
        target = int(row.target_id)
        original_out[(dataset, source)].append(target)
        base_source = mapping.get((dataset, source))
        base_target = mapping.get((dataset, target))
        if base_source is None or base_target is None:
            continue
        edge = (dataset, base_source, base_target)
        translated.add(edge)
        translated_out[(dataset, base_source)].add(base_target)

    complete_sources: set[Node] = set()
    for tracker_source, targets in original_out.items():
        dataset, source = tracker_source
        base_source = mapping.get(tracker_source)
        if base_source is None:
            continue
        if all((dataset, target) in mapping for target in targets):
            complete_sources.add((dataset, base_source))
    return translated, complete_sources, translated_out


def edge_sets(frame: pd.DataFrame) -> tuple[set[Edge], dict[Node, set[int]], dict[Node, int]]:
    edges: set[Edge] = set()
    outgoing: dict[Node, set[int]] = defaultdict(set)
    incoming: dict[Node, int] = {}
    for row in frame[frame["row_type"] == "edge"].itertuples():
        edge = (str(row.dataset), int(row.source_id), int(row.target_id))
        edges.add(edge)
        outgoing[(edge[0], edge[1])].add(edge[2])
        if (edge[0], edge[2]) in incoming:
            raise AssertionError(f"base graph has multiple parents at {(edge[0], edge[2])}")
        incoming[(edge[0], edge[2])] = edge[1]
    return edges, outgoing, incoming


def build(
    base_path: Path,
    tracker_a_path: Path,
    tracker_b_path: Path,
    output_path: Path,
    gate_um: float,
    scale: np.ndarray,
) -> dict:
    base = pd.read_csv(base_path, index_col=0)
    tracker_a = pd.read_csv(tracker_a_path, index_col=0)
    tracker_b = pd.read_csv(tracker_b_path, index_col=0)
    for name, frame in (("base", base), ("tracker_a", tracker_a), ("tracker_b", tracker_b)):
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError({name: list(frame.columns)})

    base_nodes = base[base["row_type"] == "node"]
    map_a, map_a_counts = node_mapping(
        base_nodes, tracker_a[tracker_a["row_type"] == "node"], scale, gate_um
    )
    map_b, map_b_counts = node_mapping(
        base_nodes, tracker_b[tracker_b["row_type"] == "node"], scale, gate_um
    )
    _, complete_a, outgoing_a = translate_edges(tracker_a, map_a)
    _, complete_b, outgoing_b = translate_edges(tracker_b, map_b)
    base_edges, base_outgoing, base_incoming = edge_sets(base)

    proposals: dict[Node, int] = {}
    for source in sorted(complete_a & complete_b):
        a_targets = outgoing_a.get(source, set())
        b_targets = outgoing_b.get(source, set())
        base_targets = base_outgoing.get(source, set())
        if (
            len(a_targets) == 1
            and a_targets == b_targets
            and len(base_targets) == 1
            and a_targets != base_targets
        ):
            proposals[source] = next(iter(a_targets))

    accepted: dict[Node, int] = {}
    reserved_targets: set[Node] = set()
    for source, target in proposals.items():
        dataset, source_id = source
        existing_parent = base_incoming.get((dataset, target))
        target_key = (dataset, target)
        if (
            (existing_parent is not None and existing_parent != source_id)
            or target_key in reserved_targets
        ):
            continue
        accepted[source] = target
        reserved_targets.add(target_key)

    result_edges = set(base_edges)
    changed = []
    for source, new_target in accepted.items():
        dataset, source_id = source
        old_target = next(iter(base_outgoing[source]))
        result_edges.remove((dataset, source_id, old_target))
        result_edges.add((dataset, source_id, new_target))
        changed.append(
            {
                "dataset": dataset,
                "source_id": source_id,
                "old_target_id": old_target,
                "new_target_id": new_target,
            }
        )
    if len(result_edges) != len(base_edges):
        raise AssertionError("edge count changed")

    incoming_degree: dict[Node, int] = defaultdict(int)
    outgoing_degree: dict[Node, int] = defaultdict(int)
    for dataset, source, target in result_edges:
        outgoing_degree[(dataset, source)] += 1
        incoming_degree[(dataset, target)] += 1
    if max(incoming_degree.values(), default=0) > 1 or max(outgoing_degree.values(), default=0) > 2:
        raise AssertionError("degree contract failed")

    node_rows = base_nodes.copy()
    edge_rows = pd.DataFrame(
        [
            (dataset, "edge", -1, -1, -1, -1, -1, source, target)
            for dataset, source, target in sorted(result_edges)
        ],
        columns=EXPECTED_COLUMNS,
    )
    result = pd.concat([node_rows, edge_rows], ignore_index=True)
    result.index.name = "id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)

    receipt = {
        "status": "PASS",
        "method": "two-independent-tracker unanimous one-to-one edge replacement",
        "base_sha256": sha256(base_path),
        "tracker_a_sha256": sha256(tracker_a_path),
        "tracker_b_sha256": sha256(tracker_b_path),
        "output_sha256": sha256(output_path),
        "gate_um": gate_um,
        "scale_zyx_um": scale.tolist(),
        "mapped_nodes_a": len(map_a),
        "mapped_nodes_b": len(map_b),
        "mapped_nodes_a_by_dataset": map_a_counts,
        "mapped_nodes_b_by_dataset": map_b_counts,
        "unanimous_alternative_proposals": len(proposals),
        "accepted_replacements": len(accepted),
        "rejected_target_conflicts": len(proposals) - len(accepted),
        "changed_edges": changed,
        "nodes": len(node_rows),
        "edges": len(result_edges),
        "node_rows_unchanged": True,
        "edge_count_unchanged": True,
        "max_in_degree": max(incoming_degree.values(), default=0),
        "max_out_degree": max(outgoing_degree.values(), default=0),
    }
    output_path.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tracker-a", type=Path, required=True)
    parser.add_argument("--tracker-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-um", type=float, default=2.0)
    parser.add_argument("--scale-zyx", type=float, nargs=3, default=(1.625, 0.40625, 0.40625))
    args = parser.parse_args()
    receipt = build(
        args.base,
        args.tracker_a,
        args.tracker_b,
        args.output,
        args.gate_um,
        np.asarray(args.scale_zyx, dtype=float),
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
