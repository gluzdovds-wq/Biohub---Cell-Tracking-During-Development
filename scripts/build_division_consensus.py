"""Add only physically guarded second daughters agreed by two clean trackers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_coordinate_ensemble import EXPECTED_COLUMNS, SPATIAL_COLUMNS
from build_edge_consensus import Node, edge_sets, node_mapping, translate_edges


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected.lower():
        raise ValueError(f"{label} SHA mismatch: {observed} != {expected}")
    return observed


def build(
    base_path: Path,
    tracker_a_path: Path,
    tracker_b_path: Path,
    output_path: Path,
    expected_base_sha: str,
    expected_tracker_a_sha: str,
    expected_tracker_b_sha: str,
    mapping_gate_um: float,
    edge_gate_um: float,
    maximum_cosine: float,
    scale: np.ndarray,
) -> dict:
    input_shas = {
        "base": verify_sha(base_path, expected_base_sha, "base"),
        "tracker_a": verify_sha(tracker_a_path, expected_tracker_a_sha, "tracker_a"),
        "tracker_b": verify_sha(tracker_b_path, expected_tracker_b_sha, "tracker_b"),
    }
    base = pd.read_csv(base_path, index_col=0)
    tracker_a = pd.read_csv(tracker_a_path, index_col=0)
    tracker_b = pd.read_csv(tracker_b_path, index_col=0)
    for name, frame in (("base", base), ("tracker_a", tracker_a), ("tracker_b", tracker_b)):
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError({name: list(frame.columns)})

    base_nodes = base[base["row_type"] == "node"].copy()
    map_a, map_a_counts = node_mapping(
        base_nodes,
        tracker_a[tracker_a["row_type"] == "node"],
        scale,
        mapping_gate_um,
    )
    map_b, map_b_counts = node_mapping(
        base_nodes,
        tracker_b[tracker_b["row_type"] == "node"],
        scale,
        mapping_gate_um,
    )
    _, complete_a, outgoing_a = translate_edges(tracker_a, map_a)
    _, complete_b, outgoing_b = translate_edges(tracker_b, map_b)
    base_edges, base_outgoing, base_incoming = edge_sets(base)
    node_rows = {
        (str(row.dataset), int(row.node_id)): row
        for row in base_nodes.itertuples()
    }

    rejection_counts: Counter[str] = Counter()
    raw_proposals = 0
    guarded_proposals: list[dict] = []
    for source in sorted(complete_a & complete_b):
        targets_a = outgoing_a.get(source, set())
        targets_b = outgoing_b.get(source, set())
        base_targets = base_outgoing.get(source, set())
        if not (
            len(targets_a) == 2
            and targets_a == targets_b
            and len(base_targets) == 1
            and base_targets < targets_a
        ):
            continue
        raw_proposals += 1
        dataset, source_id = source
        existing_target = next(iter(base_targets))
        proposed_target = next(iter(targets_a - base_targets))

        if (dataset, proposed_target) in base_incoming:
            rejection_counts["proposed_target_already_parented"] += 1
            continue
        if len(base_outgoing.get((dataset, existing_target), set())) != 1 or len(
            base_outgoing.get((dataset, proposed_target), set())
        ) != 1:
            rejection_counts["daughter_does_not_continue"] += 1
            continue

        source_row = node_rows[(dataset, source_id)]
        existing_row = node_rows[(dataset, existing_target)]
        proposed_row = node_rows[(dataset, proposed_target)]
        if not (
            int(existing_row.t) == int(source_row.t) + 1
            and int(proposed_row.t) == int(source_row.t) + 1
        ):
            rejection_counts["nonconsecutive_time"] += 1
            continue

        source_xyz = np.asarray([source_row.z, source_row.y, source_row.x], dtype=float)
        existing_xyz = np.asarray(
            [existing_row.z, existing_row.y, existing_row.x], dtype=float
        )
        proposed_xyz = np.asarray(
            [proposed_row.z, proposed_row.y, proposed_row.x], dtype=float
        )
        existing_displacement = (existing_xyz - source_xyz) * scale
        proposed_displacement = (proposed_xyz - source_xyz) * scale
        existing_distance = float(np.linalg.norm(existing_displacement))
        proposed_distance = float(np.linalg.norm(proposed_displacement))
        norm_product = existing_distance * proposed_distance
        cosine = (
            float(np.dot(existing_displacement, proposed_displacement) / norm_product)
            if norm_product > 0
            else 1.0
        )
        if max(existing_distance, proposed_distance) > edge_gate_um:
            rejection_counts["edge_gate"] += 1
            continue
        if cosine > maximum_cosine:
            rejection_counts["nondivergent_motion"] += 1
            continue

        guarded_proposals.append(
            {
                "dataset": dataset,
                "source_id": source_id,
                "existing_target_id": existing_target,
                "proposed_target_id": proposed_target,
                "source_t": int(source_row.t),
                "existing_edge_um": existing_distance,
                "proposed_edge_um": proposed_distance,
                "daughter_separation_um": float(
                    np.linalg.norm((existing_xyz - proposed_xyz) * scale)
                ),
                "displacement_cosine": cosine,
            }
        )

    accepted: list[dict] = []
    reserved_targets: set[Node] = set()
    for proposal in guarded_proposals:
        target = (proposal["dataset"], proposal["proposed_target_id"])
        if target in reserved_targets:
            rejection_counts["duplicate_proposed_target"] += 1
            continue
        reserved_targets.add(target)
        accepted.append(proposal)

    result_edges = set(base_edges)
    for proposal in accepted:
        result_edges.add(
            (
                proposal["dataset"],
                proposal["source_id"],
                proposal["proposed_target_id"],
            )
        )
    if len(result_edges) != len(base_edges) + len(accepted):
        raise AssertionError("added-edge count contract failed")

    incoming_degree: defaultdict[Node, int] = defaultdict(int)
    outgoing_degree: defaultdict[Node, int] = defaultdict(int)
    for dataset, source_id, target_id in result_edges:
        outgoing_degree[(dataset, source_id)] += 1
        incoming_degree[(dataset, target_id)] += 1
    if max(incoming_degree.values(), default=0) > 1:
        raise AssertionError("maximum in-degree exceeds one")
    if max(outgoing_degree.values(), default=0) > 2:
        raise AssertionError("maximum out-degree exceeds two")

    edge_rows = pd.DataFrame(
        [
            (dataset, "edge", -1, -1, -1, -1, -1, source_id, target_id)
            for dataset, source_id, target_id in sorted(result_edges)
        ],
        columns=EXPECTED_COLUMNS,
    )
    result = pd.concat([base_nodes, edge_rows], ignore_index=True)
    result.index.name = "id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)

    receipt = {
        "status": "PASS",
        "method": "two-tracker unanimous orphan-second-daughter consensus",
        "input_sha256": input_shas,
        "output_sha256": sha256(output_path),
        "mapping_gate_um": mapping_gate_um,
        "edge_gate_um": edge_gate_um,
        "maximum_cosine": maximum_cosine,
        "scale_zyx_um": scale.tolist(),
        "mapped_nodes_a": len(map_a),
        "mapped_nodes_b": len(map_b),
        "mapped_nodes_a_by_dataset": map_a_counts,
        "mapped_nodes_b_by_dataset": map_b_counts,
        "raw_proposals": raw_proposals,
        "guarded_proposals": len(guarded_proposals),
        "accepted_additions": len(accepted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "added_edges": accepted,
        "nodes": len(base_nodes),
        "edges": len(result_edges),
        "base_edges": len(base_edges),
        "node_rows_unchanged": True,
        "existing_edges_preserved": base_edges <= result_edges,
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
    parser.add_argument("--expected-base-sha", required=True)
    parser.add_argument("--expected-tracker-a-sha", required=True)
    parser.add_argument("--expected-tracker-b-sha", required=True)
    parser.add_argument("--mapping-gate-um", type=float, default=2.0)
    parser.add_argument("--edge-gate-um", type=float, default=7.0)
    parser.add_argument("--maximum-cosine", type=float, default=0.0)
    parser.add_argument(
        "--scale-zyx", type=float, nargs=3, default=(1.625, 0.40625, 0.40625)
    )
    args = parser.parse_args()
    receipt = build(
        base_path=args.base,
        tracker_a_path=args.tracker_a,
        tracker_b_path=args.tracker_b,
        output_path=args.output,
        expected_base_sha=args.expected_base_sha,
        expected_tracker_a_sha=args.expected_tracker_a_sha,
        expected_tracker_b_sha=args.expected_tracker_b_sha,
        mapping_gate_um=args.mapping_gate_um,
        edge_gate_um=args.edge_gate_um,
        maximum_cosine=args.maximum_cosine,
        scale=np.asarray(args.scale_zyx, dtype=float),
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
