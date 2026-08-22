"""Prune only physically implausible EXP006 divisions rejected by two frozen trackers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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
    minimum_bad_residual_um: float,
    minimum_residual_margin_um: float,
    scale: np.ndarray,
) -> dict[str, object]:
    input_sha256 = {
        "base": verify_sha(base_path, expected_base_sha, "base"),
        "tracker_a": verify_sha(tracker_a_path, expected_tracker_a_sha, "tracker_a"),
        "tracker_b": verify_sha(tracker_b_path, expected_tracker_b_sha, "tracker_b"),
    }
    frames = {
        "base": pd.read_csv(base_path, index_col=0),
        "tracker_a": pd.read_csv(tracker_a_path, index_col=0),
        "tracker_b": pd.read_csv(tracker_b_path, index_col=0),
    }
    for name, frame in frames.items():
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError({name: list(frame.columns)})

    base = frames["base"]
    base_nodes = base[base["row_type"] == "node"].copy()
    base_edges, base_outgoing, base_incoming = edge_sets(base)
    node_rows = {
        (str(row.dataset), int(row.node_id)): row for row in base_nodes.itertuples()
    }

    donor_contracts: dict[str, dict[str, object]] = {}
    for name in ("tracker_a", "tracker_b"):
        donor = frames[name]
        mapping, mapped_by_dataset = node_mapping(
            base_nodes,
            donor[donor["row_type"] == "node"],
            scale,
            mapping_gate_um,
        )
        _, complete_sources, translated_outgoing = translate_edges(donor, mapping)
        donor_contracts[name] = {
            "mapping": mapping,
            "mapped_by_dataset": mapped_by_dataset,
            "complete_sources": complete_sources,
            "outgoing": translated_outgoing,
        }

    consensus_candidates = 0
    predecessor_candidates = 0
    physical_candidates = 0
    changes: list[dict[str, object]] = []
    for source in sorted(base_outgoing):
        base_targets = base_outgoing[source]
        if len(base_targets) != 2:
            continue
        if any(source not in donor_contracts[name]["complete_sources"] for name in donor_contracts):
            continue
        outgoing_a = donor_contracts["tracker_a"]["outgoing"].get(source, set())
        outgoing_b = donor_contracts["tracker_b"]["outgoing"].get(source, set())
        if not (
            len(outgoing_a) == 1
            and outgoing_a == outgoing_b
            and outgoing_a < base_targets
        ):
            continue
        consensus_candidates += 1

        dataset, source_id = source
        predecessor_id = base_incoming.get(source)
        if predecessor_id is None:
            continue
        predecessor_candidates += 1
        retained_target = next(iter(outgoing_a))
        removed_target = next(iter(base_targets - outgoing_a))

        predecessor = node_rows[(dataset, predecessor_id)]
        parent = node_rows[(dataset, source_id)]
        retained = node_rows[(dataset, retained_target)]
        removed = node_rows[(dataset, removed_target)]
        if not (
            int(parent.t) == int(predecessor.t) + 1
            and int(retained.t) == int(parent.t) + 1
            and int(removed.t) == int(parent.t) + 1
        ):
            continue

        def point(row: object) -> np.ndarray:
            return np.asarray([row.z, row.y, row.x], dtype=float) * scale

        predecessor_point = point(predecessor)
        parent_point = point(parent)
        predicted_point = parent_point + (parent_point - predecessor_point)
        retained_point = point(retained)
        removed_point = point(removed)
        retained_residual = float(np.linalg.norm(retained_point - predicted_point))
        removed_residual = float(np.linalg.norm(removed_point - predicted_point))
        residual_margin = removed_residual - retained_residual
        if (
            removed_residual < minimum_bad_residual_um
            or residual_margin < minimum_residual_margin_um
        ):
            continue
        physical_candidates += 1
        retained_displacement = retained_point - parent_point
        removed_displacement = removed_point - parent_point
        denominator = float(
            np.linalg.norm(retained_displacement) * np.linalg.norm(removed_displacement)
        )
        displacement_cosine = (
            float(np.dot(retained_displacement, removed_displacement) / denominator)
            if denominator > 0
            else None
        )
        changes.append(
            {
                "dataset": dataset,
                "parent_id": source_id,
                "predecessor_id": predecessor_id,
                "retained_target_id": retained_target,
                "removed_target_id": removed_target,
                "parent_t": int(parent.t),
                "retained_edge_um": float(np.linalg.norm(retained_displacement)),
                "removed_edge_um": float(np.linalg.norm(removed_displacement)),
                "retained_cv_residual_um": retained_residual,
                "removed_cv_residual_um": removed_residual,
                "residual_margin_um": residual_margin,
                "daughter_separation_um": float(np.linalg.norm(retained_point - removed_point)),
                "displacement_cosine": displacement_cosine,
            }
        )

    result_edges = set(base_edges)
    for change in changes:
        edge = (change["dataset"], change["parent_id"], change["removed_target_id"])
        if edge not in result_edges:
            raise AssertionError(f"planned removal is absent: {edge}")
        result_edges.remove(edge)
    if len(result_edges) != len(base_edges) - len(changes):
        raise AssertionError("removed-edge count contract failed")

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

    receipt: dict[str, object] = {
        "status": "PASS",
        "experiment": "EXP040",
        "method": "two-frozen-tracker unanimous singleton plus constant-velocity division prune",
        "input_sha256": input_sha256,
        "output_sha256": sha256(output_path),
        "mapping_gate_um": mapping_gate_um,
        "minimum_bad_residual_um": minimum_bad_residual_um,
        "minimum_residual_margin_um": minimum_residual_margin_um,
        "scale_zyx_um": scale.tolist(),
        "mapped_nodes": {
            name: len(contract["mapping"]) for name, contract in donor_contracts.items()
        },
        "mapped_nodes_by_dataset": {
            name: contract["mapped_by_dataset"] for name, contract in donor_contracts.items()
        },
        "base_divisions": sum(len(targets) == 2 for targets in base_outgoing.values()),
        "consensus_singleton_candidates": consensus_candidates,
        "candidates_with_predecessor": predecessor_candidates,
        "accepted_prunes": physical_candidates,
        "changes": changes,
        "nodes": len(base_nodes),
        "base_edges": len(base_edges),
        "edges": len(result_edges),
        "node_rows_unchanged": True,
        "only_edge_removals": result_edges < base_edges if changes else result_edges == base_edges,
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
    parser.add_argument("--minimum-bad-residual-um", type=float, default=4.0)
    parser.add_argument("--minimum-residual-margin-um", type=float, default=2.0)
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
        minimum_bad_residual_um=args.minimum_bad_residual_um,
        minimum_residual_margin_um=args.minimum_residual_margin_um,
        scale=np.asarray(args.scale_zyx, dtype=float),
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
