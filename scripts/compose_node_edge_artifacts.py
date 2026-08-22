"""Compose node rows from one audited artifact with edges from another."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_coordinate_ensemble import EXPECTED_COLUMNS, SPATIAL_COLUMNS


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
    experiment: str,
    node_artifact_path: Path,
    edge_artifact_path: Path,
    output_path: Path,
    expected_node_sha: str,
    expected_edge_sha: str,
) -> dict[str, object]:
    input_sha256 = {
        "node_artifact": verify_sha(node_artifact_path, expected_node_sha, "node artifact"),
        "edge_artifact": verify_sha(edge_artifact_path, expected_edge_sha, "edge artifact"),
    }
    node_artifact = pd.read_csv(node_artifact_path, index_col=0)
    edge_artifact = pd.read_csv(edge_artifact_path, index_col=0)
    for name, frame in (("node_artifact", node_artifact), ("edge_artifact", edge_artifact)):
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError({name: list(frame.columns)})

    node_rows = node_artifact[node_artifact["row_type"] == "node"].copy()
    edge_reference_nodes = edge_artifact[edge_artifact["row_type"] == "node"].copy()
    edge_rows = edge_artifact[edge_artifact["row_type"] == "edge"].copy()
    identity_columns = ["dataset", "node_id", "t"]
    left_identity = node_rows[identity_columns].sort_values(identity_columns).reset_index(drop=True)
    right_identity = edge_reference_nodes[identity_columns].sort_values(identity_columns).reset_index(drop=True)
    if not left_identity.equals(right_identity):
        raise AssertionError("node identity/time contract differs between artifacts")

    coordinate_rows = node_rows.sort_values(identity_columns).reset_index(drop=True)
    reference_rows = edge_reference_nodes.sort_values(identity_columns).reset_index(drop=True)
    coordinate_delta = (
        coordinate_rows[SPATIAL_COLUMNS].to_numpy(dtype=float)
        - reference_rows[SPATIAL_COLUMNS].to_numpy(dtype=float)
    )
    moved_mask = np.any(coordinate_delta != 0.0, axis=1)

    node_keys = set(zip(node_rows["dataset"].astype(str), node_rows["node_id"].astype(int)))
    source_keys = set(zip(edge_rows["dataset"].astype(str), edge_rows["source_id"].astype(int)))
    target_keys = set(zip(edge_rows["dataset"].astype(str), edge_rows["target_id"].astype(int)))
    if not source_keys <= node_keys or not target_keys <= node_keys:
        raise AssertionError("topology artifact contains dangling endpoints")
    maximum_in = int(edge_rows.groupby(["dataset", "target_id"]).size().max())
    outgoing_sizes = edge_rows.groupby(["dataset", "source_id"]).size()
    maximum_out = int(outgoing_sizes.max())
    divisions = int((outgoing_sizes == 2).sum())
    if maximum_in > 1 or maximum_out > 2:
        raise AssertionError({"maximum_in": maximum_in, "maximum_out": maximum_out})

    result = pd.concat([node_rows, edge_rows], ignore_index=True)
    result.index.name = "id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    receipt: dict[str, object] = {
        "status": "PASS_DISJOINT_NODE_EDGE_COMPOSITION",
        "experiment": experiment,
        "input_sha256": input_sha256,
        "output_sha256": sha256(output_path),
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "divisions": divisions,
        "datasets": int(node_rows["dataset"].nunique()),
        "moved_nodes_vs_edge_artifact": int(moved_mask.sum()),
        "node_identity_and_time_exact": True,
        "node_rows_exact_from_node_artifact": True,
        "edge_rows_exact_from_edge_artifact": True,
        "maximum_in_degree": maximum_in,
        "maximum_out_degree": maximum_out,
        "promotion_gate": {
            "coordinate_parent_lb_at_least_exp006": False,
            "physical_prune_nonnegative_on_both_untouched_folds": False,
            "submission_allowed": False,
        },
    }
    output_path.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--node-artifact", type=Path, required=True)
    parser.add_argument("--edge-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-node-sha", required=True)
    parser.add_argument("--expected-edge-sha", required=True)
    args = parser.parse_args()
    receipt = build(
        experiment=args.experiment,
        node_artifact_path=args.node_artifact,
        edge_artifact_path=args.edge_artifact,
        output_path=args.output,
        expected_node_sha=args.expected_node_sha,
        expected_edge_sha=args.expected_edge_sha,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
