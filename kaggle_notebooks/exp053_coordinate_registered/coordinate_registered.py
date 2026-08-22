"""Rebuild the promotion-gated EXP053 coordinate/topology composition."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)

NODE_SHA256 = "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0"
EDGE_SHA256 = "3791f74f9247be99d3a9e673cd2ff9fd942764f1ad0b1d0a597d150b7a7c9fab"
OUTPUT_SHA256 = "8103351bf371b7a0654ae87a384e82862a75d33ed83759500d7507c40ee802bc"
EXPECTED_NODES = 122_266
EXPECTED_EDGES = 117_708
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
IDENTITY_COLUMNS = ["dataset", "node_id", "t"]
SPATIAL_COLUMNS = ["z", "y", "x"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(expected_sha: str) -> Path:
    matches = []
    observed = []
    for path in sorted(INPUT.rglob("submission.csv")):
        digest = sha256(path)
        observed.append({"path": str(path), "sha256": digest})
        if digest == expected_sha:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            {"expected_sha256": expected_sha, "matches": list(map(str, matches)), "observed": observed}
        )
    return matches[0]


def main() -> None:
    node_path = locate(NODE_SHA256)
    edge_path = locate(EDGE_SHA256)
    node_artifact = pd.read_csv(node_path, index_col=0)
    edge_artifact = pd.read_csv(edge_path, index_col=0)
    for name, frame in (("node", node_artifact), ("edge", edge_artifact)):
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise AssertionError({name: list(frame.columns)})

    nodes = node_artifact[node_artifact["row_type"] == "node"].copy()
    edge_reference_nodes = edge_artifact[edge_artifact["row_type"] == "node"].copy()
    edges = edge_artifact[edge_artifact["row_type"] == "edge"].copy()
    node_identity = nodes[IDENTITY_COLUMNS].sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    edge_identity = (
        edge_reference_nodes[IDENTITY_COLUMNS]
        .sort_values(IDENTITY_COLUMNS)
        .reset_index(drop=True)
    )
    if not node_identity.equals(edge_identity):
        raise AssertionError("parent node identity/time contracts differ")

    coordinate_rows = nodes.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    reference_rows = edge_reference_nodes.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    coordinate_delta = (
        coordinate_rows[SPATIAL_COLUMNS].to_numpy(dtype=float)
        - reference_rows[SPATIAL_COLUMNS].to_numpy(dtype=float)
    )
    moved_nodes = int(np.any(coordinate_delta != 0.0, axis=1).sum())

    node_keys = set(zip(nodes["dataset"].astype(str), nodes["node_id"].astype(int)))
    source_keys = set(zip(edges["dataset"].astype(str), edges["source_id"].astype(int)))
    target_keys = set(zip(edges["dataset"].astype(str), edges["target_id"].astype(int)))
    if not source_keys <= node_keys or not target_keys <= node_keys:
        raise AssertionError("topology parent has dangling endpoints")
    node_times = {
        (str(row.dataset), int(row.node_id)): int(row.t) for row in nodes.itertuples()
    }
    if any(
        node_times[(str(row.dataset), int(row.target_id))]
        != node_times[(str(row.dataset), int(row.source_id))] + 1
        for row in edges.itertuples()
    ):
        raise AssertionError("nonconsecutive edge")
    maximum_in = int(edges.groupby(["dataset", "target_id"]).size().max())
    maximum_out = int(edges.groupby(["dataset", "source_id"]).size().max())
    divisions = int((edges.groupby(["dataset", "source_id"]).size() == 2).sum())
    if maximum_in != 1 or maximum_out != 1 or divisions != 0:
        raise AssertionError(
            {"maximum_in": maximum_in, "maximum_out": maximum_out, "divisions": divisions}
        )

    result = pd.concat([nodes, edges], ignore_index=True)
    result.index.name = "id"
    output = WORK / "submission.csv"
    result.to_csv(output, lineterminator="\n")
    observed_output_sha = sha256(output)
    if (
        observed_output_sha != OUTPUT_SHA256
        or len(nodes) != EXPECTED_NODES
        or len(edges) != EXPECTED_EDGES
        or nodes["dataset"].nunique() != 4
        or edges["dataset"].nunique() != 4
        or moved_nodes != 93_630
    ):
        raise AssertionError(
            {
                "output_sha256": observed_output_sha,
                "nodes": len(nodes),
                "edges": len(edges),
                "datasets": int(nodes["dataset"].nunique()),
                "moved_nodes": moved_nodes,
            }
        )

    receipt = {
        "status": "PASS_IMMUTABLE_EXP053_COMPOSITION",
        "hypothesis": "H053",
        "node_parent": str(node_path),
        "edge_parent": str(edge_path),
        "node_parent_sha256": NODE_SHA256,
        "edge_parent_sha256": EDGE_SHA256,
        "output": str(output),
        "output_sha256": observed_output_sha,
        "nodes": len(nodes),
        "edges": len(edges),
        "divisions": divisions,
        "moved_nodes_vs_edge_parent": moved_nodes,
        "node_identity_and_time_exact": True,
        "node_rows_exact_from_exp014": True,
        "edge_rows_exact_from_exp052": True,
        "maximum_in_degree": maximum_in,
        "maximum_out_degree": maximum_out,
        "promotion_gate": {
            "exp014_public_lb_at_least_0_919": False,
            "registered_nonnegative_vs_greedy_on_each_untouched_fold": False,
            "registered_positive_vs_greedy_pooled": False,
            "submission_allowed": False,
        },
        "submission_allowed_by_this_receipt": False,
    }
    (WORK / "exp053_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
