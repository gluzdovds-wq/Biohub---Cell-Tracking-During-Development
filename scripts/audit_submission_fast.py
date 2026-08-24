"""Vectorized structural audit for Biohub submission CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path: Path, expected_datasets: int = 0) -> dict:
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise AssertionError({"columns": list(frame.columns)})
    if not np.array_equal(frame["id"].to_numpy(), np.arange(len(frame))):
        raise AssertionError("id is not consecutive from zero")
    if not frame["row_type"].isin(["node", "edge"]).all():
        raise AssertionError("unexpected row_type")

    nodes = frame[frame["row_type"] == "node"].copy()
    edges = frame[frame["row_type"] == "edge"].copy()
    datasets = sorted(frame["dataset"].unique())
    if expected_datasets and len(datasets) != expected_datasets:
        raise AssertionError({"expected_datasets": expected_datasets, "datasets": datasets})
    if nodes.duplicated(["dataset", "node_id"]).any():
        raise AssertionError("duplicate node_id within dataset")
    if edges.duplicated(["dataset", "source_id", "target_id"]).any():
        raise AssertionError("duplicate edge")
    if (nodes["t"] < 0).any() or not np.isfinite(nodes[["t", "z", "y", "x"]]).all().all():
        raise AssertionError("invalid node time/coordinate")
    if (nodes[["z", "y", "x"]] < 0).any().any():
        raise AssertionError("negative node coordinate")
    if not (nodes[["source_id", "target_id"]] == -1).all().all():
        raise AssertionError("node edge sentinel mismatch")
    if not (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all():
        raise AssertionError("edge node sentinel mismatch")
    if (edges["source_id"] == edges["target_id"]).any():
        raise AssertionError("self-loop")

    lookup = nodes.set_index(["dataset", "node_id"])["t"]
    source_index = pd.MultiIndex.from_frame(edges[["dataset", "source_id"]])
    target_index = pd.MultiIndex.from_frame(edges[["dataset", "target_id"]])
    source_times = lookup.reindex(source_index)
    target_times = lookup.reindex(target_index)
    if source_times.isna().any() or target_times.isna().any():
        raise AssertionError("orphan edge endpoint")
    if not np.array_equal(target_times.to_numpy(), source_times.to_numpy() + 1):
        raise AssertionError("edge does not advance exactly one frame")

    max_in = int(edges.groupby(["dataset", "target_id"]).size().max()) if len(edges) else 0
    max_out = int(edges.groupby(["dataset", "source_id"]).size().max()) if len(edges) else 0
    if max_in > 1 or max_out > 2:
        raise AssertionError({"max_in_degree": max_in, "max_out_degree": max_out})
    return {
        "status": "PASS",
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "datasets": datasets,
        "nodes": len(nodes),
        "edges": len(edges),
        "divisions": int((edges.groupby(["dataset", "source_id"]).size() == 2).sum()),
        "max_in_degree": max_in,
        "max_out_degree": max_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-datasets", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(
        [audit(path, args.expected_datasets) for path in args.paths],
        indent=2,
    ))


if __name__ == "__main__":
    main()
