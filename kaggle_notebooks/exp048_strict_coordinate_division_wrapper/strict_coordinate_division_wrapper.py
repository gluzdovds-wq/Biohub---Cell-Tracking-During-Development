"""Verify and expose the immutable EXP047 strict composition artifact."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)

SLUG = "biohub-exp047-strict-coordinate-division"
EXPECTED_SHA256 = "5dd662d8d12f91120425a11a7667059529ce53ad7eab4f756879e9477cf363f2"
EXPECTED_NODE_INPUT_SHA256 = "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0"
EXPECTED_EDGE_INPUT_SHA256 = "21a42ffa33c8af7ef44b28f7edaea6a3d9666745139c9c51e132fed41a8fe114"
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
EXPECTED_NODES = 122266
EXPECTED_EDGES = 118101
EXPECTED_DIVISIONS = 400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_root() -> Path:
    candidates = [
        INPUT / "datasets" / "dmitriigluzdov" / SLUG,
        INPUT / SLUG,
    ]
    matches = [root for root in candidates if (root / "submission.csv").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            {"slug": SLUG, "candidates": list(map(str, candidates)), "matches": list(map(str, matches))}
        )
    return matches[0]


root = locate_root()
source = root / "submission.csv"
build_path = root / "submission.receipt.json"
if not build_path.is_file():
    raise FileNotFoundError(str(build_path))

observed_sha = sha256(source)
build = json.loads(build_path.read_text(encoding="utf-8"))
if observed_sha != EXPECTED_SHA256:
    raise AssertionError(f"EXP047 source SHA mismatch: {observed_sha}")
if (
    build.get("status") != "PASS_DISJOINT_NODE_EDGE_COMPOSITION"
    or build.get("experiment") != "EXP047"
    or build.get("output_sha256") != observed_sha
    or build.get("input_sha256", {}).get("node_artifact") != EXPECTED_NODE_INPUT_SHA256
    or build.get("input_sha256", {}).get("edge_artifact") != EXPECTED_EDGE_INPUT_SHA256
    or build.get("node_identity_and_time_exact") is not True
    or build.get("node_rows_exact_from_node_artifact") is not True
    or build.get("edge_rows_exact_from_edge_artifact") is not True
    or build.get("promotion_gate", {}).get("submission_allowed") is not False
):
    raise AssertionError(build)

frame = pd.read_csv(source, index_col=0)
if list(frame.columns) != EXPECTED_COLUMNS:
    raise AssertionError({"columns": list(frame.columns)})
nodes = frame[frame["row_type"] == "node"]
edges = frame[frame["row_type"] == "edge"]
if len(nodes) != EXPECTED_NODES or len(edges) != EXPECTED_EDGES:
    raise AssertionError({"nodes": len(nodes), "edges": len(edges)})
if nodes["dataset"].nunique() != 4 or edges["dataset"].nunique() != 4:
    raise AssertionError("dataset-count contract failed")

node_keys = set(zip(nodes["dataset"].astype(str), nodes["node_id"].astype(int)))
source_keys = set(zip(edges["dataset"].astype(str), edges["source_id"].astype(int)))
target_keys = set(zip(edges["dataset"].astype(str), edges["target_id"].astype(int)))
if not source_keys <= node_keys or not target_keys <= node_keys:
    raise AssertionError("EXP047 has dangling edges")

maximum_in = int(edges.groupby(["dataset", "target_id"]).size().max())
maximum_out = int(edges.groupby(["dataset", "source_id"]).size().max())
divisions = int((edges.groupby(["dataset", "source_id"]).size() == 2).sum())
if maximum_in > 1 or maximum_out > 2 or divisions != EXPECTED_DIVISIONS:
    raise AssertionError(
        {"maximum_in": maximum_in, "maximum_out": maximum_out, "divisions": divisions}
    )

output = WORK / "exp047_submission.csv"
shutil.copyfile(source, output)
if sha256(output) != observed_sha:
    raise AssertionError("EXP047 output serialization drift")

receipt = {
    "status": "PASS_IMMUTABLE_STRICT_COORDINATE_DIVISION_ARTIFACT",
    "submission_allowed_by_this_receipt": False,
    "source": str(source),
    "output": str(output),
    "sha256": observed_sha,
    "nodes": len(nodes),
    "edges": len(edges),
    "divisions": divisions,
    "maximum_in_degree": maximum_in,
    "maximum_out_degree": maximum_out,
    "build_receipt_sha256": sha256(build_path),
}
(WORK / "exp048_artifact_receipt.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, indent=2))
