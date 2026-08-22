"""Verify and expose the immutable EXP045 composition artifact."""

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

SLUG = "biohub-exp045-coordinate-division-compose"
EXPECTED_SHA256 = "4d93515ed72e76ea5be0d84c7a20d1e268e20ba37a8e4ce1ff50459d21399f88"
EXPECTED_NODE_INPUT_SHA256 = "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0"
EXPECTED_EDGE_INPUT_SHA256 = "9f0b0711b5ac0b078c5fb24332c2604c09118013116bc6fbe4d6f4e2eaa4a5e3"
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
EXPECTED_EDGES = 117996
EXPECTED_DIVISIONS = 295


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
    raise AssertionError(f"EXP045 source SHA mismatch: {observed_sha}")
if (
    build.get("status") != "PASS_DISJOINT_NODE_EDGE_COMPOSITION"
    or build.get("experiment") != "EXP045"
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
    raise AssertionError("EXP045 has dangling edges")

maximum_in = int(edges.groupby(["dataset", "target_id"]).size().max())
maximum_out = int(edges.groupby(["dataset", "source_id"]).size().max())
divisions = int((edges.groupby(["dataset", "source_id"]).size() == 2).sum())
if maximum_in > 1 or maximum_out > 2 or divisions != EXPECTED_DIVISIONS:
    raise AssertionError(
        {"maximum_in": maximum_in, "maximum_out": maximum_out, "divisions": divisions}
    )

output = WORK / "exp045_submission.csv"
shutil.copyfile(source, output)
if sha256(output) != observed_sha:
    raise AssertionError("EXP045 output serialization drift")

receipt = {
    "status": "PASS_IMMUTABLE_COORDINATE_DIVISION_ARTIFACT",
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
(WORK / "exp046_artifact_receipt.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, indent=2))
