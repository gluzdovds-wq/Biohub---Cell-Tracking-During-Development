"""Verify and expose immutable EXP040/041 artifacts for possible submission."""

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
CANDIDATES = {
    "EXP040": {
        "slug": "biohub-exp040-division-prune-consensus",
        "sha256": "9f0b0711b5ac0b078c5fb24332c2604c09118013116bc6fbe4d6f4e2eaa4a5e3",
        "nodes": 122266,
        "edges": 117996,
        "divisions": 295,
        "output": "exp040_submission.csv",
    },
    "EXP041": {
        "slug": "biohub-exp041-strict-division-prune",
        "sha256": "21a42ffa33c8af7ef44b28f7edaea6a3d9666745139c9c51e132fed41a8fe114",
        "nodes": 122266,
        "edges": 118101,
        "divisions": 400,
        "output": "exp041_submission.csv",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_root(slug: str) -> Path:
    matches = [path for path in INPUT.rglob("submission.csv") if path.parent.name == slug]
    if len(matches) != 1:
        raise FileNotFoundError({"slug": slug, "matches": list(map(str, matches))})
    return matches[0].parent


receipts = {}
for experiment, contract in CANDIDATES.items():
    root = locate_root(contract["slug"])
    source = root / "submission.csv"
    build_path = root / "submission.receipt.json"
    topology_path = root / "topology_audit.json"
    if not build_path.is_file() or not topology_path.is_file():
        raise FileNotFoundError({"build": str(build_path), "topology": str(topology_path)})

    observed_sha = sha256(source)
    build = json.loads(build_path.read_text())
    topology = json.loads(topology_path.read_text())
    if observed_sha != contract["sha256"]:
        raise AssertionError(f"{experiment} source SHA mismatch: {observed_sha}")
    if (
        build.get("status") != "PASS"
        or build.get("experiment") != experiment
        or build.get("output_sha256") != observed_sha
    ):
        raise AssertionError({"experiment": experiment, "build": build})
    if (
        topology.get("status") != "PASS"
        or topology.get("candidate_sha256") != observed_sha
        or topology.get("candidate_only_edges") != 0
        or topology.get("nodes") != contract["nodes"]
        or topology.get("candidate_edges") != contract["edges"]
        or topology.get("candidate_divisions") != contract["divisions"]
    ):
        raise AssertionError({"experiment": experiment, "topology": topology})

    frame = pd.read_csv(source, index_col=0)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise AssertionError({"experiment": experiment, "columns": list(frame.columns)})
    nodes = frame[frame["row_type"] == "node"]
    edges = frame[frame["row_type"] == "edge"]
    if len(nodes) != contract["nodes"] or len(edges) != contract["edges"]:
        raise AssertionError({"experiment": experiment, "nodes": len(nodes), "edges": len(edges)})
    if nodes["dataset"].nunique() != 4 or edges["dataset"].nunique() != 4:
        raise AssertionError(f"{experiment} dataset-count contract failed")
    node_keys = set(zip(nodes["dataset"].astype(str), nodes["node_id"].astype(int)))
    source_keys = set(zip(edges["dataset"].astype(str), edges["source_id"].astype(int)))
    target_keys = set(zip(edges["dataset"].astype(str), edges["target_id"].astype(int)))
    if not source_keys <= node_keys or not target_keys <= node_keys:
        raise AssertionError(f"{experiment} has dangling edges")
    maximum_in = int(edges.groupby(["dataset", "target_id"]).size().max())
    maximum_out = int(edges.groupby(["dataset", "source_id"]).size().max())
    divisions = int((edges.groupby(["dataset", "source_id"]).size() == 2).sum())
    if maximum_in > 1 or maximum_out > 2 or divisions != contract["divisions"]:
        raise AssertionError(
            {
                "experiment": experiment,
                "maximum_in": maximum_in,
                "maximum_out": maximum_out,
                "divisions": divisions,
            }
        )

    output = WORK / contract["output"]
    shutil.copyfile(source, output)
    if sha256(output) != observed_sha:
        raise AssertionError(f"{experiment} output serialization drift")
    receipts[experiment] = {
        "source": str(source),
        "output": str(output),
        "sha256": observed_sha,
        "nodes": len(nodes),
        "edges": len(edges),
        "divisions": divisions,
        "maximum_in_degree": maximum_in,
        "maximum_out_degree": maximum_out,
        "build_receipt_sha256": sha256(build_path),
        "topology_receipt_sha256": sha256(topology_path),
    }

receipt = {
    "status": "PASS_IMMUTABLE_DIVISION_PRUNE_ARTIFACTS",
    "submission_allowed_by_this_receipt": False,
    "candidates": receipts,
}
(WORK / "exp044_artifact_receipt.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, indent=2))
