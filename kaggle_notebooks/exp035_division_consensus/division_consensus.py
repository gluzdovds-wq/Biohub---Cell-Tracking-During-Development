"""Reproduce the single pre-registered EXP035 division edge fail-closed."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
OUTPUT = WORK / "submission.csv"
RECEIPT = WORK / "exp035_receipt.json"
WORK.mkdir(parents=True, exist_ok=True)

BASE_SHA = "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d"
EXPECTED_OUTPUT_SHAS = {
    # Local pandas runtime.
    "db19d213c89995fce16add28c5d699d9f853b947ff9776e99d47805ddc43f953",
    # Kaggle pandas runtime; independently audited and semantically identical.
    "7376bd3c4056ee7c7f82fadd2db3bb37230ad09e399ae1f815c3c53a51374bd4",
}
EXPECTED_COLUMNS = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]
DATASET = "6bba_05db0fb1"
SOURCE = 65_628
EXISTING_TARGET = 66_324
PROPOSED_TARGET = 66_302
SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_base() -> Path:
    observed = []
    matches = []
    for path in INPUT.rglob("submission.csv"):
        digest = sha256(path)
        observed.append({"path": str(path), "sha256": digest})
        if digest == BASE_SHA:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError({"expected_sha": BASE_SHA, "matches": list(map(str, matches)), "observed": observed})
    return matches[0]


base_path = locate_base()
base = pd.read_csv(base_path, index_col=0)
if list(base.columns) != EXPECTED_COLUMNS:
    raise ValueError({"columns": list(base.columns)})

nodes = base[base["row_type"] == "node"].copy()
node_lookup = {
    (str(row.dataset), int(row.node_id)): row
    for row in nodes.itertuples()
}
edges = {
    (str(row.dataset), int(row.source_id), int(row.target_id))
    for row in base[base["row_type"] == "edge"].itertuples()
}
existing_edge = (DATASET, SOURCE, EXISTING_TARGET)
proposed_edge = (DATASET, SOURCE, PROPOSED_TARGET)
if existing_edge not in edges or proposed_edge in edges:
    raise AssertionError("Frozen base-edge contract failed")

incoming = defaultdict(int)
outgoing = defaultdict(set)
for dataset, source, target in edges:
    incoming[(dataset, target)] += 1
    outgoing[(dataset, source)].add(target)
if incoming[(DATASET, PROPOSED_TARGET)] != 0:
    raise AssertionError("Proposed daughter is not orphaned")
if outgoing[(DATASET, SOURCE)] != {EXISTING_TARGET}:
    raise AssertionError("Frozen parent outgoing set changed")
if len(outgoing[(DATASET, EXISTING_TARGET)]) != 1 or len(outgoing[(DATASET, PROPOSED_TARGET)]) != 1:
    raise AssertionError("Frozen daughter-continuation guard failed")

source_row = node_lookup[(DATASET, SOURCE)]
existing_row = node_lookup[(DATASET, EXISTING_TARGET)]
proposed_row = node_lookup[(DATASET, PROPOSED_TARGET)]
if int(existing_row.t) != int(source_row.t) + 1 or int(proposed_row.t) != int(source_row.t) + 1:
    raise AssertionError("Frozen consecutive-time guard failed")
source_xyz = np.asarray([source_row.z, source_row.y, source_row.x], dtype=float)
existing_xyz = np.asarray([existing_row.z, existing_row.y, existing_row.x], dtype=float)
proposed_xyz = np.asarray([proposed_row.z, proposed_row.y, proposed_row.x], dtype=float)
existing_displacement = (existing_xyz - source_xyz) * SCALE
proposed_displacement = (proposed_xyz - source_xyz) * SCALE
existing_distance = float(np.linalg.norm(existing_displacement))
proposed_distance = float(np.linalg.norm(proposed_displacement))
cosine = float(
    np.dot(existing_displacement, proposed_displacement)
    / (existing_distance * proposed_distance)
)
if max(existing_distance, proposed_distance) > 7.0 or cosine > 0.0:
    raise AssertionError("Frozen physical division guard failed")

edges.add(proposed_edge)
edge_rows = pd.DataFrame(
    [
        (dataset, "edge", -1, -1, -1, -1, -1, source, target)
        for dataset, source, target in sorted(edges)
    ],
    columns=EXPECTED_COLUMNS,
)
result = pd.concat([nodes, edge_rows], ignore_index=True)
result.index.name = "id"
result.to_csv(OUTPUT)
output_sha = sha256(OUTPUT)
if output_sha not in EXPECTED_OUTPUT_SHAS:
    raise AssertionError(f"Unexpected output SHA: {output_sha} not in {sorted(EXPECTED_OUTPUT_SHAS)}")

receipt = {
    "status": "PASS_FROZEN_DIVISION_CONSENSUS",
    "base_path": str(base_path),
    "base_sha256": BASE_SHA,
    "output_sha256": output_sha,
    "nodes": len(nodes),
    "base_edges": len(edges) - 1,
    "output_edges": len(edges),
    "added_edge": list(proposed_edge),
    "existing_edge_um": existing_distance,
    "proposed_edge_um": proposed_distance,
    "displacement_cosine": cosine,
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
