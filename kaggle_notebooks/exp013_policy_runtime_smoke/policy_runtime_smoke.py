"""Kaggle-runtime smoke test for graph construction, ILP policies, and GEFF I/O."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    raise FileNotFoundError(SUPPORT)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--find-links",
        str(SUPPORT / "wheels"),
        "-r",
        str(SUPPORT / "requirements-unet-ilp-kaggle-predownload.txt"),
    ]
)

REPO = WORK / "tracking_repo"
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(SUPPORT / "repo", REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import tracksdata as td

from biohub_tracking.io import save_graph
from predict_unet_transformer import build_graph

coords = [
    (0, 0.0, 0.0, 0.0),
    (0, 0.0, 10.0, 0.0),
    (0, 0.0, 20.0, 0.0),
    (1, 0.0, 0.0, 5.0),
    (1, 0.0, 10.0, 5.0),
    (1, 0.0, 20.0, 5.0),
]
candidate_edges = [
    (0, 3, 0.95, 5.0),
    (0, 4, 0.10, 11.2),
    (1, 4, 0.94, 5.0),
    (1, 3, 0.11, 11.2),
    (2, 5, 0.93, 5.0),
    (2, 4, 0.12, 11.2),
]

results = {}
for policy, appearance_weight, disappearance_weight in (
    ("ilp_public", 0.0, 1.4),
    ("ilp_support", 0.1, 0.1),
):
    graph = build_graph(coords, candidate_edges)
    solver = td.solvers.ILPSolver(
        edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
        division_weight=1.0,
    )
    solved = solver.solve(graph)
    node_rows = list(solved.node_attrs().iter_rows(named=True))
    edge_rows = list(solved.edge_attrs().iter_rows(named=True))
    incoming = {}
    outgoing = {}
    for edge in edge_rows:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        outgoing[source] = outgoing.get(source, 0) + 1
        incoming[target] = incoming.get(target, 0) + 1
    assert max(incoming.values(), default=0) <= 1
    assert max(outgoing.values(), default=0) <= 2

    path = WORK / f"{policy}.geff"
    save_graph(solved, path)
    restored = td.graph.IndexedRXGraph.from_geff(path)
    restored_graph = restored[0] if isinstance(restored, tuple) else restored
    assert restored_graph.num_nodes() == solved.num_nodes()
    assert restored_graph.num_edges() == solved.num_edges()
    results[policy] = {
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "max_in_degree": max(incoming.values(), default=0),
        "max_out_degree": max(outgoing.values(), default=0),
        "geff_roundtrip": True,
    }

result_path = WORK / "policy_runtime_smoke.json"
result_path.write_text(json.dumps({"status": "PASS", "policies": results}, indent=2), encoding="utf-8")
print(result_path.read_text())
