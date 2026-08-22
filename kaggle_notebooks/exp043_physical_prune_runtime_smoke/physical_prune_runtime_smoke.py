"""Runtime-contract smoke test for the frozen LOEO physical division arms."""

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

import numpy as np
import tracksdata as td

from biohub_tracking.io import save_graph
from predict_unet_transformer import build_graph


def physical_division_prune(graph, minimum_bad_residual_um: float, minimum_margin_um: float, scale):
    node_rows = list(graph.node_attrs().iter_rows(named=True))
    edge_rows = list(graph.edge_attrs().iter_rows(named=True))
    id_to_row = {int(row["node_id"]): row for row in node_rows}
    id_to_index = {int(row["node_id"]): index for index, row in enumerate(node_rows)}
    incoming = {}
    outgoing = {}
    for row in edge_rows:
        source = int(row["source_id"])
        target = int(row["target_id"])
        incoming.setdefault(target, []).append(source)
        outgoing.setdefault(source, []).append(target)

    scale_array = np.asarray(scale, dtype=float)

    def point(node_id: int) -> np.ndarray:
        row = id_to_row[node_id]
        return np.asarray([row["z"], row["y"], row["x"]], dtype=float) * scale_array

    removals = set()
    changes = []
    for parent_id, children in sorted(outgoing.items()):
        if len(children) != 2 or len(incoming.get(parent_id, [])) != 1:
            continue
        predecessor_id = incoming[parent_id][0]
        predecessor = id_to_row[predecessor_id]
        parent = id_to_row[parent_id]
        child_rows = [id_to_row[child_id] for child_id in children]
        if not (
            int(parent["t"]) == int(predecessor["t"]) + 1
            and all(int(child["t"]) == int(parent["t"]) + 1 for child in child_rows)
        ):
            continue
        predicted = point(parent_id) + (point(parent_id) - point(predecessor_id))
        residuals = [(float(np.linalg.norm(point(child_id) - predicted)), child_id) for child_id in children]
        residuals.sort()
        retained_residual, retained_id = residuals[0]
        removed_residual, removed_id = residuals[1]
        residual_margin = removed_residual - retained_residual
        if removed_residual < minimum_bad_residual_um or residual_margin < minimum_margin_um:
            continue
        removals.add((parent_id, removed_id))
        changes.append(
            {
                "parent_id": parent_id,
                "predecessor_id": predecessor_id,
                "retained_target_id": retained_id,
                "removed_target_id": removed_id,
                "retained_cv_residual_um": retained_residual,
                "removed_cv_residual_um": removed_residual,
                "residual_margin_um": residual_margin,
            }
        )

    coords = np.asarray(
        [[row["t"], row["z"], row["y"], row["x"]] for row in node_rows], dtype=float
    )
    kept_edges = []
    for row in edge_rows:
        source = int(row["source_id"])
        target = int(row["target_id"])
        if (source, target) in removals:
            continue
        probability = float(row.get("edge_prob", 1.0))
        distance = float(row.get("edge_dist", np.linalg.norm(point(source) - point(target))))
        kept_edges.append((id_to_index[source], id_to_index[target], probability, distance))
    pruned = build_graph(coords, kept_edges)
    if pruned.num_nodes() != graph.num_nodes() or pruned.num_edges() != graph.num_edges() - len(removals):
        raise AssertionError("physical division prune graph contract failed")
    return pruned, {
        "minimum_bad_residual_um": minimum_bad_residual_um,
        "minimum_margin_um": minimum_margin_um,
        "base_edges": graph.num_edges(),
        "edges": pruned.num_edges(),
        "accepted_prunes": len(changes),
        "changes": changes,
    }


# Two independent synthetic forks. The first bad child has residual 6, the
# second residual 8, while both retained children follow constant velocity.
coords = np.asarray(
    [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [2, 2, 0, 0],
        [2, 8, 0, 0],
        [0, 0, 10, 0],
        [1, 0, 11, 0],
        [2, 0, 12, 0],
        [2, 0, 20, 0],
    ],
    dtype=float,
)
edges = [
    (0, 1, 0.9, 1.0),
    (1, 2, 0.9, 1.0),
    (1, 3, 0.8, 7.0),
    (4, 5, 0.9, 1.0),
    (5, 6, 0.9, 1.0),
    (5, 7, 0.8, 9.0),
]
base = build_graph(coords, edges)
broad, broad_receipt = physical_division_prune(base, 4.0, 2.0, (1.0, 1.0, 1.0))
strict, strict_receipt = physical_division_prune(base, 7.0, 4.0, (1.0, 1.0, 1.0))
assert base.num_nodes() == broad.num_nodes() == strict.num_nodes() == 8
assert (base.num_edges(), broad.num_edges(), strict.num_edges()) == (6, 4, 5)
assert broad_receipt["accepted_prunes"] == 2
assert strict_receipt["accepted_prunes"] == 1
assert {(row["parent_id"], row["removed_target_id"]) for row in broad_receipt["changes"]} == {(1, 3), (5, 7)}
assert {(row["parent_id"], row["removed_target_id"]) for row in strict_receipt["changes"]} == {(5, 7)}

roundtrip = {}
for name, graph in (("broad", broad), ("strict", strict)):
    path = WORK / f"{name}.geff"
    save_graph(graph, path)
    restored = td.graph.IndexedRXGraph.from_geff(path)
    restored_graph = restored[0] if isinstance(restored, tuple) else restored
    assert restored_graph.num_nodes() == graph.num_nodes()
    assert restored_graph.num_edges() == graph.num_edges()
    roundtrip[name] = {"nodes": graph.num_nodes(), "edges": graph.num_edges()}

receipt = {
    "status": "PASS_PHYSICAL_PRUNE_RUNTIME_CONTRACT",
    "base": {"nodes": base.num_nodes(), "edges": base.num_edges()},
    "physical_prune_4_2": broad_receipt,
    "physical_prune_7_4": strict_receipt,
    "strict_removals_are_subset": {
        (row["parent_id"], row["removed_target_id"]) for row in strict_receipt["changes"]
    }
    < {
        (row["parent_id"], row["removed_target_id"]) for row in broad_receipt["changes"]
    },
    "geff_roundtrip": roundtrip,
}
(WORK / "exp043_physical_prune_runtime_smoke.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, indent=2))
