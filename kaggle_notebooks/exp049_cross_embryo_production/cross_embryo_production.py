"""H049: route each test embryo through a model trained only on the other embryo."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SEED = 314159
COMPETITION = "biohub-cell-tracking-during-development"
EDGE_CANDIDATE_THRESHOLD = 0.5
ALLOWED_THRESHOLDS = (0.95, 0.97, 0.985, 0.99, 0.995)
ALLOWED_POLICIES = (
    "greedy",
    "ilp_public",
    "ilp_support",
    "registered_hungarian",
    "registered_prob_hungarian",
)
# Frozen while EXP011 had only seven of 63 audit movie rows visible and before
# either untouched aggregate or any EXP012 selection/confirmation/audit value.
MIN_UNTOUCHED_SCORE = 0.55
MIN_UNTOUCHED_NODE_RECALL = 0.85
MAX_CONFIRMATION_TO_AUDIT_DROP = 0.10

FOLDS = {
    "44b6": {
        "parent_slug": "biohub-exp009-loeo-holdout-44b6",
        "audit_slug": "biohub-exp011-audit-loeo-44b6",
        "epochs": 5,
        "split_sizes": {"train": 128, "checkpoint_validation": 4, "calibration": 8, "audit": 63},
    },
    "6bba": {
        "parent_slug": "biohub-exp010-loeo-holdout-6bba",
        "audit_slug": "biohub-exp012-audit-loeo-6bba",
        "epochs": 10,
        "split_sizes": {"train": 71, "checkpoint_validation": 4, "calibration": 8, "audit": 120},
    },
}

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TEST_DIR = INPUT / "competitions" / COMPETITION / "test"
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"


def locate_notebook(slug: str) -> Path:
    candidates = [INPUT / "notebooks" / "dmitriigluzdov" / slug, INPUT / slug]
    matches = [path for path in candidates if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError({"slug": slug, "candidates": list(map(str, candidates)), "matches": list(map(str, matches))})
    return matches[0]


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

import torch
import tracksdata as td
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from biohub_tracking.io import open_dataset
from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_for_policy(coords, edges, policy: str, scale):
    if policy in ("registered_hungarian", "registered_prob_hungarian"):
        coords_array = np.asarray(coords, dtype=float)
        scale_array = np.asarray(scale, dtype=float)
        probability_weight = 0.7 if policy == "registered_prob_hungarian" else 0.0
        linked = []
        times = sorted(set(coords_array[:, 0].astype(int))) if len(coords_array) else []
        for timepoint in times[:-1]:
            source_ids = np.flatnonzero(coords_array[:, 0].astype(int) == timepoint)
            target_ids = np.flatnonzero(coords_array[:, 0].astype(int) == timepoint + 1)
            if not len(source_ids) or not len(target_ids):
                continue
            source = coords_array[source_ids, 1:4] * scale_array
            target = coords_array[target_ids, 1:4] * scale_array
            tree = cKDTree(target)
            _, nearest = tree.query(source, k=1)
            displacement = target[np.asarray(nearest, dtype=int)] - source
            shift = np.median(displacement, axis=0)
            residual_to_shift = np.linalg.norm(displacement - shift, axis=1)
            inliers = residual_to_shift <= 4.0
            if int(inliers.sum()) >= 3:
                shift = np.median(displacement[inliers], axis=0)

            residual = np.linalg.norm(
                source[:, None, :] + shift[None, None, :] - target[None, :, :], axis=2
            )
            gate_um = 7.0
            motion_score = np.exp(-residual / 3.0)
            if probability_weight:
                source_rows = {int(node_id): row for row, node_id in enumerate(source_ids)}
                target_columns = {int(node_id): column for column, node_id in enumerate(target_ids)}
                learned_probability = np.zeros_like(residual)
                for edge_source, edge_target, edge_probability, _ in edges:
                    row = source_rows.get(int(edge_source))
                    column = target_columns.get(int(edge_target))
                    if row is not None and column is not None:
                        learned_probability[row, column] = max(
                            learned_probability[row, column], float(edge_probability)
                        )
                score = probability_weight * learned_probability + (1.0 - probability_weight) * motion_score
                valid = (residual < gate_um) & (learned_probability > 0.0)
                minimum_score = 0.55
            else:
                score = motion_score
                valid = residual < gate_um
                minimum_score = float(np.exp(-gate_um / 3.0))
            cost = np.where(valid, 1.0 - score, 1e6)
            augmented = np.concatenate(
                [cost, np.full((len(source), len(source)), 1.0 - minimum_score, dtype=float)], axis=1
            )
            rows, columns = linear_sum_assignment(augmented)
            for row, column in zip(rows, columns):
                if column >= len(target) or not valid[row, column] or score[row, column] < minimum_score:
                    continue
                raw_distance = float(np.linalg.norm(source[row] - target[column]))
                linked.append((int(source_ids[row]), int(target_ids[column]), float(score[row, column]), raw_distance))
        return build_graph(coords, linked)

    if policy == "greedy":
        kept = []
        parent_count = {}
        child_count = {}
        for edge in sorted(edges, key=lambda item: (-float(item[2]), int(item[0]), int(item[1]))):
            source, target = int(edge[0]), int(edge[1])
            if child_count.get(source, 0) >= 2 or parent_count.get(target, 0) >= 1:
                continue
            kept.append(edge)
            child_count[source] = child_count.get(source, 0) + 1
            parent_count[target] = parent_count.get(target, 0) + 1
        return build_graph(coords, kept)

    graph = build_graph(coords, edges)
    if not graph.num_edges():
        return graph
    if policy == "ilp_public":
        appearance_weight, disappearance_weight = 0.0, 1.5
    elif policy == "ilp_support":
        appearance_weight, disappearance_weight = 0.1, 0.1
    else:
        raise ValueError(policy)
    solver = td.solvers.ILPSolver(
        edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
        division_weight=1.0,
    )
    return solver.solve(graph).detach()


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
use_data_parallel = gpu_count > 1
unet_batch_size = 8 if use_data_parallel else 4

contracts = {}
for embryo, settings in FOLDS.items():
    parent = locate_notebook(settings["parent_slug"])
    audit = locate_notebook(settings["audit_slug"])
    contract_path = parent / f"loeo_{embryo}_contract.json"
    weight_path = parent / f"loeo_holdout_{embryo}" / "edge_predictor_best.pth"
    selection_path = audit / f"loeo_{embryo}_selection.json"
    result_path = audit / f"loeo_{embryo}_audit_result.json"
    if not all(path.is_file() for path in (contract_path, weight_path, selection_path, result_path)):
        raise FileNotFoundError({"embryo": embryo, "required": list(map(str, (contract_path, weight_path, selection_path, result_path)))})

    contract = json.loads(contract_path.read_text())
    selection = json.loads(selection_path.read_text())
    result = json.loads(result_path.read_text())
    split_sizes = {
        "train": len(contract.get("train", [])),
        "checkpoint_validation": len(contract.get("checkpoint_validation", [])),
        "calibration": len(contract.get("calibration", [])),
        "audit": len(contract.get("audit", [])),
    }
    weight_sha = sha256(weight_path)
    contract_sha = sha256(contract_path)
    weight_receipt = contract.get("artifacts", {}).get(weight_path.name, {})
    if (
        contract.get("status") != "training_complete"
        or contract.get("holdout_embryo") != embryo
        or contract.get("seed") != SEED
        or contract.get("epochs") != settings["epochs"]
        or split_sizes != settings["split_sizes"]
        or weight_receipt.get("bytes") != weight_path.stat().st_size
        or weight_receipt.get("sha256") != weight_sha
    ):
        raise RuntimeError({"reason": "parent contract drift", "embryo": embryo, "contract": contract, "split_sizes": split_sizes})
    if (
        selection.get("status") != "selection_frozen_before_confirmation_and_audit"
        or selection.get("holdout_embryo") != embryo
        or selection.get("weights_sha256") != weight_sha
        or selection.get("parent_contract_sha256") != contract_sha
        or selection.get("selected_threshold") not in ALLOWED_THRESHOLDS
        or selection.get("selected_policy") not in ALLOWED_POLICIES
        or selection.get("audit_movies") != contract.get("audit")
    ):
        raise RuntimeError({"reason": "selection drift", "embryo": embryo, "selection": selection})
    audit_summary = result.get("audit_summary", {})
    confirmation_summary = result.get("confirmation_summary", {})
    audit_score = float(audit_summary.get("score", float("nan")))
    confirmation_score = float(confirmation_summary.get("score", float("nan")))
    audit_recall = float(audit_summary.get("node_recall", float("nan")))
    gate_pass = (
        result.get("status") == "audit_complete"
        and math.isfinite(audit_score)
        and math.isfinite(confirmation_score)
        and math.isfinite(audit_recall)
        and audit_score >= MIN_UNTOUCHED_SCORE
        and audit_recall >= MIN_UNTOUCHED_NODE_RECALL
        and audit_score >= confirmation_score - MAX_CONFIRMATION_TO_AUDIT_DROP
    )
    if not gate_pass:
        raise RuntimeError(
            {
                "reason": "H049 frozen generalization gate failed",
                "embryo": embryo,
                "audit_score": audit_score,
                "confirmation_score": confirmation_score,
                "audit_node_recall": audit_recall,
            }
        )
    contracts[embryo] = {
        "parent": parent,
        "audit": audit,
        "contract_path": contract_path,
        "weight_path": weight_path,
        "selection_path": selection_path,
        "result_path": result_path,
        "weight_sha256": weight_sha,
        "parent_contract_sha256": contract_sha,
        "selection_sha256": sha256(selection_path),
        "audit_result_sha256": sha256(result_path),
        "selected_threshold": float(selection["selected_threshold"]),
        "selected_policy": selection["selected_policy"],
        "confirmation_score": confirmation_score,
        "audit_score": audit_score,
        "audit_node_recall": audit_recall,
    }

test_paths = sorted(TEST_DIR.glob("*.zarr"))
test_names = [path.name for path in test_paths]
if not test_paths or any(name.split("_", 1)[0] not in FOLDS for name in test_names):
    raise RuntimeError({"unexpected_test_names": test_names})

graphs = {}
run_stats = []
for embryo in FOLDS:
    fold = contracts[embryo]
    model, window_size, downsample = load_model(fold["weight_path"], device)
    if use_data_parallel:
        model.unet = torch.nn.DataParallel(model.unet)
    model.eval()
    cfg = PredictConfig(
        det_threshold=fold["selected_threshold"],
        det_tta=True,
        pool_kernel_um=3.0,
        edge_activation="softmax",
        threshold=EDGE_CANDIDATE_THRESHOLD,
        use_ilp=True,
    )
    for path in test_paths:
        if path.name.split("_", 1)[0] != embryo:
            continue
        coords, edges = predict_video(
            model,
            path,
            device,
            cfg=cfg,
            window_size=window_size,
            unet_batch_size=unet_batch_size,
            downsample=downsample,
        )
        scale = open_dataset(path, require_tracks=False).scale
        graph = graph_for_policy(coords, edges, fold["selected_policy"], scale)
        graphs[path.stem] = graph
        run_stats.append(
            {
                "dataset": path.stem,
                "embryo": embryo,
                "weights_sha256": fold["weight_sha256"],
                "selected_threshold": fold["selected_threshold"],
                "selected_policy": fold["selected_policy"],
                "candidate_nodes": int(len(coords)),
                "candidate_edges": int(len(edges)),
                "output_nodes": int(graph.num_nodes()),
                "output_edges": int(graph.num_edges()),
            }
        )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

if set(graphs) != {path.stem for path in test_paths}:
    raise RuntimeError({"expected": sorted(path.stem for path in test_paths), "actual": sorted(graphs)})

columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
submission_path = WORK / "submission.csv"
row_id = 0
total_nodes = 0
total_edges = 0
total_divisions = 0
with submission_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for dataset in sorted(graphs):
        graph = graphs[dataset]
        node_rows = sorted(graph.node_attrs().iter_rows(named=True), key=lambda row: int(row["node_id"]))
        edge_rows = sorted(
            graph.edge_attrs().iter_rows(named=True),
            key=lambda row: (int(row["source_id"]), int(row["target_id"])),
        )
        node_by_id = {int(row["node_id"]): row for row in node_rows}
        if len(node_by_id) != len(node_rows):
            raise AssertionError(f"{dataset}: duplicate node ids")
        shape = open_dataset(TEST_DIR / f"{dataset}.zarr", require_tracks=False).image_shape
        outdegree = {}
        indegree = {}
        for row in node_rows:
            values = [float(row[key]) for key in ("z", "y", "x")]
            if not all(math.isfinite(value) for value in values):
                raise AssertionError(f"{dataset}: non-finite coordinate")
            if not (0 <= int(row["t"]) < shape[0] and all(0 <= value < limit for value, limit in zip(values, shape[1:4]))):
                raise AssertionError(f"{dataset}: node outside test volume")
            writer.writerow(
                {
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(row["node_id"]),
                    "t": int(row["t"]),
                    "z": format(values[0], ".17g"),
                    "y": format(values[1], ".17g"),
                    "x": format(values[2], ".17g"),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
            row_id += 1
        for row in edge_rows:
            source = int(row["source_id"])
            target = int(row["target_id"])
            if source not in node_by_id or target not in node_by_id:
                raise AssertionError(f"{dataset}: dangling edge")
            if int(node_by_id[target]["t"]) != int(node_by_id[source]["t"]) + 1:
                raise AssertionError(f"{dataset}: nonconsecutive edge")
            outdegree[source] = outdegree.get(source, 0) + 1
            indegree[target] = indegree.get(target, 0) + 1
            writer.writerow(
                {
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": source,
                    "target_id": target,
                }
            )
            row_id += 1
        if max(indegree.values(), default=0) > 1 or max(outdegree.values(), default=0) > 2:
            raise AssertionError(f"{dataset}: degree contract failed")
        total_nodes += len(node_rows)
        total_edges += len(edge_rows)
        total_divisions += sum(count == 2 for count in outdegree.values())

receipt = {
    "status": "PASS_H049_CROSS_EMBRYO_PRODUCTION",
    "hypothesis": "H049",
    "training_scope": "each test embryo is routed only to a model trained on the other embryo",
    "frozen_gate": {
        "minimum_untouched_score": MIN_UNTOUCHED_SCORE,
        "minimum_untouched_node_recall": MIN_UNTOUCHED_NODE_RECALL,
        "maximum_confirmation_to_audit_drop": MAX_CONFIRMATION_TO_AUDIT_DROP,
    },
    "folds": {
        embryo: {key: value for key, value in fold.items() if key not in {"parent", "audit", "contract_path", "weight_path", "selection_path", "result_path"}}
        for embryo, fold in contracts.items()
    },
    "inference_gpu_count": gpu_count,
    "inference_data_parallel": use_data_parallel,
    "inference_unet_batch_size": unet_batch_size,
    "datasets": sorted(graphs),
    "run_stats": run_stats,
    "submission_path": str(submission_path),
    "submission_sha256": sha256(submission_path),
    "nodes": total_nodes,
    "edges": total_edges,
    "divisions": total_divisions,
    "rows": row_id,
    "submission_ready": True,
    "submission_slot_contract": "one supporting LB slot only; source receipt does not itself prove a medal improvement",
}
(WORK / "exp049_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
