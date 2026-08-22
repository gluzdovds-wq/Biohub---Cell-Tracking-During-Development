"""Calibrate on frozen 44b6 movies, then evaluate once on untouched 44b6 audit movies."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

HOLDOUT_EMBRYO = "44b6"
PARENT_SLUG = "biohub-exp009-loeo-holdout-44b6"
SEED = 314159
CALIBRATION_THRESHOLDS = (0.95, 0.97, 0.985, 0.99, 0.995)
COMPETITION = "biohub-cell-tracking-during-development"

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TRAIN_DIR = INPUT / "competitions" / COMPETITION / "train"
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"
PARENT = INPUT / PARENT_SLUG
if not PARENT.exists():
    matches = [path for path in INPUT.glob("*") if path.name.endswith(PARENT_SLUG)]
    if len(matches) != 1:
        raise FileNotFoundError({"parent": str(PARENT), "matches": [str(p) for p in matches]})
    PARENT = matches[0]

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
os.environ["BIOHUB_DATA_DIR"] = str(TRAIN_DIR)

import numpy as np
import torch
import tracksdata as td
from geff import GeffMetadata

from biohub_tracking.io import open_dataset, save_graph
from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise
from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def estimated_nodes(name: str) -> float:
    metadata = GeffMetadata.read(TRAIN_DIR / f"{name}.geff")
    value = (metadata.extra or {}).get("estimated_number_of_nodes")
    return float(value) if value is not None else float("nan")


def score_graph(name: str, graph) -> dict:
    dataset = open_dataset(TRAIN_DIR / name, require_tracks=True)
    result = evaluate(graph, dataset.tracks, scale=dataset.scale)
    recall = node_recall(graph, dataset.tracks) if graph.num_nodes() and graph.num_edges() else 0.0
    return {"dataset": name, **per_sample_metrics(result, estimated_nodes(name), recall)}


contract_path = PARENT / f"loeo_{HOLDOUT_EMBRYO}_contract.json"
weight_path = PARENT / f"loeo_holdout_{HOLDOUT_EMBRYO}" / "edge_predictor_best.pth"
if not contract_path.is_file() or not weight_path.is_file():
    raise RuntimeError({"contract": str(contract_path), "weight": str(weight_path)})
contract = json.loads(contract_path.read_text())
if contract.get("status") != "training_complete" or contract.get("holdout_embryo") != HOLDOUT_EMBRYO:
    raise RuntimeError(contract)
train_names = set(contract["train"])
calibration_names = list(contract["calibration"])
audit_names = list(contract["audit"])
assert calibration_names and audit_names
assert not train_names.intersection(calibration_names + audit_names)
assert not set(calibration_names).intersection(audit_names)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, window_size, downsample = load_model(weight_path, device)
model.eval()


def infer_candidates(name: str, threshold: float):
    cfg = PredictConfig(
        det_threshold=threshold,
        det_tta=True,
        pool_kernel_um=3.0,
        edge_activation="softmax",
        threshold=0.5,
        use_ilp=True,
    )
    return predict_video(
        model,
        TRAIN_DIR / name,
        device,
        cfg=cfg,
        window_size=window_size,
        unet_batch_size=4,
        downsample=downsample,
    )


def graph_for_policy(coords, edges, policy: str):
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
        appearance_weight, disappearance_weight = 0.0, 1.4
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
    return solver.solve(graph)


calibration = []
for threshold in CALIBRATION_THRESHOLDS:
    policy_rows = {policy: [] for policy in ("greedy", "ilp_public", "ilp_support")}
    for name in calibration_names:
        coords, edges = infer_candidates(name, threshold)
        for policy, rows in policy_rows.items():
            rows.append(score_graph(name, graph_for_policy(coords, edges, policy)))
    for policy, rows in policy_rows.items():
        item = {"threshold": threshold, "policy": policy, "summary": summarise(rows), "per_movie": rows}
        calibration.append(item)
        print(json.dumps(jsonable(item)))

selected = max(
    calibration,
    key=lambda item: (
        item["summary"]["score"],
        -abs(item["threshold"] - 0.99),
        {"greedy": 2, "ilp_public": 1, "ilp_support": 0}[item["policy"]],
    ),
)
selection = {
    "status": "threshold_frozen_before_audit",
    "holdout_embryo": HOLDOUT_EMBRYO,
    "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
    "calibration_movies": calibration_names,
    "audit_movies": audit_names,
    "threshold_grid": CALIBRATION_THRESHOLDS,
    "selected_threshold": selected["threshold"],
    "selected_policy": selected["policy"],
    "calibration_results": calibration,
}
selection_path = WORK / f"loeo_{HOLDOUT_EMBRYO}_selection.json"
selection_path.write_text(json.dumps(jsonable(selection), indent=2), encoding="utf-8")
print(f"Frozen threshold {selected['threshold']} and policy {selected['policy']} before audit", flush=True)

audit_dir = WORK / f"loeo_{HOLDOUT_EMBRYO}_audit_predictions"
audit_dir.mkdir(exist_ok=True)
audit_rows = []
for name in audit_names:
    coords, edges = infer_candidates(name, selected["threshold"])
    graph = graph_for_policy(coords, edges, selected["policy"])
    save_graph(graph, audit_dir / f"{name}.geff")
    row = score_graph(name, graph)
    audit_rows.append(row)
    print(json.dumps(jsonable(row)))

result = {
    **selection,
    "status": "audit_complete",
    "audit_summary": summarise(audit_rows),
    "audit_per_movie": audit_rows,
}
(WORK / f"loeo_{HOLDOUT_EMBRYO}_audit_result.json").write_text(
    json.dumps(jsonable(result), indent=2), encoding="utf-8"
)
print(json.dumps(jsonable(result["audit_summary"]), indent=2))
