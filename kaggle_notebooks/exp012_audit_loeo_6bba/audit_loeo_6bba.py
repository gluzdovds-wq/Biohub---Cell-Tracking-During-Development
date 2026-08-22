"""Calibrate on frozen 6bba movies, then evaluate once on untouched 6bba audit movies."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

HOLDOUT_EMBRYO = "6bba"
PARENT_SLUG = "biohub-exp010-loeo-holdout-6bba"
SEED = 314159
EXPECTED_EPOCHS = 10
EXPECTED_SPLIT_SIZES = {"train": 71, "checkpoint_validation": 4, "calibration": 8, "audit": 120}
CALIBRATION_THRESHOLDS = (0.95, 0.97, 0.985, 0.99, 0.995)
EDGE_CANDIDATE_THRESHOLD = 0.5
COMPETITION = "biohub-cell-tracking-during-development"

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TRAIN_DIR = INPUT / "competitions" / COMPETITION / "train"
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"
parent_candidates = [
    INPUT / "notebooks" / "dmitriigluzdov" / PARENT_SLUG,
    INPUT / PARENT_SLUG,
]
matches = [path for path in parent_candidates if path.exists()]
if len(matches) != 1:
    raise FileNotFoundError(
        {"parent_candidates": [str(path) for path in parent_candidates], "matches": [str(path) for path in matches]}
    )
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

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
    metadata = GeffMetadata.read(TRAIN_DIR / f"{Path(name).stem}.geff")
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
if (
    contract.get("status") != "training_complete"
    or contract.get("holdout_embryo") != HOLDOUT_EMBRYO
    or contract.get("seed") != SEED
    or contract.get("epochs") != EXPECTED_EPOCHS
):
    raise RuntimeError(contract)
train_names = set(contract["train"])
checkpoint_names = list(contract["checkpoint_validation"])
calibration_names = list(contract["calibration"])
audit_names = list(contract["audit"])
confirmation_names = sorted(set(calibration_names) - set(checkpoint_names))
assert calibration_names and audit_names
assert not train_names.intersection(calibration_names + audit_names)
assert not set(calibration_names).intersection(audit_names)
assert set(checkpoint_names) <= set(calibration_names)
assert len(confirmation_names) == len(checkpoint_names) == 4
assert set(checkpoint_names).isdisjoint(confirmation_names)
assert set(checkpoint_names) | set(confirmation_names) == set(calibration_names)
assert all(name.startswith(f"{HOLDOUT_EMBRYO}_") for name in calibration_names + audit_names)
assert all(not name.startswith(f"{HOLDOUT_EMBRYO}_") for name in train_names)
actual_split_sizes = {
    "train": len(train_names),
    "checkpoint_validation": len(checkpoint_names),
    "calibration": len(calibration_names),
    "audit": len(audit_names),
}
if actual_split_sizes != EXPECTED_SPLIT_SIZES:
    raise RuntimeError({"expected": EXPECTED_SPLIT_SIZES, "actual": actual_split_sizes})
weight_sha256 = hashlib.sha256(weight_path.read_bytes()).hexdigest()
weight_receipt = contract.get("artifacts", {}).get(weight_path.name, {})
if (
    weight_receipt.get("bytes") != weight_path.stat().st_size
    or weight_receipt.get("sha256") != weight_sha256
):
    raise RuntimeError({"contract_weight": weight_receipt, "actual_sha256": weight_sha256})

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, window_size, downsample = load_model(weight_path, device)
INFERENCE_GPU_COUNT = torch.cuda.device_count() if device.type == "cuda" else 0
INFERENCE_DATA_PARALLEL = INFERENCE_GPU_COUNT > 1
INFERENCE_UNET_BATCH_SIZE = 8 if INFERENCE_DATA_PARALLEL else 4
if INFERENCE_DATA_PARALLEL:
    model.unet = torch.nn.DataParallel(model.unet)
model.eval()
print(
    json.dumps(
        {
            "inference_gpu_count": INFERENCE_GPU_COUNT,
            "inference_data_parallel": INFERENCE_DATA_PARALLEL,
            "inference_unet_batch_size": INFERENCE_UNET_BATCH_SIZE,
        }
    ),
    flush=True,
)


def infer_candidates(name: str, threshold: float):
    cfg = PredictConfig(
        det_threshold=threshold,
        det_tta=True,
        pool_kernel_um=3.0,
        edge_activation="softmax",
        threshold=EDGE_CANDIDATE_THRESHOLD,
        use_ilp=True,
    )
    return predict_video(
        model,
        TRAIN_DIR / name,
        device,
        cfg=cfg,
        window_size=window_size,
        unet_batch_size=INFERENCE_UNET_BATCH_SIZE,
        downsample=downsample,
    )


def graph_for_policy(coords, edges, policy: str, scale):
    if policy in ("registered_hungarian", "registered_prob_hungarian"):
        coords_array = np.asarray(coords, dtype=float)
        scale = np.asarray(scale, dtype=float)
        probability_weight = 0.7 if policy == "registered_prob_hungarian" else 0.0
        linked = []
        times = sorted(set(coords_array[:, 0].astype(int))) if len(coords_array) else []
        for timepoint in times[:-1]:
            source_ids = np.flatnonzero(coords_array[:, 0].astype(int) == timepoint)
            target_ids = np.flatnonzero(coords_array[:, 0].astype(int) == timepoint + 1)
            if not len(source_ids) or not len(target_ids):
                continue
            source = coords_array[source_ids, 1:4] * scale
            target = coords_array[target_ids, 1:4] * scale
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
                if (
                    column >= len(target)
                    or not valid[row, column]
                    or score[row, column] < minimum_score
                ):
                    continue
                raw_distance = float(np.linalg.norm(source[row] - target[column]))
                probability = float(score[row, column])
                linked.append((int(source_ids[row]), int(target_ids[column]), probability, raw_distance))
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


PHYSICAL_PRUNE_ARMS = {
    "selected_base": None,
    "physical_prune_4_2": (4.0, 2.0),
    "physical_prune_7_4": (7.0, 4.0),
}


def physical_division_prune(graph, minimum_bad_residual_um: float, minimum_margin_um: float, scale):
    """Remove only the worse child of a high-residual two-child fork.

    This isolates the fixed physical gate used inside H040/H041. It does not
    claim to reproduce their independent EXP005/008 donor-consensus gate.
    """
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


def frozen_physical_arms(graph, scale):
    arms = {"selected_base": graph}
    telemetry = {"selected_base": {"accepted_prunes": 0}}
    for arm, thresholds in PHYSICAL_PRUNE_ARMS.items():
        if thresholds is None:
            continue
        arms[arm], telemetry[arm] = physical_division_prune(graph, *thresholds, scale)
    return arms, telemetry


tuning = []
for threshold in CALIBRATION_THRESHOLDS:
    policy_rows = {
        policy: []
        for policy in (
            "greedy",
            "ilp_public",
            "ilp_support",
            "registered_hungarian",
            "registered_prob_hungarian",
        )
    }
    for name in checkpoint_names:
        coords, edges = infer_candidates(name, threshold)
        scale = open_dataset(TRAIN_DIR / name, require_tracks=False).scale
        for policy, rows in policy_rows.items():
            rows.append(score_graph(name, graph_for_policy(coords, edges, policy, scale)))
    for policy, rows in policy_rows.items():
        item = {"threshold": threshold, "policy": policy, "summary": summarise(rows), "per_movie": rows}
        tuning.append(item)
        print(json.dumps(jsonable(item)))

selected = max(
    tuning,
    key=lambda item: (
        item["summary"]["score"],
        -abs(item["threshold"] - 0.99),
        {
            "greedy": 4,
            "ilp_public": 3,
            "ilp_support": 2,
            "registered_prob_hungarian": 1,
            "registered_hungarian": 0,
        }[item["policy"]],
    ),
)
selection = {
    "status": "selection_frozen_before_confirmation_and_audit",
    "holdout_embryo": HOLDOUT_EMBRYO,
    "weights_sha256": weight_sha256,
    "parent_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "checkpoint_validation_movies": checkpoint_names,
    "tuning_movies": checkpoint_names,
    "confirmation_movies": confirmation_names,
    "audit_movies": audit_names,
    "threshold_grid": CALIBRATION_THRESHOLDS,
    "edge_candidate_threshold": EDGE_CANDIDATE_THRESHOLD,
    "candidate_pool_degree_limits": None,
    "inference_gpu_count": INFERENCE_GPU_COUNT,
    "inference_data_parallel": INFERENCE_DATA_PARALLEL,
    "inference_unet_batch_size": INFERENCE_UNET_BATCH_SIZE,
    "ilp_public_appearance_disappearance": [0.0, 1.5],
    "ilp_support_appearance_disappearance": [0.1, 0.1],
    "selected_threshold": selected["threshold"],
    "selected_policy": selected["policy"],
    "postselection_physical_arms": PHYSICAL_PRUNE_ARMS,
    "postselection_arm_status": "frozen_before_confirmation_and_untouched_audit",
    "postselection_scope": "mechanism-only; independent EXP005/008 donor consensus is not reproduced",
    "tuning_results": tuning,
}
selection_path = WORK / f"loeo_{HOLDOUT_EMBRYO}_selection.json"
selection_path.write_text(json.dumps(jsonable(selection), indent=2), encoding="utf-8")
print(
    f"Frozen threshold {selected['threshold']} and policy {selected['policy']} "
    "before confirmation and audit",
    flush=True,
)

confirmation_rows_by_arm = {arm: [] for arm in PHYSICAL_PRUNE_ARMS}
confirmation_prune_telemetry = []
for name in confirmation_names:
    coords, edges = infer_candidates(name, selected["threshold"])
    scale = open_dataset(TRAIN_DIR / name, require_tracks=False).scale
    graph = graph_for_policy(coords, edges, selected["policy"], scale)
    arms, telemetry = frozen_physical_arms(graph, scale)
    confirmation_prune_telemetry.append({"dataset": name, "arms": telemetry})
    for arm, arm_graph in arms.items():
        confirmation_rows_by_arm[arm].append(score_graph(name, arm_graph))
confirmation = {
    "status": "confirmation_complete_without_reselection",
    "selected_threshold": selected["threshold"],
    "selected_policy": selected["policy"],
    "summary": summarise(confirmation_rows_by_arm["selected_base"]),
    "per_movie": confirmation_rows_by_arm["selected_base"],
    "summary_by_physical_arm": {
        arm: summarise(rows) for arm, rows in confirmation_rows_by_arm.items()
    },
    "per_movie_by_physical_arm": confirmation_rows_by_arm,
    "physical_prune_telemetry": confirmation_prune_telemetry,
}
(WORK / f"loeo_{HOLDOUT_EMBRYO}_confirmation.json").write_text(
    json.dumps(jsonable(confirmation), indent=2), encoding="utf-8"
)
print(json.dumps(jsonable(confirmation["summary"])), flush=True)

audit_dir = WORK / f"loeo_{HOLDOUT_EMBRYO}_audit_predictions"
audit_dir.mkdir(exist_ok=True)
audit_rows_by_arm = {arm: [] for arm in PHYSICAL_PRUNE_ARMS}
audit_prune_telemetry = []
for name in audit_names:
    coords, edges = infer_candidates(name, selected["threshold"])
    scale = open_dataset(TRAIN_DIR / name, require_tracks=False).scale
    graph = graph_for_policy(coords, edges, selected["policy"], scale)
    save_graph(graph, audit_dir / f"{name}.geff")
    arms, telemetry = frozen_physical_arms(graph, scale)
    audit_prune_telemetry.append({"dataset": name, "arms": telemetry})
    movie_rows = {}
    for arm, arm_graph in arms.items():
        row = score_graph(name, arm_graph)
        audit_rows_by_arm[arm].append(row)
        movie_rows[arm] = row
    print(json.dumps(jsonable({"dataset": name, "physical_arms": movie_rows})))

result = {
    **selection,
    "status": "audit_complete",
    "confirmation_summary": confirmation["summary"],
    "confirmation_per_movie": confirmation_rows_by_arm["selected_base"],
    "confirmation_summary_by_physical_arm": confirmation["summary_by_physical_arm"],
    "confirmation_per_movie_by_physical_arm": confirmation_rows_by_arm,
    "confirmation_physical_prune_telemetry": confirmation_prune_telemetry,
    "audit_summary": summarise(audit_rows_by_arm["selected_base"]),
    "audit_per_movie": audit_rows_by_arm["selected_base"],
    "audit_summary_by_physical_arm": {
        arm: summarise(rows) for arm, rows in audit_rows_by_arm.items()
    },
    "audit_per_movie_by_physical_arm": audit_rows_by_arm,
    "audit_physical_prune_telemetry": audit_prune_telemetry,
}
(WORK / f"loeo_{HOLDOUT_EMBRYO}_audit_result.json").write_text(
    json.dumps(jsonable(result), indent=2), encoding="utf-8"
)
print(json.dumps(jsonable(result["audit_summary_by_physical_arm"]), indent=2))
