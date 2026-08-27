"""EXP077: bounded CPU, movie-held-out paired audit; not exact LB-model OOF.

Own contribution: robust leave-self-out local deformation linking.
Detector/transformer architecture and metric: pilkwang's Biohub support pack.
Weights: our reciprocal EXP009/010 training, never public all-data weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

COMPETITION = "biohub-cell-tracking-during-development"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RESULT_DIR = WORK / "exp077"
REPO = WORK / "exp077_tracking_repo"
SEED = 314159
RUN_SECONDS = 6 * 3600
MOVIE_SECONDS = 90 * 60
CPU_THREADS = 4
ARMS = ("registered", "registered_weak", "local_flow_weak", "ilp_public_style")
FLOW = {
    "anchor_gate_um": 3.5, "anchor_margin_um": 1.0,
    "radius_um": 20.0, "neighbors": 12, "minimum_neighbors": 3,
    "maximum_mad_um": 1.0, "maximum_residual_um": 2.5,
    "maximum_blend": 0.75, "full_support_neighbors": 8,
    "association_gate_um": 7.0, "motion_scale_um": 3.0,
    "learned_weight": 0.1,
}
FOLDS = {
    "44b6": {
        "parent": "biohub-exp009-loeo-holdout-44b6",
        "selection": "biohub-exp011-audit-loeo-44b6",
        "contract_sha256": "7646ce1e877e6bd62d9fedd24cd6bb2f7e6786ee824ec02a9ac1831e422c6df7",
        "selection_sha256": "9b32581a538458642b36574d01e23f58ec2de0df347253f4c9c74040f16ad7e1",
        "epochs": 5, "train_count": 128, "audit_count": 63,
        "movies": ["44b6_415c0a3a.zarr", "44b6_abf82518.zarr"],
    },
    "6bba": {
        "parent": "biohub-exp010-loeo-holdout-6bba",
        "selection": "biohub-exp012-audit-loeo-6bba",
        "contract_sha256": "66de0e8591950ad05c72ca614169dca13dadc98c5839dea0a36ba1e20dfef6ed",
        "selection_sha256": "1bb9c4675513543b733d2917f8e884ef8870e0f5cae5cdd1e6a68e332eaa4ef9",
        "epochs": 10, "train_count": 71, "audit_count": 120,
        "movies": ["6bba_96833384.zarr", "6bba_55c70843.zarr"],
    },
}
LIMITATIONS = [
    "Four full movies are a runtime/regression pilot, not a private-score estimate.",
    "Training excludes the evaluated embryo, but checkpoint/calibration used other movies of that embryo.",
    "Audit movies were evaluated in earlier experiments: not a fresh untouched final holdout.",
    "No-TTA single-seed abbreviated-training pipeline differs from the submitted LB leaders.",
    "Two embryos cannot identify the distribution of unseen private embryos.",
    "Timeouts and failed movies must remain visible; no complete-fold claim on partial output.",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps(clean_json({"event": event, **values})), flush=True)


def unique_mount(candidates):
    matches = [p for p in candidates if p.is_dir()]
    if len(matches) != 1:
        raise RuntimeError({"mount_candidates": list(map(str, candidates)), "matches": list(map(str, matches))})
    return matches[0]


def notebook_mount(slug):
    return unique_mount([INPUT / "notebooks" / "dmitriigluzdov" / slug, INPUT / slug])


def validate_fold(embryo, parent, selection_dir):
    cfg = FOLDS[embryo]
    contract_path = parent / f"loeo_{embryo}_contract.json"
    selection_path = selection_dir / f"loeo_{embryo}_selection.json"
    assert sha256(contract_path) == cfg["contract_sha256"], "Parent contract changed"
    assert sha256(selection_path) == cfg["selection_sha256"], "Frozen selection changed"
    contract = json.loads(contract_path.read_text())
    selection = json.loads(selection_path.read_text())
    assert contract["status"] == "training_complete"
    assert contract["holdout_embryo"] == embryo and contract["seed"] == SEED
    assert contract["epochs"] == cfg["epochs"]
    train, audit, calibration, checkpoint = [set(contract[k]) for k in (
        "train", "audit", "calibration", "checkpoint_validation")]
    assert len(train) == cfg["train_count"] and len(audit) == cfg["audit_count"]
    assert len(calibration) == 8 and len(checkpoint) == 4 and checkpoint <= calibration
    assert not (train & (audit | calibration)) and not (audit & calibration)
    assert all(not n.startswith(embryo + "_") for n in train)
    assert all(n.startswith(embryo + "_") for n in audit | calibration)
    assert set(cfg["movies"]) <= audit
    weight_dir = parent / f"loeo_holdout_{embryo}"
    for name in ("edge_predictor_best.pth", "config.json"):
        receipt = contract["artifacts"][name]
        assert (weight_dir / name).stat().st_size == receipt["bytes"]
        assert sha256(weight_dir / name) == receipt["sha256"], name
    assert selection["status"] == "selection_frozen_before_confirmation_and_audit"
    assert selection["holdout_embryo"] == embryo
    assert selection["parent_contract_sha256"] == cfg["contract_sha256"]
    assert selection["weights_sha256"] == contract["artifacts"]["edge_predictor_best.pth"]["sha256"]
    assert selection["audit_movies"] == contract["audit"]
    assert selection["checkpoint_validation_movies"] == contract["checkpoint_validation"]
    assert selection["confirmation_movies"] == sorted(calibration - checkpoint)
    assert selection["selected_policy"] == "registered_hungarian"
    threshold = float(selection["selected_threshold"])
    assert threshold in (0.95, 0.97, 0.985, 0.99, 0.995)
    return {"weight_path": str(weight_dir / "edge_predictor_best.pth"),
            "weight_sha256": selection["weights_sha256"], "threshold": threshold,
            "contract_sha256": cfg["contract_sha256"], "selection_sha256": cfg["selection_sha256"]}


def local_corrections(source, target, shift):
    """Infer a smooth residual field from mutual unambiguous anchors, no labels.

    Each query excludes its own anchor to prevent a nearest-neighbor decision
    from reinforcing itself. Uncertain/sparse neighborhoods fall back to zero.
    All coordinates and distances here are in physical microns.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    correction = np.zeros_like(source, dtype=float)
    if len(source) < 4 or len(target) < 2:
        return correction, {"anchors": 0, "adjusted_nodes": 0}
    shifted = source + shift
    distance, nearest = cKDTree(target).query(shifted, k=2)
    reverse = cKDTree(shifted).query(target, k=1)[1]
    mask = ((distance[:, 0] < FLOW["anchor_gate_um"])
            & ((distance[:, 1] - distance[:, 0]) >= FLOW["anchor_margin_um"])
            & (reverse[nearest[:, 0]] == np.arange(len(source))))
    anchor_ids = np.flatnonzero(mask)
    if len(anchor_ids) < FLOW["minimum_neighbors"] + 1:
        return correction, {"anchors": len(anchor_ids), "adjusted_nodes": 0}
    residual = target[nearest[anchor_ids, 0]] - shifted[anchor_ids]
    tree = cKDTree(source[anchor_ids])
    k = min(FLOW["neighbors"] + 1, len(anchor_ids))
    distances, neighbors = tree.query(source, k=k)
    for row in range(len(source)):
        near = neighbors[row][(distances[row] <= FLOW["radius_um"])
                              & (anchor_ids[neighbors[row]] != row)][:FLOW["neighbors"]]
        if len(near) < FLOW["minimum_neighbors"]:
            continue
        median = np.median(residual[near], axis=0)
        mad = float(np.median(np.linalg.norm(residual[near] - median, axis=1)))
        if mad > FLOW["maximum_mad_um"] or np.linalg.norm(median) > FLOW["maximum_residual_um"]:
            continue
        blend = min(FLOW["maximum_blend"], len(near) / FLOW["full_support_neighbors"])
        correction[row] = blend * median
    norms = np.linalg.norm(correction, axis=1)
    return correction, {"anchors": len(anchor_ids), "adjusted_nodes": int((norms > 1e-8).sum()),
                        "maximum_correction_um": float(norms.max(initial=0.0))}


def link_motion(coords, edges, scale, policy):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    if policy not in ("registered", "registered_weak", "local_flow_weak"):
        raise ValueError(policy)
    coords = np.asarray(coords, dtype=float).reshape(-1, 4)
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or not np.all(scale > 0) or not np.isfinite(coords).all():
        raise ValueError("Invalid physical coordinates/scale")
    probability = {}
    for s, t, p, _ in edges:
        key = (int(s), int(t))
        probability[key] = max(probability.get(key, 0.0), float(p))
    linked, telemetry = [], []
    times = coords[:, 0].astype(int)
    for timepoint in sorted(set(times)):
        source_ids = np.flatnonzero(times == timepoint)
        target_ids = np.flatnonzero(times == timepoint + 1)
        if not len(target_ids):
            continue
        source, target = coords[source_ids, 1:] * scale, coords[target_ids, 1:] * scale
        nearest = cKDTree(target).query(source, k=1)[1]
        displacement = target[nearest] - source
        shift = np.median(displacement, axis=0)
        inliers = np.linalg.norm(displacement - shift, axis=1) <= 4.0
        if inliers.sum() >= 3:
            shift = np.median(displacement[inliers], axis=0)
        correction = np.zeros_like(source)
        info = {"anchors": 0, "adjusted_nodes": 0}
        if policy == "local_flow_weak":
            correction, info = local_corrections(source, target, shift)
        global_residual = np.linalg.norm(source[:, None] + shift - target[None], axis=2)
        residual = np.linalg.norm(source[:, None] + shift + correction[:, None] - target[None], axis=2)
        weight = 0.0 if policy == "registered" else FLOW["learned_weight"]
        score = (1.0 - weight) * np.exp(-residual / FLOW["motion_scale_um"])
        if weight:
            rows = {int(s): i for i, s in enumerate(source_ids)}
            columns = {int(t): j for j, t in enumerate(target_ids)}
            for (s, t), p in probability.items():
                if s in rows and t in columns:
                    score[rows[s], columns[t]] += weight * p
        gate = FLOW["association_gate_um"]
        valid = (residual < gate) & (global_residual < gate)
        minimum = (1.0 - weight) * np.exp(-gate / FLOW["motion_scale_um"])
        cost = np.where(valid, 1.0 - score, 1e6)
        augmented = np.concatenate([cost, np.full((len(source), len(source)), 1.0 - minimum)], axis=1)
        row_indices, column_indices = linear_sum_assignment(augmented)
        for row, col in zip(row_indices, column_indices):
            if col < len(target) and valid[row, col] and score[row, col] >= minimum:
                linked.append((int(source_ids[row]), int(target_ids[col]), float(score[row, col]),
                               float(np.linalg.norm(source[row] - target[col]))))
        telemetry.append({"t": int(timepoint), **info})
    return linked, telemetry


def worker(embryo, name, receipt, train_dir):
    # Imports happen inside a fresh fork, before any parent Torch work.
    import random
    import numpy as np
    import torch
    import tracksdata as td
    from geff import GeffMetadata
    from biohub_tracking.io import open_dataset
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics
    from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video

    start = time.monotonic()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available(), "CPU-only metadata contract violated"
    model, window, downsample = load_model(Path(receipt["weight_path"]), torch.device("cpu"))
    dataset = open_dataset(train_dir / name, require_tracks=False, load_image=False)
    emit("inference_start", embryo=embryo, dataset=name, image_shape=dataset.image_shape,
         threads=torch.get_num_threads(), det_tta=False, weights=receipt["weight_sha256"])
    original_encode = model.encode
    encode_times = []

    def timed_encode(*args, **kwargs):
        encode_start = time.monotonic()
        result = original_encode(*args, **kwargs)
        encode_times.append(time.monotonic() - encode_start)
        if len(encode_times) <= 3 or len(encode_times) % 10 == 0:
            emit("encode_progress", dataset=name, windows=len(encode_times),
                 last_seconds=encode_times[-1], elapsed_seconds=time.monotonic() - start)
        return result

    model.encode = timed_encode
    cfg = PredictConfig(det_threshold=receipt["threshold"], det_tta=False,
                        pool_kernel_um=3.0, edge_activation="softmax", threshold=0.5, use_ilp=True)
    infer_start = time.monotonic()
    coords, edges = predict_video(model, train_dir / name, torch.device("cpu"), cfg=cfg,
                                 window_size=window, unet_batch_size=1, downsample=downsample)
    inference_seconds = time.monotonic() - infer_start
    del model
    edge_array = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    cache = RESULT_DIR / f"{Path(name).stem}_candidates.npz"
    np.savez_compressed(cache, coords=np.asarray(coords, dtype=np.float32),
                        edge_source=edge_array[:, 0].astype(np.int64), edge_target=edge_array[:, 1].astype(np.int64),
                        edge_probability=edge_array[:, 2].astype(np.float32), edge_distance=edge_array[:, 3].astype(np.float32),
                        scale=np.asarray(dataset.scale))
    emit("cache_saved", dataset=name, inference_seconds=inference_seconds, nodes=len(coords), edges=len(edges))
    graph_edges, telemetry = {}, {}
    for arm in ARMS[:-1]:
        graph_edges[arm], telemetry[arm] = link_motion(coords, edges, dataset.scale, arm)
    graphs = {arm: build_graph(coords, links) for arm, links in graph_edges.items()}
    graph = build_graph(coords, edges)
    if graph.num_edges():
        solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0, disappearance_weight=1.5, division_weight=1.0)
        graph = solver.solve(graph).detach()
    graphs[ARMS[-1]] = graph
    # Freeze every graph before any GEFF label is opened. Motion arms preserve
    # all original nodes; ILP may select a subset, so save its actual node IDs too.
    graph_receipts = {}
    for arm, graph in graphs.items():
        rows = list(graph.edge_attrs().iter_rows(named=True)) if graph.num_edges() else []
        node_rows = list(graph.node_attrs().iter_rows(named=True)) if graph.num_nodes() else []
        graph_path = RESULT_DIR / f"{Path(name).stem}_{arm}_graph.npz"
        np.savez_compressed(graph_path,
                            node_id=np.asarray([r["node_id"] for r in node_rows], dtype=np.int64),
                            coords=np.asarray([[r["t"], r["z"], r["y"], r["x"]] for r in node_rows], dtype=np.float32).reshape(-1, 4),
                            source=np.asarray([r["source_id"] for r in rows], dtype=np.int64),
                            target=np.asarray([r["target_id"] for r in rows], dtype=np.int64))
        graph_receipts[arm] = {"path": graph_path.name, "sha256": sha256(graph_path),
                               "nodes": len(node_rows), "edges": len(rows)}
    frozen_at = time.time()
    metrics = {}
    for arm, graph in graphs.items():
        truth = open_dataset(train_dir / name, require_tracks=True, load_image=False)
        metadata = GeffMetadata.read(train_dir / f"{Path(name).stem}.geff")
        n_total = float((metadata.extra or {})["estimated_number_of_nodes"])
        assert n_total > 0 and math.isfinite(n_total)
        result = evaluate(graph, truth.tracks, scale=truth.scale)
        recall = node_recall(graph, truth.tracks) if graph.num_nodes() and graph.num_edges() else 0.0
        metrics[arm] = {"dataset": name, **per_sample_metrics(result, n_total, recall)}
        assert math.isfinite(metrics[arm]["adj_edge_jaccard"]), "No silently unscored movies"
    output = {"status": "complete", "embryo": embryo, "dataset": name, "receipt": receipt,
              "det_tta": False, "scale": dataset.scale, "image_shape": dataset.image_shape,
              "inference_seconds": inference_seconds, "wall_seconds": time.monotonic() - start,
              "encode_seconds": encode_times, "cache_file": cache.name, "cache_sha256": sha256(cache),
              "cache_bytes": cache.stat().st_size, "graphs_frozen_before_label_read_unix": frozen_at,
              "graph_receipts": graph_receipts, "per_arm": metrics, "telemetry": telemetry}
    write_json(RESULT_DIR / f"{Path(name).stem}_result.json", output)
    emit("movie_complete", dataset=name, wall_seconds=output["wall_seconds"], metrics=metrics)


def guarded_worker(*args):
    try:
        worker(*args)
    except BaseException:
        traceback.print_exc()
        raise


def main():
    started = time.monotonic()
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[key] = str(CPU_THREADS)
    train_dir = unique_mount([INPUT / "competitions" / COMPETITION / "train", INPUT / COMPETITION / "train"])
    support = unique_mount([INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1",
                            INPUT / "biohub-tracking-support-pack-50ep-v1"])
    receipts = {e: validate_fold(e, notebook_mount(c["parent"]), notebook_mount(c["selection"])) for e, c in FOLDS.items()}
    jobs = [(e, FOLDS[e]["movies"][i]) for i in range(2) for e in FOLDS]
    frozen = {"experiment": "EXP077", "hypothesis": "H065", "flow": FLOW, "arms": ARMS,
              "jobs": jobs, "receipts": receipts, "det_tta": False, "cpu_threads": CPU_THREADS,
              "run_limit_seconds": RUN_SECONDS, "movie_limit_seconds": MOVIE_SECONDS,
              "selection_rule": "first two per embryo from frozen hash-selected 24-movie pilot",
              "limitations": LIMITATIONS}
    write_json(RESULT_DIR / "frozen_contract.json", frozen)
    emit("contracts_verified", jobs=jobs, det_tta=False, device="cpu")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                    "--find-links", str(support / "wheels"), "-r",
                    str(support / "requirements-unet-ilp-kaggle-predownload.txt")], check=True, timeout=600)
    shutil.copytree(support / "repo", REPO)
    sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
    os.environ["BIOHUB_DATA_DIR"] = str(train_dir)
    write_json(RESULT_DIR / "support_source_hashes.json", {
        str(p.relative_to(REPO)): sha256(p) for p in sorted(REPO.rglob("*.py"))})
    manifest = []
    # Linux fork keeps the notebook's function definitions, without relying on __file__.
    # No numerical libraries or models have been imported by this parent process.
    context = multiprocessing.get_context("fork")
    for embryo, name in jobs:
        remaining = RUN_SECONDS - (time.monotonic() - started) - 120
        if remaining < 120:
            manifest.append({"dataset": name, "embryo": embryo, "status": "skipped_budget"})
            continue
        timeout = min(MOVIE_SECONDS, remaining)
        process = context.Process(target=guarded_worker, args=(embryo, name, receipts[embryo], train_dir))
        process.start()
        process.join(timeout)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(15)
            if process.is_alive():
                process.kill()
                process.join(10)
        result_path = RESULT_DIR / f"{Path(name).stem}_result.json"
        status = "complete" if process.exitcode == 0 and result_path.is_file() else ("timeout" if timed_out else "error")
        manifest.append({"dataset": name, "embryo": embryo, "status": status, "exitcode": process.exitcode})
        write_json(RESULT_DIR / "progress.json", {"movies": manifest, "wall_seconds": time.monotonic() - started})
        emit("worker_finished", **manifest[-1])
    # Aggregation uses the same supplied official metric, not mean movie scores.
    from biohub_tracking.metrics import summarise
    completed = [json.loads((RESULT_DIR / f"{Path(r['dataset']).stem}_result.json").read_text())
                 for r in manifest if r["status"] == "complete"]
    summaries, deltas = {}, {}
    for group in (*FOLDS, "pooled"):
        rows = [r for r in completed if group == "pooled" or r["embryo"] == group]
        if not rows:
            continue
        summaries[group] = {arm: summarise([r["per_arm"][arm] for r in rows]) for arm in ARMS}
        deltas[group] = summaries[group]["local_flow_weak"]["score"] - summaries[group]["registered_weak"]["score"]
    result = {"status": "complete_pilot" if len(completed) == len(jobs) else "partial_pilot",
              "movies": manifest, "summary_by_embryo_and_arm": summaries,
              "paired_local_flow_minus_registered_weak": deltas,
              "wall_seconds": time.monotonic() - started, "limitations": LIMITATIONS}
    write_json(RESULT_DIR / "pilot_result.json", result)
    emit("pilot_finished", **result)


if __name__ == "__main__":
    main()
