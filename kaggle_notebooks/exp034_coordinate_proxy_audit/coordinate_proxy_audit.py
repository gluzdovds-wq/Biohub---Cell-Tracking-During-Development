"""Calibrated reused-label diagnostic for frozen coordinate/topology candidates.

This diagnostic may reject a harmful candidate.  It may not promote one because
the four labelled movies were already used by the public pipeline family.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
SAMPLE_STEMS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)
MATCH_RADIUS_UM = 7.0
NODE_COUNT_PENALTY = 0.1
DIVISION_WEIGHT = 0.1
EXPECTED_VISIBLE_OVERLAP_BASE = 0.8870825255187538
EXPECTED_COLUMNS = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]
CANDIDATES = {
    "EXP006": "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d",
    "EXP014": "c970d9433e68a91060894515714ae7f027b05457b98b412b625fe84482544de0",
    "EXP019": "7487ecb7de8c110caffd35bd043902b484ee4634ec58d020caebabfad9296c6d",
    "EXP022": "91e24e750dc2a305943713618bbaa3f0de95283cbeb2de9e9b2d6ecef3f8fb6a",
    "EXP023": "8bff01ab65cc2f9e022684822cd09240265417567abd5406387b808f7e052de3",
    "EXP031": "fcdc9a0a208c8666046ae304a2adc9fcd96c90fc7906dfc13941f2ceb83fa93d",
    "EXP035": "7376bd3c4056ee7c7f82fadd2db3bb37230ad09e399ae1f815c3c53a51374bd4",
    "EXP040": "9f0b0711b5ac0b078c5fb24332c2604c09118013116bc6fbe4d6f4e2eaa4a5e3",
    "EXP041": "21a42ffa33c8af7ef44b28f7edaea6a3d9666745139c9c51e132fed41a8fe114",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_tracksdata():
    try:
        return importlib.import_module("tracksdata")
    except Exception as initial_error:
        wheel_dirs = sorted({path.parent for path in INPUT.rglob("tracksdata*.whl")})
        if not wheel_dirs:
            raise ImportError("tracksdata unavailable and no offline wheel directory found") from initial_error
        specs = [
            "tracksdata", "zarr>=3.0.10,<4", "geff>=1.1.3.1.1", "geff-spec<1.2",
            "polars>=1.36", "polars-runtime-32", "rustworkx>=0.17.1", "networkx>=3.2.1",
            "ilpy>=0.5.1", "pyscipopt",
            "pydantic>=2.11", "pydantic-core", "annotated-types", "typing-extensions>=4.13",
            "typing-inspection", "bidict>=0.23.1", "psygnal>=0.14", "rich",
            "markdown-it-py", "pygments", "numcodecs>=0.13,<0.16", "donfig>=0.8",
            "google-crc32c>=1.5", "deprecated", "wrapt", "sqlalchemy>=2", "pyarrow",
        ]
        command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall"]
        for directory in wheel_dirs:
            command.extend(["--find-links", str(directory)])
        command.extend(specs)
        print("Installing graph dependencies from", list(map(str, wheel_dirs)), flush=True)
        subprocess.run(command, check=True)
        importlib.invalidate_caches()
        return importlib.import_module("tracksdata")


def locate_competition_train() -> Path:
    candidates = [
        INPUT / "competitions" / "biohub-cell-tracking-during-development" / "train",
        INPUT / "biohub-cell-tracking-during-development" / "train",
    ]
    for path in candidates:
        if all((path / f"{stem}.geff").exists() for stem in SAMPLE_STEMS):
            return path
    observed = [str(path) for path in INPUT.rglob(f"{SAMPLE_STEMS[0]}.geff")]
    if observed:
        path = Path(observed[0]).parent
        if all((path / f"{stem}.geff").exists() for stem in SAMPLE_STEMS):
            return path
    raise FileNotFoundError({"competition_train_candidates": list(map(str, candidates)), "observed": observed})


def locate_submissions() -> tuple[dict[str, Path], dict[str, list[str]]]:
    wanted = {digest: name for name, digest in CANDIDATES.items()}
    matches: dict[str, list[Path]] = defaultdict(list)
    observed: dict[str, str] = {}
    for path in INPUT.rglob("submission.csv"):
        digest = sha256(path)
        observed[str(path)] = digest
        name = wanted.get(digest)
        if name is not None:
            matches[name].append(path)
    missing = sorted(set(CANDIDATES) - set(matches))
    if missing:
        raise FileNotFoundError({"missing_candidates": missing, "observed": observed})
    selected = {name: sorted(paths, key=lambda item: (len(str(item)), str(item)))[0] for name, paths in matches.items()}
    all_matches = {name: list(map(str, paths)) for name, paths in matches.items()}
    return selected, all_matches


def graph_from_geff(td, path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def graph_to_plain(graph) -> tuple[dict[int, tuple[int, float, float, float]], list[tuple[int, int]]]:
    nodes = {
        int(row["node_id"]): (int(row["t"]), float(row["z"]), float(row["y"]), float(row["x"]))
        for row in graph.node_attrs().iter_rows(named=True)
    }
    edges = [
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    ]
    return nodes, edges


def submission_graphs(path: Path) -> dict[str, tuple[dict[int, tuple[int, float, float, float]], list[tuple[int, int]]]]:
    frame = pd.read_csv(path, index_col=0)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError({"path": str(path), "columns": list(frame.columns)})
    result = {}
    for dataset, sample in frame.groupby("dataset", sort=False):
        nodes = {
            int(row.node_id): (int(row.t), float(row.z), float(row.y), float(row.x))
            for row in sample[sample["row_type"] == "node"].itertuples()
        }
        edges = [
            (int(row.source_id), int(row.target_id))
            for row in sample[sample["row_type"] == "edge"].itertuples()
        ]
        result[str(dataset)] = (nodes, edges)
    if set(result) != set(SAMPLE_STEMS):
        raise ValueError({"path": str(path), "datasets": sorted(result)})
    return result


def recursive_find(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = recursive_find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find(value, key)
            if found is not None:
                return found
    return None


def estimated_node_count(geff_path: Path) -> float:
    for candidate in (geff_path / "zarr.json", geff_path / ".zattrs"):
        if not candidate.exists():
            continue
        found = recursive_find(json.loads(candidate.read_text()), "estimated_number_of_nodes")
        if found is not None:
            return float(found)
    raise ValueError(f"Missing estimated_number_of_nodes in {geff_path}")


def match_nodes(pred_nodes: dict, gt_nodes: dict) -> tuple[dict[int, int], dict[int, int]]:
    pred_by_t: dict[int, list[int]] = defaultdict(list)
    gt_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, (timepoint, *_coords) in pred_nodes.items():
        pred_by_t[int(timepoint)].append(node_id)
    for node_id, (timepoint, *_coords) in gt_nodes.items():
        gt_by_t[int(timepoint)].append(node_id)
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for timepoint in sorted(pred_by_t):
        pred_ids = sorted(pred_by_t[timepoint])
        gt_ids = sorted(gt_by_t.get(timepoint, []))
        if not gt_ids:
            continue
        pred_positions = np.asarray([pred_nodes[node][1:] for node in pred_ids]) * SCALE
        gt_positions = np.asarray([gt_nodes[node][1:] for node in gt_ids]) * SCALE
        distances = np.linalg.norm(pred_positions[:, None, :] - gt_positions[None, :, :], axis=-1)
        gated = np.where(distances <= MATCH_RADIUS_UM, distances, 1e6)
        pred_indices, gt_indices = linear_sum_assignment(gated)
        for pred_index, gt_index in zip(pred_indices, gt_indices):
            if gated[pred_index, gt_index] >= 1e6:
                continue
            pred_to_gt[pred_ids[pred_index]] = gt_ids[gt_index]
            gt_to_pred[gt_ids[gt_index]] = pred_ids[pred_index]
    return pred_to_gt, gt_to_pred


def edge_confusion(pred_edges, gt_edges, pred_to_gt) -> tuple[int, int, int]:
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = defaultdict(set)
    gt_incoming: dict[int, int] = {}
    for source, target in gt_edge_set:
        gt_outgoing[source].add(target)
        gt_incoming[target] = source
    tp = 0
    fp = 0
    matched = set()
    for source, target in pred_edges:
        mapped_source = pred_to_gt.get(source)
        mapped_target = pred_to_gt.get(target)
        if mapped_source is not None and mapped_target in gt_outgoing.get(mapped_source, ()):
            tp += 1
            matched.add((mapped_source, mapped_target))
        elif ((mapped_target is not None and mapped_target in gt_incoming) or
              (mapped_source is not None and bool(gt_outgoing.get(mapped_source)))):
            fp += 1
    return tp, fp, len(gt_edge_set - matched)


def union_find_components(node_ids, edges) -> dict[int, int]:
    parent = {node: node for node in node_ids}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for source, target in edges:
        if source not in parent or target not in parent:
            continue
        root_source, root_target = find(source), find(target)
        if root_source != root_target:
            parent[root_source] = root_target
    return {node: find(node) for node in node_ids}


def division_confusion(pred_nodes, pred_edges, gt_edges, pred_to_gt, gt_to_pred) -> tuple[int, int, int]:
    gt_out: dict[int, set[int]] = defaultdict(set)
    gt_in: dict[int, int] = {}
    pred_out: dict[int, set[int]] = defaultdict(set)
    for source, target in gt_edges:
        gt_out[source].add(target)
        gt_in[target] = source
    for source, target in pred_edges:
        pred_out[source].add(target)
    components = union_find_components(list(pred_nodes), pred_edges)
    fork_components = {components[node] for node, targets in pred_out.items() if len(targets) >= 2 and node in components}
    gt_divisions = [source for source, targets in gt_out.items() if len(targets) >= 2]

    def descendants(root):
        seen = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for target in gt_out.get(current, ()):
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return seen

    tp = 0
    fn = 0
    true_sources = set()
    for gt_source in gt_divisions:
        children = sorted(gt_out[gt_source])[:2]
        anchors = [gt_source] + ([gt_in[gt_source]] if gt_source in gt_in else [])
        anchor_predictions = [gt_to_pred[node] for node in anchors if node in gt_to_pred]
        lineage_components = []
        for child in children:
            lineage_components.append({
                components[prediction]
                for node in descendants(child)
                if (prediction := gt_to_pred.get(node)) is not None and prediction in components
            })
        anchor_components = {components[node] for node in anchor_predictions if node in components}
        found = bool(anchor_components and all(lineage_components)) and any(
            component in lineage_components[0]
            and component in lineage_components[1]
            and component in fork_components
            for component in anchor_components
        )
        if found:
            tp += 1
            true_sources.add(gt_source)
        else:
            fn += 1
    fp = 0
    for node, targets in pred_out.items():
        if len(targets) < 2:
            continue
        mapped = pred_to_gt.get(node)
        if mapped is not None and mapped in gt_out and mapped not in true_sources:
            fp += 1
    return tp, fp, fn


def jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator else 0.0


td = ensure_tracksdata()
train_dir = locate_competition_train()
paths, all_matches = locate_submissions()
print("Competition train:", train_dir, flush=True)
print("Candidate paths:", json.dumps({name: str(path) for name, path in paths.items()}, indent=2), flush=True)

ground_truth = {}
true_counts = {}
for stem in SAMPLE_STEMS:
    geff_path = train_dir / f"{stem}.geff"
    ground_truth[stem] = graph_to_plain(graph_from_geff(td, geff_path))
    true_counts[stem] = estimated_node_count(geff_path)

sample_rows = []
summaries = []
for candidate in CANDIDATES:
    predictions = submission_graphs(paths[candidate])
    candidate_rows = []
    for stem in SAMPLE_STEMS:
        pred_nodes, pred_edges = predictions[stem]
        gt_nodes, gt_edges = ground_truth[stem]
        pred_to_gt, gt_to_pred = match_nodes(pred_nodes, gt_nodes)
        edge_tp, edge_fp, edge_fn = edge_confusion(pred_edges, gt_edges, pred_to_gt)
        raw_edge = jaccard(edge_tp, edge_fp, edge_fn)
        adjusted_edge = max(
            0.0,
            raw_edge * (1.0 - NODE_COUNT_PENALTY * (len(pred_nodes) - true_counts[stem]) / true_counts[stem]),
        )
        div_tp, div_fp, div_fn = division_confusion(
            pred_nodes, pred_edges, gt_edges, pred_to_gt, gt_to_pred
        )
        row = {
            "candidate": candidate,
            "dataset": stem,
            "matched_nodes": len(pred_to_gt),
            "pred_nodes": len(pred_nodes),
            "true_node_estimate": true_counts[stem],
            "edge_tp": edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "edge_jaccard": raw_edge,
            "adjusted_edge_jaccard": adjusted_edge,
            "div_tp": div_tp,
            "div_fp": div_fp,
            "div_fn": div_fn,
            "weight": edge_tp + edge_fp + edge_fn,
        }
        sample_rows.append(row)
        candidate_rows.append(row)
    total_weight = sum(row["weight"] for row in candidate_rows)
    adjusted = sum(row["adjusted_edge_jaccard"] * row["weight"] for row in candidate_rows) / total_weight
    div_tp = sum(row["div_tp"] for row in candidate_rows)
    div_fp = sum(row["div_fp"] for row in candidate_rows)
    div_fn = sum(row["div_fn"] for row in candidate_rows)
    division = jaccard(div_tp, div_fp, div_fn)
    summaries.append({
        "candidate": candidate,
        "sha256": CANDIDATES[candidate],
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "proxy_score": adjusted + DIVISION_WEIGHT * division,
        "matched_nodes": sum(row["matched_nodes"] for row in candidate_rows),
        "edge_tp": sum(row["edge_tp"] for row in candidate_rows),
        "edge_fp": sum(row["edge_fp"] for row in candidate_rows),
        "edge_fn": sum(row["edge_fn"] for row in candidate_rows),
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
    })

base = next(row for row in summaries if row["candidate"] == "EXP006")
if not np.isclose(
    base["adjusted_edge_jaccard"],
    EXPECTED_VISIBLE_OVERLAP_BASE,
    rtol=0.0,
    atol=1e-12,
):
    raise AssertionError({"base_adjusted_edge_calibration_failed": base})
if not np.isclose(base["division_jaccard"], 0.0, rtol=0.0, atol=1e-12):
    raise AssertionError({"base_division_calibration_failed": base})
if not np.isclose(
    base["proxy_score"],
    EXPECTED_VISIBLE_OVERLAP_BASE,
    rtol=0.0,
    atol=1e-12,
):
    raise AssertionError({"base_proxy_calibration_failed": base})
for row in summaries:
    row["delta_adjusted_edge_vs_exp006"] = row["adjusted_edge_jaccard"] - base["adjusted_edge_jaccard"]
    row["delta_proxy_vs_exp006"] = row["proxy_score"] - base["proxy_score"]
    row["delta_matched_nodes_vs_exp006"] = row["matched_nodes"] - base["matched_nodes"]

summary_frame = pd.DataFrame(summaries).sort_values("proxy_score", ascending=False)
sample_frame = pd.DataFrame(sample_rows)
summary_frame.to_csv(WORK / "coordinate_proxy_summary.csv", index=False)
sample_frame.to_csv(WORK / "coordinate_proxy_per_sample.csv", index=False)
receipt = {
    "status": "PASS_FROZEN_VISIBLE_OVERLAP_DIFFERENTIAL",
    "promotion_allowed": False,
    "negative_delta_may_reject": True,
    "match_radius_um": MATCH_RADIUS_UM,
    "voxel_scale_zyx_um": SCALE.tolist(),
    "sample_stems": SAMPLE_STEMS,
    "candidate_paths": {name: str(path) for name, path in paths.items()},
    "candidate_input_matches": all_matches,
    "summaries": summaries,
}
(WORK / "exp034_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(summary_frame.to_string(index=False), flush=True)
print(json.dumps(receipt, indent=2), flush=True)
