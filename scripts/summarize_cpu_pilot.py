"""Archive completed EXP077 evidence and verify the downloaded candidate caches.

No inference, training, downloads or LB writes. Preserve the Kaggle run's supplied
official-metric aggregation, not an arithmetic mean of per-movie scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder):
    pilot_path = folder / "pilot_result.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    contract_path = folder / "frozen_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {(embryo, name) for embryo, name in contract["jobs"]}
    observed = {(r["embryo"], r["dataset"]) for r in pilot["movies"]}
    if (pilot["status"] != "complete_pilot" or len(expected) != 4
            or observed != expected or len(pilot["movies"]) != 4
            or any(r["status"] != "complete" or r["exitcode"] != 0 for r in pilot["movies"])):
        raise ValueError("Not a complete four-movie pilot; retain failures explicitly")
    movies = []
    for item in pilot["movies"]:
        path = folder / f"{Path(item['dataset']).stem}_result.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row["status"] != "complete" or row["dataset"] != item["dataset"]
                or row["embryo"] != item["embryo"] or row["det_tta"]
                or row["receipt"] != contract["receipts"][item["embryo"]]
                or set(row["per_arm"]) != set(contract["arms"])):
            raise ValueError(f"Contract drift: {path}")
        cache = folder / row["cache_file"]
        if sha256(cache) != row["cache_sha256"] or cache.stat().st_size != row["cache_bytes"]:
            raise ValueError(f"Cache mismatch: {cache}")
        with np.load(cache, allow_pickle=False) as arrays:
            cache_shape = {"nodes": len(arrays["coords"]), "candidate_edges": len(arrays["edge_source"])}
        retained = {key: row[key] for key in (
            "embryo", "dataset", "receipt", "det_tta", "scale", "image_shape",
            "inference_seconds", "wall_seconds", "cache_file", "cache_sha256",
            "cache_bytes", "graph_receipts", "per_arm", "graphs_frozen_before_label_read_unix")}
        retained.update({
            "result_json_sha256": sha256(path), "cache_verified_locally": True,
            "cache_shape": cache_shape,
            "local_flow_adjusted_nodes_summed_over_frames": sum(
                r["adjusted_nodes"] for r in row["telemetry"]["local_flow_weak"]),
            "local_flow_graph_differs_from_weak": (
                row["graph_receipts"]["local_flow_weak"]["sha256"]
                != row["graph_receipts"]["registered_weak"]["sha256"]),
        })
        movies.append(retained)
    movie_seconds = sum(r["wall_seconds"] for r in movies)
    return {
        "experiment": "EXP077", "checked_date": "2026-08-28",
        "kernel": "dmitriigluzdov/biohub-exp077-cpu-held-out-local-flow-pilot", "version": 1,
        "pilot_json_sha256": sha256(pilot_path), "contract_json_sha256": sha256(contract_path),
        "frozen_contract": contract, "pilot_result": pilot, "per_movie": movies,
        "cache_bytes_verified_locally": sum(r["cache_bytes"] for r in movies),
        "microscopy_images_downloaded_locally": False,
        "linear_runtime_estimate": {
            "based_on_movies": 4, "movie_wall_seconds_sum": movie_seconds,
            "hours_24_movies": movie_seconds / 4 * 24 / 3600,
            "hours_183_movies": movie_seconds / 4 * 183 / 3600,
            "caveat": "Excludes repeated setup; four-movie extrapolation, not a full-model benchmark.",
        },
        "continuation_launched": False,
        "exact_submitted_model_oof": None,
        "decision": "CPU feasible; no measured local-flow gain; no division-recall evidence; do not promote.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/exp077_kaggle_v1/exp077")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cpu_pilot_result_20260828.json")
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(args.output),
                      "cache_bytes": result["cache_bytes_verified_locally"],
                      "runtime_estimate": result["linear_runtime_estimate"]}, indent=2))


if __name__ == "__main__":
    main()
