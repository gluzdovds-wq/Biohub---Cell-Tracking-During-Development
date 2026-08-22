"""Build a bounded, physically aligned Zebrahub pretraining set.

The source is explicitly authorized by the Biohub competition host.  This CPU
stage downloads the public track table temporarily, selects deterministic and
time-disjoint crop anchors, reads only the corresponding remote OME-Zarr chunks,
and emits compact two-frame shards.  The raw table is deleted before completion.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "zarr==3.3.0",
        "fsspec==2026.7.0",
    ]
)

import fsspec
import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import map_coordinates

SEED = 314159
TRACK_URL = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/"
    "ZSNS001_tail_tracks.csv"
)
ZARR_ROOT = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/"
    "ZSNS001_tail.ome.zarr"
)
EXPECTED_TRACK_ROWS = 7_505_357
EXPECTED_TIME_RANGE = (0, 790)
LEVEL = 1
SOURCE_VOXEL_UM = np.array([2.48, 0.878, 0.878], dtype=np.float64)
OUTPUT_SHAPE = np.array([64, 64, 64], dtype=np.int64)
OUTPUT_VOXEL_UM = 1.625
N_TRAIN = 256
N_VALID = 64
TRAIN_TIMES = np.linspace(10, 629, N_TRAIN, dtype=np.int64)
VALID_TIMES = np.linspace(650, 789, N_VALID, dtype=np.int64)
CHUNK_ROWS = 250_000

WORK = Path("/kaggle/working")
OUT = WORK / "zebrahub_training_set"
TRACK_PATH = WORK / "ZSNS001_tail_tracks.csv"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "biohub-exp025/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response, destination.open("wb") as output:
        total = 0
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
            total += len(block)
            if total // (128 * 1024 * 1024) != (total - len(block)) // (128 * 1024 * 1024):
                print(f"downloaded {total / 2**20:.0f} MiB", flush=True)


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Deterministic vectorized 64-bit mixer used for anchor sampling."""
    x = values.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def select_anchors(path: Path, selected_times: set[int]) -> tuple[dict[int, dict[str, float]], dict[str, int]]:
    anchors: dict[int, dict[str, float]] = {}
    rows = 0
    t_min = 10**9
    t_max = -1
    usecols = ["NodeID", "t", "z", "y", "x"]
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS), 1
    ):
        rows += len(chunk)
        t_min = min(t_min, int(chunk["t"].min()))
        t_max = max(t_max, int(chunk["t"].max()))
        subset = chunk[chunk["t"].isin(selected_times)].copy()
        if len(subset):
            subset["score"] = splitmix64(
                subset["NodeID"].to_numpy(np.uint64) ^ np.uint64(SEED)
            )
            for timepoint, frame in subset.groupby("t", sort=False):
                row = frame.loc[frame["score"].idxmin()]
                candidate = {
                    "score": int(row["score"]),
                    "node_id": int(row["NodeID"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }
                existing = anchors.get(int(timepoint))
                if existing is None or candidate["score"] < existing["score"]:
                    anchors[int(timepoint)] = candidate
        if chunk_index % 8 == 0:
            print(f"anchor pass: {rows:,} rows; {len(anchors)}/{len(selected_times)} times", flush=True)
    stats = {"rows": rows, "time_min": t_min, "time_max": t_max}
    return anchors, stats


def load_selected_frames(path: Path, wanted_times: set[int]) -> dict[int, pd.DataFrame]:
    columns = ["track_id", "NodeID", "ParentTrackID", "t", "z", "y", "x"]
    retained = []
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=CHUNK_ROWS), 1
    ):
        subset = chunk[chunk["t"].isin(wanted_times)].copy()
        if len(subset):
            retained.append(subset)
        if chunk_index % 8 == 0:
            print(
                f"frame pass: {chunk_index * CHUNK_ROWS:,} rows; "
                f"{sum(map(len, retained)):,} retained",
                flush=True,
            )
    data = pd.concat(retained, ignore_index=True)
    data["track_id"] = data["track_id"].astype(np.int64)
    data["NodeID"] = data["NodeID"].astype(np.int64)
    data["t"] = data["t"].astype(np.int16)
    data["ParentTrackID"] = data["ParentTrackID"].fillna(-1).astype(np.int64)
    frames = {int(t): frame.reset_index(drop=True) for t, frame in data.groupby("t", sort=True)}
    missing = sorted(wanted_times - set(frames))
    if missing:
        raise RuntimeError({"missing_selected_frames": missing})
    return frames


def physical_crop(array, timepoint: int, center_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin_um = center_um - 0.5 * (OUTPUT_SHAPE - 1) * OUTPUT_VOXEL_UM
    axes_um = [origin_um[i] + np.arange(OUTPUT_SHAPE[i]) * OUTPUT_VOXEL_UM for i in range(3)]
    source_axes = [axis / SOURCE_VOXEL_UM[i] for i, axis in enumerate(axes_um)]
    spatial_shape = np.asarray(array.shape[2:], dtype=np.int64)
    start = np.maximum([int(np.floor(axis.min())) - 2 for axis in source_axes], 0)
    stop = np.minimum([int(np.ceil(axis.max())) + 3 for axis in source_axes], spatial_shape)
    if np.any(stop - start < 4):
        raise ValueError({"timepoint": timepoint, "start": start.tolist(), "stop": stop.tolist()})
    last_error = None
    for attempt in range(3):
        try:
            raw = np.asarray(
                array[
                    timepoint,
                    0,
                    start[0] : stop[0],
                    start[1] : stop[1],
                    start[2] : stop[2],
                ]
            )
            break
        except Exception as error:  # remote chunk reads occasionally time out
            last_error = error
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    if last_error is not None:
        print(f"remote read recovered after retry: {last_error!r}", flush=True)
    local_axes = [axis - start[i] for i, axis in enumerate(source_axes)]
    grid = np.meshgrid(*local_axes, indexing="ij")
    sampled = map_coordinates(raw.astype(np.float32), grid, order=1, mode="constant", cval=0.0)
    volume = np.clip(np.rint(sampled), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if tuple(volume.shape) != tuple(OUTPUT_SHAPE):
        raise RuntimeError(volume.shape)
    return volume, origin_um


def annotated_crop(
    source: pd.DataFrame,
    target: pd.DataFrame,
    origin_um: np.ndarray,
) -> dict[str, np.ndarray]:
    combined = pd.concat([source.assign(local_t=0), target.assign(local_t=1)], ignore_index=True)
    pixel = (combined[["z", "y", "x"]].to_numpy(np.float64) - origin_um) / OUTPUT_VOXEL_UM
    inside = np.all(pixel >= 0.0, axis=1) & np.all(pixel <= OUTPUT_SHAPE - 1, axis=1)
    kept = combined.loc[inside].copy().reset_index(drop=True)
    kept_pixel = pixel[inside].astype(np.float32)
    if not set(kept["local_t"].unique()) == {0, 1}:
        raise ValueError("Crop lost every annotation from one frame")

    local_by_node = {int(node): index for index, node in enumerate(kept["NodeID"])}
    source_track_to_node = dict(zip(source["track_id"].astype(int), source["NodeID"].astype(int)))
    pairs: set[tuple[int, int, int]] = set()
    for row in target.itertuples(index=False):
        target_node = int(row.NodeID)
        if target_node not in local_by_node:
            continue
        continuation = source_track_to_node.get(int(row.track_id))
        if continuation in local_by_node:
            pairs.add((local_by_node[continuation], local_by_node[target_node], 0))
        parent_track = int(row.ParentTrackID)
        parent_node = source_track_to_node.get(parent_track)
        if parent_track >= 0 and parent_node in local_by_node:
            pairs.add((local_by_node[parent_node], local_by_node[target_node], 1))

    ordered = sorted(pairs)
    edges = np.asarray([[s, t] for s, t, _ in ordered], dtype=np.int64).reshape(-1, 2)
    edge_kinds = np.asarray([kind for _, _, kind in ordered], dtype=np.uint8)
    coords = np.column_stack([kept["local_t"].to_numpy(np.float32), kept_pixel]).astype(np.float32)
    if len(edges) == 0:
        raise ValueError("Crop has no fully contained lineage edge")
    return {
        "coords_tzyx": coords,
        "node_ids": kept["NodeID"].to_numpy(np.int64),
        "track_ids": kept["track_id"].to_numpy(np.int64),
        "parent_track_ids": kept["ParentTrackID"].to_numpy(np.int64),
        "edges": edges,
        "edge_kinds": edge_kinds,
    }


def main() -> None:
    selected = {int(t) for t in np.concatenate([TRAIN_TIMES, VALID_TIMES])}
    if len(selected) != N_TRAIN + N_VALID or max(selected) + 1 > EXPECTED_TIME_RANGE[1]:
        raise AssertionError("Time split contract is invalid")
    download(TRACK_URL, TRACK_PATH)
    track_bytes = TRACK_PATH.stat().st_size
    track_sha = sha256(TRACK_PATH)
    anchors, track_stats = select_anchors(TRACK_PATH, selected)
    if track_stats != {
        "rows": EXPECTED_TRACK_ROWS,
        "time_min": EXPECTED_TIME_RANGE[0],
        "time_max": EXPECTED_TIME_RANGE[1],
    }:
        raise AssertionError({"observed_track_stats": track_stats})
    if set(anchors) != selected:
        raise AssertionError({"missing_anchors": sorted(selected - set(anchors))})

    wanted_times = selected | {t + 1 for t in selected}
    frames = load_selected_frames(TRACK_PATH, wanted_times)
    mapper = fsspec.get_mapper(f"{ZARR_ROOT}/{LEVEL}")
    array = zarr.open_array(store=mapper, mode="r")
    if tuple(array.shape) != (791, 1, 210, 609, 546):
        raise AssertionError({"zarr_shape": tuple(array.shape)})

    shard_receipts = []
    for index, timepoint in enumerate(sorted(selected)):
        split = "train" if timepoint in set(map(int, TRAIN_TIMES)) else "valid"
        center = np.array([anchors[timepoint][axis] for axis in "zyx"], dtype=np.float64)
        volume0, origin0 = physical_crop(array, timepoint, center)
        volume1, origin1 = physical_crop(array, timepoint + 1, center)
        if not np.array_equal(origin0, origin1):
            raise AssertionError("Two-frame crop origins differ")
        annotations = annotated_crop(frames[timepoint], frames[timepoint + 1], origin0)
        volumes = np.stack([volume0, volume1])
        name = f"{split}_{timepoint:04d}.npz"
        path = OUT / name
        np.savez_compressed(
            path,
            volumes=volumes,
            timepoint=np.int16(timepoint),
            center_um_zyx=center,
            origin_um_zyx=origin0,
            output_voxel_um_zyx=np.full(3, OUTPUT_VOXEL_UM, dtype=np.float64),
            **annotations,
        )
        shard_receipts.append(
            {
                "file": name,
                "split": split,
                "timepoint": timepoint,
                "nodes": int(len(annotations["node_ids"])),
                "edges": int(len(annotations["edges"])),
                "division_edges": int(annotations["edge_kinds"].sum()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        print(
            f"{index + 1}/{len(selected)} {name}: "
            f"{len(annotations['node_ids'])} nodes, {len(annotations['edges'])} edges",
            flush=True,
        )

    TRACK_PATH.unlink()
    train_rows = [row for row in shard_receipts if row["split"] == "train"]
    valid_rows = [row for row in shard_receipts if row["split"] == "valid"]
    receipt = {
        "status": "PASS",
        "purpose": "bounded authorized external pretraining data; not validation evidence",
        "track_url": TRACK_URL,
        "track_bytes": track_bytes,
        "track_sha256": track_sha,
        "track_stats": track_stats,
        "zarr_root": ZARR_ROOT,
        "zarr_level": LEVEL,
        "zarr_shape_tczyx": list(map(int, array.shape)),
        "source_voxel_um_zyx": SOURCE_VOXEL_UM.tolist(),
        "output_shape_zyx": OUTPUT_SHAPE.tolist(),
        "output_voxel_um_zyx": [OUTPUT_VOXEL_UM] * 3,
        "anchor_rule": "minimum splitmix64(NodeID xor seed) independently at each selected time",
        "seed": SEED,
        "time_split": {
            "train_range": [int(TRAIN_TIMES.min()), int(TRAIN_TIMES.max())],
            "valid_range": [int(VALID_TIMES.min()), int(VALID_TIMES.max())],
            "gap_frames": int(VALID_TIMES.min() - TRAIN_TIMES.max()),
        },
        "train_shards": len(train_rows),
        "valid_shards": len(valid_rows),
        "train_nodes": sum(row["nodes"] for row in train_rows),
        "valid_nodes": sum(row["nodes"] for row in valid_rows),
        "train_edges": sum(row["edges"] for row in train_rows),
        "valid_edges": sum(row["edges"] for row in valid_rows),
        "train_division_edges": sum(row["division_edges"] for row in train_rows),
        "valid_division_edges": sum(row["division_edges"] for row in valid_rows),
        "raw_track_table_deleted": not TRACK_PATH.exists(),
        "shards": shard_receipts,
    }
    receipt_path = OUT / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "shards"}, indent=2))


if __name__ == "__main__":
    main()
