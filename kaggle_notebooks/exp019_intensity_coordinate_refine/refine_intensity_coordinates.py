"""Conservatively refine EXP-006 coordinates from the raw 3D intensity signal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_SHA256 = "5c852379cbf2a0b8a007a1bee32bfadafc2759ab2978750b16252b7f37211f4d"
EXPECTED_COLUMNS = [
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
SPATIAL_COLUMNS = ["z", "y", "x"]
SCALE_ZYX_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
RADIUS_Z = 2
RADIUS_YX = 5
BACKGROUND_PERCENTILE = 20.0
MAX_RAW_SHIFT_UM = 1.5
ALPHA = 0.35


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_source() -> Path:
    candidates = sorted(Path("/kaggle/input").glob("**/submission.csv"))
    matches = [path for path in candidates if sha256(path) == SOURCE_SHA256]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one EXP-006 source with SHA {SOURCE_SHA256}, "
            f"found {len(matches)} among {[str(path) for path in candidates]}"
        )
    return matches[0]


def read_frame(zarr_path: Path, timepoint: int) -> np.ndarray:
    import blosc2

    metadata_path = zarr_path / "0" / "zarr.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(metadata["shape"])
    dtype = np.dtype(metadata["data_type"])
    chunk_path = zarr_path / "0" / "c" / str(timepoint) / "0" / "0" / "0"
    decompressed = blosc2.decompress(chunk_path.read_bytes())
    frame = np.frombuffer(decompressed, dtype=dtype).reshape(shape[1:])
    return frame


def intensity_center(frame: np.ndarray, coordinate: np.ndarray) -> np.ndarray | None:
    center = np.rint(coordinate).astype(int)
    lower = np.maximum(center - np.asarray([RADIUS_Z, RADIUS_YX, RADIUS_YX]), 0)
    upper = np.minimum(
        center + np.asarray([RADIUS_Z, RADIUS_YX, RADIUS_YX]) + 1,
        np.asarray(frame.shape),
    )
    crop = frame[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ].astype(np.float32)
    if not crop.size:
        return None
    background = float(np.percentile(crop, BACKGROUND_PERCENTILE))
    weights = np.clip(crop - background, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    zz, yy, xx = np.mgrid[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ]
    refined = np.asarray(
        [
            float((zz * weights).sum() / total),
            float((yy * weights).sum() / total),
            float((xx * weights).sum() / total),
        ]
    )
    return refined if np.isfinite(refined).all() else None


def main() -> None:
    source_path = locate_source()
    competition_root = Path(
        "/kaggle/input/competitions/biohub-cell-tracking-during-development"
    )
    test_dir = competition_root / "test"
    output_path = Path("/kaggle/working/submission.csv")

    source = pd.read_csv(source_path, index_col=0)
    if list(source.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected source columns: {list(source.columns)}")
    if len(source) != 240422:
        raise RuntimeError(f"Unexpected source row count: {len(source)}")
    result = source.copy(deep=True)
    result[SPATIAL_COLUMNS] = result[SPATIAL_COLUMNS].astype(float)
    nodes = source[source["row_type"] == "node"]
    if len(nodes) != 122266 or int((source["row_type"] == "edge").sum()) != 118156:
        raise RuntimeError("EXP-006 node/edge contract failed")

    accepted_distances: list[float] = []
    output_distances: list[float] = []
    rejected_large_shift = 0
    rejected_empty_signal = 0
    processed = 0

    for (dataset, timepoint), frame_nodes in nodes.groupby(["dataset", "t"], sort=True):
        zarr_path = test_dir / f"{dataset}.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(zarr_path)
        frame = read_frame(zarr_path, int(timepoint))
        for row_index, coordinate in zip(
            frame_nodes.index,
            frame_nodes[SPATIAL_COLUMNS].to_numpy(dtype=float),
        ):
            processed += 1
            candidate = intensity_center(frame, coordinate)
            if candidate is None:
                rejected_empty_signal += 1
                continue
            distance_um = float(np.linalg.norm((candidate - coordinate) * SCALE_ZYX_UM))
            if distance_um > MAX_RAW_SHIFT_UM:
                rejected_large_shift += 1
                continue
            result.loc[row_index, SPATIAL_COLUMNS] = (
                (1.0 - ALPHA) * coordinate + ALPHA * candidate
            )
            accepted_distances.append(distance_um)
            output_distances.append(ALPHA * distance_um)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "t": int(timepoint),
                    "processed": processed,
                    "accepted": len(accepted_distances),
                }
            ),
            flush=True,
        )

    source_identity = nodes[["dataset", "node_id", "t"]].reset_index(drop=True)
    result_identity = result[result["row_type"] == "node"][
        ["dataset", "node_id", "t"]
    ].reset_index(drop=True)
    if not source_identity.equals(result_identity):
        raise AssertionError("node identity or time changed")
    source_edges = source[source["row_type"] == "edge"][
        ["dataset", "source_id", "target_id"]
    ].reset_index(drop=True)
    result_edges = result[result["row_type"] == "edge"][
        ["dataset", "source_id", "target_id"]
    ].reset_index(drop=True)
    if not source_edges.equals(result_edges):
        raise AssertionError("edge topology changed")

    result.index.name = "id"
    result.to_csv(output_path)
    accepted = np.asarray(accepted_distances, dtype=float)
    displaced = np.asarray(output_distances, dtype=float)
    receipt = {
        "status": "PASS",
        "method": "gated raw-intensity center-of-mass coordinate refinement",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "output_sha256": sha256(output_path),
        "nodes": int((result["row_type"] == "node").sum()),
        "edges": int((result["row_type"] == "edge").sum()),
        "radius_zyx_voxels": [RADIUS_Z, RADIUS_YX, RADIUS_YX],
        "background_percentile": BACKGROUND_PERCENTILE,
        "max_raw_shift_um": MAX_RAW_SHIFT_UM,
        "alpha": ALPHA,
        "processed": processed,
        "accepted": len(accepted),
        "rejected_large_shift": rejected_large_shift,
        "rejected_empty_signal": rejected_empty_signal,
        "raw_shift_um": {
            "mean": float(accepted.mean()) if len(accepted) else None,
            "p95": float(np.quantile(accepted, 0.95)) if len(accepted) else None,
            "max": float(accepted.max()) if len(accepted) else None,
        },
        "output_shift_um": {
            "mean": float(displaced.mean()) if len(displaced) else None,
            "p95": float(np.quantile(displaced, 0.95)) if len(displaced) else None,
            "max": float(displaced.max()) if len(displaced) else None,
        },
        "topology_unchanged": True,
        "node_ids_and_times_unchanged": True,
    }
    Path("/kaggle/working/exp019_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
