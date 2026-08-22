"""Read and physically resample one remote Zebrahub OME-Zarr crop.

This is the alignment primitive for the future bounded external-data extractor.
Coordinates and voxel sizes are in micrometres; no raw timelapse is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fsspec
import numpy as np
import zarr
from scipy.ndimage import map_coordinates

DEFAULT_ROOT = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/"
    "ZSNS001_tail.ome.zarr"
)
LEVEL_SCALE_UM = {
    0: np.array([1.24, 0.439, 0.439], dtype=np.float64),
    1: np.array([2.48, 0.878, 0.878], dtype=np.float64),
    2: np.array([4.96, 1.756, 1.756], dtype=np.float64),
}


def extract_crop(
    root: str,
    level: int,
    timepoint: int,
    center_um: np.ndarray,
    output_shape: tuple[int, int, int],
    output_voxel_um: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if level not in LEVEL_SCALE_UM:
        raise ValueError(f"Unsupported level: {level}")
    source_scale = LEVEL_SCALE_UM[level]
    mapper = fsspec.get_mapper(f"{root.rstrip('/')}/{level}")
    array = zarr.open_array(store=mapper, mode="r")
    if array.ndim != 5 or array.shape[1] != 1:
        raise ValueError(f"Expected TCZYX array, received {array.shape}")
    if not 0 <= timepoint < array.shape[0]:
        raise ValueError(f"Timepoint {timepoint} outside [0,{array.shape[0]})")

    shape = np.asarray(output_shape, dtype=np.int64)
    if np.any(shape <= 0) or output_voxel_um <= 0:
        raise ValueError({"output_shape": output_shape, "output_voxel_um": output_voxel_um})
    center_um = np.asarray(center_um, dtype=np.float64)
    origin_um = center_um - 0.5 * (shape - 1) * output_voxel_um
    output_axes_um = [origin_um[i] + np.arange(shape[i]) * output_voxel_um for i in range(3)]
    source_axes = [axis / source_scale[i] for i, axis in enumerate(output_axes_um)]

    spatial_shape = np.asarray(array.shape[2:], dtype=np.int64)
    start = np.array([int(np.floor(axis.min())) - 2 for axis in source_axes], dtype=np.int64)
    stop = np.array([int(np.ceil(axis.max())) + 3 for axis in source_axes], dtype=np.int64)
    start = np.maximum(start, 0)
    stop = np.minimum(stop, spatial_shape)
    if np.any(stop - start < 4):
        raise ValueError({"start": start.tolist(), "stop": stop.tolist(), "shape": spatial_shape.tolist()})

    raw = np.asarray(
        array[
            timepoint,
            0,
            start[0] : stop[0],
            start[1] : stop[1],
            start[2] : stop[2],
        ]
    )
    local_axes = [axis - start[i] for i, axis in enumerate(source_axes)]
    grid = np.meshgrid(*local_axes, indexing="ij")
    sampled = map_coordinates(raw.astype(np.float32), grid, order=1, mode="constant", cval=0.0)
    volume = np.clip(np.rint(sampled), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if volume.shape != output_shape:
        raise RuntimeError(f"Unexpected resampled shape: {volume.shape}")

    center_index = 0.5 * (shape - 1)
    cz, cy, cx = np.rint(center_index).astype(int)
    local = volume[max(0, cz - 2) : cz + 3, max(0, cy - 4) : cy + 5, max(0, cx - 4) : cx + 5]
    receipt = {
        "source_root": root,
        "source_level": level,
        "source_shape_tczyx": list(map(int, array.shape)),
        "source_chunks_tczyx": list(map(int, array.chunks)),
        "source_voxel_um_zyx": source_scale.tolist(),
        "timepoint": timepoint,
        "center_um_zyx": center_um.tolist(),
        "output_origin_um_zyx": origin_um.tolist(),
        "output_shape_zyx": list(output_shape),
        "output_voxel_um_zyx": [output_voxel_um] * 3,
        "source_read_start_zyx": start.tolist(),
        "source_read_stop_zyx": stop.tolist(),
        "raw_shape_zyx": list(map(int, raw.shape)),
        "output_quantiles": np.percentile(volume, [0, 50, 99, 99.9, 100]).tolist(),
        "center_neighborhood_max": int(local.max()),
        "volume_sha256": hashlib.sha256(volume.tobytes()).hexdigest(),
    }
    return volume, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--level", type=int, default=1, choices=sorted(LEVEL_SCALE_UM))
    parser.add_argument("--timepoint", type=int, required=True)
    parser.add_argument("--center-um", type=float, nargs=3, metavar=("Z", "Y", "X"), required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 64, 64), metavar=("Z", "Y", "X"))
    parser.add_argument("--voxel-um", type=float, default=1.625)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    volume, receipt = extract_crop(
        root=args.root,
        level=args.level,
        timepoint=args.timepoint,
        center_um=np.asarray(args.center_um),
        output_shape=tuple(args.shape),
        output_voxel_um=args.voxel_um,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, volume=volume, receipt_json=json.dumps(receipt))
        receipt_path = args.output.with_suffix(".json")
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        receipt["artifact"] = str(args.output)
        receipt["receipt"] = str(receipt_path)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
