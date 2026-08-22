"""Stream a Zebrahub track CSV and emit lineage/coordinate invariants as JSON."""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/"
    "ZSNS001_tail_tracks.csv"
)
REQUIRED = {"track_id", "NodeID", "ParentTrackID", "t", "z", "y", "x"}


def rows_from_source(source: str):
    path = Path(source)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return

    with urllib.request.urlopen(source, timeout=600) as response:
        with io.TextIOWrapper(response, encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--progress-every", type=int, default=500_000)
    args = parser.parse_args()

    rows = 0
    tracks: set[int] = set()
    division_parents: set[int] = set()
    division_children: set[int] = set()
    times: set[int] = set()
    minima = {axis: float("inf") for axis in "zyx"}
    maxima = {axis: float("-inf") for axis in "zyx"}
    header_checked = False

    for row in rows_from_source(args.source):
        if not header_checked:
            missing = REQUIRED - set(row)
            if missing:
                raise ValueError(f"Missing columns: {sorted(missing)}")
            header_checked = True

        track_id = int(float(row["track_id"]))
        tracks.add(track_id)
        parent = row["ParentTrackID"].strip()
        if parent:
            division_parents.add(int(float(parent)))
            division_children.add(track_id)
        times.add(int(float(row["t"])))
        for axis in "zyx":
            value = float(row[axis])
            minima[axis] = min(minima[axis], value)
            maxima[axis] = max(maxima[axis], value)

        rows += 1
        if args.progress_every and rows % args.progress_every == 0:
            print(json.dumps({"rows": rows, "tracks": len(tracks)}), flush=True)
        if args.max_rows is not None and rows >= args.max_rows:
            break

    if not rows:
        raise RuntimeError("No rows read")

    complete_source = args.max_rows is None
    result = {
        "source": args.source,
        "complete_source": complete_source,
        "rows": rows,
        "tracks": len(tracks),
        "division_parent_tracks": len(division_parents),
        "division_child_tracks": len(division_children),
        "children_per_division_parent": (
            len(division_children) / len(division_parents) if division_parents else None
        ),
        "time_min": min(times),
        "time_max": max(times),
        "time_count": len(times),
        "coordinate_min_um": minima,
        "coordinate_max_um": maxima,
    }
    if complete_source and len(division_children) != 2 * len(division_parents):
        raise ValueError("Expected exactly two child tracks per division parent")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
