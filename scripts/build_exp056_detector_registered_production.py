"""Build hidden-compatible EXP056 from the exact EXP054 and EXP008 inference sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "kaggle_notebooks" / "exp054_registered_production"
DONOR_DIR = ROOT / "kaggle_notebooks" / "exp008_three_unet_flip_tta"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "exp056_detector_registered_production"
BASE = BASE_DIR / "registered_production.ipynb"
DONOR = DONOR_DIR / "best-score.ipynb"
BASE_METADATA = BASE_DIR / "kernel-metadata.json"
DONOR_METADATA = DONOR_DIR / "kernel-metadata.json"
OUTPUT = OUTPUT_DIR / "detector_registered_production.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"
RECEIPT = OUTPUT_DIR / "build_receipt.json"

BASE_SHA256 = "a6ad5103e5707873d6cf9644e226e6f5f1f577a4f23b81730924109a88092706"
DONOR_SHA256 = "52064d5e3a7ce0ca36853df0abc7da70afe1c79a5f0a5e6bdfdc1c29a13b50d4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path, expected in ((BASE, BASE_SHA256), (DONOR, DONOR_SHA256)):
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"source drift: {path}: {observed} != {expected}")

base_notebook = json.loads(BASE.read_text(encoding="utf-8"))
donor_notebook = json.loads(DONOR.read_text(encoding="utf-8"))
if len(donor_notebook["cells"]) != 1 or donor_notebook["cells"][0]["cell_type"] != "code":
    raise AssertionError("EXP008 source contract changed")
donor_source = "".join(donor_notebook["cells"][0]["source"])
old_out = 'OUT = "submission.csv"'
new_out = 'OUT = "/kaggle/working/exp056_donor_submission.csv"'
if donor_source.count(old_out) != 1:
    raise AssertionError({"donor_output_anchor_count": donor_source.count(old_out)})
donor_source = donor_source.replace(old_out, new_out, 1)

cleanup_source = '''# EXP056 phase boundary: EXP054 has already emitted registered topology.
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("EXP056: starting independent detector donor inference")
'''

blend_source = r'''# EXP056: exact EXP014 coordinate policy on dynamic EXP054/EXP008 predictions.
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_PATH = Path("/kaggle/working/submission.csv")
DONOR_PATH = Path("/kaggle/working/exp056_donor_submission.csv")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")
RECEIPT_PATH = Path("/kaggle/working/exp056_receipt.json")
SCALE_EXP056 = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
SPATIAL_EXP056 = ["z", "y", "x"]
GATE_UM_EXP056 = 2.0
ALPHA_EXP056 = 0.5


def _exp056_mutual_matches(base_points, donor_points):
    if not len(base_points) or not len(donor_points):
        return []
    base_physical = base_points * SCALE_EXP056
    donor_physical = donor_points * SCALE_EXP056
    donor_tree = cKDTree(donor_physical)
    base_tree = cKDTree(base_physical)
    distances, base_to_donor = donor_tree.query(base_physical, k=1)
    _, donor_to_base = base_tree.query(donor_physical, k=1)
    return [
        (base_index, int(donor_index), float(distance))
        for base_index, (distance, donor_index) in enumerate(zip(distances, base_to_donor))
        if distance <= GATE_UM_EXP056 and int(donor_to_base[int(donor_index)]) == base_index
    ]


base = pd.read_csv(BASE_PATH, index_col=0)
donor = pd.read_csv(DONOR_PATH, index_col=0)
expected_columns = [
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]
if list(base.columns) != expected_columns or list(donor.columns) != expected_columns:
    raise AssertionError({"base_columns": list(base.columns), "donor_columns": list(donor.columns)})

result = base.copy(deep=True)
result[SPATIAL_EXP056] = result[SPATIAL_EXP056].astype(float)
base_nodes = base[base["row_type"] == "node"]
donor_nodes = donor[donor["row_type"] == "node"]
donor_frames = {
    key: frame for key, frame in donor_nodes.groupby(["dataset", "t"], sort=False)
}
match_distances = []
match_count_by_dataset = {}
for (dataset, timepoint), base_frame in base_nodes.groupby(["dataset", "t"], sort=True):
    donor_frame = donor_frames.get((dataset, timepoint))
    if donor_frame is None:
        match_count_by_dataset.setdefault(dataset, 0)
        continue
    base_coordinates = base_frame[SPATIAL_EXP056].to_numpy(dtype=float)
    donor_coordinates = donor_frame[SPATIAL_EXP056].to_numpy(dtype=float)
    matches = _exp056_mutual_matches(base_coordinates, donor_coordinates)
    if matches:
        base_offsets = np.fromiter((item[0] for item in matches), dtype=int)
        donor_offsets = np.fromiter((item[1] for item in matches), dtype=int)
        row_indices = base_frame.index.to_numpy()[base_offsets]
        result.loc[row_indices, SPATIAL_EXP056] = (
            (1.0 - ALPHA_EXP056) * base_coordinates[base_offsets]
            + ALPHA_EXP056 * donor_coordinates[donor_offsets]
        )
        match_distances.extend(item[2] for item in matches)
    match_count_by_dataset[dataset] = match_count_by_dataset.get(dataset, 0) + len(matches)

base_edges = base[base["row_type"] == "edge"][["dataset", "source_id", "target_id"]].reset_index(drop=True)
result_edges = result[result["row_type"] == "edge"][["dataset", "source_id", "target_id"]].reset_index(drop=True)
result_nodes = result[result["row_type"] == "node"]
if not base_edges.equals(result_edges):
    raise AssertionError("registered topology changed")
if not base_nodes[["dataset", "node_id", "t"]].equals(result_nodes[["dataset", "node_id", "t"]]):
    raise AssertionError("base node identity or time changed")
if not match_distances:
    raise AssertionError("detector consensus produced zero mutual matches")

result.index.name = "id"
result.to_csv(OUTPUT_PATH, float_format="%.17g", lineterminator="\n", na_rep="")
output_sha = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
distance_array = np.asarray(match_distances, dtype=float)
receipt = {
    "status": "PASS_EXP056_HIDDEN_DETECTOR_REGISTERED_PRODUCTION",
    "method": "dynamic EXP054 registered graph plus dynamic EXP008 mutual-nearest coordinate blend",
    "output_sha256": output_sha,
    "nodes": int(len(result_nodes)),
    "edges": int(len(result_edges)),
    "matched_nodes": int(len(match_distances)),
    "matched_fraction": float(len(match_distances) / len(result_nodes)),
    "match_count_by_dataset": match_count_by_dataset,
    "match_distance_um_mean": float(distance_array.mean()),
    "match_distance_um_p95": float(np.quantile(distance_array, 0.95)),
    "alpha": ALPHA_EXP056,
    "gate_um": GATE_UM_EXP056,
    "registered_topology_unchanged": True,
    "node_ids_and_times_unchanged": True,
    "public_artifact_input": False,
}
RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
'''

for source in (cleanup_source, donor_source, blend_source):
    base_notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    )

metadata = base_notebook.setdefault("metadata", {})
metadata.setdefault("kaggle", {})["title"] = "Biohub EXP056 Detector Registered Production"
metadata["title"] = "Biohub EXP056 Detector Registered Production"

base_kernel = json.loads(BASE_METADATA.read_text(encoding="utf-8"))
donor_kernel = json.loads(DONOR_METADATA.read_text(encoding="utf-8"))
dataset_sources = list(dict.fromkeys(base_kernel["dataset_sources"] + donor_kernel["dataset_sources"]))
kernel_metadata = dict(base_kernel)
kernel_metadata.update(
    {
        "id": "dmitriigluzdov/biohub-exp056-detector-registered-production",
        "title": "Biohub EXP056 Detector Registered Production",
        "code_file": OUTPUT.name,
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
    }
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(base_notebook, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
OUTPUT_METADATA.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
receipt = {
    "status": "PASS_EXP056_BUILD",
    "hypothesis": "H056",
    "base": str(BASE.relative_to(ROOT)),
    "base_sha256": BASE_SHA256,
    "donor": str(DONOR.relative_to(ROOT)),
    "donor_sha256": DONOR_SHA256,
    "output": str(OUTPUT.relative_to(ROOT)),
    "output_sha256": sha256(OUTPUT),
    "appended_cells": 3,
    "dataset_sources": dataset_sources,
    "hidden_test_dynamic": {
        "public_submission_artifact_input": False,
        "base_full_inference": True,
        "donor_full_inference": True,
        "test_stem_discovery_preserved": True,
    },
}
RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
