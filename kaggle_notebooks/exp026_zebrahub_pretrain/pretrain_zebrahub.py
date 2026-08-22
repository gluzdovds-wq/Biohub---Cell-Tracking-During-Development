"""Pretrain the exact public Biohub model on authorized real Zebrahub crops.

This external validation chooses a checkpoint only.  Promotion still requires a
compute-matched competition-data fine-tune and positive reciprocal frozen LOEO
deltas against random initialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SEED = 314159
EPOCHS = 12
BATCH_SIZE = 2
EXPECTED_TRAIN_SHARDS = 256
EXPECTED_VALID_SHARDS = 64
EXPECTED_TRACK_ROWS = 7_505_357
CROP_SHAPE = (64, 64, 64)

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
support_override = os.environ.get("BIOHUB_SUPPORT_ROOT")
SUPPORT = (
    Path(support_override)
    if support_override
    else INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
)
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"
receipt_paths = sorted(INPUT.glob("**/zebrahub_training_set/receipt.json"))
if not SUPPORT.exists() or len(receipt_paths) != 1:
    raise FileNotFoundError(
        {"support": str(SUPPORT), "zebrahub_receipts": list(map(str, receipt_paths))}
    )
DATA = receipt_paths[0].parent
SOURCE_RECEIPT_PATH = receipt_paths[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_receipt_bytes = SOURCE_RECEIPT_PATH.read_bytes()
source_receipt = json.loads(source_receipt_bytes)
required_receipt = {
    "status": "PASS",
    "train_shards": EXPECTED_TRAIN_SHARDS,
    "valid_shards": EXPECTED_VALID_SHARDS,
    "raw_track_table_deleted": True,
}
for key, expected in required_receipt.items():
    if source_receipt.get(key) != expected:
        raise AssertionError({"receipt_key": key, "observed": source_receipt.get(key), "expected": expected})
if source_receipt.get("track_stats", {}).get("rows") != EXPECTED_TRACK_ROWS:
    raise AssertionError({"track_stats": source_receipt.get("track_stats")})
if source_receipt.get("output_shape_zyx") != list(CROP_SHAPE):
    raise AssertionError({"output_shape_zyx": source_receipt.get("output_shape_zyx")})
if source_receipt.get("output_voxel_um_zyx") != [1.625, 1.625, 1.625]:
    raise AssertionError({"output_voxel_um_zyx": source_receipt.get("output_voxel_um_zyx")})

source_rows = {row["file"]: row for row in source_receipt["shards"]}
train_files = sorted(DATA.glob("train_*.npz"))
valid_files = sorted(DATA.glob("valid_*.npz"))
if len(train_files) != EXPECTED_TRAIN_SHARDS or len(valid_files) != EXPECTED_VALID_SHARDS:
    raise AssertionError({"train": len(train_files), "valid": len(valid_files)})
for path in train_files + valid_files:
    row = source_rows.get(path.name)
    if row is None or sha256(path) != row["sha256"]:
        raise AssertionError({"bad_or_unregistered_shard": path.name})

if os.environ.get("BIOHUB_SKIP_DEP_INSTALL") != "1":
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_unet_transformer import (
    FrameWindowData,
    TemporalUNet3D,
    UNetNodeTransformer,
    _POS_EMBED_DIM,
    evaluate,
    extract_pos_features,
    pad_window,
    train_epoch,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_window(path: Path) -> tuple[FrameWindowData, dict[str, int]]:
    with np.load(path) as shard:
        volumes_shape = tuple(shard["volumes"].shape)
        coords = shard["coords_tzyx"].astype(np.float32)
        edges = shard["edges"].astype(np.int64)
        edge_kinds = shard["edge_kinds"].astype(np.uint8)
    if volumes_shape != (2,) + CROP_SHAPE:
        raise ValueError({"file": path.name, "volumes_shape": volumes_shape})
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError({"file": path.name, "coords_shape": coords.shape})
    if np.any(coords[:, 1:] < 0) or np.any(coords[:, 1:] > 63):
        raise ValueError({"file": path.name, "coordinate_range": [coords[:, 1:].min(), coords[:, 1:].max()]})
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(edge_kinds):
        raise ValueError({"file": path.name, "edges_shape": edges.shape})
    if len(edges) and (edges.min() < 0 or edges.max() >= len(coords)):
        raise ValueError({"file": path.name, "edge_index_range": [edges.min(), edges.max()]})

    ids_by_frame = [np.flatnonzero(coords[:, 0] == frame) for frame in (0, 1)]
    if any(len(ids) == 0 for ids in ids_by_frame):
        raise ValueError({"file": path.name, "nodes_per_frame": list(map(len, ids_by_frame))})
    local_offsets = [{int(global_id): offset for offset, global_id in enumerate(ids)} for ids in ids_by_frame]
    target = torch.zeros(len(ids_by_frame[0]), len(ids_by_frame[1]), dtype=torch.float32)
    for source, destination in edges:
        source = int(source)
        destination = int(destination)
        if source not in local_offsets[0] or destination not in local_offsets[1]:
            raise ValueError({"file": path.name, "nonconsecutive_edge": [source, destination]})
        target[local_offsets[0][source], local_offsets[1][destination]] = 1.0

    coordinates = [torch.from_numpy(coords[ids, 1:].copy()) for ids in ids_by_frame]
    pos_feats = [
        torch.from_numpy(extract_pos_features(coords[ids], (2,) + CROP_SHAPE))
        for ids in ids_by_frame
    ]
    window = FrameWindowData(
        t_start=0,
        n_frames=2,
        pos_feats=pos_feats,
        coords=coordinates,
        node_counts=list(map(len, ids_by_frame)),
        targets=[target],
    )
    return window, {
        "nodes": len(coords),
        "edges": len(edges),
        "division_edges": int(edge_kinds.sum()),
    }


def materialize(files: list[Path]):
    rows = []
    stats = []
    for path in files:
        window, row = load_window(path)
        rows.append((path, window))
        stats.append(row)
    return rows, stats


train_raw, train_stats = materialize(train_files)
valid_raw, valid_stats = materialize(valid_files)
max_nodes = max(max(window.node_counts) for _, window in train_raw + valid_raw)
print(
    json.dumps(
        {
            "train_windows": len(train_raw),
            "valid_windows": len(valid_raw),
            "max_nodes": max_nodes,
            "train_nodes": sum(row["nodes"] for row in train_stats),
            "valid_nodes": sum(row["nodes"] for row in valid_stats),
        }
    ),
    flush=True,
)

if os.environ.get("BIOHUB_VALIDATE_ONLY") == "1":
    validation_receipt = {
        "status": "PASS",
        "mode": "validate_only_no_training",
        "source_receipt_sha256": hashlib.sha256(source_receipt_bytes).hexdigest(),
        "train_windows": len(train_raw),
        "valid_windows": len(valid_raw),
        "max_nodes_per_frame": max_nodes,
        "train_nodes": sum(row["nodes"] for row in train_stats),
        "valid_nodes": sum(row["nodes"] for row in valid_stats),
        "train_edges": sum(row["edges"] for row in train_stats),
        "valid_edges": sum(row["edges"] for row in valid_stats),
        "train_division_edges": sum(row["division_edges"] for row in train_stats),
        "valid_division_edges": sum(row["division_edges"] for row in valid_stats),
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "exp026_validation_receipt.json").write_text(
        json.dumps(validation_receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation_receipt, indent=2), flush=True)
    raise SystemExit(0)


class ZebrahubDataset(Dataset):
    def __init__(self, rows):
        self.rows = [(path, pad_window(window, max_nodes)) for path, window in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, meta = self.rows[index]
        with np.load(path) as shard:
            raw = shard["volumes"].astype(np.float32)
        q_low, q_high = np.percentile(raw, [0.1, 99.9])
        images = torch.from_numpy(np.clip((raw - q_low) / (q_high - q_low + 1e-6), 0, None))
        return {
            **meta,
            "imgs": images,
            "image_shape": torch.tensor((2,) + CROP_SHAPE, dtype=torch.long),
            "voxel_size": torch.tensor((1.625, 1.625, 1.625), dtype=torch.float32),
            # Coordinates live on the isotropic pooled grid; restore native XY
            # scale only for the edge-transformer spatial features.
            "downsample": torch.tensor((1.0, 4.0, 4.0), dtype=torch.float32),
        }


train_ds = ZebrahubDataset(train_raw)
valid_ds = ZebrahubDataset(valid_raw)
generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    generator=generator,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("EXP026 is a GPU experiment")
unet = TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
model = UNetNodeTransformer(
    unet=unet,
    unet_out_channels=32,
    pos_feat_dim=4 * _POS_EMBED_DIM,
).to(device)
if torch.cuda.device_count() > 1:
    model.unet = nn.DataParallel(model.unet)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

artifact_dir = WORK / "zebrahub_pretrain"
artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "config.json").write_text(
    json.dumps(
        {
            "unet_out_channels": 32,
            "unet_layers": [32, 64, 128],
            "downsample": [1, 4, 4],
            "window_size": 2,
            "pool_kernel_um": 5.0,
        },
        indent=2,
    ),
    encoding="utf-8",
)

best_score = -1.0
checkpoint = artifact_dir / "edge_predictor_best.pth"
history = []
for epoch in range(EPOCHS):
    edge_loss, det_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        device,
        det_loss_weight=1.0,
        det_neg_weight=1e-2,
        pool_kernel_um=5.0,
    )
    valid_loss, valid_acc, valid_recall = evaluate(
        model, valid_loader, device, pool_kernel_um=5.0
    )
    score = valid_acc * valid_recall
    row = {
        "epoch": epoch,
        "edge_loss": edge_loss,
        "det_loss": det_loss,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_recall": valid_recall,
        "selection_score": score,
    }
    history.append(row)
    print(json.dumps(row), flush=True)
    if score >= best_score:
        best_score = score
        state = {
            key.replace("unet.module.", "unet.", 1): value
            for key, value in model.state_dict().items()
        }
        torch.save(state, checkpoint)

summary = {
    "status": "pretraining_complete_not_promotion_evidence",
    "seed": SEED,
    "source_receipt_sha256": hashlib.sha256(source_receipt_bytes).hexdigest(),
    "source_track_sha256": source_receipt["track_sha256"],
    "source_time_split": source_receipt["time_split"],
    "train_shards": len(train_ds),
    "valid_shards": len(valid_ds),
    "train_nodes": sum(row["nodes"] for row in train_stats),
    "valid_nodes": sum(row["nodes"] for row in valid_stats),
    "train_edges": sum(row["edges"] for row in train_stats),
    "valid_edges": sum(row["edges"] for row in valid_stats),
    "train_division_edges": sum(row["division_edges"] for row in train_stats),
    "valid_division_edges": sum(row["division_edges"] for row in valid_stats),
    "max_nodes_per_frame": max_nodes,
    "best_external_score": best_score,
    "history": history,
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_sha256": sha256(checkpoint),
    "promotion_gate": "compute-matched real fine-tune plus positive reciprocal frozen LOEO deltas",
}
(WORK / "exp026_receipt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
