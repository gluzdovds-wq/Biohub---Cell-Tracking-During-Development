"""Pretrain the public Biohub detector/edge model on fully labelled synthetic movies.

This stage produces a checkpoint only. Promotion requires a compute-matched real-data
fine-tune and reciprocal LOEO evaluation against the no-pretraining control.
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
N_TRAIN_SEQUENCES = 512
N_VALID_SEQUENCES = 64
EPOCHS = 8
CROP_SHAPE = (32, 32, 32)
BATCH_SIZE = 8

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"

synthetic_roots = sorted(INPUT.glob("**/biohub_synthetic"))
if not SUPPORT.exists() or not synthetic_roots:
    raise FileNotFoundError({"support": str(SUPPORT), "synthetic_roots": list(map(str, synthetic_roots))})
SYNTHETIC = synthetic_roots[0]
SEQUENCES = SYNTHETIC / "sequences"
SOURCE_METADATA = SYNTHETIC / "metadata.json"
if not SEQUENCES.exists() or not SOURCE_METADATA.exists():
    raise FileNotFoundError({"sequences": str(SEQUENCES), "metadata": str(SOURCE_METADATA)})

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

source_metadata_bytes = SOURCE_METADATA.read_bytes()
source_metadata = json.loads(source_metadata_bytes)
assert source_metadata["pooling"].startswith("stride 4x in XY")
assert source_metadata["voxel_native_um"] == [1.625, 0.40625, 0.40625]

all_files = sorted(SEQUENCES.glob("seq_*.npz"))
if len(all_files) < N_TRAIN_SEQUENCES + N_VALID_SEQUENCES:
    raise RuntimeError(f"Only {len(all_files)} sequence files found")
rng = random.Random(SEED)
rng.shuffle(all_files)
train_files = sorted(all_files[:N_TRAIN_SEQUENCES])
valid_files = sorted(all_files[N_TRAIN_SEQUENCES : N_TRAIN_SEQUENCES + N_VALID_SEQUENCES])


def make_windows(path: Path) -> tuple[list[tuple[FrameWindowData, torch.Tensor]], dict[str, int | float]]:
    """Convert one six-frame synthetic sequence into five fixed spatial-crop windows.

    The public builder stores pooled 64^3 images but node y/x in the native 256-grid.
    Dividing y/x by four is therefore mandatory before cropping or supervision.
    """

    with np.load(path) as data:
        volumes = data["volumes"]
        nodes = data["nodes"].astype(np.float32)
        edges = data["edges"].astype(np.int64)
        divisions = data["divisions"].astype(np.int64)
        voxel = data["voxel_um_pooled"].astype(np.float32)

    if volumes.ndim != 4 or tuple(volumes.shape[1:]) != (64, 64, 64):
        raise ValueError(f"Unexpected volume shape in {path.name}: {volumes.shape}")
    if not np.allclose(voxel, [1.625, 1.625, 1.625]):
        raise ValueError(f"Unexpected pooled voxel scale in {path.name}: {voxel}")
    if nodes.ndim != 2 or nodes.shape[1] != 5:
        raise ValueError(f"Unexpected nodes in {path.name}: {nodes.shape}")

    raw_max_yx = float(nodes[:, 2:4].max())
    if raw_max_yx <= 64.0:
        raise ValueError("Source coordinate contract changed; refusing an accidental second XY /4")
    pooled_nodes = nodes.copy()
    pooled_nodes[:, 2:4] /= 4.0
    if pooled_nodes[:, 1:4].min() < 0 or pooled_nodes[:, 1:4].max() >= 64:
        raise ValueError(f"Converted coordinates outside the pooled image in {path.name}")

    file_index = int(path.stem.split("_")[-1])
    crop_rng = np.random.default_rng(SEED + file_index)
    origin = np.array([crop_rng.integers(0, 33) for _ in range(3)], dtype=np.float32)
    stop = origin + np.array(CROP_SHAPE, dtype=np.float32)
    z0, y0, x0 = origin.astype(int)
    cz, cy, cx = CROP_SHAPE

    raw_crop = volumes[:, z0 : z0 + cz, y0 : y0 + cy, x0 : x0 + cx].astype(np.float32)
    q_low, q_high = np.percentile(raw_crop, [0.1, 99.9])
    images = torch.from_numpy(np.clip((raw_crop - q_low) / (q_high - q_low + 1e-6), 0, None)).half()

    edge_set = {(int(s), int(t)) for s, t in edges}
    output: list[tuple[FrameWindowData, torch.Tensor]] = []
    image_shape = (int(volumes.shape[0]),) + CROP_SHAPE
    kept_divisions = 0
    kept_nodes = 0
    kept_edges = 0

    for t in range(volumes.shape[0] - 1):
        global_ids: list[np.ndarray] = []
        coords_list: list[torch.Tensor] = []
        pos_feats: list[torch.Tensor] = []
        node_counts: list[int] = []
        for frame in (t, t + 1):
            mask = pooled_nodes[:, 0] == frame
            spatial = pooled_nodes[:, 1:4]
            mask &= np.all(spatial >= origin, axis=1) & np.all(spatial < stop, axis=1)
            ids = np.flatnonzero(mask)
            coords = spatial[ids] - origin
            if len(ids) == 0:
                break
            full = np.column_stack([np.full(len(ids), frame, np.float32), coords]).astype(np.float32)
            global_ids.append(ids)
            coords_list.append(torch.from_numpy(coords.astype(np.float32)))
            pos_feats.append(torch.from_numpy(extract_pos_features(full, image_shape)))
            node_counts.append(len(ids))
        if len(global_ids) != 2:
            continue

        src_lookup = {int(g): i for i, g in enumerate(global_ids[0])}
        dst_lookup = {int(g): i for i, g in enumerate(global_ids[1])}
        target = torch.zeros(len(global_ids[0]), len(global_ids[1]), dtype=torch.float32)
        for source, target_id in edge_set:
            if source in src_lookup and target_id in dst_lookup:
                target[src_lookup[source], dst_lookup[target_id]] = 1.0
                kept_edges += 1
        kept_divisions += sum(1 for div in divisions if int(div) in src_lookup and target[src_lookup[int(div)]].sum() >= 2)
        kept_nodes += sum(node_counts)
        output.append(
            (
                FrameWindowData(
                    t_start=t,
                    n_frames=2,
                    pos_feats=pos_feats,
                    coords=coords_list,
                    node_counts=node_counts,
                    targets=[target],
                ),
                images[t : t + 2],
            )
        )

    return output, {
        "windows": len(output),
        "kept_nodes_with_repetition": kept_nodes,
        "kept_edges": kept_edges,
        "kept_divisions": kept_divisions,
        "raw_max_yx": raw_max_yx,
        "converted_max_zyx": float(pooled_nodes[:, 1:4].max()),
    }


def build_items(files: list[Path], label: str):
    rows = []
    stats = []
    for i, path in enumerate(files, 1):
        windows, row = make_windows(path)
        rows.extend(windows)
        stats.append(row)
        if i % 64 == 0:
            print(f"{label}: {i}/{len(files)} files, {len(rows)} windows", flush=True)
    if not rows:
        raise RuntimeError(f"No usable {label} windows")
    return rows, stats


train_raw, train_stats = build_items(train_files, "train")
valid_raw, valid_stats = build_items(valid_files, "valid")
max_nodes = max(max(window.node_counts) for window, _ in train_raw + valid_raw)
print(f"max_nodes={max_nodes}", flush=True)


class MaterializedSyntheticDataset(Dataset):
    def __init__(self, rows):
        self.rows = [(pad_window(window, max_nodes), images) for window, images in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        meta, images = self.rows[index]
        return {
            **meta,
            "imgs": images,
            "image_shape": torch.tensor((6,) + CROP_SHAPE, dtype=torch.long),
            "voxel_size": torch.tensor((1.625, 1.625, 1.625), dtype=torch.float32),
            # The edge model was designed to see native-grid coordinates after this multiplication.
            "downsample": torch.tensor((1.0, 4.0, 4.0), dtype=torch.float32),
        }


train_ds = MaterializedSyntheticDataset(train_raw)
valid_ds = MaterializedSyntheticDataset(valid_raw)
generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("EXP024 is a GPU experiment")

unet = TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
model = UNetNodeTransformer(
    unet=unet,
    unet_out_channels=32,
    pos_feat_dim=4 * _POS_EMBED_DIM,
).to(device)
if torch.cuda.device_count() > 1:
    model.unet = nn.DataParallel(model.unet)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

artifact_dir = WORK / "synthetic_pretrain"
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
    )
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
    valid_loss, valid_acc, valid_recall = evaluate(model, valid_loader, device, pool_kernel_um=5.0)
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
        state = {key.replace("unet.module.", "unet.", 1): value for key, value in model.state_dict().items()}
        torch.save(state, checkpoint)

summary = {
    "status": "pretraining_complete_not_promotion_evidence",
    "seed": SEED,
    "source_metadata_sha256": hashlib.sha256(source_metadata_bytes).hexdigest(),
    "source_metadata": source_metadata,
    "coordinate_adapter": "z unchanged; native y/x divided by 4 before pooled-image cropping",
    "crop_shape": CROP_SHAPE,
    "train_sequences": len(train_files),
    "valid_sequences": len(valid_files),
    "train_windows": len(train_ds),
    "valid_windows": len(valid_ds),
    "max_nodes": max_nodes,
    "train_kept_divisions": sum(int(row["kept_divisions"]) for row in train_stats),
    "valid_kept_divisions": sum(int(row["kept_divisions"]) for row in valid_stats),
    "best_synthetic_score": best_score,
    "history": history,
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "promotion_gate": "compute-matched real fine-tune plus reciprocal frozen LOEO audit",
}
(WORK / "exp024_receipt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
