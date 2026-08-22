"""Train the public TemporalUNet3D + node transformer with embryo 44b6 held out."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

HOLDOUT_EMBRYO = "44b6"
SEED = 314159
EPOCHS = 50
N_CALIBRATION_MOVIES = 8
COMPETITION = "biohub-cell-tracking-during-development"

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TRAIN_DIR = INPUT / "competitions" / COMPETITION / "train"
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"
if not TRAIN_DIR.exists() or not SUPPORT.exists():
    raise FileNotFoundError({"train": str(TRAIN_DIR), "support": str(SUPPORT)})

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

available = sorted(
    path.name
    for path in TRAIN_DIR.glob("*.zarr")
    if (TRAIN_DIR / f"{path.stem}.geff").exists()
)
train_names = [name for name in available if not name.startswith(f"{HOLDOUT_EMBRYO}_")]
heldout_names = [name for name in available if name.startswith(f"{HOLDOUT_EMBRYO}_")]
rng = random.Random(SEED)
rng.shuffle(heldout_names)
calibration_names = sorted(heldout_names[:N_CALIBRATION_MOVIES])
audit_names = sorted(heldout_names[N_CALIBRATION_MOVIES:])
train_names = sorted(train_names)

assert train_names and calibration_names and audit_names
assert all(not name.startswith(f"{HOLDOUT_EMBRYO}_") for name in train_names)
assert all(name.startswith(f"{HOLDOUT_EMBRYO}_") for name in calibration_names + audit_names)
assert not (set(train_names) & set(calibration_names) or set(train_names) & set(audit_names))

splits_path = WORK / f"loeo_{HOLDOUT_EMBRYO}_splits.json"
splits_path.write_text(
    json.dumps([{"train": train_names, "test": calibration_names}], indent=2),
    encoding="utf-8",
)
contract = {
    "status": "frozen_before_training",
    "holdout_embryo": HOLDOUT_EMBRYO,
    "seed": SEED,
    "epochs": EPOCHS,
    "checkpoint_selection": "8 deterministic heldout-embryo calibration movies",
    "audit_policy": "all remaining heldout-embryo movies; never loaded during training",
    "train": train_names,
    "calibration": calibration_names,
    "audit": audit_names,
}
contract_path = WORK / f"loeo_{HOLDOUT_EMBRYO}_contract.json"
contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
print(json.dumps({key: value if not isinstance(value, list) else len(value) for key, value in contract.items()}))

import numpy as np
import torch
from train_unet_transformer import train

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

method = f"loeo_holdout_{HOLDOUT_EMBRYO}"
train(
    data_dir=TRAIN_DIR,
    fold=0,
    splits_file=splits_path,
    method=method,
    n_epochs=EPOCHS,
    lr=1e-4,
    batch_size=16,
    num_workers=8,
    unet_out_channels=32,
    unet_layers=[32, 64, 128],
    downsample=(1, 4, 4),
    det_loss_weight=1.0,
    det_neg_weight=1e-2,
    seed=SEED,
    window_size=2,
    pool_kernel_um=5.0,
    data_parallel=True,
)

model_dir = REPO / "weights" / method / "split_0"
artifact_dir = WORK / method
if artifact_dir.exists():
    shutil.rmtree(artifact_dir)
shutil.copytree(model_dir, artifact_dir)
for path in artifact_dir.rglob("*"):
    if path.is_file():
        contract.setdefault("artifacts", {})[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
contract["status"] = "training_complete"
contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
print(json.dumps(contract["artifacts"], indent=2))
