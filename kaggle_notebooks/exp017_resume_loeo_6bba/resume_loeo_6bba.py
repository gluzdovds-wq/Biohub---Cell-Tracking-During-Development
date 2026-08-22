"""Continue the full 6bba-held-out model from the canonical EXP010 checkpoint."""

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

HOLDOUT_EMBRYO = "6bba"
PARENT_SLUG = "biohub-exp010-loeo-holdout-6bba"
SEED = 314159
PARENT_EPOCHS = 10
ADDITIONAL_EPOCHS = 10
LEARNING_RATE = 5e-5
EXPECTED_SPLIT_SIZES = {"train": 71, "checkpoint_validation": 4, "calibration": 8, "audit": 120}
COMPETITION = "biohub-cell-tracking-during-development"

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TRAIN_DIR = INPUT / "competitions" / COMPETITION / "train"
SUPPORT = INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1"
if not SUPPORT.exists():
    SUPPORT = INPUT / "biohub-tracking-support-pack-50ep-v1"
PARENT = INPUT / PARENT_SLUG
if not PARENT.exists():
    matches = [path for path in INPUT.glob("*") if path.name.endswith(PARENT_SLUG)]
    if len(matches) != 1:
        raise FileNotFoundError({"parent": str(PARENT), "matches": [str(path) for path in matches]})
    PARENT = matches[0]

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

parent_contract_path = PARENT / f"loeo_{HOLDOUT_EMBRYO}_contract.json"
parent_weight_path = PARENT / f"loeo_holdout_{HOLDOUT_EMBRYO}" / "edge_predictor_best.pth"
if not parent_contract_path.is_file() or not parent_weight_path.is_file():
    raise FileNotFoundError(
        {"contract": str(parent_contract_path), "weights": str(parent_weight_path)}
    )
parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
if (
    parent_contract.get("status") != "training_complete"
    or parent_contract.get("holdout_embryo") != HOLDOUT_EMBRYO
    or parent_contract.get("seed") != SEED
    or parent_contract.get("epochs") != PARENT_EPOCHS
):
    raise RuntimeError(parent_contract)

split_sizes = {
    key: len(parent_contract[key])
    for key in ("train", "checkpoint_validation", "calibration", "audit")
}
if split_sizes != EXPECTED_SPLIT_SIZES:
    raise RuntimeError({"expected": EXPECTED_SPLIT_SIZES, "actual": split_sizes})
parent_weight_sha256 = hashlib.sha256(parent_weight_path.read_bytes()).hexdigest()
parent_weight_receipt = parent_contract.get("artifacts", {}).get(parent_weight_path.name, {})
if (
    parent_weight_receipt.get("bytes") != parent_weight_path.stat().st_size
    or parent_weight_receipt.get("sha256") != parent_weight_sha256
):
    raise RuntimeError(
        {"contract_weight": parent_weight_receipt, "actual_sha256": parent_weight_sha256}
    )

REPO = WORK / "tracking_repo"
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(SUPPORT / "repo", REPO)
train_source_path = REPO / "scripts" / "train_unet_transformer.py"
train_source = train_source_path.read_text(encoding="utf-8")
import_needle = "import argparse\n"
model_needle = """    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)
"""
resume_block = model_needle + """
    resume_weights = os.environ.get("BIOHUB_FULL_MODEL_WEIGHTS")
    if resume_weights:
        resume_state = torch.load(Path(resume_weights), map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(resume_state, strict=False)
        if missing or unexpected:
            raise RuntimeError({"resume_missing": missing, "resume_unexpected": unexpected})
        print(f"Full-model resume loaded: {resume_weights}", flush=True)
"""
if train_source.count(import_needle) != 1 or train_source.count(model_needle) != 1:
    raise RuntimeError("support training source no longer matches the audited patch anchors")
train_source = train_source.replace(import_needle, import_needle + "import os\n", 1)
train_source = train_source.replace(model_needle, resume_block, 1)
compile(train_source, str(train_source_path), "exec")
train_source_path.write_text(train_source, encoding="utf-8")

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ["BIOHUB_FULL_MODEL_WEIGHTS"] = str(parent_weight_path)

splits_path = WORK / f"loeo_{HOLDOUT_EMBRYO}_resume_splits.json"
splits_path.write_text(
    json.dumps(
        [
            {
                "train": parent_contract["train"],
                "test": parent_contract["checkpoint_validation"],
            }
        ],
        indent=2,
    ),
    encoding="utf-8",
)
contract = {
    "status": "staged_before_training",
    "holdout_embryo": HOLDOUT_EMBRYO,
    "seed": SEED,
    "parent_epochs": PARENT_EPOCHS,
    "additional_epochs": ADDITIONAL_EPOCHS,
    "total_epochs_after_stage": PARENT_EPOCHS + ADDITIONAL_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "parent_contract_sha256": hashlib.sha256(parent_contract_path.read_bytes()).hexdigest(),
    "parent_weight_sha256": parent_weight_sha256,
    "resume_scope": "full UNetNodeTransformer state; optimizer deliberately restarted",
    "train": parent_contract["train"],
    "checkpoint_validation": parent_contract["checkpoint_validation"],
    "calibration": parent_contract["calibration"],
    "audit": parent_contract["audit"],
}
contract_path = WORK / f"loeo_{HOLDOUT_EMBRYO}_resume_contract.json"
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

method = f"loeo_holdout_{HOLDOUT_EMBRYO}_resume_v1"
train(
    data_dir=TRAIN_DIR,
    fold=0,
    splits_file=splits_path,
    method=method,
    n_epochs=ADDITIONAL_EPOCHS,
    lr=LEARNING_RATE,
    batch_size=8,
    num_workers=4,
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
