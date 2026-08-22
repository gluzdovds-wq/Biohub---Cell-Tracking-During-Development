"""Compare the staged own-seed state with both frozen EXP006 learned seeds."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch

INPUT = Path(os.environ.get("BIOHUB_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK_ROOT", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
RECEIPT = WORK / "exp038_checkpoint_audit.json"

SHAS = {
    "own": "b1507f6918192c0f5c15fd5091d97ff565b1fab14e67e01101446534abb6a7b7",
    "primary": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    "secondary": "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


inventory = []
matches: dict[str, list[Path]] = {name: [] for name in SHAS}
for path in sorted([*INPUT.rglob("*.pth"), *INPUT.rglob("*.pt")]):
    digest = sha256(path)
    inventory.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    for name, expected in SHAS.items():
        if digest == expected:
            matches[name].append(path)
for name, paths in matches.items():
    if len(paths) != 1:
        raise RuntimeError({"checkpoint": name, "matches": list(map(str, paths)), "inventory": inventory})


def tensor_state(path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    wrappers = []
    while isinstance(payload, dict):
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            break
        selected = next(
            (key for key in ("model_state", "state_dict", "model") if isinstance(payload.get(key), dict)),
            None,
        )
        if selected is None:
            raise TypeError({"path": str(path), "payload_keys": list(payload)[:50]})
        wrappers.append(selected)
        payload = payload[selected]
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"No tensor state in {path}")
    return {str(key): value.detach().cpu() for key, value in payload.items()}, wrappers


states = {}
wrappers = {}
for name in SHAS:
    states[name], wrappers[name] = tensor_state(matches[name][0])


def comparison(reference_name: str) -> dict[str, object]:
    own = states["own"]
    reference = states[reference_name]
    own_keys = set(own)
    reference_keys = set(reference)
    missing = sorted(reference_keys - own_keys)
    unexpected = sorted(own_keys - reference_keys)
    if missing or unexpected:
        raise AssertionError({"reference": reference_name, "missing": missing, "unexpected": unexpected})

    shape_mismatches = []
    dtype_mismatches = []
    nonfinite = []
    equal_keys = 0
    parameters = 0
    dot = 0.0
    reference_norm_sq = 0.0
    own_norm_sq = 0.0
    difference_norm_sq = 0.0
    for key in sorted(reference_keys):
        left = reference[key]
        right = own[key]
        if tuple(left.shape) != tuple(right.shape):
            shape_mismatches.append(key)
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append(key)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            nonfinite.append(key)
            continue
        equal_keys += int(torch.equal(left, right))
        left64 = left.to(torch.float64).reshape(-1)
        right64 = right.to(torch.float64).reshape(-1)
        parameters += left64.numel()
        dot += float(torch.dot(left64, right64))
        reference_norm_sq += float(torch.dot(left64, left64))
        own_norm_sq += float(torch.dot(right64, right64))
        delta = left64 - right64
        difference_norm_sq += float(torch.dot(delta, delta))
    if shape_mismatches or dtype_mismatches or nonfinite:
        raise AssertionError(
            {
                "reference": reference_name,
                "shape_mismatches": shape_mismatches,
                "dtype_mismatches": dtype_mismatches,
                "nonfinite": nonfinite,
            }
        )
    if parameters <= 0 or reference_norm_sq <= 0 or own_norm_sq <= 0:
        raise AssertionError(f"Degenerate comparison with {reference_name}")
    if equal_keys == len(reference_keys) or difference_norm_sq == 0:
        raise AssertionError(f"Own checkpoint duplicates {reference_name}")
    return {
        "reference": reference_name,
        "state_keys": len(reference_keys),
        "parameters": parameters,
        "exact_equal_keys": equal_keys,
        "different_keys": len(reference_keys) - equal_keys,
        "global_cosine_similarity": dot / math.sqrt(reference_norm_sq * own_norm_sq),
        "relative_l2_difference": math.sqrt(difference_norm_sq / reference_norm_sq),
    }


comparisons = [comparison("primary"), comparison("secondary")]
if comparisons[0]["parameters"] != comparisons[1]["parameters"]:
    raise AssertionError("Existing seeds disagree on architecture size")

receipt = {
    "status": "PASS_NEW_COMPATIBLE_CHECKPOINT",
    "checkpoints": {
        name: {
            "path": str(matches[name][0]),
            "sha256": SHAS[name],
            "bytes": matches[name][0].stat().st_size,
            "wrappers": wrappers[name],
        }
        for name in SHAS
    },
    "comparisons": comparisons,
    "promotion_allowed": False,
    "inference_allowed_by_this_receipt": False,
    "input_inventory": inventory,
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
