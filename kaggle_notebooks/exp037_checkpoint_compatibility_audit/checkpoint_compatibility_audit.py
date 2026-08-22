"""Fail-closed provenance and state-shape audit for the public own-seed checkpoint."""

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
RECEIPT = WORK / "exp037_checkpoint_audit.json"

OWN_SHA = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
BASE_SHA = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(expected_sha: str) -> tuple[Path, list[dict[str, object]]]:
    observed = []
    matches = []
    for path in sorted([*INPUT.rglob("*.pth"), *INPUT.rglob("*.pt")]):
        digest = sha256(path)
        observed.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
        if digest == expected_sha:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            {"expected_sha": expected_sha, "matches": list(map(str, matches)), "observed": observed}
        )
    return matches[0], observed


def tensor_state(path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    wrappers = []
    while isinstance(payload, dict):
        tensor_values = [value for value in payload.values() if isinstance(value, torch.Tensor)]
        if tensor_values and len(tensor_values) == len(payload):
            break
        selected = None
        for key in ("model_state", "state_dict", "model"):
            if isinstance(payload.get(key), dict):
                selected = key
                break
        if selected is None:
            raise TypeError({"path": str(path), "payload_keys": list(payload)[:50]})
        wrappers.append(selected)
        payload = payload[selected]
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"No tensor state found in {path}")
    state = {str(key): value.detach().cpu() for key, value in payload.items()}
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"Non-tensor state entry in {path}")
    return state, wrappers


own_path, observed = locate(OWN_SHA)
base_path, observed_again = locate(BASE_SHA)
if observed != observed_again:
    raise AssertionError("Input inventory changed during hashing")
own, own_wrappers = tensor_state(own_path)
base, base_wrappers = tensor_state(base_path)

own_keys = set(own)
base_keys = set(base)
missing = sorted(base_keys - own_keys)
unexpected = sorted(own_keys - base_keys)
if missing or unexpected:
    raise AssertionError({"missing": missing, "unexpected": unexpected})

shape_mismatches = []
dtype_mismatches = []
nonfinite = []
exact_equal_keys = []
parameters = 0
dot = 0.0
base_norm_sq = 0.0
own_norm_sq = 0.0
difference_norm_sq = 0.0
for key in sorted(base_keys):
    base_tensor = base[key]
    own_tensor = own[key]
    if tuple(base_tensor.shape) != tuple(own_tensor.shape):
        shape_mismatches.append(
            {"key": key, "base": list(base_tensor.shape), "own": list(own_tensor.shape)}
        )
        continue
    if base_tensor.dtype != own_tensor.dtype:
        dtype_mismatches.append(
            {"key": key, "base": str(base_tensor.dtype), "own": str(own_tensor.dtype)}
        )
    if not torch.isfinite(base_tensor).all() or not torch.isfinite(own_tensor).all():
        nonfinite.append(key)
        continue
    if torch.equal(base_tensor, own_tensor):
        exact_equal_keys.append(key)
    base_flat = base_tensor.to(torch.float64).reshape(-1)
    own_flat = own_tensor.to(torch.float64).reshape(-1)
    parameters += base_flat.numel()
    dot += float(torch.dot(base_flat, own_flat))
    base_norm_sq += float(torch.dot(base_flat, base_flat))
    own_norm_sq += float(torch.dot(own_flat, own_flat))
    delta = base_flat - own_flat
    difference_norm_sq += float(torch.dot(delta, delta))

if shape_mismatches or dtype_mismatches or nonfinite:
    raise AssertionError(
        {
            "shape_mismatches": shape_mismatches,
            "dtype_mismatches": dtype_mismatches,
            "nonfinite": nonfinite,
        }
    )
if parameters <= 0 or base_norm_sq <= 0 or own_norm_sq <= 0:
    raise AssertionError("Degenerate parameter state")
if len(exact_equal_keys) == len(base_keys) or difference_norm_sq == 0:
    raise AssertionError("Own-seed checkpoint is an exact parameter duplicate")

receipt = {
    "status": "PASS_EXACT_ARCHITECTURE_NON_DUPLICATE",
    "own_path": str(own_path),
    "own_sha256": OWN_SHA,
    "own_bytes": own_path.stat().st_size,
    "base_path": str(base_path),
    "base_sha256": BASE_SHA,
    "base_bytes": base_path.stat().st_size,
    "own_wrappers": own_wrappers,
    "base_wrappers": base_wrappers,
    "state_keys": len(base_keys),
    "parameters": parameters,
    "shape_mismatches": 0,
    "dtype_mismatches": 0,
    "nonfinite_keys": 0,
    "exact_equal_keys": len(exact_equal_keys),
    "different_keys": len(base_keys) - len(exact_equal_keys),
    "global_cosine_similarity": dot / math.sqrt(base_norm_sq * own_norm_sq),
    "relative_l2_difference": math.sqrt(difference_norm_sq / base_norm_sq),
    "promotion_allowed": False,
    "input_inventory": observed,
}
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2), flush=True)
