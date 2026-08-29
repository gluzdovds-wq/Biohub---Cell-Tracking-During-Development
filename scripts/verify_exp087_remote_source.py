"""Verify that Kaggle's normalized EXP087 source has only the intended code change.

Kaggle may rewrite notebook metadata and append empty cells, so byte equality of
the uploaded and API-returned ipynb is not a useful guard. Non-empty code cells
must still match the locally generated controlled fork exactly and in order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import ApiGetKernelRequest, KaggleApi

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "kaggle_notebooks/exp087_sdw90/sdw90.ipynb"
KERNEL = "dmitriigluzdov/biohub-exp087-controlled-sdw90"
EXPECTED_LOCAL_SHA256 = "1105cc968c3e410807c6638e198021ca3cd42a09fc22cd6cab900fe797589111"
ANCHOR = 'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.90"'


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def nonempty_code_cells(notebook: dict) -> list[str]:
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).replace("\r\n", "\n")
        if source.strip():
            cells.append(source)
    return cells


def main() -> None:
    local_bytes = LOCAL.read_bytes()
    if sha256_bytes(local_bytes) != EXPECTED_LOCAL_SHA256:
        raise RuntimeError("Local controlled notebook drift")
    local = json.loads(local_bytes)
    local_cells = nonempty_code_cells(local)
    if sum(cell.count(ANCHOR) for cell in local_cells) != 1:
        raise RuntimeError("Local SDW90 anchor is not unique")

    owner, slug = KERNEL.split("/")
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        result = client.kernels.kernels_api_client.get_kernel(request)
    if result.metadata.current_version_number != 1:
        raise RuntimeError("Unexpected remote EXP087 version")
    remote_source = result.blob.source.replace("\r\n", "\n")
    remote = json.loads(remote_source)
    remote_cells = nonempty_code_cells(remote)
    if remote_cells != local_cells:
        first = next(
            (i for i, pair in enumerate(zip(remote_cells, local_cells)) if pair[0] != pair[1]),
            min(len(remote_cells), len(local_cells)),
        )
        raise RuntimeError(
            "Remote executable source differs from controlled fork: "
            f"remote_cells={len(remote_cells)} local_cells={len(local_cells)} first_mismatch={first}"
        )
    if sum(cell.count(ANCHOR) for cell in remote_cells) != 1:
        raise RuntimeError("Remote SDW90 anchor is not unique")
    print(json.dumps({
        "status": "PASS_REMOTE_EXECUTABLE_CODE_EXACT",
        "kernel": KERNEL,
        "version": result.metadata.current_version_number,
        "nonempty_code_cells": len(remote_cells),
        "local_notebook_sha256": EXPECTED_LOCAL_SHA256,
        "remote_source_sha256": sha256_bytes(remote_source.encode("utf-8")),
        "remote_code_sha256": sha256_bytes("\n\n".join(remote_cells).encode("utf-8")),
        "controlled_anchor_count": 1,
    }, indent=2))


if __name__ == "__main__":
    main()
