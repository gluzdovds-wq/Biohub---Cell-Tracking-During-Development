"""Download a public kernel and freeze its API version/source for manual audit.

Never executes notebook code or submits anything. An explicit version must match
the server metadata; do not infer the best-scoring version from a notebook title.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import ApiGetKernelRequest, KaggleApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("refs", nargs="+")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    for ref in args.refs:
        parts = ref.split("/")
        owner, slug = parts[:2]
        wanted_version = int(parts[2]) if len(parts) == 3 else None
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        with api.build_kaggle_client() as client:
            result = client.kernels.kernels_api_client.get_kernel(request)
            if wanted_version is not None and result.metadata.current_version_number != wanted_version:
                request.version_label = str(wanted_version)
                result = client.kernels.kernels_api_client.get_kernel(request)
        meta = result.metadata
        version = meta.current_version_number
        if wanted_version is not None and version != wanted_version:
            raise ValueError(f"Version mismatch: requested {wanted_version}, got {version}")
        target = args.root / f"{owner}__{slug}__{version}"
        target.mkdir(parents=True, exist_ok=True)
        source = result.blob.source
        extension = ".ipynb" if meta.kernel_type == "notebook" else ".py"
        original = target / (slug + extension)
        # These are generated research snapshots, not edits to the original code.
        original.write_text(source, encoding="utf-8")
        cells = json.loads(source)["cells"] if extension == ".ipynb" else []
        code = "\n\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
        ) if cells else source
        (target / "extracted_code.py").write_text(code, encoding="utf-8")
        record = {
            "ref": ref, "version": version,
            "last_run_time": str(meta.last_run_time),
            "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "source_path": str(original),
            "status": str(api.kernels_status(ref).status),
            "internet": meta.enable_internet,
            "dataset_sources": list(meta.dataset_data_sources),
            "kernel_sources": list(meta.kernel_data_sources),
        }
        (target / "source_receipt.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
