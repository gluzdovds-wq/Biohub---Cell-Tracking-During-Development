"""Freeze and audit the August 29 frontier before any submissions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import ApiGetKernelRequest, KaggleApi

from audit_submission_fast import audit

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "outputs/research/frontier_20260829"
FROZEN = ROOT / "research/frozen_sources_20260829"
MANIFEST = ROOT / "reports/submission_batch_20260829.json"
PUBLIC_CASES = [
    ("EXP083", "stephennedumpally/pls-upvote-share-higher-scoring-ideas", 1, 344481175, 0.931,
     "wider division geometry plus reverse weight 0.15", "EXP-083 source-attributed clean Stephen frontier v1"),
    ("EXP084", "rishabhr0y/biohub-938-sdw85", 1, 345526668, 0.929,
     "secondary detector mixture 0.85", "EXP-084 source-attributed SDW85 detector-weight085 v1"),
    ("EXP085", "evgendvorkin/biohub-0-928-lb", 15, 345594775, 0.928,
     "detector mixture 0.70 plus gap3/6.5um", "EXP-085 source-attributed Evgen current 0928 v15"),
    ("EXP086", "anvithpothula/biohub-dual-seed-harmonic-bidirectional-fusion", 1, 345253418, 0.928,
     "detector mixture 0.60 plus wide safe divisions", "EXP-086 source-attributed Anvith bidirectional fusion v1"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_notebook(path, label):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{label}:cell{index}", "exec")
            count += 1
    return count


def nonempty_code_cells(source):
    notebook = json.loads(source)
    return [
        "".join(cell["source"]).replace("\r\n", "\n")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).strip()
    ]


def get_current(api, ref):
    owner, slug = ref.split("/")
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(request)


def main():
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if any(row.get("submission_ref") for row in existing.get("candidates", [])):
            raise RuntimeError("Batch already registered; refusing to overwrite receipts")
    api = KaggleApi()
    api.authenticate()
    FROZEN.mkdir(parents=True, exist_ok=True)
    candidates = []
    for exp, ref, version, version_id, author_lb, mechanism, description in PUBLIC_CASES:
        owner, slug = ref.split("/")
        folder = RESEARCH / f"{owner}__{slug}__{version}"
        source = folder / f"{slug}.ipynb"
        source_receipt = json.loads((folder / "source_receipt.json").read_text(encoding="utf-8"))
        if source_receipt["version"] != version or sha256(source) != source_receipt["source_sha256"]:
            raise RuntimeError(f"Local source receipt mismatch: {ref}")
        current = get_current(api, ref)
        if current.metadata.current_version_number != version:
            raise RuntimeError(f"Remote version drift: {ref}")
        if current.blob.source.replace("\r\n", "\n") != source.read_text(encoding="utf-8"):
            raise RuntimeError(f"Remote source drift: {ref}")
        frozen = FROZEN / f"{exp.lower()}_{slug}_v{version}.ipynb"
        if frozen.exists() and frozen.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Refusing to replace different frozen source: {frozen}")
        if not frozen.exists():
            frozen.write_bytes(source.read_bytes())
        artifact = folder / "output/submission.csv"
        graph = audit(artifact, expected_datasets=4)
        candidates.append({
            "experiment": exp, "kernel": ref, "version": version,
            "script_version_id": version_id, "description": description,
            "source_path": frozen.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(frozen),
            "artifact_path": artifact.relative_to(ROOT).as_posix(),
            "artifact_sha256": graph["sha256"],
            "artifact_version_verified": "API latest/source checked; current output downloaded with submission-only filter",
            "source_review": "PASS_FULL_INFERENCE_NO_METRIC_HACK",
            "mechanism": mechanism, "author_lb": author_lb, "account_lb": None,
            "honest_oof": None,
            "attribution": "public-source reproduction; not a newly trained model of ours",
            "notebook_code_cells_compiled": compile_notebook(frozen, ref),
            "graph_audit": graph, "submission_ref": None, "account_status": "NOT_SUBMITTED",
        })
    own = ROOT / "kaggle_notebooks/exp087_sdw90/sdw90.ipynb"
    own_current = get_current(api, "dmitriigluzdov/biohub-exp087-controlled-sdw90")
    if own_current.metadata.current_version_number != 1:
        raise RuntimeError("Remote EXP087 version drift")
    remote_own_source = own_current.blob.source.replace("\r\n", "\n")
    if nonempty_code_cells(remote_own_source) != nonempty_code_cells(own.read_text(encoding="utf-8")):
        raise RuntimeError("Remote EXP087 executable code drift")
    candidates.append({
        "experiment": "EXP087", "kernel": "dmitriigluzdov/biohub-exp087-controlled-sdw90",
        "version": 1, "script_version_id": None,
        "description": "EXP-087 own controlled SDW90 detector-weight090 v1",
        "source_path": own.relative_to(ROOT).as_posix(), "source_sha256": sha256(own),
        "source_validation_mode": "nonempty_notebook_code_cells_exact",
        "remote_normalized_source_sha256": hashlib.sha256(remote_own_source.encode()).hexdigest(),
        "remote_executable_code_sha256": hashlib.sha256(
            "\n\n".join(nonempty_code_cells(remote_own_source)).encode()
        ).hexdigest(),
        "artifact_path": "outputs/exp087_kaggle_v1/submission.csv", "artifact_sha256": "PENDING_RUNTIME",
        "artifact_version_verified": False, "source_review": "PASS_FULL_INFERENCE_NO_METRIC_HACK",
        "mechanism": "our controlled continuation of SDW85: only secondary detection weight 0.85 -> 0.90",
        "author_lb": None, "account_lb": None, "honest_oof": None,
        "attribution": "our controlled parameter fork; public architecture and weights remain attributed",
        "notebook_code_cells_compiled": compile_notebook(own, "EXP087"),
        "graph_audit": None, "submission_ref": None, "account_status": "KERNEL_RUNNING",
    })
    if len({row["description"] for row in candidates}) != 5:
        raise RuntimeError("Duplicate descriptions")
    payload = {
        "date": "2026-08-29", "authorization": "User explicitly authorized the next five submissions",
        "known_best_before_batch": 0.928, "clean_frontier_found": 0.931,
        "source_policy": "Pinned full-code notebooks only; metric hacks and public-CSV wrappers excluded",
        "candidates": candidates,
        "correlation_interpretation": {
            "sdw70_vs_sdw75_physical_node_edge_jaccard": [0.970384, 0.963701],
            "sdw70_vs_flex22_physical_node_edge_jaccard": [0.861317, 0.829558],
            "stephen0931_vs_flex22_physical_node_edge_jaccard": [0.925603, 0.910946],
            "sdw85_vs_sdw75_physical_node_edge_jaccard": [0.951036, 0.940384],
            "conclusion": "All clean 0.928-0.931 models share the same public dual-seed/harmonic family; equal scores are correlated evidence, not independent private-stability confirmation.",
        },
        "excluded": [
            {"kernel": "anhadmahajan06/biohub-track-your-cells-development", "version": 24,
             "reason": "Current score 0.924; best v21 0.927 was already submitted."},
            {"kernel": "jaslee2/divcv-best", "version": 1, "reason": "Clean but displayed score 0.917."},
            {"kernel": "saitejabandaruin/biohub-masterpiece-tracker-version-21", "version": 1,
             "reason": "Current kernel status ERROR."},
        ],
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "experiments": [r["experiment"] for r in candidates]}, indent=2))


if __name__ == "__main__":
    main()
