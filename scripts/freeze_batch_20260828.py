"""Freeze five reviewed public sources + verified artifacts, without submitting.

Only creates reproducibility snapshots/receipts from existing downloaded files.
No source execution, model training, leaderboard writes or dataset image fetches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import ApiGetKernelRequest, KaggleApi

from audit_submission_fast import audit

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "outputs/research/frontier_20260828"
FROZEN = ROOT / "research/frozen_sources_20260828"
CASES = [
    ("EXP078", "rishabhr0y/biohub-934-sdw70", 1, 345468238, 0.928,
     "SDW70 detection weight 0.70", "EXP-078 source-attributed SDW70 det-weight070 v1",
     "Exact reviewed EXP073 source except secondary detection blend 0.60 -> 0.70. UI score is 0.928, not the 0.934 suggested by the title."),
    ("EXP079", "flexonafft/biohub-harmonic-fusion", 22, 345465485, 0.928,
     "DeepCenter epoch2 plus safe-division veto", "EXP-079 source-attributed Flex best-epoch2 division-veto v22",
     "Compared with downloaded Flex v17: restore epoch-2 best.pt and activate the safe-division DeepCenter veto. Gap3/6.5um family is inherited. Added bidirectional env flags alone do not establish an active reverse-logit implementation."),
    ("EXP080", "arnav170/biohub-sdw75", 1, 345508997, None,
     "SDW75 detection weight 0.75", "EXP-080 source-attributed SDW75 detector-mixture extrapolation v1",
     "Exact reviewed EXP073 source except secondary detection blend 0.60 -> 0.75. No author score was displayed; exploratory continuation beyond SDW70."),
    ("EXP081", "arnav170/biohub-vel10", 1, 345503800, None,
     "Constant-velocity relinking", "EXP-081 source-attributed full-velocity relink v1",
     "Set motion velocity weight to 1.0 instead of default 0.5 on the 0.475 detector blend. Not a one-variable comparison against EXP073, whose blend is 0.60. No author score displayed."),
    ("EXP082", "arnav170/biohub-mtl8", 1, 345255516, 0.923,
     "Minimum track length eight", "EXP-082 source-attributed track-length08 sensitivity control v1",
     "Minimum retained component length 8 rather than 6 on the 0.475 detector blend; division components remain protected. Lower-priority sensitivity reference, not an evidence-backed record candidate or private-stability claim."),
]


def dump(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    manifest_path = ROOT / "reports/submission_batch_20260828.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(row.get("submission_ref") for row in existing.get("candidates", [])):
            raise RuntimeError("Batch already registered; refusing to overwrite submission receipts")
    api = KaggleApi()
    api.authenticate()
    FROZEN.mkdir(parents=True, exist_ok=True)
    candidates = []
    for exp, ref, version, version_id, author_lb, mechanism, description, review in CASES:
        owner, slug = ref.split("/")
        folder = RESEARCH / f"{owner}__{slug}__{version}"
        source = folder / f"{slug}.ipynb"
        receipt = json.loads((folder / "source_receipt.json").read_text(encoding="utf-8"))
        assert receipt["version"] == version
        assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt["source_sha256"]
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.get_kernel(request)
        assert response.metadata.current_version_number == version
        assert response.blob.source.replace("\r\n", "\n") == source.read_text(encoding="utf-8")
        assert not response.metadata.enable_internet
        notebook = json.loads(source.read_text(encoding="utf-8"))
        code_cells = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                compile(code, f"{ref}:cell{index}", "exec")
                code_cells += 1
        artifact = folder / "output/submission.csv"
        structural = audit(artifact, expected_datasets=4)
        # Mechanical exact-byte archival of already inspected external source.
        frozen = FROZEN / f"{exp.lower()}_{slug}_v{version}.ipynb"
        if frozen.exists() and frozen.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Refusing to overwrite different frozen source: {frozen}")
        if not frozen.exists():
            frozen.write_bytes(source.read_bytes())
        row = {
            "experiment": exp, "kernel": ref, "version": version,
            "script_version_id": version_id, "description": description,
            "source_path": frozen.relative_to(ROOT).as_posix(),
            "source_sha256": receipt["source_sha256"],
            "artifact_path": artifact.relative_to(ROOT).as_posix(),
            "artifact_sha256": structural["sha256"],
            "artifact_version_verified": "API latest matches before and after download; source equals frozen source; UI version checked",
            "source_review": "PASS_FULL_INFERENCE_NO_METRIC_HACK",
            "mechanism": mechanism, "review_notes": review,
            "attribution": "public-source reproduction; not a newly trained model of ours",
            "author_lb": author_lb, "honest_oof": None,
            "notebook_code_cells_compiled": code_cells,
            "graph_audit": structural,
            "submission_ref": None, "account_status": "NOT_SUBMITTED",
        }
        candidates.append(row)
    assert len({r["artifact_sha256"] for r in candidates}) == 5
    assert len({r["description"] for r in candidates}) == 5
    manifest = {
        "date": "2026-08-28", "authorization": "User explicitly authorized the next five submissions",
        "max_authorized_submissions": 5, "known_best_before_batch": 0.927,
        "source_policy": "Full-code submissions of pinned public versions; no CSV-output wrappers",
        "gpu_training_launched": False,
        "candidates": candidates,
        "excluded": [
            {"kernel": "anhadmahajan06/biohub-track-your-cells-development", "version": 22,
             "reason": "Executable source is unchanged from already submitted v21."},
            {"kernel": "evgendvorkin/biohub-0-927-lb", "version": 13,
             "reason": "Latest is 0.923, not best v11 0.927; also carries a private notebook source."},
            {"kernel": "lucifer19/biohub-black-cat-b-lineage-recall-guard", "version": 1,
             "reason": "Verified 0.884, substantially below the frontier."},
            {"kernel": "ericwang03/biohub-daily-probe-lane-5", "version": 14,
             "reason": "No verified frontier result; displayed best 0.911 belongs to historical v2."},
            {"kernel": "mtoshidesu/test-biohub-harmonic-fusion", "version": 1,
             "reason": "Despite title, replaces learned model with crude threshold/nearest-neighbor baseline."},
        ],
    }
    dump(manifest_path, manifest)
    print(json.dumps({"status": "FROZEN_NOT_SUBMITTED", "candidates": [r["experiment"] for r in candidates]}, indent=2))


if __name__ == "__main__":
    main()
