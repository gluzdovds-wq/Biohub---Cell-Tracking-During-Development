"""Build reciprocal honest-OOF linker comparisons with reusable candidate caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    {
        "experiment": "EXP063",
        "embryo": "44b6",
        "source_dir": "exp050_weak_tiebreak_44b6",
        "source_file": "weak_tiebreak_44b6.py",
        "source_sha256": "7c24821a12baf64f82df5ce0f59d2b9f24ccdb2d0f7e62819540c897577b76d2",
    },
    {
        "experiment": "EXP064",
        "embryo": "6bba",
        "source_dir": "exp051_weak_tiebreak_6bba",
        "source_file": "weak_tiebreak_6bba.py",
        "source_sha256": "37492dec18d8e2cbbb171171034af9b4db7bd58ff350d9a9aa81b97db471ca50",
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CACHE_SOURCE = r'''
CACHE_DIR = WORK / f"oof_candidate_cache_{HOLDOUT_EMBRYO}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MANIFEST = []


def save_candidate_cache(name, coords, edges):
    coordinate_array = np.asarray(coords, dtype=np.float32)
    edge_array = np.asarray(edges, dtype=np.float64)
    if edge_array.size:
        edge_array = edge_array.reshape((-1, 4))
    else:
        edge_array = np.empty((0, 4), dtype=np.float64)
    path = CACHE_DIR / f"{Path(name).stem}.npz"
    np.savez_compressed(
        path,
        coords=coordinate_array,
        edge_source=edge_array[:, 0].astype(np.int64),
        edge_target=edge_array[:, 1].astype(np.int64),
        edge_probability=edge_array[:, 2].astype(np.float32),
        edge_distance=edge_array[:, 3].astype(np.float32),
    )
    CACHE_MANIFEST.append(
        {
            "dataset": name,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "nodes": int(len(coordinate_array)),
            "candidate_edges": int(len(edge_array)),
            "bytes": path.stat().st_size,
        }
    )
'''


for config in CONFIGS:
    source_dir = ROOT / "kaggle_notebooks" / config["source_dir"]
    source_path = source_dir / config["source_file"]
    if sha256(source_path) != config["source_sha256"]:
        raise RuntimeError(f"source drift: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    arms_anchor = 'ARMS = ("registered_hungarian", "registered_weak_hungarian")'
    expanded_arms = '''ARMS = (
    "registered_hungarian",
    "registered_weak_hungarian",
    "registered_prob_hungarian",
    "ilp_public",
    "ilp_support",
)'''
    if source.count(arms_anchor) != 1:
        raise AssertionError("ARMS anchor drift")
    source = source.replace(arms_anchor, expanded_arms)
    evaluate_anchor = "def evaluate_pair(names):"
    if source.count(evaluate_anchor) != 1:
        raise AssertionError("evaluate anchor drift")
    source = source.replace(evaluate_anchor, CACHE_SOURCE + "\n\n" + evaluate_anchor)
    infer_anchor = "        coords, edges = infer_candidates(name, selected_threshold)\n"
    if source.count(infer_anchor) != 1:
        raise AssertionError("inference anchor drift")
    source = source.replace(
        infer_anchor,
        infer_anchor + "        save_candidate_cache(name, coords, edges)\n",
    )
    source = source.replace('"status": "paired_weak_tiebreak_audit_complete",',
                            '"status": "paired_oof_model_comparison_complete",')
    source = source.replace('"hypothesis": "H050",', '"hypothesis": "H063",')
    source = source.replace(
        '"comparison_arms": ARMS,',
        '"comparison_arms": ARMS,\n    "candidate_cache_manifest": CACHE_MANIFEST,\n'
        '    "candidate_cache_scope": "embryo-disjoint frozen detections and learned edge candidates; safe for CPU-only downstream linker comparisons",',
    )
    source = source.replace(
        'f"loeo_{HOLDOUT_EMBRYO}_weak_tiebreak_result.json"',
        'f"loeo_{HOLDOUT_EMBRYO}_model_comparison_result.json"',
    )

    output_dir = ROOT / "kaggle_notebooks" / f"exp{config['experiment'][3:]}_oof_comparison_{config['embryo']}"
    output_file = output_dir / f"oof_comparison_{config['embryo']}.py"
    metadata = json.loads((source_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    title = f"Biohub {config['experiment']} OOF Comparison {config['embryo']}"
    metadata.update(
        {
            "id": f"dmitriigluzdov/biohub-{config['experiment'].lower()}-oof-comparison-{config['embryo']}",
            "title": title,
            "code_file": output_file.name,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file.write_text(source, encoding="utf-8")
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "status": f"PASS_{config['experiment']}_BUILD",
        "experiment": config["experiment"],
        "holdout_embryo": config["embryo"],
        "source": str(source_path.relative_to(ROOT)),
        "source_sha256": config["source_sha256"],
        "output": str(output_file.relative_to(ROOT)),
        "output_sha256": sha256(output_file),
        "comparison_arms": [
            "registered_hungarian", "registered_weak_hungarian",
            "registered_prob_hungarian", "ilp_public", "ilp_support",
            "greedy_base", "greedy_prune_4_2", "greedy_prune_7_4",
        ],
        "candidate_cache_export": True,
        "labels_used_only_after_each_arm_graph_is_frozen": True,
    }
    (output_dir / "build_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
