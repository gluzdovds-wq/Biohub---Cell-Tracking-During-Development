"""Size a deterministic validation pilot using metadata and existing run logs.

Downloads only the Kaggle file inventory, never image chunks or competition data.
The proposed pilot is a screening set, not a new unseen-embryo test population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = {
    "44b6": ("exp050_kaggle_v1", "biohub-exp050-weak-tie-break-44b6.log"),
    "6bba": ("exp051_kaggle_v1", "biohub-exp051-weak-tie-break-6bba.log"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/validation_budget_20260827.json")
    args = parser.parse_args()
    result = {"date": "2026-08-27", "images_downloaded": False, "folds": {}}
    for embryo, (directory, log_name) in FOLDS.items():
        base = ROOT / "outputs" / directory
        payload = json.loads((base / f"loeo_{embryo}_weak_tiebreak_result.json").read_text())
        events = json.loads((base / log_name).read_text())
        seconds = max(float(row["time"]) for row in events)
        names = payload["audit_movies"]
        ordered = sorted(names, key=lambda name: hashlib.sha256(f"pilot-20260827:{name}".encode()).hexdigest())
        pilot = ordered[:12]
        all_count = len(names) + len(payload["confirmation_movies"])
        result["folds"][embryo] = {
            "audit_movies": len(names), "evaluated_movies": all_count,
            "observed_wall_seconds": seconds,
            "gpu_devices": 2,
            "pilot_movies": pilot,
            "pilot_wall_seconds_linear_estimate": seconds * len(pilot) / all_count,
            "source_log_sha256": hashlib.sha256((base / log_name).read_bytes()).hexdigest(),
        }
    result["existing_full_run_wall_hours"] = sum(r["observed_wall_seconds"] for r in result["folds"].values()) / 3600
    result["existing_full_run_device_gpu_hours"] = 2 * result["existing_full_run_wall_hours"]
    result["pilot_wall_hours_linear_estimate"] = sum(r["pilot_wall_seconds_linear_estimate"] for r in result["folds"].values()) / 3600
    result["limits"] = [
        "Measured runtime is for the old single-seed LOEO detector, not the exact public dual-seed/DeepCenter pipeline.",
        "The 24-movie pilot screens large regressions; it cannot establish a 0.001 private gain.",
        "Bootstrap over movies is conditional on only two embryos, not an uncertainty interval for unseen private embryos.",
        "Kaggle quota billing units must not be inferred from physical GPU-device hours.",
    ]
    if args.inventory:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        cached = ROOT / "outputs/research/frontier_20260827/competition_inventory.json"
        if cached.exists():
            files = json.loads(cached.read_text())
        else:
            partial = cached.with_suffix(".partial.json")
            checkpoint = json.loads(partial.read_text()) if partial.exists() else {}
            files = checkpoint.get("files", [])
            token = checkpoint.get("next_page_token")
            for page in range(150):
                try:
                    response = api.competition_list_files(
                        "biohub-cell-tracking-during-development", page_token=token, page_size=200,
                    )
                except Exception:
                    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    raise
                files.extend({"name": f.name, "bytes": f.total_bytes} for f in response.files)
                token = response.next_page_token
                partial.write_text(json.dumps({"files": files, "next_page_token": token}), encoding="utf-8")
                if page % 10 == 0 or not token:
                    print(f"Inventory page {page + 1}: {len(files)} file records, no images downloaded", flush=True)
                if not token:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Inventory pagination exceeded limit")
            cached.write_text(json.dumps(files), encoding="utf-8")
        by_movie = defaultdict(int)
        by_split = defaultdict(int)
        for item in files:
            parts = Path(item["name"]).parts
            by_split[parts[0]] += item["bytes"]
            if len(parts) > 1 and parts[0] == "train":
                by_movie[parts[1].split(".")[0]] += item["bytes"]
        result["inventory"] = {"file_count": len(files), "bytes_by_split": dict(by_split)}
        pilot_bytes = 0
        for fold in result["folds"].values():
            sizes = {name: by_movie[Path(name).stem] for name in fold["pilot_movies"]}
            if any(size <= 0 for size in sizes.values()):
                raise ValueError(f"Missing movie sizes: {sizes}")
            fold["pilot_movie_bytes"] = sizes
            pilot_bytes += sum(sizes.values())
        result["pilot_image_and_label_bytes"] = pilot_bytes
        result["pilot_image_and_label_GiB"] = pilot_bytes / 1024**3
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
