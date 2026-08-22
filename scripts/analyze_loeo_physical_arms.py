"""Fail-closed paired analysis of frozen physical-prune arms on both LOEO audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_ARMS = {
    "selected_base": None,
    "physical_prune_4_2": [4.0, 2.0],
    "physical_prune_7_4": [7.0, 4.0],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value}")
    return number


def analyse_fold(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "audit_complete":
        raise ValueError({"path": str(path), "status": payload.get("status")})
    if payload.get("postselection_arm_status") != "frozen_before_confirmation_and_untouched_audit":
        raise ValueError({"path": str(path), "postselection_arm_status": payload.get("postselection_arm_status")})
    if payload.get("postselection_physical_arms") != EXPECTED_ARMS:
        raise ValueError(
            {
                "path": str(path),
                "expected_arms": EXPECTED_ARMS,
                "observed_arms": payload.get("postselection_physical_arms"),
            }
        )
    summaries = payload.get("audit_summary_by_physical_arm")
    per_movie = payload.get("audit_per_movie_by_physical_arm")
    telemetry = payload.get("audit_physical_prune_telemetry")
    if not isinstance(summaries, dict) or set(summaries) != set(EXPECTED_ARMS):
        raise ValueError({"path": str(path), "summary_arms": None if summaries is None else list(summaries)})
    if not isinstance(per_movie, dict) or set(per_movie) != set(EXPECTED_ARMS):
        raise ValueError({"path": str(path), "per_movie_arms": None if per_movie is None else list(per_movie)})
    base = summaries["selected_base"]
    n_movies = int(base["n"])
    if n_movies <= 0 or any(len(per_movie[arm]) != n_movies for arm in EXPECTED_ARMS):
        raise ValueError({"path": str(path), "n_movies": n_movies})

    arms = {}
    for arm in ("physical_prune_4_2", "physical_prune_7_4"):
        summary = summaries[arm]
        accepted_prunes = sum(
            int(movie["arms"][arm]["accepted_prunes"]) for movie in telemetry
        )
        score_delta = finite_number(summary["score"], f"{arm}.score") - finite_number(
            base["score"], "base.score"
        )
        adjusted_edge_delta = finite_number(
            summary["adj_edge_jaccard"], f"{arm}.adj_edge_jaccard"
        ) - finite_number(base["adj_edge_jaccard"], "base.adj_edge_jaccard")
        division_tp_delta = int(summary["division_tp"]) - int(base["division_tp"])
        division_fp_delta = int(summary["division_fp"]) - int(base["division_fp"])
        division_fn_delta = int(summary["division_fn"]) - int(base["division_fn"])
        fold_gate = (
            accepted_prunes > 0
            and score_delta >= 0.0
            and adjusted_edge_delta >= 0.0
            and division_tp_delta >= 0
        )
        arms[arm] = {
            "accepted_prunes": accepted_prunes,
            "base_summary": base,
            "summary": summary,
            "delta_score": score_delta,
            "delta_adj_edge_jaccard": adjusted_edge_delta,
            "delta_division_tp": division_tp_delta,
            "delta_division_fp": division_fp_delta,
            "delta_division_fn": division_fn_delta,
            "fold_gate": fold_gate,
        }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "holdout_embryo": payload.get("holdout_embryo"),
        "audit_movies": n_movies,
        "selected_threshold": payload.get("selected_threshold"),
        "selected_policy": payload.get("selected_policy"),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-44b6", type=Path, required=True)
    parser.add_argument("--fold-6bba", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folds = {
        "44b6": analyse_fold(args.fold_44b6),
        "6bba": analyse_fold(args.fold_6bba),
    }
    if folds["44b6"]["holdout_embryo"] != "44b6" or folds["6bba"]["holdout_embryo"] != "6bba":
        raise ValueError({"fold_embryos": {name: fold["holdout_embryo"] for name, fold in folds.items()}})
    gates = {
        arm: all(fold["arms"][arm]["fold_gate"] for fold in folds.values())
        for arm in ("physical_prune_4_2", "physical_prune_7_4")
    }
    result = {
        "status": "PASS_LOEO_PHYSICAL_ANALYSIS" if any(gates.values()) else "HOLD_LOEO_PHYSICAL_ANALYSIS",
        "scope": "physical mechanism only; does not reproduce EXP005/008 donor consensus",
        "folds": folds,
        "mechanism_gate_by_arm": gates,
        "broad_mechanism_gate": gates["physical_prune_4_2"],
        "strict_mechanism_gate": gates["physical_prune_7_4"],
        "submission_allowed_by_this_receipt": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
