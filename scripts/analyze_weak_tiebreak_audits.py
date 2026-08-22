"""Fail-closed reciprocal analysis for EXP050/051 and their amortized mechanism arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DIVISION_WEIGHT = 0.1
H050_ARMS = ("registered_hungarian", "registered_weak_hungarian")
MECHANISM_MAP = {
    "greedy_prune_4_2": "physical_prune_4_2",
    "greedy_prune_7_4": "physical_prune_7_4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarise(rows: list[dict]) -> dict:
    if not rows or any(not math.isfinite(float(row["edge_tp"])) for row in rows):
        raise ValueError("all pooled rows must contain finite edge counts")
    count_keys = (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
        "num_pred_nodes",
    )
    totals = {key: sum(int(row[key]) for row in rows) for key in count_keys}
    weights = [int(row["edge_tp"]) + int(row["edge_fp"]) + int(row["edge_fn"]) for row in rows]
    if not all(math.isfinite(float(row["adj_edge_jaccard"])) for row in rows) or sum(weights) <= 0:
        raise ValueError("all pooled rows must contain a finite adjusted edge Jaccard and positive weight")
    adj = sum(weight * float(row["adj_edge_jaccard"]) for weight, row in zip(weights, rows)) / sum(weights)
    edge_denominator = totals["edge_tp"] + totals["edge_fp"] + totals["edge_fn"]
    division_denominator = totals["division_tp"] + totals["division_fp"] + totals["division_fn"]
    edge = totals["edge_tp"] / edge_denominator
    division = totals["division_tp"] / division_denominator if division_denominator else float("nan")
    score = adj + DIVISION_WEIGHT * division if division_denominator else adj
    return {
        "n": len(rows),
        **totals,
        "edge_jaccard": edge,
        "adj_edge_jaccard": adj,
        "division_jaccard": division,
        "score": score,
    }


def metric_delta(candidate: dict, base: dict) -> dict:
    return {
        key: float(candidate[key]) - float(base[key])
        for key in ("score", "adj_edge_jaccard", "edge_jaccard")
    } | {
        key: int(candidate[key]) - int(base[key])
        for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    }


def validate_result(payload: dict, expected_embryo: str) -> None:
    required = {
        "status": "paired_weak_tiebreak_audit_complete",
        "hypothesis": "H050",
        "holdout_embryo": expected_embryo,
        "selected_policy": "registered_hungarian",
        "comparison_arms": list(H050_ARMS),
        "weak_probability_weight": 0.1,
        "motion_probability_weight": 0.9,
        "registered_residual_gate_um": 7.0,
        "registered_motion_scale_um": 3.0,
        "missing_learned_probability": 0.0,
        "candidate_requirement": "registered_residual_below_7um_only",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError({"embryo": expected_embryo, "field": key, "expected": expected, "actual": payload.get(key)})
    if payload.get("selected_threshold") not in (0.95, 0.97, 0.985, 0.99, 0.995):
        raise ValueError("unexpected selected threshold")
    if not payload.get("weights_sha256") or not payload.get("parent_contract_sha256") or not payload.get("upstream_selection_sha256"):
        raise ValueError("missing provenance SHA")
    expected_audit_count = 63 if expected_embryo == "44b6" else 120
    if len(payload.get("confirmation_movies", [])) != 4 or len(payload.get("audit_movies", [])) != expected_audit_count:
        raise ValueError("split-size drift")
    if set(payload["audit_summary_by_arm"]) != set(H050_ARMS):
        raise ValueError("H050 arm drift")
    if set(payload["audit_greedy_physical_summary_by_arm"]) != {"greedy_base", *MECHANISM_MAP}:
        raise ValueError("physical mechanism arm drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_44b6", type=Path)
    parser.add_argument("result_6bba", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {"44b6": args.result_44b6, "6bba": args.result_6bba}
    payloads = {embryo: json.loads(path.read_text()) for embryo, path in paths.items()}
    for embryo, payload in payloads.items():
        validate_result(payload, embryo)

    fold_deltas = {
        embryo: payload["audit_delta_weak_minus_motion"] for embryo, payload in payloads.items()
    }
    pooled_h050 = {
        arm: summarise(
            [
                row
                for payload in payloads.values()
                for row in payload["audit_per_movie_by_arm"][arm]
            ]
        )
        for arm in H050_ARMS
    }
    pooled_h050_delta = metric_delta(
        pooled_h050["registered_weak_hungarian"], pooled_h050["registered_hungarian"]
    )
    h050_promote = (
        all(float(delta["score"]) >= 0.0 for delta in fold_deltas.values())
        and pooled_h050_delta["score"] > 0.0
    )

    mechanism = {}
    for candidate_arm, telemetry_arm in MECHANISM_MAP.items():
        fold_rows = {}
        supportive_each_fold = True
        for embryo, payload in payloads.items():
            base = payload["audit_greedy_physical_summary_by_arm"]["greedy_base"]
            candidate = payload["audit_greedy_physical_summary_by_arm"][candidate_arm]
            delta = metric_delta(candidate, base)
            prunes = sum(
                int(movie["arms"][telemetry_arm]["accepted_prunes"])
                for movie in payload["audit_greedy_physical_telemetry"]
            )
            fold_rows[embryo] = {"accepted_prunes": prunes, "delta": delta}
            supportive_each_fold = supportive_each_fold and (
                prunes > 0
                and delta["score"] >= 0.0
                and delta["adj_edge_jaccard"] >= 0.0
                and delta["division_tp"] >= 0
            )
        pooled_base = summarise(
            [
                row
                for payload in payloads.values()
                for row in payload["audit_greedy_physical_per_movie_by_arm"]["greedy_base"]
            ]
        )
        pooled_candidate = summarise(
            [
                row
                for payload in payloads.values()
                for row in payload["audit_greedy_physical_per_movie_by_arm"][candidate_arm]
            ]
        )
        pooled_delta = metric_delta(pooled_candidate, pooled_base)
        mechanism[candidate_arm] = {
            "folds": fold_rows,
            "pooled_base": pooled_base,
            "pooled_candidate": pooled_candidate,
            "pooled_delta": pooled_delta,
            "mechanism_supportive": supportive_each_fold and pooled_delta["score"] > 0.0,
            "submission_authority": False,
        }

    receipt = {
        "status": "PASS_RECIPROCAL_H050_ANALYSIS",
        "inputs": {
            embryo: {"path": str(path), "sha256": sha256(path)} for embryo, path in paths.items()
        },
        "fold_h050_deltas": fold_deltas,
        "pooled_h050_summary_by_arm": pooled_h050,
        "pooled_h050_delta": pooled_h050_delta,
        "h050_decision": "promote" if h050_promote else "reject",
        "physical_mechanism": mechanism,
        "physical_mechanism_scope": "geometry-only on greedy graphs; no EXP005/008 donor consensus",
        "physical_mechanism_submission_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
