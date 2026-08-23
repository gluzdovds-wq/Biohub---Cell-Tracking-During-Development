"""Quantify Biohub OOF stability without leaking embryo identity across folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from analyze_weak_tiebreak_audits import summarise

REGISTERED = "registered_hungarian"
WEAK = "registered_weak_hungarian"
GREEDY = "greedy_base"
SEED = 20260823


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "q025": quantile(values, 0.025),
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "q75": quantile(values, 0.75),
        "q975": quantile(values, 0.975),
        "min": min(values),
        "max": max(values),
    }


def row_maps(payload: dict, arm: str, greedy: bool = False) -> dict[str, dict]:
    key = "audit_greedy_physical_per_movie_by_arm" if greedy else "audit_per_movie_by_arm"
    rows = payload[key][arm]
    mapped = {row["dataset"]: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"duplicate dataset in {arm}")
    return mapped


def paired_bootstrap(
    groups: dict[str, dict[str, dict[str, dict]]],
    candidate: str,
    baseline: str | None,
    iterations: int,
    rng: random.Random,
) -> list[float]:
    values: list[float] = []
    for _ in range(iterations):
        candidate_rows: list[dict] = []
        baseline_rows: list[dict] = []
        for arms in groups.values():
            names = sorted(arms[candidate])
            if baseline is not None and names != sorted(arms[baseline]):
                raise ValueError("paired dataset identities differ")
            sampled = [rng.choice(names) for _ in names]
            candidate_rows.extend(arms[candidate][name] for name in sampled)
            if baseline is not None:
                baseline_rows.extend(arms[baseline][name] for name in sampled)
        candidate_score = summarise(candidate_rows)["score"]
        if baseline is None:
            values.append(candidate_score)
        else:
            values.append(candidate_score - summarise(baseline_rows)["score"])
    return values


def sample_size_bootstrap(
    rows: list[dict], sample_size: int, iterations: int, rng: random.Random
) -> list[float]:
    return [
        summarise([rng.choice(rows) for _ in range(sample_size)])["score"]
        for _ in range(iterations)
    ]


def repeated_random_kfold(
    rows: list[dict], folds: int, repeats: int, rng: random.Random
) -> list[float]:
    scores: list[float] = []
    for _ in range(repeats):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        for fold in range(folds):
            selected = shuffled[fold::folds]
            scores.append(summarise(selected)["score"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_44b6", type=Path)
    parser.add_argument("result_6bba", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()

    input_paths = {"44b6": args.result_44b6, "6bba": args.result_6bba}
    payloads = {name: json.loads(path.read_text()) for name, path in input_paths.items()}
    if any(payload.get("holdout_embryo") != embryo for embryo, payload in payloads.items()):
        raise ValueError("holdout embryo provenance mismatch")

    groups: dict[str, dict[str, dict[str, dict]]] = {}
    for embryo, payload in payloads.items():
        groups[embryo] = {
            REGISTERED: row_maps(payload, REGISTERED),
            WEAK: row_maps(payload, WEAK),
            GREEDY: row_maps(payload, GREEDY, greedy=True),
        }
        identities = [set(rows) for rows in groups[embryo].values()]
        if identities[0] != identities[1] or identities[0] != identities[2]:
            raise ValueError(f"paired identities drift for {embryo}")

    all_registered = [
        row for arms in groups.values() for row in arms[REGISTERED].values()
    ]
    all_greedy = [row for arms in groups.values() for row in arms[GREEDY].values()]
    all_weak = [row for arms in groups.values() for row in arms[WEAK].values()]
    rng = random.Random(SEED)

    embryo_folds = {}
    for embryo, arms in groups.items():
        registered_summary = summarise(list(arms[REGISTERED].values()))
        greedy_summary = summarise(list(arms[GREEDY].values()))
        weak_summary = summarise(list(arms[WEAK].values()))
        fold_rng = random.Random(SEED + sum(ord(char) for char in embryo))
        fold_boot = sample_size_bootstrap(
            list(arms[REGISTERED].values()), len(arms[REGISTERED]), args.iterations, fold_rng
        )
        embryo_folds[embryo] = {
            "movies": len(arms[REGISTERED]),
            "registered_score": registered_summary["score"],
            "greedy_score": greedy_summary["score"],
            "weak_score": weak_summary["score"],
            "registered_minus_greedy": registered_summary["score"] - greedy_summary["score"],
            "weak_minus_registered": weak_summary["score"] - registered_summary["score"],
            "registered_movie_bootstrap": distribution(fold_boot),
        }

    stratified_absolute = paired_bootstrap(
        groups, REGISTERED, None, args.iterations, rng
    )
    registered_minus_greedy = paired_bootstrap(
        groups, REGISTERED, GREEDY, args.iterations, rng
    )
    weak_minus_registered = paired_bootstrap(
        groups, WEAK, REGISTERED, args.iterations, rng
    )
    public_four = sample_size_bootstrap(all_registered, 4, args.iterations, rng)
    private_130 = sample_size_bootstrap(all_registered, 130, args.iterations, rng)
    random_folds = repeated_random_kfold(all_registered, folds=5, repeats=200, rng=rng)

    pooled_registered = summarise(all_registered)
    pooled_greedy = summarise(all_greedy)
    pooled_weak = summarise(all_weak)
    fold_scores = [fold["registered_score"] for fold in embryo_folds.values()]
    fold_gap = max(fold_scores) - min(fold_scores)
    per_movie = []
    for embryo, arms in groups.items():
        for dataset in sorted(arms[REGISTERED]):
            registered_row = arms[REGISTERED][dataset]
            greedy_row = arms[GREEDY][dataset]
            weak_row = arms[WEAK][dataset]
            registered_score = summarise([registered_row])["score"]
            greedy_score = summarise([greedy_row])["score"]
            weak_score = summarise([weak_row])["score"]
            per_movie.append(
                {
                    "embryo": embryo,
                    "dataset": dataset,
                    "registered_score": registered_score,
                    "greedy_score": greedy_score,
                    "weak_score": weak_score,
                    "registered_minus_greedy": registered_score - greedy_score,
                    "weak_minus_registered": weak_score - registered_score,
                    "registered_edge_tp": int(registered_row["edge_tp"]),
                    "registered_edge_fp": int(registered_row["edge_fp"]),
                    "registered_edge_fn": int(registered_row["edge_fn"]),
                    "registered_node_recall": float(registered_row["node_recall"]),
                }
            )

    receipt = {
        "status": "PASS_OOF_STABILITY_ANALYSIS",
        "seed": SEED,
        "bootstrap_iterations": args.iterations,
        "inputs": {
            embryo: {"path": str(path), "sha256": sha256(path)}
            for embryo, path in input_paths.items()
        },
        "validation_unit": "movie grouped inside leave-one-embryo-out holdouts",
        "embryo_folds": embryo_folds,
        "between_embryo": {
            "fold_score_gap": fold_gap,
            "fold_mean": statistics.fmean(fold_scores),
            "fold_stdev": statistics.stdev(fold_scores),
            "caution": "Only two embryo domains exist in train, so this gap is a stress test, not a precise population confidence interval.",
        },
        "pooled": {
            "movies": len(all_registered),
            "registered_score": pooled_registered["score"],
            "greedy_score": pooled_greedy["score"],
            "weak_score": pooled_weak["score"],
            "registered_minus_greedy": pooled_registered["score"] - pooled_greedy["score"],
            "weak_minus_registered": pooled_weak["score"] - pooled_registered["score"],
            "registered_movie_scores": distribution(
                [float(row["adj_edge_jaccard"]) for row in all_registered]
            ),
        },
        "per_movie": per_movie,
        "bootstrap": {
            "stratified_registered_absolute": distribution(stratified_absolute),
            "paired_registered_minus_greedy": distribution(registered_minus_greedy)
            | {"probability_positive": sum(value > 0 for value in registered_minus_greedy) / len(registered_minus_greedy)},
            "paired_weak_minus_registered": distribution(weak_minus_registered)
            | {"probability_positive": sum(value > 0 for value in weak_minus_registered) / len(weak_minus_registered)},
            "four_movie_public_like": distribution(public_four),
            "large_130_movie_private_like": distribution(private_130),
            "scope": "Movie resampling measures finite-movie noise conditional on the two observed embryo domains; it does not cover unseen-embryo domain shift.",
        },
        "random_five_fold": {
            "fold_scores_across_200_repeats": distribution(random_folds),
            "warning": "Random movie folds mix embryo identity and therefore understate deployment domain shift.",
        },
        "decision": {
            "algorithm_selection": "Trust paired leave-one-embryo-out deltas that agree in sign on both embryo folds.",
            "absolute_score": "Treat absolute OOF and public LB as domain-conditional estimates, not interchangeable predictions of private score.",
            "leaderboard": "Use the four-movie public LB as a broad sanity check only; do not optimize tiny deltas against it.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
