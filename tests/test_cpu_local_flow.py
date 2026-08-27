"""Pure geometry and frozen split tests; no images/GPU/Kaggle credentials."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "kaggle_notebooks/exp077_cpu_local_flow/cpu_local_flow.py"
SPEC = importlib.util.spec_from_file_location("cpu_local_flow", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalFlowTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(MODULE.link_motion([], [], (1, 1, 1), "local_flow_weak"), ([], []))

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            MODULE.link_motion([], [], (1, 1, 1), "unknown")

    def test_no_temporal_gap_bridging(self):
        edges, _ = MODULE.link_motion([[0, 1, 2, 3], [2, 1, 2, 3]], [], (1, 1, 1), "local_flow_weak")
        self.assertEqual(edges, [])

    def test_sparse_falls_back_exactly(self):
        coords = [[0, 0, 0, 0], [0, 0, 0, 10], [1, 0, 0, 1], [1, 0, 0, 11]]
        a, _ = MODULE.link_motion(coords, [], (1, 1, 1), "registered_weak")
        b, _ = MODULE.link_motion(coords, [], (1, 1, 1), "local_flow_weak")
        self.assertEqual(a, b)

    def test_physical_scale_and_no_mutation(self):
        points = np.array([[0, y, x] for y in (0, 8, 16) for x in (0, 8, 16)], dtype=float)
        coords = np.vstack([np.column_stack([np.zeros(len(points)), points]),
                            np.column_stack([np.ones(len(points)), points + [0, 0.5, 0.5]])])
        original = coords.copy()
        scaled = coords.copy()
        scaled[:, 1:] /= [2, 0.5, 0.25]
        a, _ = MODULE.link_motion(coords, [], (1, 1, 1), "local_flow_weak")
        b, _ = MODULE.link_motion(scaled, [], (2, 0.5, 0.25), "local_flow_weak")
        np.testing.assert_allclose(a, b)
        np.testing.assert_array_equal(coords, original)
        self.assertEqual(len(a), len(points))
        self.assertEqual(len({e[0] for e in a}), len(a))
        self.assertEqual(len({e[1] for e in a}), len(a))

    def test_own_field_improves_synthetic_nonrigid_motion(self):
        first = np.array([[0, y, x] for y in (0, 8, 16) for x in (0, 8, 16)], dtype=float)
        source = np.vstack([first, first + [0, 60, 0]])
        target = source.copy()
        target[:len(first), 2] += 1.5
        target[len(first):, 2] -= 1.5
        correction, info = MODULE.local_corrections(source, target, np.zeros(3))
        self.assertEqual(info["adjusted_nodes"], len(source))
        self.assertLess(np.linalg.norm(source + correction - target, axis=1).mean(), 0.9)
        self.assertLessEqual(info["maximum_correction_um"], 1.875)

    def test_self_anchor_cannot_supply_own_minimum_support(self):
        source = np.array([[0, 0, 0], [0, 0, 8], [0, 8, 0]], dtype=float)
        correction, info = MODULE.local_corrections(source, source + [0, 0, 1], np.zeros(3))
        np.testing.assert_array_equal(correction, np.zeros_like(source))
        self.assertEqual(info["adjusted_nodes"], 0)

    def test_unreliable_large_residual_falls_back(self):
        source = np.array([[0, y, x] for y in (0, 10, 20) for x in (0, 10, 20)], dtype=float)
        correction, _ = MODULE.local_corrections(source, source + [3.0, 0, 0], np.zeros(3))
        np.testing.assert_array_equal(correction, np.zeros_like(source))

    def test_frozen_pilot_membership(self):
        import json
        budget = json.loads((ROOT / "reports/validation_budget_20260827.json").read_text())
        for embryo, fold in MODULE.FOLDS.items():
            self.assertEqual(fold["movies"], budget["folds"][embryo]["pilot_movies"][:2])
        metadata = json.loads(PATH.with_name("kernel-metadata.json").read_text())
        self.assertFalse(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_tpu"])
        self.assertLess(MODULE.RUN_SECONDS, 12 * 3600)

    def test_control_matches_existing_exp063(self):
        import ast
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
        source = (ROOT / "kaggle_notebooks/exp063_oof_comparison_44b6/oof_comparison_44b6.py").read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "graph_for_policy")
        namespace = {"np": np, "linear_sum_assignment": linear_sum_assignment, "cKDTree": cKDTree,
                     "build_graph": lambda coords, edges: edges}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "existing_control", "exec"), namespace)
        rng = np.random.default_rng(77)
        for _ in range(5):
            points = rng.normal(0, 20, (25, 3))
            coords = np.vstack([np.column_stack([np.zeros(25), points]),
                                np.column_stack([np.ones(25), points + rng.normal(0, 2, (25, 3))])])
            candidates = [(i, i + 25, float(rng.uniform(0.5, 1.0)), 0.0) for i in range(25)]
            for new, old in (("registered", "registered_hungarian"), ("registered_weak", "registered_weak_hungarian")):
                expected = namespace["graph_for_policy"](coords, candidates, old, (1.625, 0.40625, 0.40625))
                actual, _ = MODULE.link_motion(coords, candidates, (1.625, 0.40625, 0.40625), new)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
