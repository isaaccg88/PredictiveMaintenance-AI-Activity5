"""Pruebas pequeñas de la lógica de evaluación, sin acceso a MLflow."""
import unittest
import numpy as np
from fuentes.evaluation import scores, select_threshold


class EvaluationTests(unittest.TestCase):
    def test_confusion_and_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.9, 0.8, 0.2])
        d = scores(y, p, .5)
        self.assertEqual((d["tn"], d["fp"], d["fn"], d["tp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(d["f1"], .5)
        self.assertAlmostEqual(d["f2"], .5)

    def test_threshold_is_valid(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.6, 0.7, 0.9])
        best, grid, feasible = select_threshold(y, p)
        self.assertTrue(feasible)
        self.assertIn(best["threshold"], [round(t, 2) for t in np.arange(.05, .951, .05)])
        self.assertEqual(len(grid), 19)


if __name__ == "__main__":
    unittest.main()
