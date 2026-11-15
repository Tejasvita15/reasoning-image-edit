"""Known-answer and independent-formulation checks for agreement statistics."""

import unittest

import numpy as np
from scipy.stats import spearmanr

from src.rq3_human.metrics import icc_a1, quadratic_kappa, spearman, summary


def contingency_kappa(x, y):
    table = np.zeros((5, 5), dtype=float)
    for a, b in zip(x, y):
        if np.isfinite(a) and np.isfinite(b):
            table[int(a) - 1, int(b) - 1] += 1
    count = table.sum()
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / count
    return 1 - np.sum(weights * table) / np.sum(weights * expected)


def anova_icc(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    matrix = np.column_stack((x[valid], y[valid]))
    n, k = matrix.shape
    overall = matrix.mean()
    rows = matrix.mean(axis=1)
    columns = matrix.mean(axis=0)
    ms_rows = k * np.sum((rows - overall) ** 2) / (n - 1)
    ms_columns = n * np.sum((columns - overall) ** 2) / (k - 1)
    residuals = matrix - rows[:, None] - columns[None, :] + overall
    ms_error = np.sum(residuals ** 2) / ((n - 1) * (k - 1))
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n)


class TestHumanMetrics(unittest.TestCase):
    def test_perfect_agreement(self):
        for metric in (quadratic_kappa, spearman, icc_a1):
            self.assertAlmostEqual(float(metric([1, 2, 3], [1, 2, 3])), 1.0)

    def test_constant_bias_preserves_correlation_but_not_agreement(self):
        x, y = [1, 2, 3], [2, 3, 4]
        self.assertAlmostEqual(float(quadratic_kappa(x, y)), 4 / 7)
        self.assertAlmostEqual(float(spearman(x, y)), 1.0)
        self.assertAlmostEqual(float(icc_a1(x, y)), 2 / 3)

    def test_tied_ranks(self):
        self.assertAlmostEqual(float(spearman([1, 1, 2, 3], [1, 2, 2, 3])), 5 / 6)

    def test_identical_constant_is_undefined(self):
        for metric in (quadratic_kappa, spearman, icc_a1):
            self.assertTrue(np.isnan(metric([3, 3, 3], [3, 3, 3])))
        self.assertTrue(np.isnan(icc_a1([1.1] * 10, [1.1] * 10)))

    def test_constant_different(self):
        self.assertEqual(float(quadratic_kappa([2, 2, 2], [3, 3, 3])), 0.0)
        self.assertEqual(float(icc_a1([2, 2, 2], [3, 3, 3])), 0.0)
        self.assertTrue(np.isnan(spearman([2, 2, 2], [3, 3, 3])))

    def test_negative_estimates_are_not_clipped(self):
        self.assertAlmostEqual(float(quadratic_kappa([1, 2, 3], [3, 2, 1])), -1.0)
        self.assertAlmostEqual(float(icc_a1([1, 2, 3], [3, 2, 1])), -3.0)

    def test_pairwise_missingness_and_empty_inputs(self):
        for metric in (quadratic_kappa, spearman, icc_a1):
            self.assertAlmostEqual(float(metric([1, np.nan, 2, 3], [1, 5, np.nan, 3])), 1.0)
            self.assertTrue(np.isnan(metric([np.nan, np.inf], [2, 3])))
            self.assertTrue(np.isnan(metric([], [])))

    def test_fixed_category_spacing_and_random_contingency_reference(self):
        x, y = np.array([1, 2, 5, 5]), np.array([1, 5, 2, 5])
        self.assertAlmostEqual(float(quadratic_kappa(x, y)), contingency_kappa(x, y))
        # Compressing the missing category gaps would incorrectly change kappa.
        compressed_x, compressed_y = np.array([1, 2, 3, 3]), np.array([1, 3, 2, 3])
        self.assertNotAlmostEqual(float(quadratic_kappa(x, y)), float(quadratic_kappa(compressed_x, compressed_y)))
        rng = np.random.default_rng(284)
        for _ in range(20):
            x, y = rng.integers(1, 6, size=(2, 40)).astype(float)
            x[4], y[7] = np.nan, np.nan
            self.assertAlmostEqual(float(quadratic_kappa(x, y)), contingency_kappa(x, y))

    def test_batch_results_match_scalar_independent_references(self):
        rng = np.random.default_rng(516)
        x, y = rng.integers(1, 6, size=(2, 3, 4, 40)).astype(float)
        x[..., 4], y[..., 9] = np.nan, np.nan
        got_kappa, got_rho, got_icc = quadratic_kappa(x, y), spearman(x, y), icc_a1(x, y)
        self.assertEqual(got_icc.shape, (3, 4))
        for i in range(3):
            for j in range(4):
                xx, yy = x[i, j], y[i, j]
                valid = np.isfinite(xx) & np.isfinite(yy)
                self.assertAlmostEqual(got_kappa[i, j], contingency_kappa(xx, yy))
                self.assertAlmostEqual(got_rho[i, j], spearmanr(xx[valid], yy[valid]).statistic)
                self.assertAlmostEqual(got_icc[i, j], anova_icc(xx, yy))

    def test_broadcasting_and_pair_order_invariance(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
        for metric in (quadratic_kappa, spearman, icc_a1):
            np.testing.assert_allclose(metric(x, y), metric(y, x))
            np.testing.assert_allclose(metric(x, y), metric(x[::-1], y[:, ::-1]))

    def test_summary_retains_undefined_counts(self):
        got = summary(1.5, [0, 1, 2, 3, np.nan, np.inf])
        self.assertEqual(got['estimate'], 1.5)
        self.assertAlmostEqual(got['ci_low'], 0.075)
        self.assertAlmostEqual(got['ci_high'], 2.925)
        self.assertEqual(got['valid_bootstrap_replicates'], 4)
        self.assertEqual(got['invalid_bootstrap_replicates'], 2)
        empty = summary(np.nan, [np.nan])
        self.assertIsNone(empty['estimate'])
        self.assertIsNone(empty['ci_low'])
        self.assertIsNone(empty['ci_high'])


if __name__ == '__main__':
    unittest.main()
