"""
Unittest Test Suite
====================
Tests for calculator functions, data processor, and linear regression
model using the unittest framework.
"""

import os
import sys
import unittest

# Ensure src is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src import calculator
from src.data_processor import DataProcessor
from src.model import LinearRegressionModel


# ═══════════════════════════════════════════════════════════════════
#  CALCULATOR TESTS (matching original Lab1 + enhancements)
# ═══════════════════════════════════════════════════════════════════


class TestCalculatorBasic(unittest.TestCase):
    """Tests for original calculator functions."""

    def test_fun1(self):
        self.assertEqual(calculator.fun1(2, 3), 5)
        self.assertEqual(calculator.fun1(5, 0), 5)
        self.assertEqual(calculator.fun1(-1, 1), 0)
        self.assertEqual(calculator.fun1(-1, -1), -2)

    def test_fun2(self):
        self.assertEqual(calculator.fun2(2, 3), -1)
        self.assertEqual(calculator.fun2(5, 0), 5)
        self.assertEqual(calculator.fun2(-1, 1), -2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3(self):
        self.assertEqual(calculator.fun3(2, 3), 6)
        self.assertEqual(calculator.fun3(5, 0), 0)
        self.assertEqual(calculator.fun3(-1, 1), -1)
        self.assertEqual(calculator.fun3(-1, -1), 1)

    def test_fun4(self):
        self.assertEqual(calculator.fun4(2, 3, 5), 10)
        self.assertEqual(calculator.fun4(5, 0, -1), 4)
        self.assertEqual(calculator.fun4(-1, -1, -1), -3)
        self.assertEqual(calculator.fun4(-1, -1, 100), 98)


class TestCalculatorEnhanced(unittest.TestCase):
    """Tests for enhanced calculator functions."""

    def test_divide(self):
        self.assertEqual(calculator.divide(10, 2), 5)
        self.assertEqual(calculator.divide(7, 2), 3.5)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            calculator.divide(10, 0)

    def test_power(self):
        self.assertEqual(calculator.power(2, 3), 8)
        self.assertEqual(calculator.power(5, 0), 1)

    def test_square_root(self):
        self.assertEqual(calculator.square_root(25), 5.0)

    def test_square_root_negative(self):
        with self.assertRaises(ValueError):
            calculator.square_root(-4)

    def test_mean(self):
        self.assertAlmostEqual(calculator.mean([10, 20, 30]), 20.0)

    def test_median(self):
        self.assertEqual(calculator.median([5, 3, 1, 4, 2]), 3)

    def test_std_dev(self):
        self.assertAlmostEqual(calculator.std_dev([10, 10, 10]), 0.0)

    def test_variance(self):
        result = calculator.variance([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(result, 4.0)

    def test_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        self.assertAlmostEqual(calculator.correlation(x, y), 1.0)

    def test_percentile(self):
        data = [1, 2, 3, 4, 5]
        self.assertEqual(calculator.percentile(data, 0), 1)
        self.assertEqual(calculator.percentile(data, 100), 5)


# ═══════════════════════════════════════════════════════════════════
#  DATA PROCESSOR TESTS
# ═══════════════════════════════════════════════════════════════════


class TestDataProcessor(unittest.TestCase):
    """Tests for the DataProcessor class."""

    def setUp(self):
        """Load the housing dataset."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "data", "housing_data.csv"
        )
        self.dp = DataProcessor().load_csv(os.path.abspath(path))

    def test_load(self):
        self.assertEqual(self.dp.shape()[0], 30)
        self.assertIn("price_thousands", self.dp.columns)

    def test_get_numeric_column(self):
        prices = self.dp.get_numeric_column("price_thousands")
        self.assertEqual(len(prices), 30)
        self.assertTrue(all(isinstance(p, float) for p in prices))

    def test_column_stats(self):
        stats = self.dp.column_stats("size_sqft")
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertGreater(stats["count"], 0)

    def test_normalize(self):
        normed = self.dp.normalize_column("price_thousands")
        self.assertAlmostEqual(min(normed), 0.0)
        self.assertAlmostEqual(max(normed), 1.0)

    def test_filter(self):
        expensive = self.dp.filter_rows("price_thousands", ">=", 400)
        self.assertTrue(len(expensive) > 0)
        for row in expensive:
            self.assertGreaterEqual(float(row["price_thousands"]), 400)


# ═══════════════════════════════════════════════════════════════════
#  LINEAR REGRESSION MODEL TESTS
# ═══════════════════════════════════════════════════════════════════


class TestLinearRegressionModel(unittest.TestCase):
    """Tests for the LinearRegressionModel class."""

    def setUp(self):
        """Train a model on y = 2x + 1."""
        self.model = LinearRegressionModel()
        self.model.fit([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])

    def test_is_trained(self):
        self.assertTrue(self.model.is_trained)

    def test_slope(self):
        self.assertAlmostEqual(self.model.slope, 2.0)

    def test_intercept(self):
        self.assertAlmostEqual(self.model.intercept, 1.0)

    def test_predict(self):
        preds = self.model.predict([6, 7])
        self.assertAlmostEqual(preds[0], 13.0)
        self.assertAlmostEqual(preds[1], 15.0)

    def test_evaluate_r_squared(self):
        metrics = self.model.evaluate(
            [1, 2, 3, 4, 5], [3, 5, 7, 9, 11]
        )
        self.assertAlmostEqual(metrics["r_squared"], 1.0)

    def test_untrained_raises(self):
        m = LinearRegressionModel()
        with self.assertRaises(RuntimeError):
            m.predict([1, 2])


if __name__ == "__main__":
    unittest.main()
