"""
Pytest Test Suite
==================
Tests for calculator functions (basic arithmetic + enhanced statistical
functions), data processor, and linear regression model.
"""

import math
import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import calculator
from src.data_processor import DataProcessor
from src.model import LinearRegressionModel


# ═══════════════════════════════════════════════════════════════════
#  CALCULATOR: BASIC ARITHMETIC (matching original Lab1 tests)
# ═══════════════════════════════════════════════════════════════════


def test_fun1():
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(5, 0) == 5
    assert calculator.fun1(-1, 1) == 0
    assert calculator.fun1(-1, -1) == -2


def test_fun2():
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5, 0) == 5
    assert calculator.fun2(-1, 1) == -2
    assert calculator.fun2(-1, -1) == 0


def test_fun3():
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5, 0) == 0
    assert calculator.fun3(-1, 1) == -1
    assert calculator.fun3(-1, -1) == 1


def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5, 0, -1) == 4
    assert calculator.fun4(-1, -1, -1) == -3
    assert calculator.fun4(-1, -1, 100) == 98


# ═══════════════════════════════════════════════════════════════════
#  CALCULATOR: ENHANCED ARITHMETIC
# ═══════════════════════════════════════════════════════════════════


def test_divide():
    assert calculator.divide(10, 2) == 5
    assert calculator.divide(7, 2) == 3.5


def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calculator.divide(10, 0)


def test_power():
    assert calculator.power(2, 3) == 8
    assert calculator.power(5, 0) == 1


def test_square_root():
    assert calculator.square_root(25) == 5.0
    assert calculator.square_root(0) == 0.0


def test_square_root_negative():
    with pytest.raises(ValueError, match="negative"):
        calculator.square_root(-4)


# ═══════════════════════════════════════════════════════════════════
#  CALCULATOR: STATISTICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def test_mean():
    assert calculator.mean([1, 2, 3, 4, 5]) == 3.0
    assert calculator.mean([42]) == 42.0


def test_mean_empty():
    with pytest.raises(ValueError, match="empty"):
        calculator.mean([])


def test_median_odd():
    assert calculator.median([3, 1, 2]) == 2


def test_median_even():
    assert calculator.median([1, 2, 3, 4]) == 2.5


def test_median_empty():
    with pytest.raises(ValueError, match="empty"):
        calculator.median([])


def test_mode_single():
    assert calculator.mode([1, 2, 2, 3]) == [2]


def test_mode_multi():
    assert calculator.mode([1, 1, 2, 2, 3]) == [1, 2]


def test_mode_empty():
    with pytest.raises(ValueError, match="empty"):
        calculator.mode([])


def test_variance_population():
    result = calculator.variance([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result - 4.0) < 1e-9


def test_variance_sample():
    result = calculator.variance([2, 4, 4, 4, 5, 5, 7, 9], population=False)
    expected = 32 / 7
    assert abs(result - expected) < 1e-6


def test_variance_empty():
    with pytest.raises(ValueError, match="empty"):
        calculator.variance([])


def test_std_dev():
    result = calculator.std_dev([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result - 2.0) < 1e-9


def test_correlation_perfect_positive():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert abs(calculator.correlation(x, y) - 1.0) < 1e-9


def test_correlation_perfect_negative():
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]
    assert abs(calculator.correlation(x, y) - (-1.0)) < 1e-9


def test_correlation_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculator.correlation([1, 2], [1, 2, 3])


def test_percentile_50():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = calculator.percentile(data, 50)
    assert abs(result - 5.5) < 1e-9


def test_percentile_bounds():
    data = [1, 2, 3, 4, 5]
    assert calculator.percentile(data, 0) == 1
    assert calculator.percentile(data, 100) == 5


def test_percentile_invalid():
    with pytest.raises(ValueError, match="between 0 and 100"):
        calculator.percentile([1, 2, 3], 150)


# ═══════════════════════════════════════════════════════════════════
#  DATA PROCESSOR TESTS
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temp CSV file for testing."""
    content = (
        "name,age,score,grade\n"
        "Alice,25,88.5,A\n"
        "Bob,30,72.0,B\n"
        "Charlie,22,95.0,A\n"
        "Diana,28,65.5,C\n"
        "Eve,35,80.0,B\n"
    )
    f = tmp_path / "test.csv"
    f.write_text(content)
    return str(f)


@pytest.fixture
def processor(sample_csv):
    """Loaded DataProcessor with sample data."""
    return DataProcessor().load_csv(sample_csv)


def test_load_csv(sample_csv):
    dp = DataProcessor().load_csv(sample_csv)
    assert len(dp.data) == 5
    assert "age" in dp.columns


def test_load_csv_missing():
    with pytest.raises(FileNotFoundError):
        DataProcessor().load_csv("/nonexistent.csv")


def test_get_numeric_column(processor):
    ages = processor.get_numeric_column("age")
    assert ages == [25, 30, 22, 28, 35]


def test_column_stats(processor):
    stats = processor.column_stats("age")
    assert stats["count"] == 5
    assert stats["min"] == 22
    assert stats["max"] == 35


def test_normalize_column(processor):
    normed = processor.normalize_column("age")
    assert min(normed) == 0.0
    assert max(normed) == 1.0


def test_filter_rows(processor):
    result = processor.filter_rows("age", ">", 28)
    assert len(result) == 2


def test_shape(processor):
    assert processor.shape() == (5, 4)


def test_head(processor):
    assert len(processor.head(3)) == 3


def test_describe(processor):
    summary = processor.describe()
    assert "age" in summary
    assert "name" not in summary


def test_housing_data():
    """Test that the included housing dataset loads correctly."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "housing_data.csv"
    )
    dp = DataProcessor().load_csv(os.path.abspath(path))
    assert dp.shape()[0] == 30
    assert "price_thousands" in dp.columns


# ═══════════════════════════════════════════════════════════════════
#  LINEAR REGRESSION MODEL TESTS
# ═══════════════════════════════════════════════════════════════════


def test_model_fit():
    m = LinearRegressionModel()
    m.fit([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])  # y = 2x + 1
    assert m.is_trained is True
    assert abs(m.slope - 2.0) < 1e-9
    assert abs(m.intercept - 1.0) < 1e-9


def test_model_predict():
    m = LinearRegressionModel()
    m.fit([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    preds = m.predict([6, 7, 8])
    assert abs(preds[0] - 13.0) < 1e-9
    assert abs(preds[1] - 15.0) < 1e-9


def test_model_predict_single():
    m = LinearRegressionModel()
    m.fit([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    assert abs(m.predict_single(10) - 21.0) < 1e-9


def test_model_evaluate():
    m = LinearRegressionModel()
    m.fit([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    metrics = m.evaluate([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    assert metrics["mae"] == 0.0
    assert abs(metrics["r_squared"] - 1.0) < 1e-9


def test_model_untrained_predict():
    with pytest.raises(RuntimeError, match="trained"):
        LinearRegressionModel().predict([1, 2])


def test_model_save_load(tmp_path):
    m = LinearRegressionModel()
    m.fit([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    path = str(tmp_path / "model.json")
    m.save_model(path)

    m2 = LinearRegressionModel().load_model(path)
    assert abs(m2.predict_single(5) - 11.0) < 1e-9
