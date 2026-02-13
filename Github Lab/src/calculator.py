"""
Calculator Module
==================
An enhanced calculator that goes beyond basic arithmetic to include
statistical functions like mean, median, standard deviation,
and correlation analysis.

This module extends the original Lab1 calculator concept with
powerful statistical computation capabilities.
"""

import math
from collections import Counter


# ═══════════════════════════════════════════════════════════════════
#  BASIC ARITHMETIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def fun1(x, y):
    """
    Adds two numbers together.

    Args:
        x (int/float): First number.
        y (int/float): Second number.

    Returns:
        int/float: Sum of x and y.

    Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x + y


def fun2(x, y):
    """
    Subtracts y from x.

    Args:
        x (int/float): First number.
        y (int/float): Second number.

    Returns:
        int/float: Difference of x and y.

    Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x - y


def fun3(x, y):
    """
    Multiplies two numbers together.

    Args:
        x (int/float): First number.
        y (int/float): Second number.

    Returns:
        int/float: Product of x and y.

    Raises:
        ValueError: If either x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x * y


def fun4(x, y, z):
    """
    Adds three numbers together.

    Args:
        x (int/float): First number.
        y (int/float): Second number.
        z (int/float): Third number.

    Returns:
        int/float: Sum of x, y, and z.
    """
    return x + y + z


# ═══════════════════════════════════════════════════════════════════
#  ENHANCED: ADDITIONAL ARITHMETIC
# ═══════════════════════════════════════════════════════════════════


def divide(x, y):
    """
    Divides x by y.

    Args:
        x (int/float): Numerator.
        y (int/float): Denominator.

    Returns:
        float: Quotient of x and y.

    Raises:
        ValueError: If y is zero or inputs are not numbers.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y


def power(base, exponent):
    """
    Raises base to the power of exponent.

    Args:
        base (int/float): The base number.
        exponent (int/float): The exponent.

    Returns:
        int/float: base raised to the power of exponent.
    """
    if not (isinstance(base, (int, float)) and isinstance(exponent, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return base ** exponent


def square_root(x):
    """
    Computes the square root of a number.

    Args:
        x (int/float): A non-negative number.

    Returns:
        float: Square root of x.

    Raises:
        ValueError: If x is negative or not a number.
    """
    if not isinstance(x, (int, float)):
        raise ValueError("Input must be a number.")
    if x < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return math.sqrt(x)


# ═══════════════════════════════════════════════════════════════════
#  ENHANCED: STATISTICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def mean(data):
    """
    Computes the arithmetic mean of a list of numbers.

    Args:
        data (list): List of int/float values.

    Returns:
        float: The arithmetic mean.

    Raises:
        ValueError: If the list is empty.
    """
    if not data:
        raise ValueError("Cannot compute mean of an empty list.")
    return sum(data) / len(data)


def median(data):
    """
    Computes the median of a list of numbers.

    Args:
        data (list): List of int/float values.

    Returns:
        float: The median value.

    Raises:
        ValueError: If the list is empty.
    """
    if not data:
        raise ValueError("Cannot compute median of an empty list.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]


def mode(data):
    """
    Computes the mode(s) of a list of numbers.

    Args:
        data (list): List of int/float values.

    Returns:
        list: Sorted list of most frequently occurring values.

    Raises:
        ValueError: If the list is empty.
    """
    if not data:
        raise ValueError("Cannot compute mode of an empty list.")
    counter = Counter(data)
    max_count = max(counter.values())
    return sorted([k for k, v in counter.items() if v == max_count])


def variance(data, population=True):
    """
    Computes the variance of a list of numbers.

    Args:
        data (list): List of int/float values.
        population (bool): If True, population variance (N);
                           if False, sample variance (N-1).

    Returns:
        float: The variance.

    Raises:
        ValueError: If the list is empty or has < 2 elements for sample.
    """
    if not data:
        raise ValueError("Cannot compute variance of an empty list.")
    if not population and len(data) < 2:
        raise ValueError("Sample variance requires at least 2 data points.")
    avg = sum(data) / len(data)
    n = len(data) if population else len(data) - 1
    return sum((x - avg) ** 2 for x in data) / n


def std_dev(data, population=True):
    """
    Computes the standard deviation of a list of numbers.

    Args:
        data (list): List of int/float values.
        population (bool): If True, population std dev; else sample.

    Returns:
        float: The standard deviation.
    """
    return math.sqrt(variance(data, population))


def correlation(x, y):
    """
    Computes the Pearson correlation coefficient between two lists.

    Args:
        x (list): First list of int/float values.
        y (list): Second list of int/float values.

    Returns:
        float: Pearson correlation coefficient (-1 to 1).

    Raises:
        ValueError: If lists differ in length or have zero std dev.
    """
    if len(x) != len(y):
        raise ValueError("Lists must have the same length.")
    if len(x) < 2:
        raise ValueError("Need at least 2 data points for correlation.")

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        raise ValueError("Standard deviation is zero; correlation undefined.")
    return numerator / (denom_x * denom_y)


def percentile(data, p):
    """
    Computes the p-th percentile of a list of numbers.

    Args:
        data (list): List of int/float values.
        p (float): Percentile to compute (0-100).

    Returns:
        float: The p-th percentile value.

    Raises:
        ValueError: If list is empty or p is out of range.
    """
    if not data:
        raise ValueError("Cannot compute percentile of an empty list.")
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
