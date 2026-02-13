"""
Data Processor Module
======================
Handles loading, cleaning, transforming, and summarizing CSV data.
Provides utilities for common data preprocessing tasks used in ML pipelines.
"""

import csv
import math
import os


class DataProcessor:
    """
    A lightweight data processor for CSV files.

    Supports loading data, handling missing values, normalization,
    column statistics, and filtering.

    Attributes:
        data (list[dict]): The loaded dataset as a list of row dicts.
        columns (list[str]): Column names from the CSV header.
        filepath (str): Path to the loaded CSV file.
    """

    def __init__(self):
        """Initialize an empty DataProcessor."""
        self.data = []
        self.columns = []
        self.filepath = None

    def load_csv(self, filepath):
        """
        Load data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            DataProcessor: self, for method chaining.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.columns = reader.fieldnames or []
            self.data = [row for row in reader]

        if not self.data:
            raise ValueError(f"The file is empty: {filepath}")

        self.filepath = filepath
        return self

    def get_column(self, column_name):
        """
        Extract all values from a specific column.

        Args:
            column_name (str): Name of the column.

        Returns:
            list[str]: Values from that column.

        Raises:
            KeyError: If the column does not exist.
        """
        if column_name not in self.columns:
            raise KeyError(
                f"Column '{column_name}' not found. "
                f"Available: {self.columns}"
            )
        return [row[column_name] for row in self.data]

    def get_numeric_column(self, column_name):
        """
        Extract numeric values from a column, skipping non-numeric entries.

        Args:
            column_name (str): Name of the column.

        Returns:
            list[float]: Numeric values.
        """
        raw = self.get_column(column_name)
        values = []
        for v in raw:
            try:
                values.append(float(v))
            except (ValueError, TypeError):
                continue
        return values

    def column_stats(self, column_name):
        """
        Compute summary statistics for a numeric column.

        Args:
            column_name (str): Name of the numeric column.

        Returns:
            dict: Dictionary with count, mean, min, max, and std.

        Raises:
            ValueError: If column has no numeric values.
        """
        values = self.get_numeric_column(column_name)
        if not values:
            raise ValueError(
                f"No numeric values in column '{column_name}'."
            )

        n = len(values)
        avg = sum(values) / n
        var = sum((x - avg) ** 2 for x in values) / n
        sd = math.sqrt(var)

        return {
            "count": n,
            "mean": round(avg, 4),
            "min": min(values),
            "max": max(values),
            "std": round(sd, 4),
        }

    def normalize_column(self, column_name):
        """
        Min-Max normalize a numeric column to [0, 1].

        Args:
            column_name (str): Name of the numeric column.

        Returns:
            list[float]: Normalized values.

        Raises:
            ValueError: If column has < 2 values or is constant.
        """
        values = self.get_numeric_column(column_name)
        if len(values) < 2:
            raise ValueError("Need at least 2 numeric values to normalize.")

        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            raise ValueError("Cannot normalize a constant column.")

        return [(v - min_val) / (max_val - min_val) for v in values]

    def filter_rows(self, column_name, condition, value):
        """
        Filter rows based on a numeric condition.

        Args:
            column_name (str): Column to filter on.
            condition (str): One of '>', '<', '>=', '<=', '==', '!='.
            value (float): Threshold to compare against.

        Returns:
            list[dict]: Rows matching the condition.

        Raises:
            ValueError: If the condition is invalid.
        """
        valid = {">", "<", ">=", "<=", "==", "!="}
        if condition not in valid:
            raise ValueError(
                f"Invalid condition '{condition}'. Use one of {valid}."
            )

        ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }

        filtered = []
        for row in self.data:
            try:
                row_val = float(row[column_name])
                if ops[condition](row_val, value):
                    filtered.append(row)
            except (ValueError, TypeError, KeyError):
                continue
        return filtered

    def shape(self):
        """Return (rows, columns) of the dataset."""
        return (len(self.data), len(self.columns))

    def head(self, n=5):
        """Return the first n rows."""
        return self.data[:n]

    def describe(self):
        """
        Generate summary statistics for all numeric columns.

        Returns:
            dict: Column name -> statistics dict.
        """
        summary = {}
        for col in self.columns:
            try:
                summary[col] = self.column_stats(col)
            except (ValueError, KeyError):
                continue
        return summary
