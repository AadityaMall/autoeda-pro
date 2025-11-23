import os
import json
import pandas as pd
from datetime import datetime

import numpy as np
import pandas as pd

def convert_to_serializable(obj):
    """
    Recursively convert objects to JSON-serializable types.
    Handles numpy types, pandas NA/NaT, timestamps, dicts, lists, sets, tuples.
    """

    # Pandas NaT
    if obj is pd.NaT:
        return None

    # Pandas NA (newer missing value type)
    if obj is pd.NA:
        return None

    # Pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # Numpy scalars → convert using item()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # Numpy booleans
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Numpy arrays → convert to list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas Series or DataFrame → convert to dict
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()

    # Python dict → recurse
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}

    # Iterable types → convert to list
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(v) for v in obj]

    # Leave primitives unchanged
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback: convert to string
    return str(obj)

def check_empty(df: pd.DataFrame) -> dict:
    """Check if dataframe is empty."""
    return {
        "empty": df.empty,
        "message": "DataFrame is empty." if df.empty else "OK"
    }


def check_duplicate_columns(df: pd.DataFrame) -> dict:
    """Check for duplicate columns."""
    duplicates = df.columns[df.columns.duplicated()].tolist()
    return {
        "duplicate_columns": duplicates,
        "message": "Duplicate columns detected." if duplicates else "OK"
    }


def infer_dtypes(df: pd.DataFrame) -> dict:
    """Return inferred data types."""
    return df.dtypes.astype(str).to_dict()


def check_mixed_types(df: pd.DataFrame) -> dict:
    """
    Detect columns with mixed data types.
    e.g., a column containing ints + strings.
    """
    mixed_columns = {}

    for col in df.columns:
        unique_types = df[col].map(type).nunique()
        if unique_types > 1:
            mixed_columns[col] = unique_types

    return {
        "mixed_type_columns": mixed_columns,
        "message": "Mixed types found." if mixed_columns else "OK"
    }


def check_missingness(df: pd.DataFrame) -> dict:
    """Count missing values per column."""
    missing_counts = df.isna().sum().to_dict()
    return {
        "missing_values": missing_counts
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    """Count duplicate rows."""
    num_dupes = df.duplicated().sum()
    return {
        "duplicate_rows": num_dupes
    }


def check_invalid_values(df: pd.DataFrame) -> dict:
    """
    Basic invalid value rules:
    - Negative ages
    - Negative numeric values where inappropriate
    """
    invalid = {}

    # Example rule: negative ages
    if "Age" in df.columns:
        invalid["negative_ages"] = int((df["Age"] < 0).sum())

    # Generic rule: negative values in all-numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    negative_counts = {}
    for col in numeric_cols:
        negative_counts[col] = int((df[col] < 0).sum())

    invalid["negative_values"] = negative_counts

    return invalid


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run all validation checks and return a combined report.
    Also saves report as JSON inside data_processed/.
    """

    report = {
        "created_at": datetime.now().isoformat(),
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": infer_dtypes(df),
    }

    # Run each check
    report.update(check_empty(df))
    report.update(check_duplicate_columns(df))
    report.update(check_mixed_types(df))
    report.update(check_missingness(df))
    report.update(check_duplicates(df))
    report.update(check_invalid_values(df))

    # Save to JSON
    safe_report = convert_to_serializable(report)
    save_path = os.path.join("data_processed", "validation_report.json")

    with open(save_path, "w") as f:
        json.dump(safe_report, f, indent=4)

    return safe_report