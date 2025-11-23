import os
import json
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

EDA_REPORTS_DIR_MISSINGNESS = "reports/eda/missingness/"
EDA_REPORTS_DIR_DISTRIBUTION = "reports/eda/distributions/"

def _missingness_summary(df: pd.DataFrame) -> Dict[str, float]:
    total = len(df)
    if total == 0:
        return {col: 0.0 for col in df.columns}
    miss_percentage = ((df.isnull().sum() / total) * 100).round(3)
    return miss_percentage.to_dict()

def generate_missingness_bar(df:pd.DataFrame, save_path: str) -> None:
    miss_data = _missingness_summary(df)
    if not miss_data:
        return
    miss_series = pd.Series(miss_data).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=miss_series.index, y=miss_series.values, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Percentage of Missing Values')
    plt.title('Missingness Bar Plot')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_missingness_heatmap(df: pd.DataFrame, save_path: str, max_cols_for_heatmap: int = 80) -> None:
    if df.empty:
        return
    if df.shape[1] > max_cols_for_heatmap:
        miss_pct = pd.Series(_missingness_summary(df))
        keep_cols = list(miss_pct.sort_values(ascending=False).head(max_cols_for_heatmap).index)
        data = df[keep_cols].isna()
    else:
        data = df.isna()

    # Convert boolean to int for heatmap
    data_int = data.astype(int)

    plt.figure(figsize=(min(16, max(6, data_int.shape[1] * 0.2)), min(10, max(4, data_int.shape[0] * 0.02))))
    sns.heatmap(data_int.T, cbar=False, cmap="Greys", linewidths=0.0)
    plt.xlabel("Row index")
    plt.ylabel("Columns")
    plt.title("Missingness heatmap (1 = missing)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_missingness_reports(df: pd.DataFrame) -> Dict[str, str]:

    summary = _missingness_summary(df)

    bar_path = os.path.join(EDA_REPORTS_DIR_MISSINGNESS, "missingness_bar.png")
    heatmap_path = os.path.join(EDA_REPORTS_DIR_MISSINGNESS, "missingness_heatmap.png")
    json_path = os.path.join(EDA_REPORTS_DIR_MISSINGNESS, "_missingness_summary.json")
    # Generate artifacts
    generate_missingness_bar(df, bar_path)
    generate_missingness_heatmap(df, heatmap_path)

    # Save summary JSON
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": json_path,
        "bar_plot": bar_path,
        "heatmap": heatmap_path
    }

def filter_columns_for_eda(df: pd.DataFrame,
                           max_unique_categorical: int = 30,
                           min_non_missing_ratio: float = 0.5) -> dict:
    """
    Returns a dict with:
      - numeric_cols: numeric columns suitable for plotting
      - categorical_cols: categorical columns suitable for plotting

    Filters out:
      - ID-like columns (unique ratio > 0.9)
      - High-cardinality categoricals (unique > max_unique_categorical)
      - Columns with too many missing values (< min_non_missing_ratio present)
      - Free-text columns (object dtype and long strings)
      - Columns with a single unique value
    """

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col]
        non_missing_ratio = 1 - series.isna().mean()
        unique_vals = series.nunique(dropna=True)

        # Rule 1: Too many missing values → skip
        if non_missing_ratio < min_non_missing_ratio:
            continue

        # Rule 2: Single unique value → skip
        if unique_vals <= 1:
            continue

        # Rule 3: ID-like (unique ratio > 0.9)
        if unique_vals / len(df) > 0.9:
            continue

        # Rule 4: Large cardinality categoricals → skip
        if series.dtype == object and unique_vals > max_unique_categorical:
            continue

        # Rule 5: Free-text columns (long strings)
        if series.dtype == object:
            sample_str = str(series.dropna().iloc[0])
            if len(sample_str) > 25:     # heuristic threshold
                continue

        # Classification
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

def generate_numeric_distribution(df:pd.DataFrame, col:str, save_path:str) -> None:
    data = df[col].dropna()
    if data.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_categorical_distribution(df: pd.DataFrame, col: str, save_path: str, max_bars: int = 20) -> None:
    data = df[col].dropna()
    if data.empty:
        return
    counts = data.value_counts().sort_values(ascending=False)
    # Limit to top N categories
    if len(counts) > max_bars:
        counts = counts.head(max_bars)
    # Safe figsize: capped so Matplotlib never explodes
    fig_width = min(12, max(6, len(counts) * 0.4))  # cap at 12
    fig_height = 4
    plt.figure(figsize=(fig_width, fig_height))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title(f"Value counts of {col} (top {min(max_bars, len(counts))})")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def generate_distribution_reports(df: pd.DataFrame) -> Dict[str, str]:
    artifacts = {}
    filtered = filter_columns_for_eda(df)
    numeric_cols = filtered["numeric_cols"]
    categorical_cols = filtered["categorical_cols"]

    for col in numeric_cols:
        save_path = os.path.join(EDA_REPORTS_DIR_DISTRIBUTION, f"dist_numeric_{col}.png")
        generate_numeric_distribution(df, col, save_path)
        artifacts[f"dist_numeric_{col}"] = save_path

    for col in categorical_cols:
        save_path = os.path.join(EDA_REPORTS_DIR_DISTRIBUTION, f"dist_categorical_{col}.png")
        generate_categorical_distribution(df, col, save_path)
        artifacts[f"dist_categorical_{col}"] = save_path

    return artifacts
