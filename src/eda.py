import os
import json
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np

EDA_REPORTS_DIR_MISSINGNESS = "reports/eda/missingness/"
EDA_REPORTS_DIR_DISTRIBUTION = "reports/eda/distributions/"
EDA_REPORTS_DIR_PAIRWISE = "reports/eda/pairwise/"

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

def pearson_corr(df, col1, col2):
    data = df[[col1, col2]].dropna()
    if len(data) < 10:
        return 0
    return data[col1].corr(data[col2])

def correlation_ratio(categories, measurements):
    if len(measurements) == 0:
        return 0

    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_mean = np.mean(measurements)

    numerator = 0
    for i in range(cat_num):
        yi = measurements[fcat == i]
        if len(yi) > 0:
            numerator += len(yi) * (np.mean(yi) - y_mean)**2

    denominator = np.sum((measurements - y_mean)**2)
    if denominator == 0:
        return 0

    return np.sqrt(numerator / denominator)

def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    if tab.empty:
        return 0
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.values.sum()
    phi2 = chi2 / n
    r, k = tab.shape
    return np.sqrt(phi2 / max(1, min(k-1, r-1)))

def select_numeric_pairs_corr(df, num_cols, threshold=0.2, max_pairs=10):
    pairs = []

    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i+1:]:
            r = pearson_corr(df, c1, c2)
            if abs(r) >= threshold:
                pairs.append((c1, c2, abs(r)))

    # strongest first
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for (a, b, r) in pairs[:max_pairs]]

def select_num_cat_pairs_corr(df, num_cols, cat_cols,
                              threshold=0.1, min_samples=5, max_pairs=10):

    pairs = []

    for ncol in num_cols:
        if df[ncol].nunique() < 5:
            continue

        for ccol in cat_cols:
            if df[ccol].value_counts().min() < min_samples:
                continue

            eta = correlation_ratio(df[ccol], df[ncol])
            if eta >= threshold:
                pairs.append((ncol, ccol, eta))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for (a, b, _) in pairs[:max_pairs]]

def select_cat_cat_pairs_corr(df, cat_cols, threshold=0.1, max_pairs=5):
    pairs = []

    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i+1:]:
            v = cramers_v(df[c1], df[c2])
            if v >= threshold:
                pairs.append((c1, c2, v))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for (a, b, _) in pairs[:max_pairs]]

def generate_numeric_numeric_scatterplot(df: pd.DataFrame, x: str, y: str, save_path: str) -> None:
    data = df[[x, y]].dropna()
    if data.empty:
        return
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=data, x=x, y=y, alpha=0.6)
    plt.title(f'Scatterplot of {x} vs {y}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_numeric_categorical_boxplot(df, num_col, cat_col, save_path, max_categories=10):
    """
    High-quality numeric vs categorical visualization:
    - Clean boxplot (no thick shading)
    - Jittered points with controlled opacity
    - Mean marker (red dot)
    - Median line is clear
    - Grid for readability
    """

    data = df[[num_col, cat_col]].dropna()
    if data.empty:
        return

    # Reduce categories
    top_cats = data[cat_col].value_counts().head(max_categories).index
    data = data[data[cat_col].isin(top_cats)]

    plt.figure(figsize=(max(7, len(top_cats) * 1), 6))

    # Clean boxplot (minimalist)
    sns.boxplot(
        data=data,
        x=cat_col,
        y=num_col,
        whis=1.5,
        linewidth=1.8,
        fliersize=0,  # hide default fliers because we show our own points
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )

    # Jittered points (reveal distribution)
    sns.stripplot(
        data=data,
        x=cat_col,
        y=num_col,
        color='blue',
        size=3,
        jitter=0.25,
        alpha=0.5
    )

    # Add mean points (red)
    mean_vals = data.groupby(cat_col)[num_col].mean()
    for i, (cat, mean_val) in enumerate(mean_vals.items()):
        plt.scatter(i, mean_val, color='red', s=60, zorder=5)
        plt.text(i, mean_val + 1, f"{mean_val:.1f}", color="red", ha='center', fontsize=10)

    plt.title(f"{num_col} by {cat_col}", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
def generate_categorical_categorical_heatmap(df: pd.DataFrame, cat1: str, cat2: str, save_path: str) -> None:
    data = df[[cat1, cat2]].dropna()
    if data.empty:
        return
    cross_tab = pd.crosstab(data[cat1], data[cat2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Heatmap of {cat1} vs {cat2}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_pairwise_relationships(df: pd.DataFrame):
    pair_dir = os.path.join(EDA_REPORTS_DIR_PAIRWISE)

    filt = filter_columns_for_eda(df)
    num_cols = filt["numeric_cols"]
    cat_cols = filt["categorical_cols"]

    output = {}

    # ⭐ 1. Numeric ↔ Numeric pairs
    nn_pairs = select_numeric_pairs_corr(df, num_cols)
    for a, b in nn_pairs:
        p = os.path.join(pair_dir, f"{a}_vs_{b}.png")
        generate_numeric_numeric_scatterplot(df, a, b, p)
        output[f"{a}_vs_{b}"] = p

    # ⭐ 2. Numeric ↔ Categorical pairs
    nc_pairs = select_num_cat_pairs_corr(df, num_cols, cat_cols)
    for n, c in nc_pairs:
        p = os.path.join(pair_dir, f"{n}_by_{c}.png")
        generate_numeric_categorical_boxplot(df, n, c, p)
        output[f"{n}_by_{c}"] = p

    # ⭐ 3. Categorical ↔ Categorical pairs
    cc_pairs = select_cat_cat_pairs_corr(df, cat_cols)
    for a, b in cc_pairs:
        p = os.path.join(pair_dir, f"{a}_vs_{b}.png")
        generate_categorical_categorical_heatmap(df, a, b, p)
        output[f"{a}_vs_{b}"] = p

    return output