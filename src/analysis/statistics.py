"""
Statistical Analysis Module cho Active Learning experiments.
Bao gồm: paired t-test, correlation analysis, result aggregation.
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
import pandas as pd


def paired_ttest(
    baseline_values: List[float],
    stratified_values: List[float]
) -> Dict:
    """
    Paired t-test để so sánh hai pipelines.
    Kiểm định H1: Stratified có entropy cao hơn baseline.

    Args:
        baseline_values: Danh sách entropy (hoặc mAP) của baseline pipeline
        stratified_values: Danh sách entropy (hoặc mAP) của stratified pipeline

    Returns:
        Dict với 't_statistic', 'p_value', 'significant', 'interpretation'
    """

    baseline = np.array(baseline_values)
    stratified = np.array(stratified_values)

    if len(baseline) != len(stratified):
        raise ValueError("Hai list phải có cùng độ dài")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(stratified, baseline)

    # One-tailed test (baseline < stratified)
    # p_value / 2 nếu t > 0, ngược lại thì 1 - p_value/2
    if t_stat > 0:
        p_one_tailed = p_value / 2
    else:
        p_one_tailed = 1 - p_value / 2

    return {
        "t_statistic": float(t_stat),
        "p_value_two_tailed": float(p_value),
        "p_value_one_tailed": float(p_one_tailed),
        "significant_0.05": p_one_tailed < 0.05,
        "significant_0.01": p_one_tailed < 0.01,
        "mean_baseline": float(np.mean(baseline)),
        "mean_stratified": float(np.mean(stratified)),
        "mean_diff": float(np.mean(stratified - baseline)),
        "interpretation": "significant" if p_one_tailed < 0.05 else "not significant"
    }


def correlation_analysis(
    entropy_values: List[float],
    map_values: List[float]
) -> Dict:
    """
    Phân tích tương quan giữa entropy và mAP.
    Kiểm định H2: Entropy cao hơn → mAP cao hơn.

    Args:
        entropy_values: Danh sách entropy values
        map_values: Danh sách mAP values (cùng độ dài)

    Returns:
        Dict với Pearson/Spearman correlation và p-values
    """

    entropy = np.array(entropy_values)
    map_arr = np.array(map_values)

    if len(entropy) != len(map_arr):
        raise ValueError("Hai list phải có cùng độ dài")

    # Pearson correlation (linear)
    pearson_r, pearson_p = stats.pearsonr(entropy, map_arr)

    # Spearman correlation (rank-based, monotonic)
    spearman_r, spearman_p = stats.spearmanr(entropy, map_arr)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "interpretation": "significant positive" if pearson_p < 0.05 and pearson_r > 0
                        else "significant negative" if pearson_p < 0.05 and pearson_r < 0
                        else "not significant"
    }


def aggregate_multi_seed_results(
    results_by_seed: Dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """
    Tổng hợp kết quả từ nhiều seeds.

    Args:
        results_by_seed: Dict {seed: DataFrame} từ mỗi seed run

    Returns:
        DataFrame với mean, std, min, max cho mỗi round
    """

    all_dfs = []
    for seed, df in results_by_seed.items():
        df_copy = df.copy()
        df_copy["seed"] = seed
        all_dfs.append(df_copy)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Group by round và tính statistics
    agg_stats = combined.groupby("round").agg({
        "map50": ["mean", "std", "min", "max"],
        "map50_95": ["mean", "std", "min", "max"],
        "labeled_count": ["mean", "std"]
    })

    # Flatten column names
    agg_stats.columns = ["_".join(col).strip() for col in agg_stats.columns.values]
    agg_stats = agg_stats.reset_index()

    return agg_stats


def format_statistical_report(
    entropy_test: Dict,
    correlation: Dict
) -> str:
    """
    Format kết quả thống kê thành report string.

    Args:
        entropy_test: Kết quả từ paired_ttest
        correlation: Kết quả từ correlation_analysis

    Returns:
        String chứa formatted report
    """

    report = []
    report.append("=" * 60)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 60)

    report.append("\n[H1] Entropy Comparison (Paired t-test)")
    report.append("-" * 40)
    report.append(f"  Baseline mean:   {entropy_test['mean_baseline']:.4f}")
    report.append(f"  Stratified mean: {entropy_test['mean_stratified']:.4f}")
    report.append(f"  Mean diff:       {entropy_test['mean_diff']:.4f}")
    report.append(f"  t-statistic:     {entropy_test['t_statistic']:.4f}")
    report.append(f"  p-value (1-tail):{entropy_test['p_value_one_tailed']:.6f}")
    report.append(f"  Significant (α=0.05): {'YES' if entropy_test['significant_0.05'] else 'NO'}")
    report.append(f"  Interpretation:  {entropy_test['interpretation']}")

    report.append("\n[H2] Entropy-mAP Correlation")
    report.append("-" * 40)
    report.append(f"  Pearson r:   {correlation['pearson_r']:.4f}")
    report.append(f"  Pearson p:   {correlation['pearson_p']:.6f}")
    report.append(f"  Spearman r:  {correlation['spearman_r']:.4f}")
    report.append(f"  Spearman p:  {correlation['spearman_p']:.6f}")
    report.append(f"  Interpretation: {correlation['interpretation']}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)
