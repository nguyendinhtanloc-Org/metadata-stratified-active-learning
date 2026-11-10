"""
Multi-seed Experiment Runner.
Chạy experiment với nhiều seeds để kiểm soát nhiễu ngẫu nhiên.
"""

from typing import List, Dict, Optional, Literal
import pandas as pd
from pathlib import Path
from datetime import datetime

from active_learning.loop import ActiveLearningLoop
from sampling.stratified import stratified_sampling
from sampling.baseline import baseline_sampling
from analysis.statistics import (
    paired_ttest,
    correlation_analysis,
    aggregate_multi_seed_results,
    format_statistical_report
)
from utils.config import DEFAULT_CONFIG


def run_single_experiment(
    config: Dict,
    metadata_path: str,
    image_dir: str,
    experiment_name: str,
    pipeline: Literal["stratified", "baseline"] = "stratified"
) -> pd.DataFrame:
    """
    Chạy một experiment với config chỉ định.

    Args:
        config: Dict chứa các tham số
        metadata_path: Đường dẫn đến annotations JSON
        image_dir: Thư mục chứa ảnh
        experiment_name: Tên experiment
        pipeline: "stratified" hoặc "baseline"

    Returns:
        DataFrame chứa kết quả
    """

    print(f"\n{'#'*60}")
    print(f"# Running: {experiment_name} ({pipeline})")
    print(f"# Seed: {config.get('seed', 'default')}")
    print(f"{'#'*60}")

    # Khởi tạo AL loop
    al_loop = ActiveLearningLoop(
        config=config,
        experiment_name=experiment_name
    )

    # Override sampling strategy nếu là baseline
    if pipeline == "baseline":
        # Monkey-patch stratified_sampling với baseline
        import active_learning.loop as loop_module
        loop_module.stratified_sampling = baseline_sampling

    # Chạy experiment
    results_df = al_loop.run(
        metadata_path=metadata_path,
        image_dir=image_dir
    )

    results_df["pipeline"] = pipeline
    results_df["seed"] = config.get("seed", 0)

    return results_df


def run_multi_seed_experiment(
    base_config: Dict,
    metadata_path: str,
    image_dir: str,
    seeds: List[int],
    pipelines: List[Literal["stratified", "baseline"]] = ["stratified", "baseline"],
    output_dir: str = "results"
) -> Dict[str, pd.DataFrame]:
    """
    Chạy experiment với nhiều seeds và nhiều pipelines.

    Args:
        base_config: Config cơ bản (sẽ được copy và update seed)
        metadata_path: Đường dẫn annotations
        image_dir: Thư mục ảnh
        seeds: Danh sách seeds
        pipelines: Danh sách pipelines ["stratified", "baseline"]
        output_dir: Thư mục lưu kết quả

    Returns:
        Dict với keys: "baseline", "stratified", "comparison"
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for pipeline in pipelines:
        pipeline_results = []

        for seed in seeds:
            # Copy config và update seed
            config = base_config.copy()
            config["seed"] = seed

            experiment_name = f"{pipeline}_seed_{seed}"

            try:
                results_df = run_single_experiment(
                    config=config,
                    metadata_path=metadata_path,
                    image_dir=image_dir,
                    experiment_name=experiment_name,
                    pipeline=pipeline
                )

                pipeline_results.append(results_df)

            except Exception as e:
                print(f"[ERROR] Failed seed {seed}, pipeline {pipeline}: {e}")
                continue

        if pipeline_results:
            combined = pd.concat(pipeline_results, ignore_index=True)
            all_results[pipeline] = combined

            # Save pipeline results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/{pipeline}_{timestamp}.csv"
            combined.to_csv(output_path, index=False)
            print(f"[SAVED] {pipeline} results: {output_path}")

    # Tổng hợp và so sánh
    if "baseline" in all_results and "stratified" in all_results:
        comparison = compare_pipelines(
            baseline_df=all_results["baseline"],
            stratified_df=all_results["stratified"]
        )
        all_results["comparison"] = comparison

    return all_results


def compare_pipelines(
    baseline_df: pd.DataFrame,
    stratified_df: pd.DataFrame
) -> Dict:
    """
    So sánh hai pipelines về entropy và mAP.

    Args:
        baseline_df: Kết quả baseline
        stratified_df: Kết quả stratified

    Returns:
        Dict chứa statistical comparison
    """

    comparison = {}

    # So sánh entropy (nếu có trong df)
    if "batch_entropy" in baseline_df.columns:
        baseline_entropy = baseline_df["batch_entropy"].tolist()
        stratified_entropy = stratified_df["batch_entropy"].tolist()

        entropy_test = paired_ttest(baseline_entropy, stratified_entropy)
        comparison["entropy_test"] = entropy_test

    # So sánh mAP
    if "map50" in baseline_df.columns:
        baseline_map = baseline_df["map50"].tolist()
        stratified_map = stratified_df["map50"].tolist()

        map_test = paired_ttest(baseline_map, stratified_map)
        comparison["map_test"] = map_test

    # Correlation: entropy vs mAP
    if "batch_entropy" in baseline_df.columns and "map50" in baseline_df.columns:
        all_entropy = list(baseline_df["batch_entropy"]) + list(stratified_df["batch_entropy"])
        all_map = list(baseline_df["map50"]) + list(stratified_df["map50"])

        corr = correlation_analysis(all_entropy, all_map)
        comparison["entropy_map_correlation"] = corr

    # Format report
    if "entropy_test" in comparison and "entropy_map_correlation" in comparison:
        report = format_statistical_report(
            comparison["entropy_test"],
            comparison["entropy_map_correlation"]
        )
        comparison["report"] = report

    return comparison


def quick_test_config() -> Dict:
    """
    Config cho test nhanh (batch nhỏ, ít epochs).
    """

    config = DEFAULT_CONFIG.copy()

    config.update({
        "seed": 42,
        "initial_budget": 100,    # Test với 100 ảnh
        "batch_size": 50,          # Thêm 50 mỗi round
        "num_rounds": 5,           # 5 rounds
        "epochs": 5,               # 5 epochs mỗi round (test nhanh)
        "device": "cuda",
        "num_classes": 10,
        "stratify_fields": ["weather", "scene", "timeofday"]
    })

    return config


def full_experiment_config() -> Dict:
    """
    Config cho experiment đầy đủ (theo proposal).
    """

    config = DEFAULT_CONFIG.copy()

    config.update({
        "seed": 42,
        "initial_budget": 1000,   # Theo proposal
        "batch_size": 500,
        "num_rounds": 5,           # Theo proposal
        "epochs": 10,              # Training đầy đủ
        "device": "cuda",
        "num_classes": 10,
        "stratify_fields": ["weather", "scene", "timeofday"]
    })

    return config
