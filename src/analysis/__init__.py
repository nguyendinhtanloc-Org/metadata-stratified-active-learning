"""
Analysis module for experiment results.
"""

from .statistics import (
    paired_ttest,
    correlation_analysis,
    aggregate_multi_seed_results,
    format_statistical_report
)

from .multi_seed_runner import (
    run_single_experiment,
    run_multi_seed_experiment,
    compare_pipelines,
    quick_test_config,
    full_experiment_config
)

__all__ = [
    "paired_ttest",
    "correlation_analysis",
    "aggregate_multi_seed_results",
    "format_statistical_report",
    "run_single_experiment",
    "run_multi_seed_experiment",
    "compare_pipelines",
    "quick_test_config",
    "full_experiment_config",
]
