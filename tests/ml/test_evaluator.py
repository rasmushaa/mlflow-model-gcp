import os

import numpy as np
import pytest
from matplotlib import pyplot as plt

from src.ml.evaluation.evaluator import Evaluator


class MockLogger:
    """Mock logger for testing metric logging."""

    def __init__(self):
        self.metrics = {}
        self.figures = {}
        self.temp_dir = "./temp_test_logs/"
        os.makedirs(self.temp_dir, exist_ok=True)

    def log_metric(self, name: str, value: float, step: int = None) -> None:
        if step is None:
            self.metrics[name] = value
        else:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((step, value))

    def log_figure(self, figure: plt.Figure, name: str) -> None:
        self.figures[name] = figure
        filepath = os.path.join(self.temp_dir, name)
        figure.savefig(filepath, dpi=150, bbox_inches="tight")


def test_evaluator_with_one_metric():
    """
    Test the Evaluator with RocAucMetric for binary classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0.1, 0.9, 0.4, 0.8, 0.35, 0.6, 0.7, 0.2, 0.55, 0.05])

    # Create Evaluator instance with RocAucMetric configuration
    evaluator = Evaluator(
        metric_configs=[
            {
                "function": "roc_auc",
                "aggregations": ["mean", "minmax"],
            },
        ]
    )

    # Evaluate
    evaluator.evaluate_probabilities(
        y_true, y_pred, classes=None
    )  # Classes are not needed for binary

    # Log metrics to a mock logger
    mock_logger = MockLogger()
    evaluator.log_metrics(mock_logger)
    print(mock_logger.metrics)

    assert mock_logger.metrics["roc_auc.mean"] == pytest.approx(0.9166, abs=0.0001)
    assert mock_logger.metrics["roc_auc.minmax"] == pytest.approx(
        0.0, abs=0.0001
    ), "Min should be 0.0 for single evaluation"
    assert (
        mock_logger.metrics["roc_auc.steps"][0][0] == 0
    ), "First step index should be 0 for single evaluation"
    assert mock_logger.metrics["roc_auc.steps"][0][1] == pytest.approx(
        0.9166, abs=0.0001
    ), "First step value should equal overall value for single evaluation"
    assert (
        mock_logger.metrics["roc_auc.steps"].__len__() == 1
    ), "There should be only one step logged for single evaluation"


def test_evaluator_with_multiple_metrics_multiclass():
    """
    Test the Evaluator with RocAucMetric for multi-class classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_prob = np.array(
        [
            [0.70, 0.20, 0.10],  # true 0
            [0.10, 0.70, 0.20],  # true 1
            [0.60, 0.30, 0.10],  # true 1 (bad prediction)
            [0.20, 0.60, 0.20],  # true 1
            [0.05, 0.20, 0.75],  # true 2
            [0.10, 0.40, 0.50],  # true 2
            [0.05, 0.30, 0.65],  # true 2
            [0.05, 0.10, 0.85],  # true 2
        ]
    )
    classes = [0, 1, 2]

    # Create Evaluator instance with RocAucMetric configuration
    evaluator = Evaluator(
        metric_configs=[
            {
                "function": "roc_auc",
                "scope": "macro",
                "aggregations": ["mean", "minmax"],
            },
            {
                "function": "roc_auc",
                "scope": "micro",
                "aggregations": ["mean", "minmax"],
            },
            {
                "function": "roc_auc",
                "scope": "class",
                "class_name": 2,
                "aggregations": ["mean", "minmax"],
            },
        ]
    )

    # Evaluate
    evaluator.evaluate_probabilities(y_true, y_prob, classes)
    print(evaluator)

    # Log metrics to a mock logger
    mock_logger = MockLogger()
    evaluator.log_metrics(mock_logger)
    evaluator.log_plots(mock_logger)

    assert mock_logger.metrics["roc_auc.macro.mean"] == pytest.approx(
        0.9666, abs=0.0001
    )
    assert mock_logger.metrics["roc_auc.macro.minmax"] == pytest.approx(
        0.0, abs=0.0001
    ), "Min should be 0.0 for single evaluation"
    assert mock_logger.metrics["roc_auc.micro.mean"] == pytest.approx(
        0.9687, abs=0.0001
    )
    assert mock_logger.metrics["roc_auc.micro.minmax"] == pytest.approx(
        0.0, abs=0.0001
    ), "Min should be 0.0 for single evaluation"
    assert mock_logger.metrics["roc_auc.class.2.mean"] == pytest.approx(1.0, abs=0.0001)
    assert (
        mock_logger.metrics["roc_auc.macro.steps"][0][0] == 0
    ), "First step index should be 0 for single evaluation"
    assert mock_logger.metrics["roc_auc.macro.steps"][0][1] == pytest.approx(
        0.9666, abs=0.0001
    ), "First step value should equal overall value for single evaluation"


def test_evaluator_with_invalid_config():
    """Test that Evaluator raises ValueError for invalid metric configurations."""
    with pytest.raises(AssertionError, match="must include 'aggregations' "):
        Evaluator(
            metric_configs=[
                {
                    "function": "roc_auc",
                    # Missing aggregations
                }
            ]
        )

    with pytest.raises(ValueError, match='Cannot convert "invalid_agg" to'):
        Evaluator(
            metric_configs=[
                {"function": "roc_auc", "aggregations": ["mean", "invalid_agg"]}
            ]
        )

    with pytest.raises(
        AssertionError, match="Non-class-specific metrics do not support class_name"
    ):
        Evaluator(
            metric_configs=[
                {
                    "function": "roc_auc",
                    "scope": "micro",
                    "class_name": 1,  # class_name should not be provided for micro
                    "aggregations": ["mean"],
                }
            ]
        )

    with pytest.raises(
        AssertionError, match="Class-specific metrics require a class_name."
    ):
        Evaluator(
            metric_configs=[
                {
                    "function": "roc_auc",
                    "scope": "class",
                    # Missing class_name for class-specific metric
                    "aggregations": ["mean"],
                }
            ]
        )
