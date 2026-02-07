import numpy as np
import pytest

from src.ml.evaluation.metrics.base_metric import MetricMultiClassScope
from src.ml.evaluation.metrics.recall_metric import RecallMetric


def test_recall_metric_binary():
    """
    Test the classification_report_metrics function for binary classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    # Create RecallMetric instance
    metric = RecallMetric()

    # Evaluate
    metric.evaluate(y_true, y_pred)
    print(metric)

    # Check that accuracy is within valid range
    assert metric.mean() == pytest.approx(0.75, abs=0.0001)
    assert metric.values()[0] == pytest.approx(0.75, abs=0.0001)
    assert metric.minmax() == 0.0, "Min-Max should be zero for single evaluation"


def test_recall_metric_multiclass_micro():
    """
    Test the classification_report_metrics function for multi-class classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 1, 2, 1, 2, 2])

    # Create RecallMetric instance
    metric = RecallMetric(
        scope=MetricMultiClassScope.MICRO,
    )

    # Evaluate
    metric.evaluate(y_true, y_pred, classes=np.array([0, 1, 2]))
    print(metric)

    # Check that accuracy is within valid range
    assert metric.mean() == pytest.approx(0.75, abs=0.0001)
    assert metric.values()[0] == pytest.approx(0.75, abs=0.0001)
    assert metric.minmax() == 0.0, "Min-Max should be zero for single evaluation"


def test_recall_metric_multiclass_macro():
    """
    Test the classification_report_metrics function for multi-class classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 1, 2, 1, 2, 2])

    # Create RecallMetric instance
    metric = RecallMetric(
        scope=MetricMultiClassScope.MACRO,
    )

    # Evaluate
    metric.evaluate(y_true, y_pred, classes=np.array([0, 1, 2]))
    print(metric)

    # Check that accuracy is within valid range
    assert metric.mean() == pytest.approx(0.8055, abs=0.0001)
    assert metric.values()[0] == pytest.approx(0.8055, abs=0.0001)
    assert metric.minmax() == 0.0, "Min-Max should be zero for single evaluation"


def test_recall_metric_multiclass_class():
    """
    Test the classification_report_metrics function for multi-class classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 1, 2, 1, 2, 2])
    classes = [0, 1, 2]

    # Create RecallMetric instance
    metric = RecallMetric(
        scope=MetricMultiClassScope.CLASS,
        class_name=1,
    )

    # Evaluate
    metric.evaluate(y_true, y_pred, classes=classes)
    print(metric)

    # Check that accuracy is within valid range
    assert metric.mean() == pytest.approx(0.6666, abs=0.0001)
    assert metric.values()[0] == pytest.approx(0.6666, abs=0.0001)
    assert metric.minmax() == 0.0, "Min-Max should be zero for single evaluation"
