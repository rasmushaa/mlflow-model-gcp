from unittest.mock import patch

import numpy as np
import pytest

from src.ml.evaluation.metrics.base_metric import MetricMultiClassScope
from src.ml.evaluation.metrics.roc_auc_metric import RocAucMetric


def test_roc_auc_metric_binary():
    """
    Test the RocAucMetric for binary classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0.1, 0.9, 0.4, 0.8, 0.35, 0.6, 0.7, 0.2, 0.55, 0.05])

    # Create RocAucMetric instance
    roc_auc_metric = RocAucMetric()

    # Evaluate
    roc_auc_metric.evaluate(y_true, y_pred)
    print(roc_auc_metric)

    assert "tpr" in roc_auc_metric.metadata
    assert "fpr" in roc_auc_metric.metadata
    assert len(roc_auc_metric.metadata["tpr"]) == 1
    assert roc_auc_metric.name == "roc_auc"
    assert roc_auc_metric.values()[0] == pytest.approx(0.9166, abs=0.0001)
    assert roc_auc_metric.mean() == pytest.approx(
        0.9166, abs=0.0001
    ), "Mean should equal single evaluation value"
    assert (
        roc_auc_metric.minmax() == 0.0
    ), "Min-Max should be zero for single evaluation"


def test_roc_auc_metric_binary_with_unsupported_class_name():
    # Create RocAucMetric instance with class name
    with pytest.raises(AssertionError, match="not support class_name"):
        RocAucMetric(
            class_name="A",  # Class name should not be used in binary classification
        )


def test_roc_auc_metric_multiclass_missing_classes():
    # Test data as numpy arrays
    y_true = np.array([0, 1, 2])
    y_prob = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )

    # Create RocAucMetric instance
    roc_auc_metric = RocAucMetric(
        scope=MetricMultiClassScope.MICRO,
    )

    # Evaluate without classes should raise ValueError
    with pytest.raises(AssertionError, match="classes must be provided"):
        roc_auc_metric.evaluate(y_true, y_prob)


def test_roc_auc_metric_multiclass_micro():
    """
    Test the prediction_report_metrics function for multi-class classification.
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

    # Create RocAucMetric instance
    roc_auc_metric = RocAucMetric(
        scope=MetricMultiClassScope.MICRO,
    )

    # Evaluate
    roc_auc_metric.evaluate(y_true, y_prob, classes=classes)
    print(roc_auc_metric)

    assert roc_auc_metric.name == "roc_auc.micro"
    assert roc_auc_metric.values()[0] == pytest.approx(0.9687, abs=0.0001)


def test_roc_auc_metric_multiclass_macro():
    """
    Test the prediction_report_metrics function for multi-class classification.
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

    # Create RocAucMetric instance
    roc_auc_metric = RocAucMetric(
        scope=MetricMultiClassScope.MACRO,
    )

    # Evaluate
    roc_auc_metric.evaluate(y_true, y_prob, classes=classes)
    print(roc_auc_metric)

    assert roc_auc_metric.name == "roc_auc.macro"
    assert roc_auc_metric.values()[0] == pytest.approx(0.9666, abs=0.0001)


def test_roc_auc_metric_multiclass_one_class():
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

    # Create RocAucMetric instance
    roc_auc_metric = RocAucMetric(scope=MetricMultiClassScope.CLASS, class_name=2)

    # Evaluate
    roc_auc_metric.evaluate(y_true, y_prob, classes=classes)
    print(roc_auc_metric)

    # Note, this is guite a vague test case, and it has not been calculated manually
    assert roc_auc_metric.name == "roc_auc.class.2"
    assert roc_auc_metric.values()[0] == pytest.approx(1.0, abs=0.0001)


@patch("src.ml.evaluation.metrics.roc_auc_metric.auc")
def test_roc_auc_binary_aggregation_metric_steps(roc_auc_mock):
    """
    Test that RocAucMetric aggregation works correctly for binary classification.
    This test uses mocking to simulate multiple evaluations.
    """
    # Setup mock to return different values on each call
    roc_auc_mock.side_effect = [0.8, 0.9, 0.85]

    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])

    # Create RocAucMetric instance
    roc_auc_metric = RocAucMetric()

    # Perform multiple evaluations
    for _ in range(3):
        roc_auc_metric.evaluate(y_true, y_pred)
    print(roc_auc_metric)

    # Check that aggregated ROC AUC is correct
    expected_mean = (0.8 + 0.9 + 0.85) / 3
    assert roc_auc_metric.mean() == pytest.approx(expected_mean, abs=0.0001)
    assert roc_auc_metric.min() == 0.8
    assert roc_auc_metric.max() == 0.9
    assert roc_auc_metric.std() == pytest.approx(np.std([0.8, 0.9, 0.85]), abs=0.0001)
    assert roc_auc_metric.minmax() == pytest.approx(0.1, abs=0.0001)
    assert roc_auc_metric.median() == 0.85

    # Check that individual values are stored correctly
    values = roc_auc_metric.values()
    assert len(values) == 3
    assert values[0] == 0.8
    assert values[1] == 0.9
    assert values[2] == 0.85

    # Check that metadata is stored correctly
    assert "tpr" in roc_auc_metric.metadata
    assert "fpr" in roc_auc_metric.metadata
    assert len(roc_auc_metric.metadata["tpr"]) == 3
    assert len(roc_auc_metric.metadata["fpr"]) == 3
    assert len(roc_auc_metric.metadata["tpr"][0]) > 0
