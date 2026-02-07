import numpy as np

from src.ml.metrics import classification_report_metrics, prediction_report_metrics


def test_classification_report_metrics_binary():
    """
    Test the classification_report_metrics function for binary classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    # Call the function
    metrics, _ = classification_report_metrics(y_true, y_pred)

    print(metrics)

    # Check that accuracy is within valid range
    assert metrics["accuracy"] == 0.7
    assert metrics["precision"] == 0.6
    assert metrics["recall"] == 0.75
    assert abs(metrics["f1"] - 0.6666666666666665) < 0.0001


def test_classification_report_metrics_multiclass():
    """
    Test the classification_report_metrics function for multi-class classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 1, 2, 1, 2, 2])

    # Call the function
    metrics, _ = classification_report_metrics(y_true, y_pred)

    print(metrics)

    tolerance = 0.0001
    assert abs(metrics["accuracy"] - 0.75) < tolerance  # Sum(Correct) / Total
    assert (
        abs(metrics["precision.0"] - 0.5) < tolerance
    )  # TP / (TP + FP) for only class 0
    assert (
        abs(metrics["precision.1"] - 0.6666) < tolerance
    )  # TP / (TP + FP) for only class 1
    assert (
        abs(metrics["precision.2"] - 1.0) < tolerance
    )  # TP / (TP + FP) for only class 2
    assert abs(metrics["recall.0"] - 1.0) < tolerance  # TP / (TP + FN) for only class 0
    assert (
        abs(metrics["recall.1"] - 0.6666) < tolerance
    )  # TP / (TP + FN) for only class 1
    assert (
        abs(metrics["recall.2"] - 0.75) < tolerance
    )  # TP / (TP + FN) for only class 2
    assert (
        abs(metrics["f1.0"] - 0.66666) < tolerance
    )  # 2 * (precision * recall) / (precision + recall) for only class 0
    assert (
        abs(metrics["f1.1"] - 0.6666) < tolerance
    )  # 2 * (precision * recall) / (precision + recall) for only class 1
    assert (
        abs(metrics["f1.2"] - 0.8571) < tolerance
    )  # 2 * (precision * recall) / (precision + recall) for only class 2
    assert abs(metrics["precision.macro"] - 0.7222) < tolerance  # (P1 + P2 + P3) / 3
    assert abs(metrics["recall.macro"] - 0.8055) < tolerance  # (R1 + R2 + R3) / 3
    assert abs(metrics["f1.macro"] - 0.73015) < tolerance  # (F1_1 + F1_2 + F1_3) / 3
    assert abs(metrics["precision.micro"] - 0.75) < tolerance  # Sum(TP) / Sum(TP + FP)
    assert abs(metrics["recall.micro"] - 0.75) < tolerance  # Sum(TP) / Sum(TP + FN)
    assert (
        abs(metrics["f1.micro"] - 0.75) < tolerance
    )  # 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)


def test_prediction_report_metrics_multi_class():
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
    classes = np.array([0, 1, 2])

    metrics, _ = prediction_report_metrics(y_true, y_prob, classes)
    print(metrics)

    tolerance = 0.0001
    assert (
        abs(metrics["roc_auc.0"] - 1.0) < tolerance
    )  # AUC for only class 0 in One VS All binary classification
    assert (
        abs(metrics["roc_auc.1"] - 0.8999) < tolerance
    )  # AUC for only class 1 in One VS All binary classification
    assert (
        abs(metrics["roc_auc.2"] - 1.0) < tolerance
    )  # AUC for only class 2 in One VS All binary classification
    assert (
        abs(metrics["roc_auc.macro"] - 0.9666) < tolerance
    )  # (AUC_1 + AUC_2 + AUC_3) / 3
    assert (
        abs(metrics["roc_auc.micro"] - 0.9687) < tolerance
    )  # Overall AUC in One VS All fashion


def test_prediction_report_metrics_binary():
    """
    Test the prediction_report_metrics function for binary classification.
    The mock values have been calculated manually,
    by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.6, 0.7, 0.3, 0.5, 0.2])
    classes = np.array([0, 1])

    metrics, _ = prediction_report_metrics(y_true, y_prob, classes)
    print(metrics)

    tolerance = 0.0001
    assert (
        abs(metrics["roc_auc"] - 0.9166) < tolerance
    )  # Normal AUC for binary classification
