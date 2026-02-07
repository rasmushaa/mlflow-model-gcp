import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .base_metric import BaseMetric
from .interface import MetricFunction, MetricMultiClassScope, MetricType

logger = logging.getLogger(__name__)


class RocAucMetric(BaseMetric):
    """Compute ROC AUC metric for binary and multi-class classification.
    Also stores TPR and FPR values to metadata for ROC curve plotting.
    """

    def __init__(
        self,
        scope: Optional[MetricMultiClassScope] = None,
        class_name: Optional[str] = None,
    ):
        """Initialize ROC AUC metric.

        Parameters
        ----------
        scope : Optional[MetricMultiClassScope]
            The scope of the metric (MICRO, MACRO, CLASS) for multi-class classification.
            Default is None for binary classification.
        class_name : Optional[str]
            The class name for CLASS scope. Ignored for other scopes."""
        super().__init__(
            type=MetricType.PROBABILITY,
            function=MetricFunction.ROC_AUC,
            class_name=class_name,
            scope=scope,
        )
        self.clear()  # Initialize metadata

    def clear(self) -> None:
        self._values = np.array([np.nan])
        self._metadata: Dict[str, list[list]] = {
            "tpr": [],
            "fpr": [],
        }

    def evaluate(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.float64],
        classes: Optional[np.ndarray[np.int_ | np.str_]] = None,
    ) -> None:
        """Evaluate the model predicted probabilities against true values.

        Parameters
        ----------
        y_true : np.ndarray[np.int_ | np.str_]
            True class labels as integers or strings.
        y_pred : np.ndarray[np.float64]
            Predicted probabilities as floats.
        classes : Optional[np.ndarray[np.int_ | np.str_]]
            Array of class labels from model.classes_ for multi-class classification.
        """
        assert (
            (y_pred >= 0.0) & (y_pred <= 1.0)
        ).all(), "Probability predictions must be in the range [0.0, 1.0]"

        if self.scope is None:
            self._compute_binary_metrics(y_true, y_pred)
        else:
            self._compute_multiclass_metrics(y_true, y_pred, classes)

    def _compute_binary_metrics(
        self, y_true: np.ndarray[np.int_ | np.str_], y_pred: np.ndarray[np.float64]
    ) -> None:
        """Compute ROC AUC for binary classification.

        This is the standard ROC AUC computation for binary classification tasks.
        """
        assert y_pred.ndim == 1, "y_pred must be a 1D array for binary classification."
        assert (
            np.unique(y_true).shape[0] == 2
        ), "y_true must contain exactly two classes for binary classification."

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        self._metadata["tpr"].append(tpr)
        self._metadata["fpr"].append(fpr)
        self._append_value(roc_auc)

    def _compute_multiclass_metrics(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.float64],
        classes: np.ndarray[np.int_ | np.str_],
    ) -> None:
        """Compute ROC AUC for multi-class classification.

        There is 3 supported scopes for multi-class ROC AUC:
        - Micro: Combine all classes as binary using one-vs-rest to compute a single ROC AUC.
        - Macro: Compute ROC AUC for each class independently and average them.
        - Class: Compute ROC AUC for a specific class as normal binary

        Details
        -------
        If the scope is class_name is not found in classes, NaN is appended with a warning.
        """
        assert (
            y_pred.ndim == 2
        ), "y_pred must be a 2D array for multi-class classification."
        assert (
            classes is not None
        ), "classes must be provided for multi-class classification from model.classes_."
        assert (
            np.unique(y_true).shape[0] > 2
        ), "y_true must contain more than two classes for multi-class classification."
        assert all(
            label in classes for label in np.unique(y_true)
        ), "All y_true labels must be present in classes."  # But all classes need not be in y_true

        # Binarize labels in a one-vs-all fashion -> shape (n_samples, n_classes).
        # Classes of [A,B,C] and 3 rows -> [[1,0,0],[0,1,0],[0,0,1]]
        n_classes = len(classes)
        y_true_binarized = label_binarize(y_true, classes=classes)

        # Single class in multi-class equals binary classification
        if self.scope == MetricMultiClassScope.CLASS:
            if self.class_name in classes:
                class_index = list(classes).index(self.class_name)
                self._compute_binary_metrics(
                    y_true_binarized[:, class_index],
                    y_pred[:, class_index],
                )
            else:
                logger.warning(
                    f"Class name '{self.class_name}' not found in classes. Skipping ROC AUC computation for this class."
                )
                self._append_value(np.nan)

        # Micro average: aggregate contributions of all classes to compute the average metric
        elif self.scope == MetricMultiClassScope.MICRO:
            fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred.ravel())
            roc_auc = auc(fpr, tpr)
            self._metadata["tpr"].append(tpr)
            self._metadata["fpr"].append(fpr)
            self._append_value(roc_auc)

        # Macro average: unweighted mean of the metrics for each class
        # This would be easy to compute with roc_auc_score(average='macro'),
        # but we need to store TPR/FPR for plotting...
        elif self.scope == MetricMultiClassScope.MACRO:
            all_fpr = np.unique(
                np.concatenate(
                    [
                        roc_curve(y_true_binarized[:, i], y_pred[:, i])[0]
                        for i in range(n_classes)
                    ]
                )
            )
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= n_classes
            roc_auc = auc(all_fpr, mean_tpr)
            self._metadata["tpr"].append(mean_tpr)
            self._metadata["fpr"].append(all_fpr)
            self._append_value(roc_auc)

        else:
            raise ValueError(
                f"Unsupported MetricMultiClassScope for multi-class ROC AUC: {self.scope}"
            )
