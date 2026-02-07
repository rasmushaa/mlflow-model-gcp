import logging
from typing import Optional

import numpy as np
from sklearn.metrics import recall_score

from .base_metric import BaseMetric
from .interface import MetricFunction, MetricMultiClassScope, MetricType

logger = logging.getLogger(__name__)


class RecallMetric(BaseMetric):

    def __init__(
        self,
        scope: Optional[MetricMultiClassScope] = None,
        class_name: Optional[str] = None,
    ):
        """Initialize Recall metric.

        Parameters
        ----------
        scope : Optional[MetricMultiClassScope]
            The scope of the metric (MICRO, MACRO, CLASS) for multi-class classification.
            Default is None for binary classification.
        class_name : Optional[str]
            The class name for CLASS scope. Ignored for other scopes."""
        super().__init__(
            type=MetricType.PROBABILITY,
            function=MetricFunction.RECALL,
            class_name=class_name,
            scope=scope,
        )
        self.clear()  # Initialize metadata

    def clear(self) -> None:
        self._values = np.array([np.nan])
        self._metadata = None

    def evaluate(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.int_ | np.str_],
        classes: Optional[np.ndarray[np.int_ | np.str_]] = None,
    ) -> None:
        """Evaluate the model predicted probabilities against true values.

        Parameters
        ----------
        y_true : np.ndarray[np.int_ | np.str_]
            True class labels as integers or strings.
        y_pred : np.ndarray[np.int_ | np.str_]
            Predicted class labels as integers or strings.
        classes : Optional[np.ndarray[np.int_ | np.str_]]
            Array of class labels from model.classes_ for multi-class classification.
        """
        assert not np.issubdtype(
            y_pred.dtype, np.floating
        ), "y_pred must contain class labels (int or str), not float probabilities"

        if self.scope is None:
            self._compute_binary_metrics(y_true, y_pred)
        else:
            self._compute_multiclass_metrics(y_true, y_pred, classes)

    def _compute_binary_metrics(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.int_ | np.str_],
    ) -> None:
        """Compute a normal recall for binary classification."""
        assert (
            np.unique(y_true).shape[0] == 2
        ), "y_true must contain exactly two classes for binary classification."
        recall = recall_score(y_true, y_pred)
        self._append_value(recall)

    def _compute_multiclass_metrics(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.int_ | np.str_],
        classes: np.ndarray[np.int_ | np.str_],
    ) -> None:
        """Compute recall for multi-class classification based on the scope."""
        assert (
            classes is not None
        ), "classes must be provided for multi-class classification from model.classes_."
        assert (
            np.unique(y_true).shape[0] > 2
        ), "y_true must contain more than two classes for multi-class classification."
        assert all(
            label in classes for label in np.unique(y_true)
        ), "All y_true labels must be present in classes."  # But all classes need not be in y_true

        # Micro averaged recall
        if self.scope == MetricMultiClassScope.MICRO:
            recall = recall_score(y_true, y_pred, average="micro", labels=classes)
            self._append_value(recall)

        # Macro averaged recall
        elif self.scope == MetricMultiClassScope.MACRO:
            recall = recall_score(y_true, y_pred, average="macro", labels=classes)
            self._append_value(recall)

        # Class-specific recall (compute for all classes, then select)
        elif self.scope == MetricMultiClassScope.CLASS:
            class_index = list(classes).index(self.class_name)
            recalls = recall_score(
                y_true, y_pred, labels=classes, average=None, zero_division=0
            )
            self._append_value(recalls[class_index])

        else:
            raise ValueError(f'Unsupported MetricMultiClassScope: "{self.scope}"')
