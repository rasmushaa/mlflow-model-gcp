from typing import Dict, Optional

import numpy as np

from .interface import MetricFunction, MetricMultiClassScope, MetricType


class BaseMetric:
    """Base class for all evaluation metrics.

    A child class implements a specific metric function,
    such as ROC AUC, Accuracy, Precision, etc.
    The base class is called by evaluate() method,
    which appends the computed value to internal storage.
    """

    def __init__(
        self,
        type: MetricType,
        function: MetricFunction,
        scope: Optional[MetricMultiClassScope] = None,
        class_name: Optional[str] = None,
    ):
        """Initialize the BaseMetric.

        Parameters
        ----------
        type : MetricType
            The type of the metric (e.g., PROBABILITY, CLASS). This defines the expected
            input type for predictions, and helps to validate the input signature for each metric type.
        function : MetricFunction
            The function type of the metric (e.g., ROC_AUC, ACCURACY). This is given by the child class.
        scope : Optional[MetricMultiClassScope], optional
            The scope of the metric (e.g., MICRO, MACRO, CLASS). This defines how the metric is computed
            across multiple classes. For Binary classification, this is always None, by default None.
        class_name : Optional[str], optional
            The class name if the metric is class-specific, else None. This is required for CLASS scope in multiclass metrics,
            by default None.
        """
        self.__type = type
        self.__class_name = class_name
        self.__function = function
        self.__scope = scope
        self._metadata: Dict[str, any] = {}
        self.__values: np.array = np.array([np.nan])
        self.__validate_configuration()

    def __repr__(self) -> str:
        return (
            f"Metric(type={self.__type}, "
            f"class_name={self.__class_name}, "
            f"function={self.__function}, "
            f"scope={self.__scope}, "
            f"values={self.__values}, "
            f"metadata={self._metadata})"
        )

    def evaluate(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.float64 | np.int_ | np.str_],
        classes: Optional[np.ndarray[np.int_ | np.str_]] = None,
    ) -> None:
        """Evaluate the metric based on true and predicted values.

        The actual computation is implemented in each subclass.
        The type of y_pred depends on the metric type,
        and optional classes are required for multi-class metrics.

        This method defines the common interface for all metrics,
        ensuring consistent input handling and validation.

        Parameters
        ----------
        y_true : np.ndarray[np.int_ | np.str_]
            The true class labels as a 1D numpy array of integers or strings.
        y_pred : np.ndarray[np.float64 | np.int_ | np.str_]
            The predicted values as a 1D numpy array. The type depends on the metric type:
            - For PROBABILITY type metrics, this should be a float array of predicted probabilities.
            - For CLASS type metrics, this should be an int or str array of predicted class labels,
              with values matching those in y_true.
        classes : Optional[np.ndarray[np.int_ | np.str_]], optional
            The array of all evaluated classes in order from model.classes_,
            required for multi-class metrics, by default None.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    @property
    def name(self) -> str:
        """Get the metric name.

        The name is constructed based on the function, scope, and class_name (if applicable).
        The format is: "{function}[.{scope}][.{class_name}]"
        Note, the name returned from items() includes additional suffixes.

        Returns
        -------
        str
            The name of the metric.
        """
        parts = []
        parts.append(self.__function.value)
        if self.__scope:
            parts.append(self.__scope.value)
        if self.__class_name:
            parts.append(str(self.__class_name))
        return ".".join(parts)

    @property
    def metadata(self) -> Dict[str, any]:
        """Get the metadata dictionary.

        Each metric can store additional metadata during evaluation,
        such as true positive rates and false positive rates for ROC AUC.
        Note, the metadata is metric-specific and may vary!

        Details
        -------
        Only this method is exposed to the users.
        Child classes can modify the _metadata attribute directly

        Returns
        -------
        dict
            The metadata dictionary associated with the metric.
        """
        return dict(self._metadata)

    @property
    def type(self) -> MetricType:
        """Get the metric type.

        Returns
        -------
        MetricType
            The type of the metric (e.g., PROBABILITY, CLASS).
        """
        return self.__type

    @property
    def class_name(self) -> Optional[str]:
        """Get the class name for class-specific metrics.

        Returns
        -------
        Optional[str]
            The class name if the metric is class-specific, else None.
        """
        return self.__class_name

    @property
    def function(self) -> MetricFunction:
        """Get the metric function.

        Returns
        -------
        MetricFunction
            The function type of the metric (e.g., ROC_AUC, ACCURACY).
        """
        return self.__function

    @property
    def scope(self) -> Optional[MetricMultiClassScope]:
        """Get the metric scope.

        Returns
        -------
        Optional[MetricMultiClassScope]
            The scope of the metric (e.g., MICRO, MACRO, CLASS, BINARY).
        """
        return self.__scope

    def values(self) -> np.array:
        """Get the evaluation values stored in the metric.

        Thsese are the raw values collected during each evaluation step,
        before any aggregation is applied.

        Returns
        -------
        np.array
            The array of evaluation values.
        """
        return self.__values.copy()

    def mean(self) -> float:
        """Get the mean of the stored evaluation values.

        Returns
        -------
        float
            The mean of the evaluation values.
        """
        return float(np.nanmean(self.__values))

    def median(self) -> float:
        """Get the median of the stored evaluation values.

        Returns
        -------
        float
            The median of the evaluation values.
        """
        return float(np.nanmedian(self.__values))

    def min(self) -> float:
        """Get the minimum of the stored evaluation values.

        Returns
        -------
        float
            The minimum of the evaluation values.
        """
        return float(np.nanmin(self.__values))

    def max(self) -> float:
        """Get the maximum of the stored evaluation values.

        Returns
        -------
        float
            The maximum of the evaluation values.
        """
        return float(np.nanmax(self.__values))

    def std(self) -> float:
        """Get the standard deviation of the stored evaluation values.

        Returns
        -------
        float
            The standard deviation of the evaluation values.
        """
        return float(np.nanstd(self.__values))

    def minmax(self) -> float:
        """Get the min-max range of the stored evaluation values.

        Returns
        -------
        float
            The difference between the maximum and minimum evaluation values.
        """
        return float(np.nanmax(self.__values) - np.nanmin(self.__values))

    def clear(self) -> None:
        """Clear the metric's stored values and metadata.

        This has to be implemented by each subclass, as the internal
        storage structure may vary between different metric types.
        """
        raise NotImplementedError("Subclasses must implement the clear method.")

    def _append_value(self, value: float) -> None:
        """Append a new evaluation value to the metric's values array.

        This method is used internally by subclasses during evaluation
        to store the computed metric value for each evaluation step.

        Parameters
        ----------
        value : float
            The new evaluation value to append.
        """
        if np.isnan(self.__values).all():
            self.__values = np.array([value])
        else:
            self.__values = np.append(self.__values, value)

    def __validate_configuration(self) -> None:
        """Validate the metric configuration based on scope, and class_name.

        Raises
        -------
        AssertionError
            If the configuration is invalid based on the rules defined for each target type.
        """
        if self.__scope == MetricMultiClassScope.CLASS:
            assert (
                self.__class_name is not None
            ), "Class-specific metrics require a class_name."
        else:
            assert (
                self.__class_name is None
            ), "Non-class-specific metrics do not support class_name."
