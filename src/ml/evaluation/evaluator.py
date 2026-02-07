from enum import Enum
from typing import Generator, Optional, Protocol

import numpy as np
from matplotlib import pyplot as plt

from .metrics.base_metric import BaseMetric
from .metrics.interface import (
    MetricAggregation,
    MetricFunction,
    MetricMultiClassScope,
    MetricType,
)
from .metrics.roc_auc_metric import RocAucMetric
from .plots import plot_roc_auc_curves


# ------------------------------- Helper Classes ------------------------------- #
class MetricLogger(Protocol):
    """Protocol for metric logging implementations. (Mlflow)"""

    def log_metric(self, name: str, value: float, step: int = None) -> None: ...
    def log_figure(self, figure: plt.Figure, name: str) -> None: ...


class MetricModelType(Protocol):
    """Protocol for ML models used in evaluation."""

    def predict(self, X: np.ndarray) -> np.ndarray[np.int_ | np.str_]: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray[np.float64]: ...
    @property
    def classes(self) -> np.ndarray[np.int_ | np.str_]: ...


class MetricAggregator:
    """Aggregate metric results based on specified aggregations."""

    def __init__(
        self, metric: BaseMetric, aggregations: list[MetricAggregation]
    ) -> None:
        self.metric = metric
        self.aggregations = aggregations

    def __repr__(self) -> str:
        return (
            f"MetricAggregator(metric={self.metric}, aggregations={self.aggregations})"
        )

    def step_items(self) -> Generator[tuple[str, list[float]], None, None]:
        """Generate step-wise metric name-values pairs."""
        for value in self.metric.values():
            yield (f"{self.metric.name}.steps", value)

    def items(self) -> Generator[tuple[str, float], None, None]:
        """Generate aggregated metric name-value pairs."""
        for agg in self.aggregations:
            match agg:
                case MetricAggregation.MEAN:
                    yield (f"{self.metric.name}.mean", self.metric.mean())
                case MetricAggregation.STD:
                    yield (f"{self.metric.name}.std", self.metric.std())
                case MetricAggregation.MEDIAN:
                    yield (f"{self.metric.name}.median", self.metric.median())
                case MetricAggregation.MIN:
                    yield (f"{self.metric.name}.min", self.metric.min())
                case MetricAggregation.MAX:
                    yield (f"{self.metric.name}.max", self.metric.max())
                case MetricAggregation.MINMAX:
                    yield (f"{self.metric.name}.minmax", self.metric.minmax())
                case _:
                    raise ValueError(f'Unsupported MetricAggregation: "{agg}"')


# ------------------------------- Evaluator Class ------------------------------- #
class Evaluator:
    def __init__(self, metric_configs: list[dict]):
        self.__metrics = self.__create_metrics(metric_configs)

    def __repr__(self) -> str:
        repr_str = "Evaluator() with metrics:\n"
        for metric in self.__metrics:
            repr_str += f" - {metric}\n"
        return repr_str

    def evaluate_model(self, model: MetricModelType, X: np.ndarray) -> None:
        """Evaluate the provided model on the given data.

        This is just a convenience method that calls both
         - evaluate_classes
         - evaluate_probabilities
        and computes all included metrics for both types.

        Parameters
        ----------
        model : MetricModelType
            The ML model to evaluate. Must implement predict and predict_proba methods.
        X : np.ndarray
            The input data for evaluation.
        """
        labels = model.predict(X)
        probs = model.predict_proba(X)
        classes = model.classes

        self.evaluate_classes(y_true=labels, y_pred=labels, classes=classes)
        self.evaluate_probabilities(y_true=labels, y_proba=probs, classes=classes)

    def evaluate_probabilities(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_proba: np.ndarray[np.float64],
        classes: Optional[np.ndarray[np.int_ | np.str_]] = None,
    ) -> None:
        """Evaluate metrics based on predicted probabilities.

        This method evaluates all included metrics
        that operate on predicted probabilities.

        Parameters
        ----------
        y_true : np.ndarray[np.int_ | np.str_]
            The true class labels.
        y_proba : np.ndarray[np.float64]
            The predicted class probabilities.
        classes : Optional[np.ndarray[np.int_ | np.str_]], optional
            The list of class labels for multi-class classification
        """
        self.__evaluate(y_true, y_proba, classes, type=MetricType.PROBABILITY)

    def evaluate_classes(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.int_ | np.str_],
        classes: Optional[np.ndarray[np.int_ | np.str_]] = None,
    ) -> None:
        """Evaluate metrics based on predicted classes.

        This method evaluates all included metrics
        that operate on predicted classes.

        Parameters
        ----------
        y_true : np.ndarray[np.int_ | np.str_]
            The true class labels.
        y_pred : np.ndarray[np.int_ | np.str_]
            The predicted class labels.
        classes : Optional[np.ndarray[np.int_ | np.str_]], optional
            The list of class labels for multi-class classification
        """
        self.__evaluate(y_true, y_pred, classes, type=MetricType.CLASS)

    def log_metrics(self, logger: MetricLogger) -> None:
        """Log all evaluated metrics to the provided logger.

        All aggregated metrics,
        and step-wise metrics are always logged.

        Parameters
        ----------
        logger : MetricLogger
            The logger instance to log metrics to. This is any object
            that implements the MetricLogger protocol (Mlflow).
        """
        for aggregator in self.__metrics:
            # Log aggregated metrics
            for name, value in aggregator.items():
                logger.log_metric(name, value)
            # Log step-wise metrics for mlflow or similar
            for i, (name, values) in enumerate(aggregator.step_items()):
                logger.log_metric(name, values, step=i)

    def log_plots(self, logger: MetricLogger) -> None:

        # Plot ROC AUC curves if any RocAucMetric is included
        auc_metrics = [
            agg.metric
            for agg in self.__metrics
            if agg.metric.function == MetricFunction.ROC_AUC
        ]
        if auc_metrics:
            fig = plot_roc_auc_curves(auc_metrics)
            logger.log_figure(fig, "roc_auc_curves.png")

    def __evaluate(
        self,
        y_true: np.ndarray[np.int_ | np.str_],
        y_pred: np.ndarray[np.float64 | np.int_ | np.str_],
        classes: Optional[np.ndarray[np.int_ | np.str_]] = None,
        type: MetricType = MetricType.PROBABILITY,
    ) -> None:
        """Evaluate all included metrics of the specified type.

        Parameters
        ----------
        y_true : np.ndarray[np.int_ | np.str_]
            The true class labels.
        y_pred : np.ndarray[np.float64 | np.int_ | np.str_]
            The predicted class probabilities or labels.
        classes : Optional[np.ndarray[np.int_ | np.str_]], optional
            The list of class labels for multi-class classification
        type : MetricType, optional
            The type of metrics to evaluate (probability or class), by default MetricType.PROBABILITY
        """
        for aggregator in self.__metrics:
            if aggregator.metric.type == type:
                aggregator.metric.evaluate(y_true, y_pred, classes)

    def __create_metrics(self, metric_configs: list[dict]) -> list[MetricAggregator]:
        """Create metric instances based on the provided configurations.

        String config values are converted to enums automatically,
        but using enums directly is also supported.
        No duplicate metrics are allowed in the configuration.

        Parameters
        ----------
        metric_configs : list[dict]
            A list of metric configuration dictionaries for one metric each.
        """
        aggregator_list = []
        for config in metric_configs:

            self.__validate_metric_config(config)
            parsed_config = self.__parse_metric_config(config)

            # Check for duplicate metrics
            for included_agg in aggregator_list:
                assert (
                    parsed_config["function"] != included_agg.metric.function
                    or parsed_config["scope"] != included_agg.metric.scope
                ), (
                    f"Duplicate metrics with the same scope are not allowed: "
                    f"\"{parsed_config['function']}\" with scope \"{parsed_config['scope']}\" already included."
                )

            # Create metric instance based on function
            match parsed_config["function"]:
                case MetricFunction.ROC_AUC:
                    metric = RocAucMetric(
                        scope=parsed_config["scope"],
                        class_name=parsed_config["class_name"],
                    )
                case _:
                    raise ValueError(
                        f"Unsupported MetricFunction: \"{parsed_config['function']}\""
                    )

            # Append the MetricAggregator to the list
            aggregator_list.append(
                MetricAggregator(
                    metric=metric, aggregations=parsed_config["aggregations"]
                )
            )

        return aggregator_list

    def __parse_metric_config(self, config: dict) -> dict:
        """Parse a single metric configuration dictionary.

        This method converts string values to enums,
        and and pads missing optional values with defaults.

        Parameters
        ----------
        config : dict
            The metric configuration dictionary to parse.

        Returns
        -------
        dict
            The parsed metric configuration with enum values.
        """
        parsed_config = config.copy()

        parsed_config["function"] = self.__to_enum(config["function"], MetricFunction)
        if "scope" in config:
            parsed_config["scope"] = self.__to_enum(
                config["scope"], MetricMultiClassScope
            )
        else:
            parsed_config["scope"] = None
        parsed_config["aggregations"] = [
            self.__to_enum(agg, MetricAggregation) for agg in config["aggregations"]
        ]
        parsed_config["class_name"] = config.get("class_name", None)

        return parsed_config

    def __validate_metric_config(self, config: dict) -> None:
        """Validate a single metric configuration dictionary.

        This method checks that the provided configuration dictionary
        contains valid values for function, scope, and aggregations.

        Parameters
        ----------
        config : dict
            The metric configuration dictionary to validate.

        Raises
        ------
        ValueError
            If any of the configuration values are invalid.
        """
        # Validate function
        assert "function" in config, "Metric configuration must include 'function' key."
        assert self.__to_enum(
            config["function"], MetricFunction
        ), f"Invalid metric function: \"{config['function']}\""

        # Validate scope if present
        if "scope" in config:
            assert self.__to_enum(
                config["scope"], MetricMultiClassScope
            ), f"Invalid metric scope: \"{config['scope']}\""
            if config["scope"] == MetricMultiClassScope.CLASS:
                assert (
                    "class_name" in config
                ), "Metric configuration with CLASS scope must include 'class_name' key."

        # Validate aggregations
        assert (
            "aggregations" in config
        ), "Metric configuration must include 'aggregations' key."
        for agg in config["aggregations"]:
            assert self.__to_enum(
                agg, MetricAggregation
            ), f'Invalid metric aggregation: "{agg}"'

    def __to_enum(self, value: str, enum_class: type[Enum]) -> Enum:
        """Convert a string values to Metric enum.

        If the value is already an enum, it is returned as is.
        This enables passing either strings or enum values in the configuration.
        In practise, the configuration is likely to be in string format from the config.yaml,
        and this method just enables more flexible usage in some other projects.
        Note, string values are matched case-insensitively, and inputs are automatically uppercased
        to match the enum member names.

        Parameters
        ----------
        value : str
            The string representation of the enum value.
        enum_class : type[Enum]
            The enum class to convert to.

        Raises
        ------
        ValueError
            If the value cannot be converted to the specified enum class,
            supported values are listed in the error message.

        Returns
        -------
        Enum
            The corresponding enum value.
        """
        if isinstance(value, enum_class):
            return value
        try:
            return enum_class[value.upper()]
        except Exception:
            raise ValueError(
                f'Cannot convert "{value}" to {enum_class}.\nSupported values: {[e.name for e in enum_class]}'
            )
