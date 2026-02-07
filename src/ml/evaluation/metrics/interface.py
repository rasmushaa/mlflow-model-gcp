"""
This module defines enumerations for all metric configurations.
Note: the config.yaml file settings must match these enums KEYS exactly.
"""

from enum import Enum


class MetricFunction(Enum):
    """Enumeration of supported metric functions.

    This enum is used to specify the metric function
    to be calculated.
    """

    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"


class MetricMultiClassScope(Enum):
    """Enumeration of supported metric scopes.

    This enum is used to specify the scope at which
    the metric is calculated.

    Details
    -------
    - MICRO: Calculate the metric globally by considering
      all instances and classes. This is the standard and only
      supported scope for binary classification.
    - MACRO: Calculate the metric for each class independently
      and then take the unweighted mean. This treats all classes
      equally, regardless of their frequency.
    - CLASS: Calculate the metric for a specific class.
      Only applicable for multi-class classification.
    """

    MICRO = "micro"
    MACRO = "macro"
    CLASS = "class"


class MetricAggregation(Enum):
    """Enumeration of supported metric aggregation methods.

    This enum is used to specify how to aggregate metric values
    over multiple evaluation steps. Note: this is always required
    even if only a single evaluation is performed.
    """

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    STD = "std"
    MEDIAN = "median"
    MINMAX = "minmax"


class MetricType(Enum):
    """Enumeration of supported metric types.

    This enum is used to specify whether the metric
    is calculated based on predicted classes or probabilities.

    This is NOT specified by user directly, but hardcoded
    in each metric implementation depending on the function.

    Details
    -------
    - CLASS: Metric is calculated using predicted class labels.
    - PROBABILITY: Metric is calculated using predicted probabilities.
    """

    PROBABILITY = "probability"
    CLASS = "class"
