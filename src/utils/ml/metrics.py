"""
A collection of machine learning model evaluation utilities.
Includes functions to compute metrics and generate plots for model predictions.
Each function always returns a dictionary of metrics and a dictionary of plots.
"""
import pandas as pd
import numpy as np  
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from .plots import plot_roc_auc, plot_confusion_matrix, plot_classification_metrics


def _compute_binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ''' Compute binary classification metrics for vanilla binary classification.

    Parameters
    ----------
    y_true: np.ndarray
        The true target labels
    y_pred: np.ndarray
        The predicted target labels

    Returns
    -------
    metrics: dict
        A dictionary containing evaluation metrics.
    '''
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    return metrics


def _compute_multiclass_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict:
    ''' Compute multi-class classification metrics for multi-class classification.

    Computes overall accuracy, micro and macro precision/recall/f1,
    and per-class precision/recall/f1 and support, in One VS All (ova) fashion.

    Parameters
    ----------
    y_true: np.ndarray
        The true target labels
    y_pred: np.ndarray
        The predicted target labels
    classes: np.ndarray
        The unique class labels

    Returns
    -------
    metrics: dict
        A dictionary containing evaluation metrics.
    '''
    metrics = {}

    # Micro and macro averages in One VS All fashion
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision.micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision.macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall.micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall.macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1.micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1.macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class metrics (precision, recall, f1) and support
    precisions = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)   
    
    # Support: count of true instances per class
    y_true_series = pd.Series(y_true)
    supports = y_true_series.value_counts().reindex(classes, fill_value=0).values

    # Metrics are ordered by the used class parameter in the function call
    for cls, p, r, f, s in zip(classes, precisions, recalls, f1s, supports):
        label = str(cls)
        metrics[f'precision.{label}'] = float(p)
        metrics[f'recall.{label}'] = float(r)
        metrics[f'f1.{label}'] = float(f)
        metrics[f'support.{label}'] = int(s)

    return metrics


def classification_report_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[dict, dict]:
    ''' Evaluate the model predicted labels against true values.

    Computes overall accuracy, micro and macro precision/recall/f1.
    Automatically detects if the problem is binary or multi-class classification.
    For multi-class, per-class precision/recall/f1 and support are also computed,
    in One VS All (ova) fashion.

    Parameters
    ----------
    y_true: pd.Series
        The true target labels
    y_pred: pd.Series
        The predicted target labels

    Returns
    -------
    metrics: dict
        A dictionary containing evaluation metrics.
    '''
    # Ensure arrays
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Determine class labels present in either true or predicted to report per-class metrics
    classes = np.unique(np.concatenate((y_true_arr, y_pred_arr)))

    # Compute metrics
    if len(classes) == 2:
        metrics = _compute_binary_classification_metrics(y_true_arr, y_pred_arr)
    else:
        metrics = _compute_multiclass_classification_metrics(y_true_arr, y_pred_arr, classes)

    # Build a list of selected metric keys
    selected_metrics = ['accuracy'] + [f"{m}.{avg}" for m in ['precision', 'recall', 'f1'] for avg in ['micro', 'macro']]

    # Plot confusion matrix, and metrics
    fig, axis = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]}, figsize=(10, 7), dpi=160)
    plot_confusion_matrix(axis[0], y_true, y_pred)
    plot_classification_metrics(axis[1], {k: v for k, v in metrics.items() if k in selected_metrics})
    plt.tight_layout()
    plt.close(fig)

    return metrics, {'classification_report': fig}


def _compute_binary_prediction_report_metrics(y_true, y_prob) -> tuple[dict, dict, dict]:
    ''' Evaluate the model predicted probabilities against true values for binary classification.

    Parameters
    ----------
    y_true: pd.Series
        The true target labels
    y_prob: pd.Series
        The predicted target probabilities

    Returns
    -------
    metrics: dict
        A dictionary containing evaluation metrics such as ROC AUC score.
    fpr: dict
        A dictionary containing false positive rates for ROC curve.
    tpr: dict
        A dictionary containing true positive rates for ROC curve.
    '''
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    metrics = {'roc_auc': roc_auc}

    return metrics, fpr, tpr


def _compute_multiclass_prediction_report_metrics(y_true, y_prob, classes) -> tuple[dict, dict, dict]:
    ''' Evaluate the model predicted probabilities against true values for multi-class classification.

    The ROC AUC is automatically computed for multi-class problems using
    the one-vs-all approach. Micro and macro averages are also calculated.

    Classes must be provided as a list of unique class labels.
    In thoery, classes can be inferred from y_true, but if the test set
    does not contain all classes, the ROC AUC calculation will fail.

    Parameters
    ----------
    y_true: pd.Series
        The true target labels
    y_prob: pd.Series
        The predicted target probabilities
    classes: list
        A list of unique class labels.

    Returns
    -------
    metrics: dict
        A dictionary containing evaluation metrics such as ROC AUC score for each class, micro-average, and macro-average.
    fpr: dict
        A dictionary containing false positive rates for ROC curve.
    tpr: dict
        A dictionary containing true positive rates for ROC curve.
    '''
    n_classes = len(classes)

    # Binarize labels in a one-vs-all fashion -> shape (n_samples, n_classes). 
    # [[1,0,0],[0,1,0],...] for 3 classes of [A,B,C] and 2 rows
    y_test_bin = label_binarize(y_true, classes=classes) 

    # For plotting ROC curves
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Compute ROC curve and AUC for each class in binary format, One-vs-All
    for i, label in enumerate(classes):
        fpr[label], tpr[label], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    # Compute micro-average ROC curve and AUC (The overall performance in One-vs-All)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[label] for label in classes])) # all false positive rates  
    mean_tpr = np.zeros_like(all_fpr)  # initialize mean true positive rates
    for i, label in enumerate(classes):
        mean_tpr += np.interp(all_fpr, fpr[label], tpr[label])  # interpolate all ROC curves at these points
    mean_tpr /= n_classes  # average it
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    metrics = {f'roc_auc.{k}': v for k, v in roc_auc.items()}
    return metrics, fpr, tpr


def prediction_report_metrics(y_true, y_prob, classes) -> tuple[dict, dict]:
    ''' Evaluate the model predicted probabilities against true values.

    The ROC AUC is automatically computed for multi-class problems using
    the one-vs-all approach. Micro and macro averages are also calculated.

    Classes must be provided as a list of unique class labels.
    In thoery, classes can be inferred from y_true, but if the test set
    does not contain all classes, the ROC AUC calculation will fail.

    Parameters
    ----------
    y_true: pd.Series
        The true target labels
    y_prob: pd.Series
        The predicted target probabilities
    classes: list
        A list of unique class labels.

    Returns
    -------
    metrics: dict
        A dictionary containing evaluation metrics such as ROC AUC score for each class, micro-average, and macro-average.
    plots: dict
        A dictionary containing ROC curve plot figure.
    classes: list
        A list of unique class labels.
    '''
    n_classes = len(classes)

    # Ensure arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Compute metrics
    if n_classes == 2:
        metrics, fpr, tpr = _compute_binary_prediction_report_metrics(y_true, y_prob)
    else:
        metrics, fpr, tpr = _compute_multiclass_prediction_report_metrics(y_true, y_prob, classes)

    # Plot ROC AUC curve
    fig = plt.figure(figsize=(12, 7), dpi=140)
    plot_roc_auc(fig.gca(), 
                 fpr=fpr, 
                 tpr=tpr, 
                 roc_auc={k.replace('roc_auc.', ''): v for k, v in metrics.items()})
    plt.close(fig)

    return metrics, {'roc_auc_curve': fig}