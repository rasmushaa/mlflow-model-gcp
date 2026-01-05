"""
A collection of plotting utilities to evaluate ML models.
Inlcudes functions to plot ROC-AUC curves, Confusion Matrices, and more.
"""

from itertools import cycle

import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray
) -> plt.Axes:
    """Plot normalized confusion matrix on the given Axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes object to plot on.
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted target labels.

    Returns
    -------
    ax : plt.Axes
        Matplotlib Axes object with the confusion matrix plotted.
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap=plt.cm.Greens,
        ax=ax,
        values_format=".2f",
        normalize="true",
        colorbar=False,
    )
    disp.ax_.set_title("Normalized True values Confusion Matrix")
    disp.ax_.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    return ax


def plot_classification_metrics(ax: plt.Axes, metrics: dict) -> plt.Axes:
    """Plot prediction metrics on the given Axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes object to plot on.
    metrics : dict
        Dictionary of overall metrics (accuracy, f1, precision, recall).
    Returns
    -------
    ax : plt.Axes
        Matplotlib Axes object with the metrics displayed.
    """
    x = list(metrics.keys())
    y = [metrics[key] for key in x]
    bar = ax.bar(x=x, height=y)
    _bars_to_gradient(bar, y, "Greens")
    _set_bar_value_labels(bar)
    ax.set_ylim(0, 1.1)
    ax.set_title("Overall Metrics")
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.tick_params(axis="x", rotation=90)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls="--", alpha=0.7, color="gray")

    plt.tight_layout()
    return ax


def plot_roc_auc(ax: plt.Axes, fpr: dict, tpr: dict, roc_auc: dict) -> plt.Axes:
    """Plot ROC-AUC curves for multiclass classification.

    The plot style is green, and a Cycler is used to vary line styles and colors for each N classes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes object to plot on.
    fpr : dict
        Dictionary of False Positive Rates for each class and averages.
    tpr : dict
        Dictionary of True Positive Rates for each class and averages.
    roc_auc : dict
        Dictionary of AUC values for each class and averages.

    Returns
    -------
    ax : plt.Axes
        Matplotlib Axes object with the ROC-AUC plot.
    """

    # Multiline ROC curve (multi-class, KFold CV etc.)
    if len(fpr) > 1:

        # Plot micro and macro average ROC curves, if available
        if "micro" in fpr and "macro" in fpr:
            ax.plot(
                fpr["micro"],
                tpr["micro"],
                label=f'Micro (AUC = {roc_auc["micro"]:.2f})',
                color="darkgreen",
                ls="--",
            )
            ax.plot(
                fpr["macro"],
                tpr["macro"],
                label=f'Macro (AUC = {roc_auc["macro"]:.2f})',
                color="forestgreen",
                ls="--",
            )
            classes = [key for key in fpr.keys() if key not in ("micro", "macro")]
        else:
            classes = list(fpr.keys())

        N = min(len(classes), 4)
        cycle_iterator = _get_color_wheel(N)

        # Plot all the Key class ROC curves
        for label in classes:
            ax.plot(
                fpr[label],
                tpr[label],
                label=f"{label} (AUC = {roc_auc[label]:.2f})",
                **next(cycle_iterator),
                alpha=0.8,
            )

    else:
        # Single line ROC curve (binary classification)
        label = list(fpr.keys())[0]
        ax.plot(
            fpr[label],
            tpr[label],
            label=f"ROC curve (AUC = {roc_auc[label]:.2f})",
            color="darkgreen",
            ls="--",
        )

    # Figure settings
    plt.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.grid(ls="--", alpha=0.7, color="gray")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig = ax.get_figure()
    fig.subplots_adjust(
        right=0.70
    )  # Leave room on the right for legend box outside the figure
    return ax


def plot_kfold_results(ax: plt.Axes, kfold_metrics: dict) -> plt.Axes:

    metrics = list(kfold_metrics.keys())

    N = min(len(metrics), 4)
    cycle_iterator = _get_color_wheel(N)

    for metric in metrics:
        ax.plot(
            kfold_metrics[metric],
            label=metric,
            **next(cycle_iterator),
        )

    ax.set_title("K-Fold Cross Validation Metrics")
    ax.set_xlabel("Fold Number")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.0)
    ax.grid(ls="--", alpha=0.7, color="gray")
    ax.legend()
    plt.tight_layout()
    return ax


def _bars_to_gradient(bar_container, values: list, colors: str = "Greens"):
    """Apply a gradient color map to a bar container.

    Parameters
    ----------
    bar_container : matplotlib.container.BarContainer
        The bar container to apply the gradient to.
    values : list
        The list of values corresponding to each bar.
    colors : str, optional
        The name of the matplotlib colormap to use, by default 'Greens'.
    """
    cmap = plt.get_cmap(colors)
    norm = plt.Normalize(0, 1)
    for bar, value in zip(bar_container, values):
        bar.set_color(cmap(norm(value)))


def _set_bar_value_labels(bar_container):
    """Set value labels on top of each bar in a bar container.

    Parameters
    ----------
    bar_container : matplotlib.container.BarContainer
        The bar container to set labels on.
    values : list
        The list of values corresponding to each bar.
    """
    for bar in bar_container:
        height = bar.get_height()
        bar.axes.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _get_color_wheel(N: int) -> cycle:
    """Get a list of colors from a matplotlib colormap.

    Parameters
    ----------
    N : int
        The number of colors to retrieve.

    Returns
    -------
    cycle
        An iterator cycling through linestyle and color combinations.
    """
    linestyles = ["-", ":", "-."]
    colors = plt.cm.Greens(np.linspace(0.3, 0.7, N))
    combined_cycle = cycler(linestyle=linestyles) * cycler(
        color=colors
    )  # First color, then ls
    return cycle(combined_cycle)  # use itertools.cycle to repeat indefinitely
