"""Plotting utilities for model evaluation (e.g. confusion matrices, ROC curves)."""

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    ax=None,
    cmap="Blues",
    colorbar=True,
    values_format="d",
):
    """ 
    Plot a single confusion matrix for multiclass (or binary) predictions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap=cmap,
        values_format=values_format,
        ax=ax,
        colorbar=colorbar,
    )
    ax.set_title(title)
    return ax


def plot_confusion_matrices(
    results_list,
    class_names=None,
    figsize=None,
    ncols=3,
    cmap="Blues",
    values_format="d",
    suptitle="Confusion matrices",
):
    """
    Plot multiple confusion matrices in a grid (e.g. one per model).
    """
    n_plots = len(results_list)
    if n_plots == 0:
        raise ValueError("results_list must contain at least one (name, y_true, y_pred).")

    nrows = (n_plots + ncols - 1) // ncols
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, (name, y_true, y_pred) in enumerate(results_list):
        ax = axes_flat[i]
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=class_names,
            title=name,
            ax=ax,
            cmap=cmap,
            colorbar=True,
            values_format=values_format,
        )

    # Hide unused subplots
    for j in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig, axes

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_multimodel_roc(models_dict, X_test, y_test):
    """
    Plots ROC curves for multiple models on a single graph.
    
    Args:
        models_dict (dict): Format {'Model Name': model_object}
        X_test: Features for testing
        y_test: True labels
    """
    plt.figure(figsize=(10, 7))
    
    for name, model in models_dict.items():
        # Get predicted probabilities for the positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    # Plot the random "luck" line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()