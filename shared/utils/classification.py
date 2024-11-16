"""Helper functions for classification tasks."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_metric_curve(
        xvalues, yvalues, thresholds, title=None,
        figsize=(8, 7), show_thresholds=True, show_legend=True,
        ylabel='X', xlabel='Y', ax=None, text_delta=0.01,
        label="Metric Curve", color="royalblue", show=False,
        fill=None,
    ):
    """Plot a metric curve, e.g., PR curve or ROC curve."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    ax.plot(xvalues, yvalues, marker='o', label=label, color=color)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    
    if fill is not None:
        yticks = ax.get_yticks()
        ax.fill_between(xvalues, yvalues, "", alpha=0.08, color=color)
        # Add `fill` inside the curve
        # Find a single (x, y) s.t. it is inside the curve
        ax.text(0.4, 0.5, fill, color=color)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks])
        ax.set_ylim(-0.08, 1.08)
    
    # Show thresholds
    if show_thresholds:
        for x, y, t in zip(xvalues, yvalues, thresholds):
            ax.text(x + text_delta, y + text_delta, np.round(t, 2), color=color, alpha=0.5)
    
    if show_legend:
        ax.legend()
    
    if show:
        plt.show()
