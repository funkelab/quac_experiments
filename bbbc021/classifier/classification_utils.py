import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, classes):
    """
    Plot the confusion matrix with annotations as fractions (e.g., 20 / 45),
    and color intensity representing the fraction. Handles division by zero.
    """
    row_sums = cm.sum(axis=1, keepdims=True)  # Row-wise totals
    # Avoid division by zero by replacing zeros with ones temporarily
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm / row_sums_safe  # Normalize by row for color intensity

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        cm_normalized,
        annot=False,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=False,
    )

    # Add annotations as fractions
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            total = int(row_sums[i][0])
            if total == 0:
                annotation = f"{count} / 0"
            else:
                annotation = f"$\\frac{{{count}}}{{{total}}}$"  # LaTeX fraction
            ax.text(
                j + 0.5,
                i + 0.5,
                annotation,
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.suptitle("Confusion Matrix", fontsize=16)
    fig.tight_layout()
    return fig


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.confusion_matrix[t, p] += 1

    def compute(self):
        return self.confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
