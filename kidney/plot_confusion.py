import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(confusion_matrix, labels, title=""):
    ax = sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                     vmin=0, vmax=1, square=True, cbar=False, 
                     xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Remove ticks, but keep labels
    ax.tick_params(left=False, bottom=False)
    return plt.gcf(), ax
