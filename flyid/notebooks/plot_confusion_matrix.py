import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cmap import Colormap


def main():
    suffix = "_selected_weights"
    # suffix = "_retry"
    # suffix = ""
    # Read the confusion matrix from CSV file
    confusion_matrix = np.loadtxt(f"confusion_matrix{suffix}.csv", delimiter=",")

    # plot the 3x3 matrix and save that
    cm_ducktales = confusion_matrix  # [:3, :3]
    # normalize the confusion matrix by "true" class
    cm_ducktales = cm_ducktales / cm_ducktales.sum(axis=1, keepdims=True)

    cmap = Colormap("colorbrewer:rdpu").to_mpl()

    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm_ducktales,
        annot=True,
        fmt=".0%",
        cmap=cmap,
        # remove x and y ticks and labels
        xticklabels=False,
        yticklabels=False,
        # font large enough to be readable
        annot_kws={"size": 18},
        # no colorbar
        cbar=False,
        linewidth=0.8,
        linecolor="black",
        # change the ends of the colormap
        vmin=-0.3,
        vmax=1.3,
    )
    plt.savefig(
        f"confusion_matrix{suffix}.png",
        dpi=250,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )


if __name__ == "__main__":
    main()
