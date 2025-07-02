import matplotlib.pyplot as plt
import seaborn as sns
import torch


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0, clip=True):
        self.std = std
        self.mean = mean
        self.clip = clip

    def __call__(self, tensor):
        if self.clip:
            return torch.clamp(
                tensor + torch.randn(tensor.size()) * self.std + self.mean, 0, 1
            )
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def plot_confusion_matrix(confusion_matrix, labels, title=""):
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        square=True,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Remove ticks, but keep labels
    ax.tick_params(left=False, bottom=False)
    return plt.gcf(), ax
