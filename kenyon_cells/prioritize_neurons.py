# %% [markdown]
# # Prioritize neurons
# In this notebook, we will order the Kenyon Cells by priority.
# This will allow us to choose, potentially, only a subset of neurons to create counterfactuals for, and run attribution methods on.
# This needs to be run *after* the classification is done.
#
# The criterion we will use is the fraction of synapses that are classified as class 5.
# Class 5 is dopaminergic neurons, which are the most interesting for us, because that is the most common neuron-level mis-classification.

# %% Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array as da
import time
from tqdm import tqdm
import seaborn as sns
import scipy

# %% Important metadata
class_names = [
    "GABA",
    "Acetylcholine",
    "Glutamate",
    "Serotonin",
    "Octopamine",
    "Dopamine",
]

confusion_matrix = np.loadtxt("source_confusion_matrix.csv", delimiter=",")
# %% Loading the results and metadata
result_path = "/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr"
syn = pd.read_csv("kenyon_cell_synapses2.csv")

print(
    len(syn)
)  # printing this to compare later, to make sure we have all of the synapses
# %%
# Count number of synapses per pre-synaptic neuron
# We could filter out some of the neurons with very few synapses
(syn["pre"].value_counts() < 100).sum()


def compute_confidence(predictions, targets, confusion_matrix):
    """
    Computes the confidence for a given neuron based on synapse predictions and targets.

    $$c(n) = \frac{1}{|S_n|}\sum_{s\in S_n} C_{\hat{y}_n, \hat{y}_s}$$
    Where:
    - $c(n)$ is the confidence of neuron $n$
    - $S_n$ is the set of (pre-)synapses of neuron $n$
    - $\hat{y}_n$ is the predicted class of neuron $n$
    - $\hat{y}_s$ is the predicted class of synapse $s$
    - $C_{i,j}$ is the confusion matrix
    """
    # Get y_hat_n as the most common class in the neuron
    y_hat_n = scipy.stats.mode(predictions)[0]

    return confusion_matrix[y_hat_n, predictions].mean()


# %% Now, we load the predictions and confidence scores for each synapse
dfs = []
confidences = []
for name, df in tqdm(syn.groupby("pre")):
    predictions = da.from_zarr(result_path, component=f"{name}/predictions")
    class_predictions = predictions.argmax(axis=1).compute()
    targets = np.ones_like(predictions)
    confidence = compute_confidence(class_predictions, targets, confusion_matrix)
    df["classes"] = class_predictions
    confidences.append(confidence)
    dfs.append(df)

syn = pd.concat(dfs)
# %% Double-check that we have all of the synapses
print(len(syn))

# %% Now, we can look at the distribution of classes
syn["classes"].value_counts()

# %%
# Get the fraction of synapses that are classified as each class, for each neuron
syn_grouped = syn.groupby("pre")
syn_class_fractions = syn_grouped["classes"].value_counts(normalize=True).unstack()
syn_class_fractions = syn_class_fractions.fillna(0)
syn_class_fractions["confidence"] = confidences
syn_class_fractions["classes"] = syn_class_fractions.idxmax(axis=1)
syn_class_fractions["classes"].value_counts()


# %% Look at the distribution of synapse classes for each neuron
# %% Plot
def plot_histogram(dataframe):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.yaxis.grid(True)
    histogram_colors = sns.color_palette("tab10")
    # Remove color 2, because it has no samples
    remove_colors = set(range(len(class_names))) - set(dataframe["classes"].unique())
    for color in remove_colors:
        histogram_colors.pop(color)
    sns.histplot(
        data=dataframe,
        x="classes",
        discrete=True,
        ax=ax,
        hue="classes",
        legend=False,
        palette=histogram_colors,
    )
    ax.set_xticks(range(6))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_xlabel("")
    sns.despine(ax=ax, left=True)
    # Add the counts as text above the bars
    colors = sns.color_palette("tab10")
    for i, _ in enumerate(class_names):
        count = (dataframe["classes"] == i).sum()
        ax.text(i, count, count, ha="center", va="bottom", color=colors[i])
    return fig, ax


def plot_confidence_histogram(dataframe):
    histogram_colors = sns.color_palette("tab10")
    # Remove color 2, because it has no samples
    remove_colors = set(range(len(class_names))) - set(dataframe["classes"].unique())
    for color in remove_colors:
        histogram_colors.pop(color)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.kdeplot(
        data=dataframe,
        x="confidence",
        ax=ax,
        palette=histogram_colors,
        hue="classes",
        common_norm=False,
        legend=False,
    )
    ax.grid(True, axis="y")
    ax.set_xlabel("")
    sns.despine(ax=ax, left=True)
    return fig, ax


# %%
fig, ax = plot_confidence_histogram(syn_class_fractions)
plt.savefig(
    "results/syn_class_confidences.png", dpi=300, bbox_inches="tight", transparent=True
)
# %%
plot_histogram(syn_class_fractions)
plt.savefig(
    "results/neuron_prediction_histogram.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
plot_histogram(syn)
plt.savefig(
    "results/synapse_prediction_histogram.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

# %% Now, we sort the neurons by the fraction of synapses that are class 5
syn_class_fractions = syn_class_fractions.sort_values("confidence", ascending=False)

# %% Save the results
syn_class_fractions.to_csv("syn_class_confidences.csv")

# %%
syn_class_fractions.head()
# %%
syn_class_fractions = syn_class_fractions.sort_values(5, ascending=False)

# %%
syn_class_fractions.head()
# %%
