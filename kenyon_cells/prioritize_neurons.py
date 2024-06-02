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

# %% Now, we load the predictions and confidence scores for each synapse
dfs = []
for name, df in tqdm(syn.groupby("pre")):
    predictions = da.from_zarr(result_path, component=f"{name}/predictions")
    df["classes"] = predictions.argmax(axis=1).compute()
    df["confidence"] = predictions.max(axis=1).compute()
    dfs.append(df)

syn = pd.concat(dfs)
# %% Double-check that we have all of the synapses
print(len(syn))

# %%
# Get the fraction of synapses that are classified as each class, for each neuron
syn_grouped = syn.groupby("pre")
syn_class_fractions = syn_grouped["classes"].value_counts(normalize=True).unstack()
syn_class_fractions = syn_class_fractions.fillna(0)

# %% Now, we sort the neurons by the fraction of synapses that are class 5
syn_class_fractions = syn_class_fractions.sort_values(5, ascending=False)

# %% Save the results
syn_class_fractions.to_csv("syn_class_fractions.csv")
