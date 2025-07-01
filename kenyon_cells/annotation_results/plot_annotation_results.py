# %%
# Quick script to plot the results of manual annotations
import json
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Load the annotations

annotation_file = "20240912_da_annotation_state.json"
column_description_file = "descriptions.json"
# Load the column descriptions

with open(column_description_file, "r") as f:
    column_description = json.load(f)

# Load the data
df = pd.read_json(annotation_file).T
df.columns = column_description.keys()

# %%
# Count the number of True annotations for each column
counts = df.sum() / len(df)
# Remove "Needs review" from the counts
counts = counts.drop("Needs review")
# plot the counts as a histogram
fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_ylabel("Fraction of Annotated")
ax.set_ylim(0, 1)
ax.set_title("All samples")
# Rotate the x labels
plt.xticks(rotation=90)
# Remove spines on right and top
sns.despine()

# %%
# Same thing, but without any of the samples that need review
# Drop the samples that have "Needs review" as True
df = df[df["Needs review"] == False]
print(len(df))

# Count the number of True annotations for each column
counts = df.sum() / len(df)
# Remove "Needs review" from the counts
counts = counts.drop("Needs review")
# plot the counts as a histogram

fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_ylabel("Fraction of Annotated")
ax.set_ylim(0, 1)
ax.set_title("Filtered (No 'Needs review' samples)")
# Rotate the x labels
plt.xticks(rotation=90)
# Remove spines on right and top
sns.despine()

# %%
# Save the counts to a CSV file
from datetime import datetime

today = datetime.today().strftime("%Y%m%d")
# Name the index and column
counts.index.name = "Annotation"
counts.name = "Fraction"
counts.to_csv(f"{today}_annotation_counts.csv")

# %%
# Pick a set of examples that describe the top 6 most annotated columns
top_columns = counts.sort_values(ascending=False).index[:6]
# Pick the first sample that has a True annotation for each of these columns
top_samples = set({})  # a set
for column in top_columns:
    # Add the first sample that has a True annotation for this column to the set
    top_samples = top_samples.union(set(df[df[column] == True].index[0:1]))
# %% Convert from "png" names to the (neuron_id, synapse_id) format
top_samples = [tuple(sample.replace(".png", "").split("_")) for sample in top_samples]
print(top_samples)

# %% Store this list of samples to a CSV file
top_samples_df = pd.DataFrame(top_samples, columns=["neuron_id", "synapse_id"])
top_samples_df.to_csv(f"{today}_top_samples.csv", index=False)
