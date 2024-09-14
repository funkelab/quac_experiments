# %%
import zarr
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# %%
results = zarr.open(
    "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr", "r"
)
labels = results["labels"][:]
preds = results["predictions"][:]

# %%
cm = confusion_matrix(labels, preds.argmax(axis=1), normalize="true")
sns.heatmap(cm, annot=True)

# %% Store the overall confusion matrix in the format used in the paper
# This is a CSV file with three columns: label,prediction,value
# The value is the percentage of the confusion matrix
with open("paper_results/source_confusion_matrix.csv", "w") as f:
    f.write("label,prediction,value\n")
    for pred in range(6):
        for lbl in range(6):
            f.write(f"{lbl},{pred},{cm[lbl, pred]}\n")

# %%
