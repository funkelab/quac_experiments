# %%
import zarr
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %%
results = zarr.open(
    "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr", "r"
)
labels = results["labels"][:]
preds = results["predictions"][:]

# %%
cm = confusion_matrix(labels, preds.argmax(axis=1), normalize="true")
sns.heatmap(cm, annot=True)

# %%
