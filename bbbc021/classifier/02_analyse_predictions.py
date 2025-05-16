# %% [markdown]
# # Analysing the predictions of the model
#
# Based on the confusion matrix ![confusion_matrix_epoch_28.png](confusion_matrix_epoch_49.png)
# it seems that the model is often confusing things for DMSO, which is the control.
#
# My current hypothesis is that the low concentrations are being mis-classified, whereas the high concentrations are being classifier correctly.
# In this notebook, I will verify this hypoythesis.

# %%
import pandas as pd
from sklearn.metrics import confusion_matrix
from classification_utils import plot_confusion_matrix
import matplotlib.pyplot as plt

predictions = pd.read_csv("predictions_epoch_49.csv")
gt = pd.read_csv("preparation/data/moa_split_test.csv")

# %% [markdown]
# In the predictions, the `filename` column can be used to match to the ground truth.
# First, I need to extract the base name from the path.
# %%
predictions.filename = predictions.filename.apply(lambda x: x.split("/")[-1])
# %% [markdown]
# Now the nams are formatted as:
# "{plate}_{well}_{table}_{image}_{cell}.tiff", where the includes an underscore as well.
# I will add all of this metadata to the predictions dataframe, by splitting the name.


# %%
def split_filename(filename):
    """
    Split the filename into its components.
    """
    # Remove the .tiff extension
    filename = filename[:-5]
    # Get the cell, image, table, and well from the filename
    well, table, image, cell = filename.split("_")[2:]
    # Get the plate from the filename by removing everything else:
    plate = "_".join(filename.split("_")[:2])
    return str(plate), str(well), int(table), int(image), int(cell)


predictions[["plate", "well", "table", "image", "cell"]] = predictions.filename.apply(
    split_filename
).apply(pd.Series)
# %%
print(predictions.dtypes)
predictions.head()
# %% [markdown]
# Now I can match the predictions to their image on the following:
# {
#   "plate": "Image_Metadata_Plate_DAPI",
#   "well": "Image_Metadata_Well_DAPI",
#   "table": "TableNumber",
#   "image": "ImageNumber",
# }
# The cell is not needed, as the predictions are made on the whole image.

# %%
joined = pd.merge(
    predictions,
    gt,
    how="left",
    left_on=["plate", "well", "table", "image"],
    right_on=[
        "Image_Metadata_Plate_DAPI",
        "Image_Metadata_Well_DAPI",
        "TableNumber",
        "ImageNumber",
    ],
)
joined.dropna(inplace=True)
joined.drop_duplicates(inplace=True)
# %% [markdown]
# Now I can get confusion matrixes, filtered for various things.
# First, I will get the confusion matrix for all predictions, making sure it mathces the one above.
# %%
cm1 = confusion_matrix(joined["labels"], joined["predictions"])
cm2 = confusion_matrix(predictions["labels"], predictions["predictions"])
assert (cm1 == cm2).all()
# %% [markdown]
# Now let's look at concentrations.
# I Will filter out all samples whose concentration is <= the mean concentration for that compound.
# %%

# %%
moa_info = pd.read_csv("preparation/data/moa.csv")
# each compound only has one MoA?
for compound, group in moa_info.groupby("compound"):
    assert (
        group["moa"].nunique() == 1
    ), f"Compound {compound} has multiple MoAs: {group['moa'].unique()}"

# Minimum concentration for each compound in moa_info
threshold_concentration = moa_info.groupby("compound")["concentration"].min()
# OR Get percentiles of concentration for each compound
# threshold_concentration = gt.groupby("Image_Metadata_Compound")[
#     "Image_Metadata_Concentration"
# ].quantile(0.5)

# %%
# Drop anything whose concentration is less than the mean concentration
filtered = joined[
    joined["Image_Metadata_Concentration"]
    >= threshold_concentration[joined["Image_Metadata_Compound"]].values
]
# And get the confusion matrix
cm = confusion_matrix(filtered["labels"], filtered["predictions"])
plot_confusion_matrix(
    cm,
    classes=filtered["moa"].unique(),
)
plt.show()

# %% [markdown]
# # Preparing the data and the model for the next step
# Given the results above, we are going to select a subset of the classes, and modify the model to predict only those classes.
# The classes are:
# - Actin disruptors
# - DMSO
# - Microtubule destabilizers
# - Microtubule stabilizers
# - Protein degradation
# - Protein synthesis
#
# First, I will create a dataset with only these classes, by using soft-links to the original images.
# %%
import json
from pathlib import Path

data_location = Path("/nrs/funke/adjavond/data/bbbc021_processed/")
output_location = Path("/nrs/funke/adjavond/data/bbbc021_subset/")
# Create the output directory
output_location.mkdir(parents=True, exist_ok=True)
# %%

classes = [
    "actin_disruptors",
    "dmso",
    "microtubule_destabilizers",
    "microtubule_stabilizers",
    "protein_degradation",
    "protein_synthesis",
]


for split in ["train", "val", "test"]:
    split_location = output_location / split
    split_location.mkdir(parents=True, exist_ok=True)
    # make a soft link to the original data folders
    for moa in classes:
        cls_location = split_location / moa
        cls_location.symlink_to(data_location / f"{split}/{moa}")

# %%
# Check that the links are working
p = list((output_location / "train").iterdir())[0]
print(p, p.is_symlink(), p.resolve())


# %% [markdown]
# To create a new version of the model, I will also need know where
# in the original list of classes the selected classes are.
#
# I will create a mapping from the selected classes to the original classes.
# %%

# Get the original classes from the data_location
original_classes = sorted([s.name for s in (data_location / "train").iterdir()])
new_classes = {c: original_classes.index(c) for c in classes}

# Save the mapping to a file
with open("class_subset.json", "w") as f:
    json.dump(new_classes, f)
