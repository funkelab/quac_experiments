# %%
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

moa = pd.read_csv("data/moa.csv")
# %%
moa.head()
# %%
# I want to check, here that each compound only has one MoA
# This would mean that I can relatively freely ignore the concentration
# and just consider drug -> moa
# How many MoAs per-compound?
for compound, group in moa.groupby("compound"):
    n_moa = group.moa.nunique()
    assert n_moa == 1
# %%
# Since there is exactly one MoA per compound, I can drop the concentration column,
# and then drop all duplicatesj
print(len(moa))
moa = moa.drop(columns=["concentration"])
moa = moa.drop_duplicates()
print(len(moa))
# I get, as I expect, 39 unique compounds: this is the 38 that
# were applied, plus DMSO which is the negative control
# %%
# So now the question is how to organize the data given this task.
# First: do a leave-one-out situation for the compounds.
# Second: make sure that the data does not have other confounders.
# To check for confounders, I have to look more closely at the data.

dataset = pd.read_csv("data/BBBC021_v1_image.csv")

# %%
# Since there is missing data in the dataset, I need to drop all the rows whose
# Image_Metadata_Plate_DAPI is not a directory in my base dir
base_dir = Path("/nrs/funke/adjavond/data/bbbc021/")
available_plates = [subdir.name for subdir in base_dir.iterdir() if subdir.is_dir()]

# I will drop all the rows whose Image_Metadata_Plate_DAPI is not in available_plates
dataset = dataset[dataset.Image_Metadata_Plate_DAPI.isin(available_plates)].reset_index(
    drop=True
)

# %%
# First question, are the compounds well split across plates and wells.
# First, I want to make sure that each compound is split over several unique plates

for compound, group in dataset.groupby("Image_Metadata_Compound"):
    n_plates = group.Image_Metadata_Plate_DAPI.nunique()
    assert n_plates > 1, f"Compound {compound} is only in one plate"
# %%
# Second, I want to make sure that each compound is split over
# a distinct set of wells
for compound, group in dataset.groupby("Image_Metadata_Compound"):
    n_wells = group.Image_Metadata_Well_DAPI.nunique()
    assert n_wells > 1, f"Compound {compound} is only in one well"

# Now, I'm fairly certain that the only reason that they are in fact
# split across wells is because of the different concentrations.
# Nevertheless, since I am going to be grouping concentrations together,
# I hope that this will not be a problem. I will continue to consider it,
# however, if something weird happens.
# %%
# Now, I want to check the number of compounds per MoA
for name, group in moa.groupby("moa"):
    n_compounds = group.compound.nunique()
    print(f"{name}: {n_compounds} compounds")
# %%
# For each case, I will select one compound to be held-out as "test"
# except for DMSO, which I will just split via the replicates (1+2 is train-val, 3 is test)
# For training and validation, I will use the rest of the compounds, mixed.
# %%
# I want to do k-fold cross-validation for which compound to hold out
# However, this will have combinatorial explosion (like 4^12 options, which is crazy).
# So I will do the following:
# 1. Choose a compound to hold out - train.
# 2. For MoAs that have low performance, change the held-out compound - keep the rest of them.
# 3. Continue until I have a good classifier.
#

# %%
dmso_df = dataset[dataset.Image_Metadata_Compound == "DMSO"]
dmso_trainidate = dmso_df[dmso_df.Replicate <= 2]
dmso_test = dmso_df[dmso_df.Replicate == 3]

# for every other MoA, just choose the first compound as test
moa_test = {}
for name, group in moa.groupby("moa"):
    compound = group.compound.iloc[0]
    moa_test[name] = compound
    print(f"{name}: {compound}")

# make the test set
moa_test_df = dataset[dataset.Image_Metadata_Compound.isin(moa_test.values())]
# add dmso test set
moa_test_df = pd.concat([moa_test_df, dmso_test])

# make the train set
moa_train_df = dataset[~dataset.Image_Metadata_Compound.isin(moa_test.values())]
moa_train_df = pd.concat([moa_train_df, dmso_trainidate])

# %%
# Add the MoA to the train and test set
moa_test_df = moa_test_df.merge(
    moa[["compound", "moa"]], left_on="Image_Metadata_Compound", right_on="compound"
)
moa_train_df = moa_train_df.merge(
    moa[["compound", "moa"]], left_on="Image_Metadata_Compound", right_on="compound"
)
# %%
# Check that no compounds are in both the train and test set, except for DMSO

for compound in moa_test_df.Image_Metadata_Compound.unique():
    if compound == "DMSO":
        continue
    assert (
        compound not in moa_train_df.Image_Metadata_Compound.unique()
    ), f"Compound {compound} is in both train and test set"

# %%
# Check that the train and test set have the same number of MoAs
assert len(moa_test_df.moa.unique()) == len(
    moa_train_df.moa.unique()
), "Train and test set have different number of MoAs"

# %%
# Split train into train and validation set
# use 70-30 split, but make sure to stratify by MoA

moa_train_df, moa_val_df = train_test_split(
    moa_train_df,
    test_size=0.3,
    stratify=moa_train_df.moa,
    random_state=42,
)
# Check that the train and validation set have the same number of MoAs
assert len(moa_train_df.moa.unique()) == len(
    moa_val_df.moa.unique()
), "Train and validation set have different number of MoAs"
# %%
# Save the train and test set
moa_test_df.to_csv("data/moa_split_test.csv", index=False)
moa_val_df.to_csv("data/moa_split_val.csv", index=False)
moa_train_df.to_csv("data/moa_split_train.csv", index=False)
# %%
