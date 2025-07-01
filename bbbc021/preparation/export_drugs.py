# %%
import pandas as pd

# %%
df_train = pd.read_csv("data/moa_split_train.csv")
df_val = pd.read_csv("data/moa_split_val.csv")
df_test = pd.read_csv("data/moa_split_test.csv")

# %%
combined_trainidate = pd.concat([df_train, df_val], ignore_index=True)
blah = combined_trainidate[["compound", "moa"]].drop_duplicates()

# %%
# Select from the classes I kept:
classes = [
    "Actin disruptors",
    "Microtubule stabilizers",
    "Microtubule destabilizers",
    "Protein synthesis",
    "Protein degradation",
]

# Drop anything where moa is not in classes
blah = blah[blah["moa"].isin(classes)]
blah["priority"] = "no"
# %%
# Do the same for the test set
blah_test = df_test[["compound", "moa"]].drop_duplicates()
blah_test = blah_test[blah_test["moa"].isin(classes)]
blah_test["priority"] = "yes"

# %%
# Concatenate the two dataframes
blah_whole = pd.concat([blah, blah_test], ignore_index=True)

# %%
# Order by MoA
blah_whole = blah_whole.sort_values(by="moa").reset_index(drop=True)

# %%
blah_whole
# %%
# Save to csv
blah_whole.to_csv("drugs_moa.csv", index=False)

# %%
