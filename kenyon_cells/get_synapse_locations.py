# TODO Clean up this script
# %%
from fafbseg import flywire

# Just to see what materializations are available
flywire.get_materialization_versions()

# %%
# Get all supervoxels (?) that are annotated as Kenyon cells
ann = flywire.search_annotations("Kenyon_Cell")
# %%
# Shows that most of them are *incorrectly* classified as Dopaminergic
ann["top_nt"].value_counts()
# %% for filtering
from joblib import Parallel, delayed
import multiprocessing


def filter_synapses(df, image_size=(12.8, 128, 128), voxel_size=(40, 4, 4)):
    tolerance_z = image_size[0] * voxel_size[0]
    tolerance_y = image_size[1] * voxel_size[1]
    tolerance_x = image_size[2] * voxel_size[2]
    blah = df.copy()
    # order by cleft score
    blah.sort_values("cleft_score", ascending=False, inplace=True)
    blah["pre_x"] = blah["pre_x"].mul(1 / tolerance_x).round()
    blah["pre_y"] = blah["pre_y"].mul(1 / tolerance_y).round()
    blah["pre_z"] = blah["pre_z"].mul(1 / tolerance_z).round()

    blah.drop_duplicates(subset=["pre_x", "pre_y", "pre_z"], keep="first", inplace=True)

    best_synapses = df.loc[blah.index]
    return best_synapses


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in dfGrouped
    )
    return pd.concat(retLst)


# %%
# Get the synapses
from tqdm import tqdm
from contextlib import redirect_stdout
from requests.exceptions import HTTPError
import navis

syn = []
pbar = tqdm(ann.iterrows(), total=len(ann))
for i, row in pbar:
    with redirect_stdout(None):
        try:
            df = flywire.get_synapses(
                row.root_id,
                pre=True,
                post=False,
                materialization="auto",
                filtered=True,
                clean=True,
            )
            fafbseg_coordinates = df[["pre_x", "pre_y", "pre_z"]].values
            to_fafb14_coordinates = navis.xform_brain(
                fafbseg_coordinates, source="FLYWIRE", target="FAFB14"
            )
            # TODO result = applyParallel(df.groupby("pre"), filter_synapses)
            result = df.copy()
            result[["x", "y", "z"]] = to_fafb14_coordinates
            syn.append(result)
        except HTTPError as e:
            print(f"Error for {row.root_id}: {e}")

# %%
import pandas as pd

synapses = pd.concat(syn)
# %%
# Just a quick check
# synapses.head()
# len(synapses)

# %%
# Save the synapses to CSV
synapses.to_csv("kenyon_cell_synapses2.csv", index=False)

# %%
