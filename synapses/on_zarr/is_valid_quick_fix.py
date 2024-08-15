"""
I hadn't added the `is_valid` array to the zarr file, adding it here.
"""

import numpy as np
import zarr


if __name__ == "__main__":
    zfile = zarr.open(
        "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr",
        mode="a",
    )
    classes = sorted(list(zfile["counterfactuals"].array_keys()))
    labels = np.array(zfile["labels"][:])
    for class_name in classes:
        # Data is only valid if the label is not the same as the target class
        is_valid = labels != classes.index(class_name)
        zfile.create_dataset(f"is_valid/{class_name}", data=is_valid)
