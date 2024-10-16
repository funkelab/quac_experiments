# %%
import dask.array as da
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import zarr


def get_confusion_matrix(experiment_dir, kind):
    filepath = Path(experiment_dir) / "output.zarr"
    zfile = zarr.open(filepath, "r")
    predictions = []
    labels = []
    for source in range(4):
        for target in range(4):
            if source == target:
                continue
            pred = da.from_zarr(
                zfile[f"{kind}/generated_predictions/{source}_{target}"]
            )
            predictions.append(pred)
            labels.append(da.ones(pred.shape[0]) * target)

    predictions = da.concatenate(predictions)
    labels = da.concatenate(labels)

    # Swap the first two axes of the predictions -> batch-first
    predictions = predictions.transpose(1, 0, 2)

    # Compute the confusion matrices
    cm = []
    for batch in predictions:
        cm.append(confusion_matrix(labels, np.argmax(batch, axis=-1)))

    return da.stack(cm)


# %%
directory = "/nrs/funke/adjavond/projects/quac/fictus/"

experiments = [
    "stargan_20241013",
    "stargan_20241014",
    "stargan_20241014_2",
]

for experiment in experiments:
    print(experiment)
    for kind in ["latent", "reference"]:
        cm = get_confusion_matrix(directory + experiment, kind)
        print("\t", kind)
        for batch in cm:
            print("\t\t", (batch.trace() / batch.sum()).compute())
        summed = cm.sum(axis=0)
        print("\t\t average:", (summed.trace() / summed.sum()).compute())
# %%
