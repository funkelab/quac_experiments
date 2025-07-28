# %%
import zarr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add random seed for multiple runs
random_seed = 24

# %%
output_location = Path(
    f"/groups/funke/funkelab/projects/quac/experiments/synapses/annotation/phase1/phase1_{random_seed}"
)
output_location.mkdir(exist_ok=True, parents=True)
dir_A = output_location / "A"
dir_B = output_location / "B"
dir_M = output_location / "M"
dir_A.mkdir(exist_ok=True, parents=True)
dir_B.mkdir(exist_ok=True, parents=True)
dir_M.mkdir(exist_ok=True, parents=True)

input_location = "/groups/funke/funkelab/projects/quac/experiments/synapses/20240808_results_test.zarr"
# %% Loading the data
results = zarr.open(input_location, "r")

evaluation = results["evaluations"]

# %% QuAC scores, method-agnostic
quac_scores = pd.DataFrame(
    {method: evaluation[f"{method}/quac_scores"][:] for method in evaluation.keys()}
)

best_method = quac_scores.idxmax(axis=1)
best_quac = quac_scores.max(axis=1)

# %%
# The indices that are "valid" (i.e., where the source and target are not the same) have been annotated using a boolean array.
# Here, we extract that information and store it in a matrix.
# Each row corresponds to a *target* class.
# Each column corresponds to one item in the original set of images.
all_valid = np.stack(
    [results[f"is_valid/{key}"][:] for key in results["is_valid"].keys()]
)
# %%
# Describe the hybrid indices
labels = results["labels"][:]
# We also want to render "invalid" items where the original image's prediction does not match the source label's prediction
well_predicted = labels == results["predictions"][:].argmax(axis=1)
mapping = []
i = 0
for target in range(all_valid.shape[0]):
    for is_valid in np.where(all_valid[target])[0]:
        index = i
        i += 1
        label = labels[is_valid]
        mapping.append((label, target, is_valid, index, well_predicted[is_valid]))

# %%
import pandas as pd

mapping = pd.DataFrame(
    mapping,
    columns=["source", "target", "source_index", "eval_index", "well_predicted"],
)
# mapping["quac_scores"] = evaluation[f"{attribution_method}/quac_scores"][:]
# Stack the best method and the best quac score
mapping["method"] = best_method
mapping["quac_scores"] = best_quac
# Drop everything that is not well predicted
mapping = mapping[mapping["well_predicted"]]

# %%
# Choose only values above a threshold
# Take the top 100 examples per source-target pair from the mapping
threshold = 0.5
selected_mapping = (
    mapping[mapping.quac_scores > threshold].groupby(["source", "target"]).head(100)
)
np.random.seed(random_seed)
# Randomly flip to decide whether A will be real or B will be real
selected_mapping["a_real"] = np.random.choice([True, False], len(selected_mapping))
# Randomize the order, and reset the index
selected_mapping = selected_mapping.sample(
    frac=1, random_state=random_seed
).reset_index(drop=True)
selected_mapping
# %%
selected_mapping.to_csv(output_location / "selected_mapping_phase1.csv", index=False)

# %% [markdown]
# We now want to *save* the selected mapping to a CSV file.
# We then want to *save* the images corresponding to the selected mapping with the following directory structure:
# Directory A: holds an image, which is the real image if a_real is True, and the counterfactual image if a_real is False.
# Directory B: holds an image, which is the real image if a_real is False, and the counterfactual image if a_real is True.
# Directory M: holds the mask image.

# %%
from PIL import Image

classes = sorted(list(results["counterfactuals"].keys()))


def save_grayscale_image(array, path):
    assert array.ndim == 2
    if np.any(array < 0):
        assert array.min() >= -1 and array.max() <= 1
        array = (array + 1) / 2  # Normalize to [0, 1]
    else:
        if array.max() > 0:
            # Clip to [0, 1]
            array = np.clip(array, 0, 1)
    assert array.min() >= 0 and array.max() <= 1, f"{array.min()}, {array.max()}"
    pil_image = Image.fromarray(
        (array * 255).astype(np.uint8)
    )  # Convert to 8-bit grayscale
    pil_image = pil_image.convert("L")  # Ensure it's in grayscale mode
    pil_image.save(path, format="PNG")


# %%
for idx, row in tqdm(selected_mapping.iterrows(), total=len(selected_mapping)):
    source_index = row["source_index"]
    image = results["images"][source_index]
    target = classes[row["target"]]
    attribution_method = row["method"]
    hybrid_index = row["eval_index"]
    counterfactual = evaluation[f"{attribution_method}/hybrids"][hybrid_index]
    mask = evaluation[f"{attribution_method}/optimal_masks"][hybrid_index]
    # Store the images
    if row["a_real"]:
        save_grayscale_image(image.squeeze(), dir_A / f"{idx}.png")
        save_grayscale_image(counterfactual.squeeze(), dir_B / f"{idx}.png")
    else:
        save_grayscale_image(counterfactual.squeeze(), dir_A / f"{idx}.png")
        save_grayscale_image(image.squeeze(), dir_B / f"{idx}.png")
    # Store the mask
    save_grayscale_image(mask.squeeze(), dir_M / f"{idx}.png")

# %% [markdown]
# Verification steps:
# - Check that images match between A, B, and M
# - Check that the image in A is real if a_real is True, and counterfactual if a_real is False
# - Check the classification results?

# %%
# 1. Check that images match between A, B, and M
random_indices = np.random.choice(selected_mapping.index, 5)

# load the data
for idx in random_indices:
    image_A = plt.imread(dir_A / f"{idx}.png")
    image_B = plt.imread(dir_B / f"{idx}.png")
    image_M = plt.imread(dir_M / f"{idx}.png")

    assert image_A.shape == image_B.shape == image_M.shape
    assert image_A.ndim == 2, image_A.ndim
    assert image_A.shape == (128, 128), image_A.shape

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_A, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(image_B, cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(image_M, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("A")
    ax[1].set_title("B")
    ax[2].set_title("M")
    plt.show()

    mask = image_M == 0
    # Check that they are the same outside the mask
    try:
        assert np.allclose(image_A[mask], image_B[mask], atol=1e-2)
    except AssertionError:
        print("Images do not match outside the mask")
        print((image_A[mask] - image_B[mask]).max())
        break

    # Check that if a_real, A is real
    a_real = selected_mapping.loc[idx, "a_real"]
    if a_real:
        # Load the real image from the results
        real_image = results["images"][selected_mapping.loc[idx, "source_index"]]
        # Scale
        real_image = (real_image + 1) / 2
        try:
            assert np.allclose(real_image, image_A, atol=1e-2)
            assert not np.allclose(real_image, image_B, atol=1e-2)
        except AssertionError:
            print("Image A is not the real image")
            break

# %%
# Load the classifier
import torch

model = torch.jit.load(
    "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
)
model = model.eval()

prediction_real = []
prediction_counterfactual = []

for idx in tqdm(selected_mapping.index, total=len(selected_mapping)):
    # Load the images
    image_A = plt.imread(dir_A / f"{idx}.png")
    image_B = plt.imread(dir_B / f"{idx}.png")
    # Stack the images
    stacked = np.stack([image_A, image_B], axis=0)
    # Add channel dimension
    stacked = np.expand_dims(stacked, axis=1)
    # convert to tensor
    tensor = torch.from_numpy(stacked).float()
    # Normalize
    tensor = (tensor - 0.5) / 0.5
    # Predict
    with torch.no_grad():
        output = model(tensor)
    # Get the predicted class
    prediction = output.argmax(dim=1)
    # add to the correct list
    if selected_mapping.loc[idx, "a_real"]:
        prediction_real.append(prediction[0].item())
        prediction_counterfactual.append(prediction[1].item())
    else:
        prediction_real.append(prediction[1].item())
        prediction_counterfactual.append(prediction[0].item())

# %%
from sklearn.metrics import confusion_matrix

confusion_matrix(prediction_real, selected_mapping["source"])
# %%
confusion_matrix(prediction_counterfactual, selected_mapping["target"])

# %%
