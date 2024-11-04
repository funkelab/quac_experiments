# %% [markdown]
# A test to find out whether we can pick thresholds with percentiles
# And get a good description of mask sizes
# Started as the end of the kornia test
# %%
import cv2
import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt
import zarr
from yaml import safe_load
from pathlib import Path
from torch.utils.data import Subset
from quac.training.config import ExperimentConfig
from quac.training.data_loader import get_test_loader
import logging

# %% [markdown]
# First we load data from the `fictus` experiment.
# %% Setup
config_path = "../configs/stargan_20241013.yml"
kind = "latent"
attribution_name = "DIntegratedGradients"
source = 0
target = 1
struct = 11  # size of the structuring element for the morphological closing
# %%
# Load metadata
with open(config_path, "r") as f:
    metadata = safe_load(f)
experiment = ExperimentConfig(**metadata)
experiment_dir = Path(experiment.solver.root_dir)
logging.info(f"Experiment directory {str(experiment_dir)}")

# Load the classifier
logging.info("Loading classifier")
classifier_checkpoint = Path(experiment.validation_config.classifier_checkpoint)
classifier = torch.jit.load(classifier_checkpoint)
classifier.eval()

# Load the data
logging.info("Loading input data")
data_config = experiment.test_data
if data_config is None:
    logging.warning("Test data not found in metadata, using validation data")
    data_config = experiment.validation_data
# Load the data
dataset = get_test_loader(
    data_config.source,
    img_size=data_config.img_size,
    mean=data_config.mean,
    std=data_config.std,
    return_dataset=True,
)
is_source = np.where(np.array(dataset.targets) == source)[0]
source_dataset = Subset(dataset, is_source)
# Get the Zarr file in which things are kept
logging.info("Loading generated data")
zarr_file = zarr.open(experiment_dir / "output.zarr", "r")
group = zarr_file[kind]
generated_images = group[f"generated_images/{source}_{target}"]
attributions = group[f"{attribution_name}/attributions/{source}_{target}"]
# %% [markdown]
# Both the attributions and the generated images have shape (N, B, C, H, W)
# This is because I made 2 generated images per input.
# For this test, I will use the first one.
# %%
generated_images = torch.from_numpy(generated_images[:, 0])
attributions = torch.from_numpy(attributions[:, 0])

# %% [markdown]
# We will apply morphological closing to the attributions
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (struct, struct))
torch_kernel = torch.tensor(kernel).float().to(device)

dataloader = torch.utils.data.DataLoader(
    attributions, batch_size=32, shuffle=False, drop_last=False, pin_memory=True
)
closed_attributions = torch.cat(
    [
        kornia.morphology.morphology.closing(attr.to(device), kernel=torch_kernel)
        for attr in dataloader
    ]
)


# %% [markdown]
# Let us look at the results, to see if they make sense
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(attributions[0].permute((1, 2, 0)))
ax2.imshow(closed_attributions[0].cpu().permute((1, 2, 0)))


# %% [markdown]
# The first thing that we want to check is whether we can use quantiles for thresholding.
# We will compare mask sizes with two different methods of choosing thresholds:
# - using a linear space of 100 thresholds between the 0 and 1
# - using the 0th to 100th percentiles of the attribution
# The goal is to compare the range of mask sizes that we get.

# %%
linear_thresholds = torch.linspace(0, 1, 100)
linear_sizes = []
quantile_sizes = []
for threshold in linear_thresholds:
    linear_sizes.append(
        (closed_attributions >= threshold).flatten(1).sum(dim=1)
        / closed_attributions.flatten(1).shape[1]
    )
    quantile_sizes.append(
        (
            closed_attributions
            >= torch.quantile(
                closed_attributions.flatten(1),
                torch.tensor(threshold).to(device),
                dim=1,
            )[:, None, None, None]
        )
        .flatten(1)
        .sum(dim=1)
        / closed_attributions.flatten(1).shape[1]
    )

linear_sizes = torch.stack(linear_sizes, dim=1).cpu()
quantile_sizes = torch.stack(quantile_sizes, dim=1).cpu()

# %% Plot the results
fig, ax = plt.subplots()
ax.plot(linear_thresholds, linear_sizes.mean(dim=0), label="Linear")
ax.fill_between(
    linear_thresholds,
    linear_sizes.mean(dim=0) - linear_sizes.std(dim=0),
    linear_sizes.mean(dim=0) + linear_sizes.std(dim=0),
    alpha=0.3,
)
ax.plot(linear_thresholds, quantile_sizes.mean(dim=0), label="Quantile")
ax.fill_between(
    linear_thresholds,
    quantile_sizes.mean(dim=0) - quantile_sizes.std(dim=0),
    quantile_sizes.mean(dim=0) + quantile_sizes.std(dim=0),
    alpha=0.3,
)
ax.set_ylabel("Mask size")
ax.set_xlabel("Threshold/Quantile")
ax.legend()

# %% [markdown]
# Conclusion - I get a much better spread of mask sizes using the quantiles.

# %% [markdown]
# The next thing to do is to create counterfactual images using these masks.
# Given a mask at a certain threshold, a counterfactual image is created as:
# mask_smoothed * generated_image + (1 - mask_smoothed) * source_image
# where mask_smoothed is the mask after a Gaussian blur.
# %% Get source images
source_images = torch.stack([im[0] for im in source_dataset])


# %% check that the images match
def rescale(image):
    return (image + 1) / 2


fig, axes = plt.subplots(3, 2, figsize=(6, 8))
for i, (ax1, ax2) in enumerate(axes):
    ax1.imshow(rescale(source_images[i].permute((1, 2, 0))).cpu())
    ax2.imshow(rescale(generated_images[i].permute((1, 2, 0))).cpu())
    ax1.axis("off")
    ax2.axis("off")
# %%
with torch.no_grad():
    source_classifications = torch.softmax(classifier(source_images.to(device)), dim=1)[
        :, target
    ].cpu()
# %% Create counterfactual images
from tqdm import tqdm

all_sizes = []
all_scores = []
optimal_counterfactuals = []
generated_images = generated_images.float().to(device)
source_images = source_images.float().to(device)
for threshold in tqdm(linear_thresholds):
    quantiles = torch.quantile(
        closed_attributions.flatten(1),
        torch.tensor(threshold).to(device),
        dim=1,
    )
    masks = closed_attributions >= quantiles[:, None, None, None]
    masks = masks.float()
    kernel_size = 11
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    mask_sizes = masks.flatten(2).any(dim=1).sum(dim=1) / masks.flatten(2).shape[2]
    all_sizes.append(mask_sizes)
    masks = kornia.filters.gaussian_blur2d(
        masks, (kernel_size, kernel_size), (sigma, sigma)
    )
    counterfactuals = masks * generated_images + (1 - masks) * source_images
    # classify
    with torch.no_grad():
        counterfactual_classifications = torch.softmax(
            classifier(counterfactuals), dim=1
        )[:, target].cpu()
    scores = counterfactual_classifications - source_classifications
    all_scores.append(scores)

all_sizes = torch.stack(all_sizes, dim=1).cpu()
all_scores = torch.stack(all_scores, dim=1).cpu()

# %% Plot the results
for size, score in zip(all_sizes, all_scores):
    plt.plot(size, score, alpha=0.5)
plt.xlabel("Mask size")
plt.ylabel("Score")

# %%
# Get QuAC scores
auc = -torch.trapz(all_scores, all_sizes, dim=1)  # baseline/ground truth
plt.hist(auc, bins=20)
# %% [markdown]
# Next, I want to see where the "optimal" threshold is.
# This is calculated by looking at the mask size - the score change, and getting the minimum.

# %%
# Do all thresholds at once, instead of all samples at once
all_sizes = []
all_scores = []
optimal_counterfactuals = []
optimal_masks = []

for source_image, image, attribution in zip(
    source_images, generated_images, closed_attributions
):
    # Repeat n_samples times
    n_samples = 100
    thresholds = torch.linspace(0, 1, n_samples)
    attributions = attributions.repeat(n_samples, 1, 1, 1)
    masks = attributions >= thresholds[:, None, None, None]
    masks = masks.float()

    kernel_size = 11
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    mask_sizes = masks.flatten(2).any(dim=1).sum(dim=1) / masks.flatten(2).shape[2]
    all_sizes.append(mask_sizes)
    masks = kornia.filters.gaussian_blur2d(
        masks, (kernel_size, kernel_size), (sigma, sigma)
    )
    # Repeat n_samples time
    images = generated_images[0].repeat(n_samples, 1, 1, 1)
    source_image = source_images[0].repeat(n_samples, 1, 1, 1)
    counterfactuals = masks * images + (1 - masks) * source_image

    # classify
    with torch.no_grad():
        counterfactual_classifications = torch.softmax(
            classifier(counterfactuals), dim=1
        )[:, target].cpu()
    scores = counterfactual_classifications - source_classifications[target]
    all_scores.append(scores)
    # Get the optimal counterfactual and the optimal mask
    optimal_index = scores.argmin()
    optimal_counterfactuals.append(counterfactuals[optimal_index])
    optimal_masks.append(masks[optimal_index])
