# %% Setup
from yaml import safe_load
import random
import torch
from utils import (
    Compose,
    RandomCrop,
    Rescale,
    IntensityAugmentation,
    ZfishDataset,
    random_rotation,
)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Metadata
metadata = safe_load(open("metadata.yaml"))
shape = (512, 512)

# %% Setting up the dataset
transform = Compose(
    [
        RandomCrop(shape),
        # Rescale(),
        # IntensityAugmentation(factor=0.3),
    ]
)
dataset = ZfishDataset(
    metadata, rotation_transform=random_rotation, transform=transform
)
# %%
output_directory = Path("blind_test")
output_directory.mkdir(exist_ok=True)
image_directory = output_directory / "images"

# %%
# Seed the randomness so that we get the same validation set every time
random.seed(42)
#
num_samples = 100
samples = {}
# Get the step from the checkpoint name
for i in tqdm(range(num_samples)):
    with torch.no_grad():
        # Getting a batch, something is weird with dataloaders
        x, y = next(iter(dataset))
    samples[f"{i}.png"] = y
    plt.imsave(image_directory / f"{i}.png", x.squeeze(), cmap="gray", vmin=0, vmax=255)
    # print(x.min(), x.max())
    # plt.imshow(x.squeeze(), cmap="gray", vmin=0, vmax=255)
    # plt.title(f"Sample {i} - {y}")
    # plt.show()
# %%
with open(output_directory / "ground_truth.csv", "w") as f:
    f.write("filename,label\n")
    for k, v in samples.items():
        f.write(f"{k},{v}\n")
