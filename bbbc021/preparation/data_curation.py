# %%
import pandas as pd
from pathlib import Path
import numpy as np
from utils import load_image, get_dapi_mask, get_base_name, get_moa
import scipy.ndimage
import tifffile
from tqdm import tqdm

# %%
dataset = pd.read_csv("data/moa_split_train.csv")
data_root = Path("/nrs/funke/adjavond/data/bbbc021/")
data_output = Path("/nrs/funke/adjavond/data/bbbc021_processed/train")

data_output.mkdir(parents=True, exist_ok=True)
# Make all the moa directories
for moa in dataset["moa"].unique():
    moa_dir = data_output / moa.lower().replace(" ", "_")
    moa_dir.mkdir(parents=True, exist_ok=True)


# %%
for i, example in tqdm(dataset.iterrows(), total=len(dataset)):
    moa = get_moa(example)
    base_name = get_base_name(example)
    moa_dir = data_output / moa

    images = load_image(example)
    labels, count = scipy.ndimage.label(get_dapi_mask(images))

    for n in range(1, count + 1):
        mask = labels == n
        # Get the bounding box of the mask
        y, x = np.where(mask)
        min_y, max_y = y.min(), y.max()
        min_x, max_x = x.min(), x.max()
        # Get the center of the mask
        center_y, center_x = (min_y + max_y) // 2, (min_x + max_x) // 2
        # Create a bounding box around the center
        box_size = 80
        min_y, max_y = center_y - box_size, center_y + box_size
        min_x, max_x = center_x - box_size, center_x + box_size
        # Skip any boxes that are out of bounds
        if (
            min_y < 0
            or max_y >= images.shape[0]
            or min_x < 0
            or max_x >= images.shape[1]
        ):
            continue
        # Save the image using tifffile
        image_path = moa_dir / f"{base_name}_{n}.tiff"
        # Save the cropped image
        tifffile.imwrite(
            image_path,
            images[min_y:max_y, min_x:max_x, :],
        )
# %%
