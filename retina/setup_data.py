"""Script used to turn the original dataset into something that can be used by the StarGAN model."""

# %%
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# %%
source_directory = Path("/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/DR_grading")
output_directory = Path("/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/DR_grading_processed")

# %%
split = "train"
for split in ["train", "test", "val"]:
    metadata = pd.read_csv(f"/nrs/funke/adjavond/data/retina/ddrdataset/{split}.csv")
    target_directory = output_directory / split
    target_directory.mkdir(parents=True, exist_ok=True)

    target_directories = {
        0: target_directory / "0",
        1: target_directory / "1",
        2: target_directory / "2",
        3: target_directory / "3",
        4: target_directory / "4",
    }

    for directory in target_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"{split} split"):
        filename = row["id_code"]
        target = row["diagnosis"]
        assert (source_directory/ filename).exists()
        assert target in target_directories
        os.symlink(source_directory / filename, target_directories[target] / filename)

