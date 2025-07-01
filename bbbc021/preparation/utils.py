from skimage.filters import threshold_otsu
import scipy
from pathlib import Path
import tifffile
import numpy as np
import pandas as pd


def get_base_name(row):
    plate = row["Image_Metadata_Plate_DAPI"]
    well = row["Image_Metadata_Well_DAPI"]
    table = row["TableNumber"]
    image = row["ImageNumber"]
    return f"{plate}_{well}_{table}_{image}"


def get_meta_from_name(name: str):
    """
    Get metadata from the image name.
    The name is expected to be in the format: plate_well_table_image.
    Note that the plate name has underscores, so we cannot directly split by underscores.
    """
    name = Path(name).stem  # Get the stem of the path to avoid issues with extensions
    well, table, image, cell = name.split("_")[-4:]
    plate = name.replace(f"_{well}_{table}_{image}_{cell}", "")
    return {
        "plate": plate,
        "well": well,
        "table": int(table),
        "image": int(image),
        "cell": cell,
    }


def get_moa(row):
    """
    Get the moa from the row.
    """
    moa = row["moa"].lower().replace(" ", "_")
    return moa


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to [0, 1] range.
    """
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def load_image(
    row: pd.Series,
    data_root: Path = Path("/nrs/funke/adjavond/data/bbbc021/"),
):
    """
    Load the image from a given row of the dataset.
    Returns images in channel-last format.
    Actin: red, tubulin: green, dapi: blue.
    """
    dapi_channel = (
        data_root / row["Image_Metadata_Plate_DAPI"] / row["Image_FileName_DAPI"]
    )
    tubulin_channel = (
        data_root / row["Image_Metadata_Plate_DAPI"] / row["Image_FileName_Tubulin"]
    )
    actin_channel = (
        data_root / row["Image_Metadata_Plate_DAPI"] / row["Image_FileName_Actin"]
    )
    dapi = normalize_image(tifffile.imread(dapi_channel))
    tubulin = normalize_image(tifffile.imread(tubulin_channel))
    actin = normalize_image(tifffile.imread(actin_channel))

    return np.stack([actin, tubulin, dapi], axis=-1)


def get_dapi_mask(images):
    """
    Get the dapi mask from the images.
    """
    dapi = images[:, :, 2].copy()
    # Gaussian filter the dapi channel
    dapi = scipy.ndimage.gaussian_filter(dapi, sigma=1)
    # Threshold the dapi channel to get the mask
    dapi_mask = dapi > threshold_otsu(dapi)
    return dapi_mask
