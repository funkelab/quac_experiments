import cv2
import dask.array as da
import kornia.morphology as morph
import logging
import numpy as np
from pathlib import Path
from quac.training.config import ExperimentConfig
from quac.training.data_loader import get_test_loader
import torch
from torch.utils.data import Subset
from yaml import safe_load
import zarr


def evaluate(
    config_path: str,
    kind: str,
    batch_size: int,
    num_samples: int,
    attribution_name: str,
    struct_size: int = 11,
):
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
    # Get the Zarr file in which things are kept
    logging.info("Loading generated data")
    zarr_file = zarr.open(experiment_dir / "output.zarr", "a")
    group = zarr_file[kind]
    generated_images = group["generated_images"]
    method_group = group[attribution_name]

    # attributions
    attributions = method_group["attributions"]

    # Morphological closing kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (struct_size, struct_size))
    kernel = torch.tensor(kernel).float()

    for source, source_name in enumerate(dataset.classes):
        is_source = np.where(np.array(dataset.targets) == source)[0]
        source_dataset = Subset(dataset, is_source)
        logging.info(f"Length of source dataset {len(source_dataset)}")
        for target, target_name in enumerate(dataset.classes):
            if source == target:
                continue
            logging.info(f"Running for source {source_name}  target {target_name}")
            generated_image_array = da.from_zarr(
                generated_images[f"{source}_{target}"]
            )  # N, B, C, H, W
            attribution_array = da.from_zarr(
                attributions[f"{source}_{target}"]
            )  # N, B, C, H, W
            for i in range(num_samples):
                for j in range(len(source_dataset)):
                    # Get source image
                    image = source_dataset[j][0]
                    # Get generated image
                    generated = generated_image_array[j, i]
                    # Get attribution
                    attribution = attribution_array[j, i]
                    # TODO evaluate
                    # Get array w/ threshold, mask-size, delta-f
                    # TODO compute scores?
                    # TODO store optimal mask
                    pass
