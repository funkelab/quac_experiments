import argparse
import dask.array as da
import logging
import numpy as np
from pathlib import Path
import quac.attribution
from quac.training.config import ExperimentConfig
from quac.training.data_loader import get_test_loader
import timm
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from yaml import safe_load
import zarr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcatDataset(torch.utils.data.Dataset):
    """
    Combines the images and the generated_images.
    """

    def __init__(self, source_dataset, generated_image_array):
        self.source_dataset = source_dataset
        self.generated_image_array = generated_image_array
        assert len(source_dataset) == generated_image_array.shape[0]

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        x, _ = self.source_dataset[idx]
        cf = self.generated_image_array[idx].compute()
        return x, cf


def resnet_from_script(script_path):
    # Note: unfortunately I can't DeepLift on the scripted model
    # Because it can't register the hooks
    # This loads the full model weights into a regular torch module
    model = timm.create_model("resnet18.a1_in1k", num_classes=4)
    scripted_model = torch.jit.load(script_path)
    scripted_model.eval()
    # Copy scripted weights to model
    model.load_state_dict(scripted_model.state_dict())
    model.eval()
    return model


def attribute(
    config_path: str,
    kind: str,
    batch_size: int = 16,
    num_samples: int = 2,
    attribution_name: str = "DIntegratedGradients",
):
    # Load metadata
    with open(config_path, "r") as f:
        metadata = safe_load(f)
    experiment = ExperimentConfig(**metadata)
    experiment_dir = Path(experiment.solver.root_dir)
    logging.info(f"Experiment directory {str(experiment_dir)}")

    # Get attribution object by name from the attribution module
    try:
        attributor = getattr(quac.attribution, attribution_name)
    except AttributeError:
        raise ValueError(f"Attribution method {attribution_name} not found")

    # Load the classifier
    logging.info("Loading classifier")
    classifier_checkpoint = Path(experiment.validation_config.classifier_checkpoint)
    # Classifier is JIT compiled
    classifier = resnet_from_script(classifier_checkpoint)
    classifier.to(device)
    classifier.eval()

    logging.info("Creating attribution object")
    attribution = attributor(classifier)

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
    method_group = group.require_group(attribution_name)
    attribution_group = method_group.require_group("attributions")

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
            # Create a place to store the attribution results
            attributions_array = attribution_group.require_dataset(
                f"{source}_{target}",
                shape=generated_image_array.shape,
                dtype=np.float32,
            )
            # Swap the batch and the number of images
            generated_image_array = generated_image_array.transpose(1, 0, 2, 3, 4)

            for i in range(num_samples):
                dataloader = torch.utils.data.DataLoader(
                    ConcatDataset(source_dataset, generated_image_array[i]),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    drop_last=False,
                )
                for batch_idx, (image, generated_image) in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    desc=f"Run {i + 1} / {num_samples}",
                ):
                    start = batch_size * batch_idx
                    end = min(start + batch_size, len(source_dataset))
                    attributions_array[start:end, i] = attribution.attribute(
                        image.float(),
                        generated_image.float(),
                        source,
                        target,
                        device=device,
                    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run attribution using generated images"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config file",
    )
    parser.add_argument(
        "-t",  # -t because we want to say type, but that's a reserved word
        "--kind",
        type=str,
        choices=["latent", "reference"],
        required=True,
        help="Kind of data to use",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for running attribution",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples to run attribution on",
    )
    parser.add_argument(
        "-a",
        "--attribution_name",
        type=str,
        default="DIntegratedGradients",
        choices=[
            "DIntegratedGradients",
            "DDeepLift",
            "VanillaDeepLift",
            "VanillaIntegratedGradients",
            # TODO: Add more attribution methods
        ],
        help="Name of the attribution method to use",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase verbosity",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    attribute(
        args.config,
        args.kind,
        args.batch_size,
        args.num_samples,
        args.attribution_name,
    )
