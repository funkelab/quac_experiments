import argparse
import datetime
import logging
import numpy as np
from pathlib import Path
from quac.generate import load_stargan
from quac.training.data_loader import get_test_loader
from quac.training.config import ExperimentConfig
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from yaml import safe_load
import zarr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcatDataset(torch.utils.data.Dataset):
    """
    Take two datasets and return the output of both.
    One of the dataset is the main, the other is secondary.
    If the secondary is smaller, it will be repeated.
    """

    def __init__(self, main_dataset, secondary_dataset):
        self.main_dataset = main_dataset
        self.secondary_dataset = secondary_dataset
        self.len_main = len(main_dataset)
        self.len_secondary = len(secondary_dataset)

    def __len__(self):
        return self.len_main

    def __getitem__(self, idx):
        x_main, _ = self.main_dataset[idx]
        x_secondary, _ = self.secondary_dataset[idx % self.len_secondary]
        return x_main, x_secondary


@torch.inference_mode()
def generate(
    config_path: str,
    checkpoint_iter: int,
    kind: str,
    num_samples: int = 2,
    batch_size: int = 16,
):
    # Load metadata
    with open(config_path, "r") as f:
        metadata = safe_load(f)
    experiment = ExperimentConfig(**metadata)

    # Load the classifier
    logging.info("Loading classifier")
    classifier_checkpoint = Path(metadata["validation_config"]["classifier_checkpoint"])
    # Classifier is JIT compiled
    classifier = torch.jit.load(classifier_checkpoint)
    classifier.to(device)
    classifier.eval()

    # Load the model
    logging.info("Loading model")
    experiment_dir = Path(experiment.solver.root_dir)
    latent_model_checkpoint_dir = experiment_dir / "checkpoints"
    inference_model = load_stargan(
        latent_model_checkpoint_dir,
        checkpoint_iter=checkpoint_iter,
        kind=kind,
        **metadata["model"],
    )
    inference_model.to(device)
    inference_model.eval()

    # Load the data
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
    # Create a Zarr file to keep things in
    zarr_file = zarr.open(experiment_dir / "output.zarr", "a")
    group = zarr_file.require_group(kind)
    images = group.require_group("generated_images")
    # Add attributes
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    images.attrs["checkpoint"] = checkpoint_iter
    images.attrs["generation_date"] = today
    images.attrs["classes"] = dataset.classes
    predictions = group.require_group("generated_predictions")
    predictions.attrs["generation_date"] = today
    predictions.attrs["classes"] = dataset.classes

    # Generate counterfactuals
    for source, source_name in enumerate(dataset.classes):
        is_source = np.where(np.array(dataset.targets) == source)[0]
        source_dataset = Subset(dataset, is_source)
        logging.info("Length of source dataset", len(source_dataset))
        for target, target_name in enumerate(dataset.classes):
            if source == target:
                continue
            y_target = torch.tensor([target] * batch_size).to(device)
            logging.info("Running for source", source_name, "and target", target_name)
            source_target_images = images.require_dataset(
                f"{source}_{target}",
                # (N, B, C, H, W)
                shape=(len(source_dataset), num_samples) + source_dataset[0][0].shape,
                dtyle=np.float32,
            )
            source_target_predictions = predictions.require_dataset(
                f"{source}_{target}",
                # (N, B, C)
                shape=(len(source_dataset), num_samples, len(dataset.classes)),
                dtyle=np.float32,
            )
            for i in range(num_samples):
                if kind == "reference":
                    is_target = np.where(np.array(dataset.targets) == target)[0]
                    # Shift the target dataset indices by i to make sure there are no repeats
                    target_dataset = Subset(dataset, np.roll(is_target, i))
                    dataloader = torch.utils.data.DataLoader(
                        ConcatDataset(source_dataset, target_dataset),
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                    )
                    for batch_idx, (x, x_ref) in tqdm(
                        enumerate(dataloader),
                        total=len(dataloader),
                        desc=f"Run {i + 1} / {num_samples}",
                    ):
                        y_target = torch.tensor([target] * x.size(0)).to(device)
                        xg = inference_model(x.to(device), x_ref.to(device), y_target)
                        # Predict the class of the generated images
                        pred = torch.softmax(classifier(xg), dim=-1)
                        # Save generated images
                        start, end = (
                            batch_idx * batch_size,
                            (batch_idx + 1) * batch_size,
                        )
                        source_target_images[start:end, i] = xg.cpu().numpy()
                        # Save generated predictions
                        source_target_predictions[start:end, i] = pred.cpu().numpy()

                else:  # kind == "latent"
                    torch.manual_seed(i)  # Trying to make this somewhat reproducible
                    dataloader = torch.utils.data.DataLoader(
                        source_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                    )
                    for batch_idx, (x, _) in tqdm(
                        enumerate(dataloader),
                        total=len(dataloader),
                        desc=f"Run {i + 1} / {num_samples}",
                    ):
                        y_target = torch.tensor([target] * x.size(0)).to(device)
                        xg = inference_model(x.to(device), y_target)
                        # Predict the class of the generated images
                        pred = torch.softmax(classifier(xg), dim=-1)
                        # Save generated images
                        start, end = (
                            batch_idx * batch_size,
                            min((batch_idx + 1) * batch_size, len(source_dataset)),
                        )
                        source_target_images[start:end, i] = xg.cpu().numpy()
                        # Save generated predictions
                        source_target_predictions[start:end, i] = pred.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file used during training.",
    )
    parser.add_argument(
        "-i",
        "--checkpoint",
        type=int,
        required=True,
        help="Checkpoint iteration to use.",
    )
    parser.add_argument(
        "-t",
        "--kind",  # We use -t because we want to say "type" but it's a reserved word
        type=str,
        required=True,
        choices=["latent", "reference"],
        help="Kind of counterfactual to generate.",
    )
    parser.add_argument("-n", "--num_samples", type=int, default=2)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
    )
    args = parse_args()
    generate(
        args.config,
        args.checkpoint,
        args.kind,
        args.num_samples,
        args.batch_size,
    )
