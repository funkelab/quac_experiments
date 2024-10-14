"""
Evaluate the classifier on the test set.
"""

import argparse
from config import ExperimentConfig
import logging
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml


# The best checkpoint is JIT compiled under best_model.pt
def eval(experiment: ExperimentConfig):
    # Load the data
    try:
        data_config = experiment.test_data
    except AttributeError:
        # No test data, use the validation data
        logging.warning("No test data, using validation data.")
        data_config = experiment.val_data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )
    dataset = ImageFolder(data_config.data_location, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=12,
    )

    # Load the jitted model
    checkpoint_dir = Path(experiment.training.checkpoint_dir)
    model = torch.jit.load(checkpoint_dir / "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        correct = 0
        total = 0
        for x, y in tqdm(dataloader, total=len(dataloader)):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        accuracy = correct / total
    logging.info(f"Accuracy: {accuracy}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    experiment = ExperimentConfig(**config)
    eval(experiment)
