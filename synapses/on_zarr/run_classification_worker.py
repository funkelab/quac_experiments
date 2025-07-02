"""
Loading the images, classifying them, and storing them to zarr.
"""

import zarr
from funlib.learn.torch.models.vgg2d import Vgg2D
import torch
import gunpowder as gp
from tqdm import tqdm
from scipy.special import softmax
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import typer


def run_worker(
    image_directory_path: str = "/nrs/funke/adjavond/data/synapses/test/",
    output_zarr_path: str = "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr",
):
    """
    image_directory_path: str
        Path to the directory containing the images
    output_zarr_path: str
        Path to the zarr file where the results will be stored
    """
    # Loading the data
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )
    dataset = ImageFolder(root=image_directory_path, transform=transform)

    # Metadata
    N = len(dataset)
    batch_size = min(512, N)
    chunk_size = min(32, N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=False
    )

    # Loading the pre-trained classifier
    model = Vgg2D(input_size=(128, 128), fmaps=12)
    model.to(device)
    model.load_state_dict(
        torch.load(
            "/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint",
            map_location=device,
        )["model_state_dict"]
    )
    model.eval()

    # Some more metadata
    group = zarr.open(output_zarr_path, "a")

    images = group.require_dataset(
        "images",
        shape=(N, 1, 128, 128),
        chunks=(chunk_size, 1, 128, 128),
        dtype="float32",
    )
    labels = group.require_dataset(
        "labels", shape=(N), chunks=(chunk_size), dtype="uint8"
    )
    predictions = group.require_dataset(
        "predictions",
        shape=(N, 6),
        chunks=(chunk_size, 6),
        dtype="float32",
    )

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, y = batch
        with torch.no_grad():
            pred = model(x.to(device)).cpu()
        # Prepare storage
        start = i * batch_size
        end = min(start + batch_size, N)
        number = end - start
        # Add data
        images[start:end] = x[:number].numpy()
        predictions[start:end] = softmax(pred[:number], axis=1)
        labels[start:end] = y.numpy().squeeze()


if __name__ == "__main__":
    typer.run(run_worker)
