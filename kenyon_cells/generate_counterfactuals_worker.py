# %% Setup
import dask.array as da
from pathlib import Path
from quac.generate import load_data, load_stargan, get_counterfactual
from quac.training.classification import ClassifierWrapper
import sys
import torch
from tqdm import tqdm
import zarr


def run_worker(zarr_path, cell_id, target=1):
    """
    target: int,
        defaults to 1: This is Acetylcholine
    """
    # %% Load metadata
    ref_directory = "/nrs/funke/adjavond/data/synapses/"
    img_size = 128
    input_dim = 1
    style_dim = 64
    latent_dim = 16
    num_domains = 6
    checkpoint_iter = 100000
    kind = "reference"
    classifier_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the data from the zarr file
    zarr_dataset = da.from_zarr(zarr_path, component=f"{cell_id}/images")
    preds = da.from_zarr(zarr_path, component=f"{cell_id}/predictions")

    counterfactuals = da.zeros_like(zarr_dataset)
    cf_preds = da.zeros_like(preds)

    # Load the reference data from the root directory
    img_size = 128
    reference_data_directory = Path(f"{ref_directory}/val/1_acetylcholine")
    reference_dataset = load_data(reference_data_directory, img_size, grayscale=True)

    # Load the classifier
    print("Loading classifier")
    mean = 0.5
    std = 0.5
    assume_normalized = False
    classifier = ClassifierWrapper(
        classifier_checkpoint,
        mean=mean,
        std=std,
        assume_normalized=assume_normalized,
    )
    classifier.to(device)
    classifier.eval()

    # Load model
    latent_model_checkpoint_dir = Path(
        "/groups/funke/home/adjavond/workspace/projects/stargan/run/synapses_v1/stargan_invariance_v0/checkpoints"
    )

    inference_model = load_stargan(
        latent_model_checkpoint_dir,
        img_size=img_size,
        input_dim=input_dim,
        style_dim=style_dim,
        latent_dim=latent_dim,
        num_domains=num_domains,
        checkpoint_iter=checkpoint_iter,
        kind=kind,
    )

    # Generate counterfactuals

    for i, (x, y) in tqdm(enumerate(zip(zarr_dataset, preds)), total=len(zarr_dataset)):
        xcf, ycf = get_counterfactual(
            classifier,
            inference_model,
            torch.from_numpy(x.compute()).to(device),
            target=target,
            kind=kind,  # Change the kind of inference being done
            dataset_ref=reference_dataset,  # Add the reference dataset
            device=device,
            max_tries=2,
            batch_size=128,
            return_pred=True,
        )
        counterfactuals[i] = xcf.cpu().numpy()
        cf_preds[i] = ycf.cpu().numpy()

    counterfactuals.to_zarr(zarr_path, component=f"{cell_id}/counterfactuals")
    cf_preds.to_zarr(zarr_path, component=f"{cell_id}/counterfactual_predictions")


def check_exists(zarr_path, cell_id):
    """
    Check if the counterfactuals already exist
    """
    zfile = zarr.open(zarr_path, mode="r")
    if (
        f"{cell_id}/counterfactuals" in zfile
        and f"{cell_id}/counterfactual_predictions" in zfile
    ):
        return True
    return False


if __name__ == "__main__":
    zarr_path = sys.argv[1]
    cell_id = sys.argv[2]
    if not check_exists(zarr_path, cell_id):
        run_worker(zarr_path, cell_id)
    else:
        print("Counterfactuals already exist.")
