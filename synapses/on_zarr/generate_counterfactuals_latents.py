# %% Setup
import logging
import dask.array as da
from pathlib import Path
from quac.generate import load_data, load_stargan, get_counterfactual
from quac.training.classification import ClassifierWrapper
import sys
import torch
from tqdm import tqdm
import zarr


def run_worker(
    zarr_path, ref_directory="/nrs/funke/adjavond/data/synapses/val/", kind="latent"
):
    """
    zarr_path: str
        Path to the zarr file containing the data, in which results will be stored
    ref_directory: str
        Path to the directory containing the reference data
    kind: str
        The kind of inference to be done. Can be 'reference' or 'latent'
    """
    # %% Load metadata
    img_size = 128
    input_dim = 1
    style_dim = 64
    latent_dim = 16
    num_domains = 6
    checkpoint_iter = 100000
    classifier_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the data from the zarr file
    logging.info("Loading data")
    zarr_dataset = da.from_zarr(zarr_path, component=f"images")
    preds = da.from_zarr(zarr_path, component=f"predictions")

    # Load the classifier
    logging.info("Loading classifier")
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
    logging.info("Loading StarGAN model")
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

    target_classes = [
        "0_gaba",
        "1_acetylcholine",
        "2_glutamate",
        "3_serotonin",
        "4_octopamine",
        "5_dopamine",
    ]

    for target, name in enumerate(target_classes):
        logging.info(f"Generating counterfactuals for {name}")
        # Load the reference data from the root directory
        reference_data_directory = Path(f"{ref_directory}/{name}")
        reference_dataset = load_data(
            reference_data_directory, img_size, grayscale=True
        )

        counterfactuals = da.zeros_like(zarr_dataset)
        cf_preds = da.zeros_like(preds)

        # Generate counterfactuals -- note that this also creates counterfactuals for the images of the same class
        for i, (x, y) in tqdm(
            enumerate(zip(zarr_dataset, preds)), total=len(zarr_dataset), desc=name
        ):
            xcf, ycf = get_counterfactual(
                classifier,
                inference_model,
                torch.from_numpy(x.compute()).to(device),
                target=target,
                kind=kind,  # Change the kind of inference being done
                device=device,
                max_tries=2,
                batch_size=128,
                return_pred=True,
            )
            counterfactuals[i] = xcf.cpu().numpy()
            cf_preds[i] = ycf.cpu().numpy()

        logging.info(f"Saving counterfactuals for {name}")
        counterfactuals.to_zarr(zarr_path, component=f"counterfactuals_latent/{name}")
        cf_preds.to_zarr(
            zarr_path, component=f"counterfactual_predictions_latent/{name}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    zarr_path = (
        "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr"
    )
    run_worker(zarr_path)
