import logging
import numpy as np
import torch
from tqdm import tqdm
import zarr

from quac.attribution import (
    DIntegratedGradients,
    VanillaIntegratedGradients,
    DDeepLift,
    VanillaDeepLift,
)


def run_worker(result_path, target_class):
    # Setting up the classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
    )
    # Load the jit model
    # No need to use the full wrapper here because the images have been saved in the correct scale
    # But we still need a wrapper because of the forward hook added by the attribution methods
    classifier = torch.nn.Sequential(torch.jit.load(classifier_checkpoint))

    classifier.to(device)
    classifier.eval()

    # Choose the neuron
    zfile = zarr.open(result_path, mode="a")
    # Load the data
    images = zfile["images"]
    predictions = zfile["predictions"]
    counterfactuals = zfile[f"counterfactuals/{target_class}"]
    counterfactual_predictions = zfile[f"counterfactual_predictions/{target_class}"]
    # Check validity of the data
    assert (
        len(images)
        == len(predictions)
        == len(counterfactuals)
        == len(counterfactual_predictions)
    ), f"The data is not valid. Lengths: {len(images)}, {len(predictions)}, {len(counterfactuals)}, {len(counterfactual_predictions)}"

    # Create the attribution objects
    attributions = {
        "DIntegratedGradients": DIntegratedGradients(classifier),
        "VanillaIntegratedGradients": VanillaIntegratedGradients(classifier),
        "DDeepLift": DDeepLift(classifier),
        "VanillaDeepLift": VanillaDeepLift(classifier),
    }

    # Run the attributions
    results = {attr_name: [] for attr_name in attributions.keys()}

    for image, counterfactual, prediction in tqdm(
        zip(images, counterfactuals, predictions),
        total=len(images),
    ):
        source_class_index = int(prediction.argmax())
        target_class_index = 1  # This is the target, acetylcholine

        for attr_name, attribution in attributions.items():
            attr = attribution.attribute(
                torch.from_numpy(image).float(),
                torch.from_numpy(counterfactual).float(),
                source_class_index,
                target_class_index,
                device=device,
            )
            results[attr_name].append(attr)
    results = {
        attr_name: np.stack(attributions) for attr_name, attributions in results.items()
    }
    # Save the attributions to zarr
    zfile.require_groups("attributions", f"attributions/{target_class}")
    for attr_name, attr in results.items():
        zfile[f"attributions/{target_class}"].create_dataset(
            attr_name, data=attr, chunks=zfile["images"].chunks
        )


def check_exists(zarr_path, target_class):
    """
    Check if the counterfactuals already exist
    """
    zfile = zarr.open(zarr_path, mode="r")
    if f"attributions/{target_class}" in zfile:
        return True
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    zarr_path = (
        "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr"
    )
    target_class = "0_gaba"
    for target_class in [
        "0_gaba",
        "1_acetylcholine",
        "2_glutamate",
        "3_serotonin",
        "4_octopamine",
        "5_dopamine",
    ]:
        if not check_exists(zarr_path, target_class):
            logging.info(f"Running attributions for {target_class}")
            run_worker(zarr_path, target_class)
        else:
            print(f"Attributions already exist for {target_class}")
