import numpy as np
import torch
from tqdm import tqdm
import zarr
import sys

from quac.attribution import (
    DIntegratedGradients,
    VanillaIntegratedGradient,
    DDeepLift,
    VanillaDeepLift,
)
from quac.training.classification import ClassifierWrapper


def run_worker(result_path, neuron_id):
    # Setting up the classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
    )
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

    # Choose the neuron
    this_neuron = zarr.open(result_path, mode="a")[neuron_id]
    # Load the data
    images = this_neuron["images"]
    predictions = this_neuron["predictions"]
    counterfactuals = this_neuron["counterfactuals"]
    counterfactual_predictions = this_neuron["counterfactual_predictions"]
    # Check validity of the data
    assert (
        len(images)
        == len(predictions)
        == len(counterfactuals)
        == len(counterfactual_predictions)
    )

    # Create the attribution objects
    attributions = {
        "DIntegratedGradients": DIntegratedGradients(classifier),
        "VanillaIntegratedGradients": VanillaIntegratedGradient(classifier),
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
    this_neuron.require_group("attributions")
    for attr_name, attr in results.items():
        this_neuron["attributions"].create_dataset(
            attr_name, data=attr, chunks=this_neuron["images"].chunks
        )


def check_exists(zarr_path, cell_id):
    """
    Check if the counterfactuals already exist
    """
    zfile = zarr.open(zarr_path, mode="r")
    if f"{cell_id}/attributions" in zfile:
        return True
    return False


if __name__ == "__main__":
    zarr_path = sys.argv[1]
    cell_id = sys.argv[2]
    if not check_exists(zarr_path, cell_id):
        run_worker(zarr_path, cell_id)
    else:
        print("Attributions already exist.")
