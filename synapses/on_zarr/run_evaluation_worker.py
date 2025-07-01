import logging
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from tqdm import tqdm
import zarr

from quac.evaluation import BaseEvaluator, Processor
from utils import (
    DatasetWithAttribution,
)


def run_worker(zarr_path):
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

    reports = {}
    zarr_file = zarr.open(zarr_path, "a")
    # The existing attribution can be found from the names of the keys in the zarr file
    classes = sorted(list(zarr_file["counterfactuals"].array_keys()))
    attribution_methods = list(zarr_file[f"attributions/{classes[0]}"].keys())

    evaluations = zarr_file.require_group("evaluations")
    images = zarr_file["images"]

    for attribution_method in attribution_methods:
        this_evaluation = evaluations.require_group(attribution_method)

        dataset = DatasetWithAttribution(
            zarr_file=zarr_file,
            attribution_method=attribution_method,
        )

        processor = Processor()
        evaluator = BaseEvaluator(
            classifier, attribution_dataset=dataset, device=device
        )
        report = evaluator.quantify(processor)
        reports[attribution_method] = report
        # Store the data in the zarr file
        mask_sizes = np.array(report.normalized_mask_sizes)
        this_evaluation["mask_sizes"] = mask_sizes
        score_changes = np.array(report.score_changes)
        this_evaluation["score_changes"] = score_changes
        # Store the optimal thresholds in the zarr file
        optimal_thresholds = report.optimal_thresholds()
        this_evaluation["optimal_thresholds"] = optimal_thresholds
        # Store the optimal masks in the zarr file

        optimal_masks = []
        hybrids = []
        inverse_hybrids = []
        hybrid_classifications = []
        inverse_hybrid_classifications = []
        for i, sample in tqdm(enumerate(dataset), desc="Create masks and hybrids"):
            attribution = sample.attribution
            image = sample.image
            counterfactual = sample.counterfactual
            mask, _ = processor.create_mask(attribution, optimal_thresholds[i])
            optimal_masks.append(mask)
            # Create images
            # Hybrid = mask *xcf + (1-mask)*x --> A majority of pixels come from image, but has the same class as xcf
            # Inv Hybrid = (1-mask)*xcf + mask*x --> A majority of pixels come from counterfactual, but has the same class as x
            # Store the hybrids in the zarr file
            hybrid = (1 - mask) * image + mask * counterfactual
            inverse_hybrid = mask * image + (1 - mask) * counterfactual

            logging.info("hybrid shape: %s", hybrid.shape)
            logging.info("inverse_hybrid shape: %s", inverse_hybrid.shape)
            hybrid_classififcation = softmax(
                classifier(torch.from_numpy(hybrid[None, ...]).float().to(device))
                .cpu()
                .detach()
                .numpy(),
                axis=1,
            )
            inverse_hybrid_classification = softmax(
                classifier(
                    torch.from_numpy(inverse_hybrid[None, ...]).float().to(device)
                )
                .cpu()
                .detach()
                .numpy(),
                axis=1,
            )
            hybrids.append(hybrid)
            inverse_hybrids.append(inverse_hybrid)
            hybrid_classifications.append(hybrid_classififcation)
            inverse_hybrid_classifications.append(inverse_hybrid_classification)

        this_evaluation.create_dataset(
            "optimal_masks", data=np.array(optimal_masks), chunks=images.chunks
        )
        this_evaluation.create_dataset(
            "hybrids", data=np.array(hybrids), chunks=images.chunks
        )
        this_evaluation.create_dataset(
            "inverse_hybrids", data=np.array(inverse_hybrids), chunks=images.chunks
        )
        prediction_chunks = zarr_file["predictions"].chunks
        this_evaluation.create_dataset(
            "hybrid_classification",
            data=np.array(hybrid_classifications),
            chunks=prediction_chunks,
        )
        this_evaluation.create_dataset(
            "inverse_hybrid_classification",
            data=np.array(inverse_hybrid_classifications),
            chunks=prediction_chunks,
        )
        # Store the scores
        quac_scores = report.compute_scores()
        this_evaluation.create_dataset("quac_scores", data=quac_scores)
        # Add the population score
        median, p25, p75 = report.get_curve()
        this_evaluation["population_curve"] = np.array([median, p25, p75])


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    zarr_path = (
        "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr"
    )
    run_worker(zarr_path)
