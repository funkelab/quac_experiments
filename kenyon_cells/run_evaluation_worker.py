import numpy as np
import pandas as pd
from scipy.special import softmax
import sys
import torch
import zarr

from quac.training.classification import ClassifierWrapper
from quac.evaluation import BaseEvaluator, Processor
from utils import (
    DatasetWithAttribution,
)


def run_worker(zarr_path, index):
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

    reports = {}
    zarr_file = zarr.open(zarr_path, "a")
    # The existing attribution can be found from the names of the keys in the zarr file
    attribution_methods = list(zarr_file[index]["attributions"].keys())

    evaluations = zarr_file[index].require_group("evaluations")

    for attribution_method in attribution_methods:
        this_evaluation = evaluations.require_group(attribution_method)

        dataset = DatasetWithAttribution(
            zarr_file=zarr_file,
            neuron_index=index,
            attribution_method=attribution_method,
            target_class_override=1,
        )

        processor = Processor()
        evaluator = BaseEvaluator(
            classifier, attribution_dataset=dataset, device=device
        )
        report = evaluator.quantify(processor)
        reports[attribution_method] = report
        # Get the optimal masks
        attributions = zarr_file[index][f"attributions/{attribution_method}"]
        images = zarr_file[index]["images"]
        counterfactuals = zarr_file[index]["counterfactuals"]
        # Store the data in the zarr file
        mask_sizes = np.array(report.normalized_mask_sizes)
        this_evaluation["mask_sizes"] = mask_sizes
        score_changes = np.array(report.score_changes)
        this_evaluation["score_changes"] = score_changes
        # Store the optimal thresholds in the zarr file
        optimal_thresholds = report.optimal_thresholds()
        this_evaluation["optimal_thresholds"] = optimal_thresholds
        # Store the optimal masks in the zarr file
        optimal_masks = np.array(
            [
                processor.create_mask(attribution, threshold, return_size=False)
                for attribution, threshold in zip(attributions, optimal_thresholds)
            ]
        )
        this_evaluation.create_dataset(
            "optimal_masks", data=optimal_masks, chunks=attributions.chunks
        )
        # FIXME!!! I've swapped Hybrid and Inverse Hybrid here!
        # Hybrid = mask *xcf + (1-mask)*x --> majority image, same class as xcf
        # Inv Hybrid = (1-mask)*xcf + mask*x --> majority counterfactual, same class as x
        # Store the hybrids in the zarr file
        hybrids = np.array(
            [
                mask * image + (1 - mask) * counterfactual
                for mask, image, counterfactual in zip(
                    optimal_masks, images, counterfactuals
                )
            ]
        )
        this_evaluation.create_dataset("hybrids", data=hybrids, chunks=images.chunks)
        inverse_hybrids = np.array(
            [
                (1 - mask) * image + mask * counterfactual
                for mask, image, counterfactual in zip(
                    optimal_masks, images, counterfactuals
                )
            ]
        )
        this_evaluation.create_dataset(
            "inverse_hybrids", data=inverse_hybrids, chunks=images.chunks
        )
        # Predict on the hybrids and store the results in the zarr file
        hybrid_classification = softmax(
            classifier(torch.from_numpy(hybrids).float().to(device))
            .cpu()
            .detach()
            .numpy(),
            axis=1,
        )
        prediction_chunks = zarr_file[index]["predictions"].chunks
        this_evaluation.create_dataset(
            "hybrid_classification",
            data=hybrid_classification,
            chunks=prediction_chunks,
        )
        inverse_hybrid_classification = softmax(
            classifier(torch.from_numpy(inverse_hybrids).float().to(device))
            .cpu()
            .detach()
            .numpy(),
            axis=1,
        )
        this_evaluation.create_dataset(
            "inverse_hybrid_classification",
            data=inverse_hybrid_classification,
            chunks=prediction_chunks,
        )
        # Store the scores
        quac_scores = report.compute_scores()
        this_evaluation.create_dataset("quac_scores", data=quac_scores)
        # Add the population score
        median, p25, p75 = report.get_curve()
        this_evaluation["population_curve"] = np.array([median, p25, p75])


if __name__ == "__main__":
    zarr_path = sys.argv[1]
    index = int(sys.argv[2])
    run_worker(zarr_path, index)
