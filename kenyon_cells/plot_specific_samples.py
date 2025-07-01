"""
Script to plot specific synapses from the results, given a CSV file that holds neuron_id and synapse_id pairs.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage import measure
import zarr

# Dilation and fill holes
from scipy.ndimage import binary_dilation


def save_one_image(image, filepath):
    """
    Takes an image in the form of a numpy array with values from -1 to 1,
    and saves it as a PNG image.
    """
    # Normalize the image
    image = (image + 1) / 2
    pil_image = Image.fromarray((image.squeeze() * 255).astype(np.uint8))
    pil_image.save(filepath)


def save_one_mask(mask, filepath, threshold=0.0):
    """
    Takes a mask in the form of a numpy array with values from 0 to 1,
    and saves it as a PNG image.
    """
    mask = mask > threshold
    pil_image = Image.fromarray((mask.squeeze() * 255).astype(np.uint8))
    pil_image.save(filepath)


def save_inverse_mask(mask, filepath, threshold=0.0):
    """
    Takes a mask in the form of a numpy array with values from 0 to 1,
    Inverts the mask and saves it as a PNG image.
    """
    mask = mask > threshold
    inv_mask = 1 - mask
    pil_image = Image.fromarray((inv_mask.squeeze() * 255).astype(np.uint8))
    pil_image.save(filepath)


def save_mask_contour(mask, filepath, threshold=0.0, linewidth=4):
    """
    Takes a mask in the form of a numpy array with values from 0 to 1,
    Gets the contour of the mask and saves it as a PNG image.
    """
    contour = np.zeros_like(mask)
    contours = measure.find_contours(mask.squeeze(), threshold)

    # Turn the contour into a binary mask
    for contour_coords in contours:
        contour_coords = np.round(contour_coords).astype(int)
        contour[0, contour_coords[:, 0], contour_coords[:, 1]] = 1
    # Dilate the contour linewidth times, then remove the interior
    interior = mask > threshold
    for _ in range(1, linewidth):
        contour = binary_dilation(contour)
    # Remove the interior from the contour
    contour = contour & ~interior
    pil_image = Image.fromarray((contour.squeeze() * 255).astype(np.uint8))
    pil_image.save(filepath)


def save_classifications(image_classification, hybrid_classification, filepath):
    """
    Save the classifications for the image and the hybrid to CSV in the required format.
    """
    assert np.isclose(sum(image_classification), 1)
    assert np.isclose(sum(hybrid_classification), 1)
    with open(filepath, "w") as f:
        f.write("filename,0,1,2,3,4,5,style\n")
        f.write(
            f"image.png,{image_classification[0]},{image_classification[1]},{image_classification[2]},{image_classification[3]},{image_classification[4]},{image_classification[5]},image\n"
        )
        f.write(
            f"hybrid.png,{hybrid_classification[0]},{hybrid_classification[1]},{hybrid_classification[2]},{hybrid_classification[3]},{hybrid_classification[4]},{hybrid_classification[5]},hybrid"
        )


def save_example_metadata(neuron_id, synapse_index, config, filepath):
    """
    Save the metadata for the example to a text file.
    """
    with open(filepath, "w") as f:
        f.write("This example corresponds to\n")
        f.write(f"Neuron ID: {neuron_id}\n")
        f.write(f"Synapse Index: {synapse_index}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def save_one_synapse(
    neuron_id,
    synapse_index,
    results,
    output_dir,
    attribution_method="DIntegratedGradients",
    threshold=0.01,
):
    """
    Save the results for one synapse.

    neuron_id: int
        The neuron ID.
    synapse_index: int
        The index of the synapse in all the synapses for that neuron.
    results: zarr.hierarchy.Group
        The Zarr group with the results.
    output_dir: str, Path
        The directory to save the results to.
    """
    neuron_results = results[neuron_id]
    neuron_evaluation = neuron_results["evaluations"][attribution_method]
    prediction = neuron_results["predictions"][synapse_index]
    image = neuron_results["images"][synapse_index]
    hybrid = neuron_evaluation["inverse_hybrids"][synapse_index]
    hybrid_classification = neuron_evaluation["inverse_hybrid_classification"][
        synapse_index
    ]
    optimal_mask = neuron_evaluation["optimal_masks"][synapse_index]

    # Save everything
    output_dir = Path(output_dir)
    this_image_dir = output_dir / f"{neuron_id}_{synapse_index}"
    this_image_dir.mkdir(exist_ok=True, parents=True)

    save_one_image(image, this_image_dir / "image.png")
    save_one_image(hybrid, this_image_dir / "hybrid.png")
    save_one_mask(
        optimal_mask,
        this_image_dir / "optimal_mask.png",
        threshold,
    )
    save_inverse_mask(optimal_mask, this_image_dir / "inverse_mask.png", threshold)
    save_mask_contour(optimal_mask, this_image_dir / "contour_mask.png", threshold)
    # Save the classification
    save_classifications(
        prediction, hybrid_classification, this_image_dir / "predictions.csv"
    )
    # Save the metadata
    config = {
        "attribution_method": attribution_method,
        "threshold": threshold,
    }
    save_example_metadata(
        neuron_id, synapse_index, config, this_image_dir / "metadata.txt"
    )


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="Path to the CSV file with neuron_id and synapse_id pairs.",
    )
    parser.add_argument(
        "--attribution_method",
        type=str,
        default="DIntegratedGradients",
        help="The attribution method to use for the mask.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="The directory to save the results to.",
    )
    parser.add_argument(
        "-i",
        "--results_path",
        type=str,
        default="/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr",
        help="The path to the results Zarr file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="The threshold for the contour mask.",
    )
    args = parser.parse_args()
    return args


def main(args):
    synapse_pairs = pd.read_csv(args.file)
    output_dir = Path(args.output_dir)

    # Load the results
    result_path = args.results_path
    results = zarr.open(result_path, mode="r")

    print(f"Loaded results from {result_path}")
    print(f"Saving to {output_dir}")

    for _, row in synapse_pairs.iterrows():
        neuron_id = row["neuron_id"]
        synapse_index = row["synapse_id"]
        print(f"Processing neuron {neuron_id}, synapse {synapse_index}")
        save_one_synapse(
            neuron_id,
            synapse_index,
            results,
            output_dir,
            attribution_method=args.attribution_method,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    args = make_parser()
    main(args)
