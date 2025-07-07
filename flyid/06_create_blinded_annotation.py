import argparse
import yaml
from quac.report import Report
from quac.data import write_image
from pathlib import Path
import random
import json
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create blinded annotations.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="blinded_annotations",
        help="Output directory for blinded annotations.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=5678,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=20,
        help="Number of samples to select per source-to-target pair.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Select top N = 20 examples per source-to-target pair
    Run blinding (with a seed, for reproducibility)
    Generate folders A, B, M which can be used with the Fiji annotation setup
    """

    args = parse_args()
    random.seed(args.seed)

    metadata = yaml.safe_load(open("config.yaml"))
    report_directory = metadata["solver"]["root_dir"] + "/reports"
    # Load the results
    report = Report.from_directory(report_directory, name="final_report")

    folder = Path(args.output)
    filename = folder / "metadata.csv"

    # Create the output folder if it doesn't exist
    a_folder = folder / "A"
    b_folder = folder / "B"
    m_folder = folder / "M"
    a_folder.mkdir(parents=True, exist_ok=True)
    b_folder.mkdir(parents=True, exist_ok=True)
    m_folder.mkdir(parents=True, exist_ok=True)

    class_names = {0: "11", 1: "15", 2: "17"}

    with open(filename, "w") as f:
        # Write the header for the metadata file
        f.write("source,target,score,a_real,hex_id\n")
        for source in 0, 1, 2:
            for target in 0, 1, 2:
                if source == target:
                    continue
                subset_report = (
                    report.from_source(source).to_target(target).top_n(args.n_samples)
                )

                # Loop over the explanations in the subset report
                for explanation in tqdm(
                    subset_report.explanations,
                    desc=f"Processing {class_names[source]} to {class_names[target]}",
                ):
                    # Get a random hex ID
                    name = hex(random.getrandbits(64))[2:]
                    # Get random binary value
                    a_real = random.getrandbits(1)
                    # Save images
                    if a_real:
                        # Save the real image to the A folder
                        write_image(explanation.query, a_folder / f"{name}.png")
                        # Save the counterfactual image to the B folder
                        write_image(
                            explanation.counterfactual, b_folder / f"{name}.png"
                        )
                    else:
                        # Save the real image to the B folder
                        write_image(explanation.query, b_folder / f"{name}.png")
                        # Save the counterfactual image to the A folder
                        write_image(
                            explanation.counterfactual, a_folder / f"{name}.png"
                        )
                    # Save the mask image to the M folder
                    write_image(explanation.mask, m_folder / f"{name}.png")

                    # Save the metadata to a file
                    f.write(
                        f"{class_names[source]},{class_names[target]},{explanation.score},{a_real},{name}\n"
                    )
    # Write a json file with an initial set of features
    features = [
        {
            "name": "Leg Length",
            "-1": "Legs are shorter in A than in B",
            "0": "Legs are equal in A and B",
            "1": "Legs are longer in A than in B",
        },
        {
            "name": "Abdomen Length",
            "-1": "Abdomen is shorter in A than in B",
            "0": "Abdomen is equal in A and B",
            "1": "Abdomen is longer in A than in B",
        },
        {
            "name": "Head size",
            "-1": "Head is smaller in A than in B",
            "0": "Head is equal in A and B",
            "1": "Head is larger in A than in B",
        },
        {
            "name": "Eye color",
            "-1": "Eyes are darker in A than in B",
            "0": "Eyes are equal in A and B",
            "1": "Eyes are lighter in A than in B",
        },
        {
            "name": "Wing shape",
            "-1": "Wings are more symmetrical in A than in B",
            "0": "Wings are equal in A and B",
            "1": "Wings are more asymmetrical in A than in B",
        },
    ]
    features_file = folder / "features.json"
    with open(features_file, "w") as f:
        json.dump(features, f, indent=4)
