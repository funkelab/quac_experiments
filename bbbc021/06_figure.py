# %% [markdown]
# # Extracting the results needed for the figure
#
# 1. Representative samples
# 2. Confusion matrices
# 3. QuAC scores

# %% Requirements
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from quac.config import ExperimentConfig
from quac.data import write_image
from quac.report import Report
from torchvision.transforms import functional as F
import yaml


def save_contours(
    mask,
    directory,
    channels=["f_actin", "b_tubulin", "dapi"],
    threshold=0.3,
):
    """
    Plot contours for the mask for each of the channels in the mask.

    mask: np.ndarray
        The mask to use to make the contours.
    directory: Path
        The directory to save the contours to.
    channels: list of str
        The names of the channels, used to name the contour files.
    """
    for i, channel in enumerate(channels):
        plt.imshow(np.zeros_like(mask[i]).squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.contour(mask[i].squeeze(), levels=[threshold], colors="white", linewidths=3)
        plt.axis("off")
        plt.savefig(directory / f"{channel}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
    return


# %%
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    experiment = ExperimentConfig(**config)
    report_directory = Path(experiment.solver.root_dir) / "reports"

print("Reading final report...")
report = Report(report_directory)
report.load(report_directory / "final_report_filtered.json")
print("...done.")
# %%
class_names = [
    "actin_disruptors",
    "dmso",
    "microtubule_destabilizers",
    "microtubule_stabilizers",
    "protein_degradation",
    "protein_synthesis",
]

for source_class in range(6):
    for target_class in range(6):
        if source_class == target_class:
            continue
        if source_class != 1 and target_class != 1:
            continue

        top10 = report.from_source(source_class).to_target(target_class).top_n(10)

        for sample, explanation in enumerate(top10.explanations):

            sample_path = Path(
                f"figures/{class_names[source_class]}/{class_names[target_class]}/{sample}"
            )
            sample_path.mkdir(parents=True, exist_ok=True)

            # Downsample query to match counterfactual, because we downsampled for training
            query = F.resize(explanation.query, explanation.counterfactual.shape[1:])

            # We don't have a use for those images, skip that
            # for i, channel in enumerate(["f_actin", "b_tubulin", "dapi"]):
                # write_image(
                    # query[i],
                    # sample_path / f"image_{channel}.png",
                # )

                # write_image(
                    # explanation.counterfactual[i],
                    # sample_path / f"cf_image_{channel}.png",
                # )

            write_image(query, sample_path / "image.png")
            write_image(explanation.counterfactual, sample_path / "hybrid.png")
            save_contours(
                explanation.mask,
                sample_path,
                channels=["contour_red", "contour_green", "contour_blue"],
                threshold=explanation._optimal_threshold,
            )

            # Store the predictions as well
            with open(sample_path / "predictions.csv", "w") as f:
                f.write("filename,A,B,C,D,E,F,style\n")

                f.write("image.png,")
                f.write(",".join([str(v) for v in explanation.query_prediction]))
                f.write(",image\n")

                f.write("hybrid.png,")
                f.write(",".join([str(v) for v in explanation.counterfactual_prediction]))
                f.write(",counterfactual\n")
