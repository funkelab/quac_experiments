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

report = Report(report_directory)
report.load(report_directory / "final_report_filtered.json")
# %%
class_names = [
    "actin_disruptors",
    "dmso",
    "microtubule_destabilizers",
    "microtubule_stabilizers",
    "protein_degradation",
    "protein_synthesis",
]
# %% [markdown]
# ## 1. Representative samples
# We extract representative samples from the annotation results.
#
# %%
plates = [
    "Week1_22141",
    "Week8_38203",
    "Week3_25461",
    "Week1_22141",
    "Week7_34341",
    "Week9_39206",
    "Week3_25701",
    "Week8_38241",
]
wells = ["B03", "E11", "B07", "F11", "C09", "D02", "C07", "G11"]
tables = [1, 8, 3, 1, 7, 9, 3, 8]
images = [247, 160, 502, 440, 71, 81, 3183, 717]
cells = [32, 44, 35, 129, 50, 9, 24, 62]

for plate, well, table, image, cell in zip(
    plates,
    wells,
    tables,
    images,
    cells,
):
    explanation = [
        ex
        for ex in report.explanations
        if Path(ex._query_path).stem == f"{plate}_{well}_{table}_{image}_{cell}"
    ][0]
    end_path = Path(
        f"figures/{class_names[explanation.source_class]}/{class_names[explanation.target_class]}"
    )
    real_path = end_path / "real"
    counterfactual_path = end_path / "counterfactual"
    mask_path = end_path / "mask"
    real_path.mkdir(parents=True, exist_ok=True)
    counterfactual_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)

    for i, channel in enumerate(["f_actin", "b_tubulin", "dapi"]):
        # Downsample query to match counterfactual, because we downsampled for training
        query = F.resize(explanation.query, explanation.counterfactual.shape[1:])
        write_image(
            query[i],
            real_path / f"{channel}.png",
        )

        write_image(
            explanation.counterfactual[i],
            counterfactual_path / f"{channel}.png",
        )
    write_image(query, real_path / "merged.png")
    write_image(explanation.counterfactual, counterfactual_path / "merged.png")
    save_contours(
        explanation.mask,
        mask_path,
        channels=["f_actin", "b_tubulin", "dapi"],
        threshold=0.3,
    )

# %%
