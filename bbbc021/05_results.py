# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
from quac.report import Report
from quac.config import ExperimentConfig
import yaml
import torch.nn.functional as F
import torch
from matplotlib.colors import LinearSegmentedColormap

# from preparation.utils import get_meta_from_name


def get_meta_from_name(name: str):
    """
    Get metadata from the image name.
    The name is expected to be in the format: plate_well_table_image.
    Note that the plate name has underscores, so we cannot directly split by underscores.
    """
    name = Path(name).stem  # Get the stem of the path to avoid issues with extensions
    well, table, image, cell = name.split("_")[-4:]
    plate = name.replace(f"_{well}_{table}_{image}_{cell}", "")
    return {
        "plate": plate,
        "well": well,
        "table": int(table),
        "image": int(image),
        "cell": cell,
    }


# %% Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


experiment = ExperimentConfig(**config)
report_directory = Path(experiment.solver.root_dir) / "reports"
# %%
# NOTE: Only do this once!
# # Filtering and re-storing the report
# report = Report.from_directory(report_directory)
# # Try making a set!
# test = set(report.explanations)
# report.explanations = sorted(list(test), key=lambda x: x.score, reverse=True)
# # %%
# # Save the report so I don't have to re-do this:
# report.name = "final_report_filtered"
# report.store(report_directory)

# %%
# From this point on, the report can just be loaded from the file,
# Instead of from the directory.
report = Report()
report.load(report_directory / "final_report_filtered.json")


# %%
class_names = {
    0: "A.D",  # "Actin disruptor",
    1: "DMSO",  # "Control",
    2: "M.D",  # "Microtubule destabilizer",
    3: "M.S",  # Microtubule stabilizer",
    4: "P.D",  # "Protein degradation",
    5: "P.S",  # "Protein synthesis",
}


def plot_contours(mask, ax, colors=["magenta", "gold", "cyan"]):
    """
    Plot contours for the mask for each of the channels in the mask.
    Plot R in magenta, G in gold, B in cyan.
    """
    color_match = {
        "magenta": 0,
        "gold": 1,
        "cyan": 2,
    }
    for color in colors:
        i = color_match[color]
        ax.contour(mask[i], levels=[0.3], colors=color, linewidths=1)
    return


def plot_cym(image, ax):
    magenta_cmap = LinearSegmentedColormap.from_list("magenta", ["black", "magenta"])
    cyan_cmap = LinearSegmentedColormap.from_list("cyan", ["black", "cyan"])
    yellow_cmap = LinearSegmentedColormap.from_list("yellow", ["black", "yellow"])
    ax.imshow(image[..., 0], cmap=magenta_cmap)
    ax.imshow(image[..., 2], cmap=cyan_cmap, alpha=0.5)
    ax.imshow(image[..., 1], cmap=yellow_cmap, alpha=0.3)


def plot_explanation(explanation, class_names=class_names):
    """
    Plot the query and counterfactual images in an explanation.

    Plots a 2x2 grid, with the query in the top left corner,
    the counterfactual in the bottom left corner,
    and the pmagentaictions next to their corresponding image.
    The mask is overlaid depending on the value of `show_mask`.
    The QuAC score is shown as a figure title.
    """
    fig, (
        (ax1r, ax1g, ax1b, ax1),
        (diffr, diffg, diffb, none),
        (ax2r, ax2g, ax2b, ax2),
    ) = plt.subplots(3, 4, gridspec_kw={"height_ratios": [1, 0.2, 1]})

    # Plot the query and its accompanying pmagentaiction
    # TODO this resize shouldn't be necessary :'(
    # Resize to the same size as the counterfactual
    query = F.interpolate(
        explanation.query.unsqueeze(0),
        size=explanation.counterfactual.shape[1:3],
        mode="bilinear",
    ).squeeze(0)
    query = query.permute(1, 2, 0)
    # Convert image to CMYK for plotting
    plot_cym(query, ax1)
    # ax1.imshow(query)
    ax1.axis("off")
    # ax1.set_title(f"Query: {class_names[explanation.source_class]}")
    # bax1.bar(
    #     np.arange(len(explanation.query_pmagentaiction)),
    #     explanation.query_pmagentaiction,
    #     color="gray",
    # )
    for i, (ax, color) in enumerate(
        zip((ax1r, ax1g, ax1b), ["magenta", "gold", "cyan"])
    ):
        ax.imshow(query[:, :, i], cmap="gray", vmin=0, vmax=1)
        plot_contours(explanation.mask, ax, colors=[color])
        ax.axis("off")
    ax1r.set_title("F-actin")
    ax1g.set_title("B-tubulin")
    ax1b.set_title("DAPI")
    ax1.set_title("Merge")

    # Plot the counterfactual and its accompanying pmagentaiction
    counterfactual = explanation.counterfactual.permute(1, 2, 0)
    # ax2.imshow(counterfactual)
    plot_cym(counterfactual, ax2)
    # ax2.set_title(f"CF: {class_names[explanation.target_class]}")
    ax2.axis("off")
    # bax2.bar(
    #     np.arange(len(explanation.counterfactual_pmagentaiction)),
    #     explanation.counterfactual_pmagentaiction,
    #     color="gray",
    # )
    for i, (ax, color) in enumerate(
        zip((ax2r, ax2g, ax2b), ["magenta", "gold", "cyan"])
    ):
        ax.imshow(counterfactual[:, :, i], cmap="gray", vmin=0, vmax=1)
        plot_contours(explanation.mask, ax, colors=[color])
        ax.axis("off")

    # Make better bar plots
    # for bax in (bax1, bax2):
    #     # x-labels are class names
    #     bax.set_xticks(np.arange(len(explanation.query_pmagentaiction)))
    #     bax.set_xticklabels(class_names.values(), rotation=90)
    #     # Remove y-ticks, set the range from 0 to 1
    #     bax.set_yticks([])
    #     bax.set_ylim(0, 1)
    #     # Remove top and right spines
    #     bax.spines["top"].set_visible(False)
    #     bax.spines["right"].set_visible(False)

    plot_contours(explanation.mask, ax1)
    plot_contours(explanation.mask, ax2)

    # intensity, scaled by mask, as a bar plot
    for i, (ax, color) in enumerate(
        zip((diffr, diffg, diffb), ["magenta", "gold", "cyan"])
    ):
        mask = (explanation.mask[i] > 0.3).float()
        mask[mask == 0] = torch.nan  # Set masked values to NaN for mean calculation
        ax.barh(
            ["xc", "x"],
            [1, 1],
            color="gray",
            alpha=0.5,
        )

        ax.barh(
            ["xc", "x"],
            [
                (mask * counterfactual[..., i]).nanmean(),
                (mask * query[..., i]).nanmean(),
            ],
            color=color,
        )
        ax.set_xlim(0, 1)
        # Remove
        ax.axis("off")

    none.axis("off")

    # fig.suptitle(f"Score: {explanation.score:.2f}", y=0.9)
    fig.tight_layout()
    return fig


def compile_metadata(explanation, full_meta):
    metadata = get_meta_from_name(Path(explanation._query_path).name)
    metadata["id"] = os.urandom(6).hex()
    metadata["source_class"] = class_names[explanation.source_class]
    metadata["target_class"] = class_names[explanation.target_class]
    row = full_meta[
        (full_meta["Image_Metadata_Plate_DAPI"] == metadata["plate"])
        & (full_meta["Image_Metadata_Well_DAPI"] == metadata["well"])
        & (full_meta["TableNumber"] == metadata["table"])
        & (full_meta["ImageNumber"] == metadata["image"])
    ].iloc[0]

    metadata["compound"] = row["compound"]
    metadata["concentration"] = row["Image_Metadata_Concentration"].item()
    metadata["moa"] = row["moa"]
    return metadata


# %%
full_meta = pd.read_csv("preparation/data/moa_split_test.csv")

# %%
n = 20  # Number of explanations to save per class
# CSV File with metadata, and folder for images
folder = Path("annotations/images")
folder.mkdir(exist_ok=True, parents=True)
file = Path("annotations") / "hidden_metadata.csv"

# From DMSO
with open(file, "w") as f:
    # Add a header to the CSV file
    f.write(
        "id,plate,well,table,image,cell,compound,concentration,moa,source_class,target_class\n"
    )
    source = 1
    for target in [0, 2, 3, 4, 5]:
        filtered_report = report.from_source(source).to_target(target)
        print(
            f"Filtered report {class_names[source]} -> {class_names[target]} has {len(filtered_report.explanations)} explanations."
        )

        for explanation in filtered_report.top_n(n).explanations:
            metadata = compile_metadata(explanation, full_meta)
            fig = plot_explanation(explanation)
            # Save the figure
            fig.savefig(folder / f"{metadata['id']}.png", dpi=300)
            plt.close(fig)
            # Write metadata to CSV
            f.write(
                f"{metadata['id']},{metadata['plate']},{metadata['well']},{metadata['table']},{metadata['image']},"
                f"{metadata['cell']},{metadata['compound']},{metadata['concentration']},{metadata['moa']},"
                f"{metadata['source_class']},{metadata['target_class']}\n"
            )

    # TO DSMO
    target = 1
    for source in [0, 2, 3, 4, 5]:
        filtered_report = report.from_source(source).to_target(target)
        print(
            f"Filtered report {class_names[source]} -> {class_names[target]} has {len(filtered_report.explanations)} explanations."
        )

        for explanation in filtered_report.top_n(n).explanations:
            metadata = compile_metadata(explanation, full_meta)
            fig = plot_explanation(explanation)
            # Save the figure
            fig.savefig(folder / f"{metadata['id']}.png", dpi=300)
            plt.close(fig)
            # Write metadata to CSV
            f.write(
                f"{metadata['id']},{metadata['plate']},{metadata['well']},{metadata['table']},{metadata['image']},"
                f"{metadata['cell']},{metadata['compound']},{metadata['concentration']},{metadata['moa']},"
                f"{metadata['source_class']},{metadata['target_class']}\n"
            )

# %%
# Create a file for the naive annotations: it should just be the first column of the CSV file
naive_annotations = pd.read_csv(file)
naive_annotations = naive_annotations[["id"]]
# Add an empty column for the annotations
naive_annotations["annotation"] = ""
# Save the naive annotations to a file
naive_annotations.to_csv("annotations/naive_annotations_template.csv", index=False)

# %%
