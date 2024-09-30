# %%
import zarr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Loading the data
results = zarr.open(
    "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr", "r"
)

evaluation = results["evaluations"]

# %% QuAC scores
for attribution_method in evaluation.keys():
    pop_curve = evaluation[f"{attribution_method}/population_curve"][:]
    plt.plot(pop_curve[0], label=attribution_method)
    p25, p75 = pop_curve[1], pop_curve[2]
    plt.fill_between(range(len(pop_curve[0])), p25, p75, alpha=0.5)
    # Store the results
    to_store = pd.DataFrame(pop_curve, index=["median", "p25", "p75"]).T
    to_store.to_csv(f"paper_results/{attribution_method}_curve.csv", index=False)
plt.legend()
plt.show()

attribution_method = "DIntegratedGradients"
pop_curve = evaluation[f"{attribution_method}/population_curve"][:]

# %%
# The indices that are "valid" (i.e., where the source and target are not the same) have been annotated using a boolean array.
# Here, we extract that information and store it in a matrix.
# Each row corresponds to a *target* class.
# Each column corresponds to one item in the original set of images.
all_valid = np.stack(
    [results[f"is_valid/{key}"][:] for key in results["is_valid"].keys()]
)
# %%
# Describe the hybrid indices
labels = results["labels"][:]
# We also want to render "invalid" items where the original image's prediction does not match the source label's prediction
well_predicted = labels == results["predictions"][:].argmax(axis=1)
mapping = []
i = 0
for target in range(all_valid.shape[0]):
    for is_valid in np.where(all_valid[target])[0]:
        index = i
        i += 1
        label = labels[is_valid]
        mapping.append((label, target, is_valid, index, well_predicted[is_valid]))

# %%
import pandas as pd

mapping = pd.DataFrame(
    mapping,
    columns=["source", "target", "source_index", "eval_index", "well_predicted"],
)
mapping["quac_scores"] = evaluation[f"{attribution_method}/quac_scores"][:]
# Drop everything that is not well predicted
mapping = mapping[mapping["well_predicted"]]
# %%
import matplotlib as mpl


# Add the inset with the classification
def add_predictions(axis, pred, colors):
    axins = axis.inset_axes([0, 1, 1, 0.5])
    # Make a bar chart, with six different colors
    axins.bar(range(6), pred.squeeze(), color=colors)
    axins.set_ylim([0, 1])
    # Hide spines
    axins.spines["right"].set_visible(False)
    axins.spines["top"].set_visible(False)
    axins.spines["bottom"].set_visible(False)
    # Remove xticks
    axins.set_xticks([])
    # Clean up yticks
    axins.set_yticks([0, 1])
    axins.set_yticklabels(["0", "1"])


for (src, target), group in mapping.groupby(["source", "target"]):
    classes = sorted(list(results["counterfactuals"].keys()))
    target_name = classes[target]
    counterfactuals = results[f"counterfactuals/{target_name}"]
    # Get the row with the highest quac score
    idx = group["quac_scores"].idxmax()
    # Get the corresponding source image
    source_index = group.loc[idx, "source_index"]
    image = results["images"][source_index]
    # Get the corresponding counterfactual image
    cf = counterfactuals[source_index]
    # Get the corresponding hybrid image
    hybrid_index = group.loc[idx, "eval_index"]
    hybrid = evaluation[f"{attribution_method}/hybrids"][hybrid_index]
    # Get the corresponding mask image
    mask = evaluation[f"{attribution_method}/optimal_masks"][hybrid_index]
    # Mask the mask
    mask = np.ma.masked_where(mask == 1, mask)
    # Load the classifications
    pred = results["predictions"][source_index]
    cf_pred = results[f"counterfactual_predictions/{target_name}"][source_index]
    hybrid_pred = evaluation[f"{attribution_method}/hybrid_classification"][
        hybrid_index
    ]
    # define barchart colors
    colors = mpl.colormaps["tab10"](np.linspace(0, 1, 6))
    # Plot the images
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image.squeeze(), cmap="gray", vmin=-1, vmax=1)
    axs[0].axis("off")
    axs[0].set_title(f"Source: {src}")
    add_predictions(axs[0], pred, colors)
    # Counterfactual (formerly hybrid)
    axs[1].imshow(hybrid.squeeze(), cmap="gray", vmin=-1, vmax=1)
    # Draw a contour plot of the mask
    threshold = 0.5
    axs[1].contour(mask.squeeze(), levels=[threshold], colors="magenta", linewidths=2)
    # Slightly darken the areas outside of the mask
    plot_mask = mask.squeeze().copy()
    plot_mask = np.ma.masked_where(mask.squeeze() > threshold, np.ones_like(plot_mask))
    axs[1].imshow(plot_mask, alpha=0.4, cmap="gray_r", vmin=0, vmax=1)
    #
    axs[1].axis("off")
    axs[1].set_title(f"Hybrid: {src} -> {target}")
    add_predictions(axs[1], hybrid_pred, colors)
    # Generated (formerly counterfactual)
    axs[2].imshow(cf.squeeze(), cmap="gray", vmin=-1, vmax=1)
    axs[2].axis("off")
    axs[2].set_title(f"Counterfactual: {target}")
    add_predictions(axs[2], cf_pred, colors)
    #
    fig.suptitle(f"QuAC score: {group.loc[idx, 'quac_scores']:.2f}")
    plt.show()


# %%
# Next, we'll create the directory structure to save the results.
#
# paper_results/ - This directory will hold the outputs exactly as they are needed for the paper
#   best_examples/
#       source_class_target_class/
#           image.png
#           hybrid.png
#           predictions.csv -> filename,0,1,2,3,4,5,style
#
# visualizations/ - This directory will hold other visualizations
# %%
from pathlib import Path

results_path = Path("paper_results/best_examples")
for (src, target), group in mapping.groupby(["source", "target"]):
    classes = sorted(list(results["counterfactuals"].keys()))
    source_name = classes[src]
    target_name = classes[target]
    this_path = results_path / f"{source_name}_{target_name}"
    this_path.mkdir(parents=True, exist_ok=True)
    #
    counterfactuals = results[f"counterfactuals/{target_name}"]
    # Get the row with the highest quac score
    idx = group["quac_scores"].idxmax()
    # Get the corresponding source image
    source_index = group.loc[idx, "source_index"]
    image = results["images"][source_index]
    # Get the corresponding hybrid image
    hybrid_index = group.loc[idx, "eval_index"]
    hybrid = evaluation[f"{attribution_method}/hybrids"][hybrid_index]
    # Get the corresponding mask image
    mask = evaluation[f"{attribution_method}/optimal_masks"][hybrid_index]
    # Mask the mask
    mask = np.ma.masked_where(mask == 1, mask)
    # Load the classifications
    pred = results["predictions"][source_index]
    hybrid_pred = evaluation[f"{attribution_method}/hybrid_classification"][
        hybrid_index
    ]
    # Plot the image
    plt.imshow(image.squeeze(), cmap="gray", vmin=-1, vmax=1)
    plt.axis("off")
    # Save and clear figure
    plt.savefig(this_path / "image.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    # Counterfactual (formerly hybrid)
    plt.imshow(hybrid.squeeze(), cmap="gray", vmin=-1, vmax=1)
    # Draw a contour plot of the mask
    threshold = 0.1
    plt.contour(mask.squeeze(), levels=[threshold], colors="magenta", linewidths=3)
    # Slightly darken the areas outside of the mask
    plot_mask = mask.squeeze().copy()
    plot_mask = np.ma.masked_where(mask.squeeze() > threshold, np.ones_like(plot_mask))
    plt.imshow(plot_mask, alpha=0.4, cmap="gray_r", vmin=0, vmax=1)
    #
    plt.axis("off")
    # Save and close figure
    plt.savefig(this_path / "hybrid.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    pred_file = pd.DataFrame(
        {
            "filename": ["image.png", "hybrid.png"],
            "0": [pred.squeeze()[0], hybrid_pred.squeeze()[0]],
            "1": [pred.squeeze()[1], hybrid_pred.squeeze()[1]],
            "2": [pred.squeeze()[2], hybrid_pred.squeeze()[2]],
            "3": [pred.squeeze()[3], hybrid_pred.squeeze()[3]],
            "4": [pred.squeeze()[4], hybrid_pred.squeeze()[4]],
            "5": [pred.squeeze()[5], hybrid_pred.squeeze()[5]],
            "style": ["image", "hybrid"],
        }
    )
    # Save
    pred_file.to_csv(this_path / "predictions.csv", index=False)

# %%
