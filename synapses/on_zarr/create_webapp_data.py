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
# attribution_method = "DIntegratedGradients"
attribution_method = "DDeepLift"

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
# Get only the subset that is from a particular source to a particular target
source = 5
target = 1

lim_mapping = mapping[(mapping["source"] == source) & (mapping["target"] == target)]
# Order by the quac score
lim_mapping = lim_mapping.sort_values("quac_scores", ascending=False)
# %%
# Plot the top 10 examples
# import matplotlib as mpl


# # Add the inset with the classification
# def add_predictions(axis, pred, colors):
#     axins = axis.inset_axes([0, 1, 1, 0.5])
#     # Make a bar chart, with six different colors
#     axins.bar(range(6), pred.squeeze(), color=colors)
#     axins.set_ylim([0, 1])
#     # Hide spines
#     axins.spines["right"].set_visible(False)
#     axins.spines["top"].set_visible(False)
#     axins.spines["bottom"].set_visible(False)
#     # Remove xticks
#     axins.set_xticks([])
#     # Clean up yticks
#     axins.set_yticks([0, 1])
#     axins.set_yticklabels(["0", "1"])


# for idx, row in lim_mapping.head(10).iterrows():
#     if row.quac_scores < 0.75:
#         continue
#     classes = sorted(list(results["counterfactuals"].keys()))
#     target_name = classes[target]
#     counterfactuals = results[f"counterfactuals/{target_name}"]
#     # Get the row with the highest quac score
#     image = results["images"][row.source_index]
#     # Get the corresponding counterfactual image
#     cf = counterfactuals[row.source_index]
#     # Get the corresponding hybrid image
#     hybrid = evaluation[f"{attribution_method}/hybrids"][row.eval_index]
#     # Get the corresponding mask image
#     mask = evaluation[f"{attribution_method}/optimal_masks"][row.eval_index]
#     # Mask the mask
#     mask = np.ma.masked_where(mask == 1, mask)
#     # Load the classifications
#     pred = results["predictions"][row.source_index]
#     cf_pred = results[f"counterfactual_predictions/{target_name}"][row.source_index]
#     hybrid_pred = evaluation[f"{attribution_method}/hybrid_classification"][
#         row.eval_index
#     ]
#     # define barchart colors
#     colors = mpl.colormaps["tab10"](np.linspace(0, 1, 6))
#     # Plot the images
#     fig, axs = plt.subplots(1, 3)
#     axs[0].imshow(image.squeeze(), cmap="gray", vmin=-1, vmax=1)
#     axs[0].axis("off")
#     add_predictions(axs[0], pred, colors)
#     # Counterfactual (formerly hybrid)
#     axs[1].imshow(hybrid.squeeze(), cmap="gray", vmin=-1, vmax=1)
#     # # Draw a contour plot of the mask
#     threshold = 0.5
#     axs[1].contour(mask.squeeze(), levels=[threshold], colors="magenta", linewidths=2)
#     # Slightly darken the areas outside of the mask
#     plot_mask = mask.squeeze().copy()
#     plot_mask = np.ma.masked_where(mask.squeeze() > threshold, np.ones_like(plot_mask))
#     axs[1].imshow(plot_mask, alpha=0.4, cmap="gray_r", vmin=0, vmax=1)
#     #
#     axs[1].axis("off")
#     add_predictions(axs[1], hybrid_pred, colors)
#     # Generated (formerly counterfactual)
#     axs[2].imshow(cf.squeeze(), cmap="gray", vmin=-1, vmax=1)
#     axs[2].axis("off")
#     axs[2].set_title(f"Counterfactual: {target}")
#     add_predictions(axs[2], cf_pred, colors)
#     #
#     fig.suptitle(f"QuAC score: {row.quac_scores:.2f}")
#     plt.show()

# %% Show the same images, except create an interpolation between the source and the hybrid
# Don't plot the classifications
# Show these as an animation in a loop
from matplotlib import animation

for idx, row in lim_mapping.head(10).iterrows():
    classes = sorted(list(results["counterfactuals"].keys()))
    target_name = classes[target]
    counterfactuals = results[f"counterfactuals/{target_name}"]
    # Get the row with the highest quac score
    image = results["images"][row.source_index]
    # Get the corresponding counterfactual image
    cf = counterfactuals[row.source_index]
    # Get the corresponding hybrid image
    hybrid = evaluation[f"{attribution_method}/hybrids"][row.eval_index]
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # , wspace=None, hspace=None)

    def update(frame):
        alpha = frame / 10
        axs.imshow(
            (1 - alpha) * image.squeeze() + alpha * hybrid.squeeze(),
            cmap="gray",
            vmin=-1,
            vmax=1,
        )
        axs.axis("off")
        # fig.suptitle(f"QuAC score: {row.quac_scores:.2f}")
        return axs

    ani = animation.FuncAnimation(fig, update, frames=10, interval=10)
    print(
        f"Saving animation for source {row.source} and target {row.target}, index {row.source_index}"
    )
    # Save animation as an animated gif,
    ani.save(
        f"animations_{attribution_method}/source_{row.source}_target_{row.target}_{row.source_index}.gif",
        writer="imagemagick",
        # bbox_inches="tight",
    )
    plt.close(fig)


# %%
# Plot the QuAC scores as quantiles per source-target pair
import seaborn as sns

sns.boxplot(
    data=mapping,
    x="source",
    y="quac_scores",
    hue="target",
    showfliers=False,
    palette="tab10",
)

# %%
# Add the classification confidences into the mapping
source_confidences = np.take_along_axis(
    results["predictions"][mapping["source_index"]],
    mapping["source"].values[:, None],
    1,
)
mapping["source_confidences"] = source_confidences

# %%
names = [
    "0_gaba",
    "1_acetylcholine",
    "2_glutamate",
    "3_serotonin",
    "4_octopamine",
    "5_dopamine",
]
added_mapping = []

for target, group in mapping.groupby("target"):
    target_confidences = np.take_along_axis(
        results[f"counterfactual_predictions/{names[target]}"][group["source_index"]],
        group["target"].values[:, None],
        1,
    )
    group["target_confidences"] = target_confidences
    added_mapping.append(group)

added_mapping = pd.concat(added_mapping)

# %%
# Plot the target confidences as quantiles per source-target pair
sns.boxplot(
    data=added_mapping,
    x="source",
    y="target_confidences",
    hue="target",
    showfliers=False,
    palette="tab10",
)

# %%
# Plot the source confidences as quantiles per source-target pair
sns.boxplot(
    data=added_mapping,
    x="source",
    y="source_confidences",
    hue="target",
    showfliers=False,
    palette="tab10",
)

# %%
# Plot the quac scores as a function of the target_confidences
# Plot a scatter plot with a linear regression line
# Really reduce the opacity of the points so that the density is visible

lm = sns.lmplot(
    data=added_mapping,
    x="target_confidences",
    y="quac_scores",
    palette="tab10",
    scatter_kws={"alpha": 0.01},
    legend=True,
    # col="source",
    # row="target",
)
lm.axes[0, 0].axhline(1 / 6, color="red")
# %%
