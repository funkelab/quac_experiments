# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import zarr
from viz_utils import plot_with_bar, plot_with_text
from matplotlib.patches import ConnectionPatch

# %%
result_path = "/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr"
syn_fraction_path = "syn_class_confidences.csv"
synapses_path = "kenyon_cell_synapses2.csv"

results = zarr.open(result_path)
syn_fraction = pd.read_csv(syn_fraction_path)
synapses = pd.read_csv(synapses_path)

# %%
attribution_method = "DIntegratedGradients"
# %%
all_scores = []
no_eval = []
for neuron_id in tqdm(syn_fraction.pre.values):
    try:
        score = pd.DataFrame(
            {
                "score": results[neuron_id]["evaluations"][attribution_method][
                    "quac_scores"
                ],
                "source": results[neuron_id]["predictions"][:].argmax(axis=1),
                "target": results[neuron_id]["evaluations"][attribution_method][
                    "inverse_hybrid_classification"
                ][:].argmax(axis=1),
                "syn_index": range(len(results[neuron_id]["predictions"])),
            }
        )
        score["neuron_id"] = neuron_id
        all_scores.append(score)
    except KeyError:
        no_eval.append(neuron_id)

all_scores = pd.concat(all_scores)
# %% FIXME 72% not run because of disk quota issues
print(len(no_eval) / len(syn_fraction))
# %% Choose only samples where source=5 and target=1
all_scores = all_scores[(all_scores.source == 5) & (all_scores.target == 1)]
# %% Order by scores, descending
all_scores = all_scores.sort_values("score", ascending=False)

# %%
# %% Take the top N scored samples
N = 10
threshold = 0.0001
top_scores = all_scores.head(N)

# group by neuron id
for neuron_id, group in top_scores.groupby("neuron_id"):
    neuron_results = results[neuron_id]
    neuron_evaluation = neuron_results["evaluations"][attribution_method]
    # FIXME I've swapped Hybrid and Inverse Hybrid in these results!
    predictions = neuron_results["predictions"]
    counterfactual_predictions = neuron_results["counterfactual_predictions"]
    images = neuron_results["images"]
    hybrids = neuron_evaluation["inverse_hybrids"]
    inverse_hybrids = neuron_evaluation["hybrids"]
    inverse_hybrid_classification = neuron_evaluation["hybrid_classification"]
    hybrid_classification = neuron_evaluation["inverse_hybrid_classification"]
    counterfactuals = neuron_results["counterfactuals"]
    optimal_masks = neuron_evaluation["optimal_masks"]

    for index_to_show, row in group.iterrows():
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        plot_with_text(
            images[index_to_show],
            predictions[index_to_show],
            axes[0],
            target=1,
            mask=optimal_masks[index_to_show],
            threshold=threshold,
        )
        plot_with_text(
            hybrids[index_to_show],
            hybrid_classification[index_to_show],
            axes[1],
            source=5,
            target=1,
            mask=optimal_masks[index_to_show],
            threshold=threshold,
        )
        # Add an arrow between the two images
        con = ConnectionPatch(
            xyA=(1, 0.5),
            coordsA=axes[0].transAxes,
            xyB=(0, 0.5),
            coordsB=axes[1].transAxes,
            arrowstyle="simple, tail_width=0.1, head_width=0.5, head_length=1",
            # lw=1,
            facecolor="black",
            edgecolor="none",
        )
        axes[0].add_artist(con)
        # fig.suptitle(f"QuAC score: {row['score']:.4f}", fontsize=12)
        # fig.tight_layout()
        # plt.show()
        plt.savefig(
            f"results/{neuron_id}_{index_to_show}_{int(row['score']*100):000d}.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
# # %%

# %% TODO keep this for plotting later

# # %% Plotting
# for i in range(1):
#     index_to_show = ordered_indices[i]
#     fig, axes = plt.subplots(
#         2, 2, gridspec_kw={"height_ratios": [1, 3], "hspace": -0.1}, figsize=(7, 5)
#     )
#     axes = axes.T
#     plot_with_bar(
#         images[index_to_show],
#         predictions[index_to_show],
#         axes[0],
#         mask=optimal_masks[index_to_show],
#         title="$x$",
#     )
#     plot_with_bar(
#         hybrids[index_to_show],
#         hybrid_classification[index_to_show],
#         axes[1],
#         "$mx_{cf} + (1 - m)x$",
#         mask=optimal_masks[index_to_show],
#         threshold=0.05,
#     )
#     # Add an arrow between the two images
#     con = ConnectionPatch(
#         xyA=(1, 0.5),
#         coordsA=axes[0, 1].transAxes,
#         xyB=(0, 0.5),
#         coordsB=axes[1, 1].transAxes,
#         arrowstyle="simple, tail_width=0.1, head_width=0.5, head_length=1",
#         # lw=1,
#         facecolor="black",
#         edgecolor="none",
#     )
#     axes[0, 1].add_artist(con)
#     # title on the bottom
#     #  fig.suptitle(f"QuAC score: {quac_scores[index_to_show]:.4f}", fontsize=12)
#     # fig.tight_layout()
#     plt.show()
#     # NExt one
#     # index_to_show = ordered_indices[i]
#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))
#     plot_with_text(
#         images[index_to_show],
#         predictions[index_to_show],
#         axes[0],
#         target=1,
#         mask=optimal_masks[index_to_show],
#         # title="Image",
#     )
#     plot_with_text(
#         hybrids[index_to_show],
#         hybrid_classification[index_to_show],
#         axes[1],
#         source=5,
#         target=1,
#         mask=optimal_masks[index_to_show],
#         # title="Necessary Perturbation",
#     )
#     # Add an arrow between the two images
#     # Add an arrow between the two images
#     con = ConnectionPatch(
#         xyA=(1, 0.5),
#         coordsA=axes[0].transAxes,
#         xyB=(0, 0.5),
#         coordsB=axes[1].transAxes,
#         arrowstyle="simple, tail_width=0.1, head_width=0.5, head_length=1",
#         # lw=1,
#         facecolor="black",
#         edgecolor="none",
#     )
#     axes[0].add_artist(con)
#     # fig.suptitle(f"QuAC score: {quac_scores[index_to_show]:.4f}", fontsize=12)
#     # fig.tight_layout()
#     plt.show()
# # %%

# %%
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(5, 5))
n_samples = 40
anim_images = np.linspace(
    images[index_to_show].squeeze(), hybrids[index_to_show].squeeze(), n_samples
)

vmin = anim_images.min()
vmax = anim_images.max()


def update(i):
    ax.imshow(anim_images[i], cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")


anim = FuncAnimation(fig, update, frames=n_samples, repeat=True, repeat_delay=5000)
anim.save("results/animation.gif", writer="imagemagick", fps=10)
plt.close()

# %%
import navis

nl = navis.example_neurons(kind="skeleton")

fig, ax = nl.plot2d()
plt.show()
# %%
from fafbseg import flywire

skeletons = [
    flywire.get_skeletons(neuron_id) for neuron_id in syn_fraction.pre.values[:50]
]
# %%
fig, ax = navis.plot2d([m, *skeletons], volume_outlines="both")
# %%
