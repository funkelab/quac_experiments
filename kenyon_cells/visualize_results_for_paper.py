# %% [markdown]
# # Visualize results for paper
#
# In this notebook, we will visualize the results of the Kenyon Cell experiments for the paper.
# This is an extension of the `visualize_results.py` script, which was used to visualize the results of the Kenyon Cell experiments for the connectomics conference.
#
# We will:
# - Choose 10 neurons that are classified as dopaminergic with the highest confidence
# - For each neuron, show the samples where the source is 5 and the target is 1, in order of their QuAC scores
# - Save the visualizations to a PDF file per neuron
#
# The PDF will be used to annotate the differences between the original image and the counterfactual image.
# This will be used as a quantitative evaluation of the features that we first found for the connectomics conference.
# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
from pathlib import Path
import json
from matplotlib.backends.backend_pdf import PdfPages
from viz_utils import plot_with_text

# %% Reading the data
result_path = "/nrs/funke/adjavond/projects/quac/kenyon_cells/results2.zarr"
syn_fraction_path = "syn_class_confidences.csv"
synapses_path = "kenyon_cell_synapses2.csv"

results = zarr.open(result_path)
syn_fraction = pd.read_csv(syn_fraction_path)
synapses = pd.read_csv(synapses_path)

# %% [markdown]
# ## Description of the data
#
# The `syn_fraction` dataframe contains a per-neuron classification, along with a confidence level computed as in the Eckstein et al. Cell paper.
# There are 5175 neurons in total.
#
# The `synapses` dataframe contains the synapse location and other metadata.
# There are 1289855 synapses in total.
#
# The `results` zarr store contains the results of the QuAC evaluation.
# The results are grouped by neuron ID.
# Due to a disk space issue, not all neurons have been evaluated.
# However, as they have been evaluated in order of confidence, the top N neurons should be available.
#
# Here we will filter out only neurons classified as dopaminergic, and then order them by confidence.
# %%
dopaminergic_neurons = syn_fraction[syn_fraction["classes"] == 5]
# Order by confidence
dopaminergic_neurons = dopaminergic_neurons.sort_values("confidence", ascending=False)
# %%
N = 10
top_neurons = dopaminergic_neurons.head(N)

# %% [markdown]
# We need to make sure that all of the top N neurons have been evaluated.
# To do this, we will check if the neuron ID is in the results zarr store.

# %%
no_eval = []
for neuron_id in top_neurons.pre.values:
    try:
        # Make sure that it is not empty
        print(list(results[neuron_id].keys()))
    except KeyError:
        no_eval.append(neuron_id)

# %% [markdown]
# # Visualize the results for one neuron
#
# Given one neuron, we can now get all of the synapses in that neuron, and order them by their QuAC scores.
#
# We will filter out only the synapses where the source is 5 and the target is 1.
#
# This corresponds to the following conditions:
# - the original synapse was indeed classified as dopaminergic
# - the counterfactual was successfully classified as cholinergic
#
# We will then keep the index, but order the rows by the QuAC score.
# %%
attribution_method = "DIntegratedGradients"
neuron_id = top_neurons.pre.values[0]


# %%
def get_scores(neuron_id, attribution_method, source=5, target=1):
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
    score = pd.DataFrame(score)
    # Choose only samples where source=5 and target=1
    score = score[(score.source == source) & (score.target == target)]
    # Order by scores, descending
    score = score.sort_values("score", ascending=False)
    return score


# %% [markdown]
# ## Make the visualizations
#
# Now for each of these synapses, we want to plot the original image and the counterfactual (formerly, hybrid) side by side, with the optimal mask overlaid.
# We will save these visualizations to a PDF file.
#
# While we write, we will also prepare the annotation file.
#
# It needs to contain the following filled columns:
# - neuron_id
# - synapse_id
# - QuAC score
# - source image Dopamine prediction value
# - source image Acetylcholine prediction value
# - hybrid image Dopamine prediction value
# - hybrid image Acetylcholine prediction value
# - neuron confidence score
#
# We will also add the following columns, unfilled, for the annotator to fill:
# - Wider cleft
# - Longer cleft
# - darker pre-synaptic densities
# - Added/larger/darker vesicles
# - Cleared DCV
# - Removed large clear vesicle
# - Added spots
# - Removed spots
# - T-bar modified
# - T-bar added
# - Needs review
#
# We will only look at the top 50 synapses per neuron

# %% Function for one neuron


def visualize_one_neuron(
    neuron_id, save_dir, attribution_method="DIntegratedGradients"
):
    neuron_results = results[neuron_id]
    neuron_evaluation = neuron_results["evaluations"][attribution_method]
    predictions = neuron_results["predictions"]
    images = neuron_results["images"]
    hybrids = neuron_evaluation["inverse_hybrids"]
    hybrid_classification = neuron_evaluation["inverse_hybrid_classification"]
    optimal_masks = neuron_evaluation["optimal_masks"]

    pdf_path = save_dir / f"results_{neuron_id}.pdf"

    score = get_scores(neuron_id, attribution_method)

    with PdfPages(pdf_path) as pdf:
        for _, row in tqdm(
            score.iterrows(), total=len(score), desc=f"Neuron {neuron_id}"
        ):
            index_to_show = int(row["syn_index"])
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            plot_with_text(
                image=images[index_to_show],
                prediction=predictions[index_to_show],
                im_ax=axes[0],
                source=5,
                target=1,
                mask=optimal_masks[index_to_show],
            )
            plot_with_text(
                image=hybrids[index_to_show],
                prediction=hybrid_classification[index_to_show],
                im_ax=axes[1],
                source=5,
                target=1,
                mask=optimal_masks[index_to_show],
            )
            fig.suptitle(f"Synapse {index_to_show}, QuAC score: {row['score']}")
            plt.tight_layout()

            # Add the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)  # Close the figure to save memory


# %% [markdown]
# ## Visualize all neurons
#
# Finally, we run this for all of the top N neurons.
# %%
from pathlib import Path

save_dir = Path("visualizations")
save_dir.mkdir(exist_ok=True)

for neuron_id in top_neurons.pre.values:
    visualize_one_neuron(neuron_id, save_dir)


# %% [markdown]
# # Interactive visualization
#
# For a limited number of these samples, I want to be able to load the images in our web-app and click through the changes we see.
# To this, we will create a set of images to load in the web app.
# We will only do this for the first M samples per neuron.
#
# We will also need to create a JSON file for state that contains the following information:
# - neuron_id
# - synapse_id
# - QuAC score
# - source image Dopamine prediction value
# - source image Acetylcholine prediction value
# - hybrid image Dopamine prediction value
# - hybrid image Acetylcholine prediction value
# - neuron confidence score
# %%
def visualize_one_neuron_interactive(
    neuron_id,
    save_dir,
    state_file,
    n_samples=10,
    attribution_method="DIntegratedGradients",
):
    """
    Parameters
    ----------

    neuron_id: int
        The neuron ID to visualize
    save_dir: Path
        The directory to save the images
    state_file: Path
        The JSON file to save the state
    n_samples: int
        The number of samples to visualize
    attribution_method: str
        The attribution method to use
    """
    with state_file.open("r") as f:
        state = json.load(f)

    neuron_results = results[neuron_id]
    neuron_evaluation = neuron_results["evaluations"][attribution_method]
    predictions = neuron_results["predictions"]
    images = neuron_results["images"]
    hybrids = neuron_evaluation["inverse_hybrids"]
    hybrid_classification = neuron_evaluation["inverse_hybrid_classification"]
    optimal_masks = neuron_evaluation["optimal_masks"]

    score = get_scores(neuron_id, attribution_method)[:n_samples]

    for _, row in tqdm(score.iterrows(), total=len(score), desc=f"Neuron {neuron_id}"):
        index_to_show = int(row["syn_index"])
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        plot_with_text(
            image=images[index_to_show],
            prediction=predictions[index_to_show],
            im_ax=axes[0],
            source=5,
            target=1,
            mask=optimal_masks[index_to_show],
        )
        plot_with_text(
            image=hybrids[index_to_show],
            prediction=hybrid_classification[index_to_show],
            im_ax=axes[1],
            source=5,
            target=1,
            mask=optimal_masks[index_to_show],
        )
        fig.suptitle(f"Synapse {index_to_show}, QuAC score: {row['score']}")
        plt.tight_layout()

        # Add the figure to the PDF
        fig.savefig(save_dir / f"{neuron_id}_{index_to_show}.png")
        plt.close(fig)  # Close the figure to save memory

        # Add metadata to the state
        state[f"{neuron_id}_{index_to_show}"] = {
            "neuron_id": neuron_id,
            "synapse_id": index_to_show,
            "QuAC score": row["score"],
            "source_dopamine": predictions[index_to_show][5],
            "source_acetylcholine": predictions[index_to_show][1],
            "hybrid_dopamine": hybrid_classification[index_to_show][5],
            "hybrid_acetylcholine": hybrid_classification[index_to_show][1],
            "neuron_confidence": top_neurons[
                top_neurons.pre == neuron_id
            ].confidence.values[0],
        }
        # Make json serializable
        for key, value in state[f"{neuron_id}_{index_to_show}"].items():
            if isinstance(value, np.int64):
                state[f"{neuron_id}_{index_to_show}"][key] = int(value)
            if isinstance(value, np.float32):
                state[f"{neuron_id}_{index_to_show}"][key] = float(value)

    with state_file.open("w") as f:
        json.dump(state, f)


# %%
M = 10
web_app_assets = Path("webapp/assets/visualizations")
web_app_assets.mkdir(exist_ok=True, parents=True)

state_file = Path("webapp/state.json")

for neuron_id in top_neurons.pre.values:
    if not state_file.exists():
        with state_file.open("w") as f:
            json.dump({}, f)
    visualize_one_neuron_interactive(neuron_id, web_app_assets, state_file, n_samples=M)

# %%
