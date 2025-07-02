# %% [markdown]
# This is a cleaned up version of the previous notebook, where we will generate all of the necessary files for
# Figure 1 of the paper.
#
# We will generate these for the top 5 examples for GABA to ACh
#
# Namely:
# Images
# - The original image
# - The counterfactual image
# - The attribution
# - The mask at 5 different thresholds
# - The image with an overlay of the optimal mask
# - The hybrids at the same 5 thresholds
#
# Classification.csv
# - Index is the Path to whatever is being classified
# - Columns are: GABA, Ach, Glut, Ser, Oct, Dop
# - The classifications are of the original, counterfactual, and the hybrids
# We will also generate a CSV for a fake QuAC plot
# %%
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from quac.report import Report
from quac.evaluation import Evaluator
import cv2
from funlib.learn.torch.models import Vgg2D
import torch
from skimage import measure

# %%
colors = [
    "#7C5099",  # GABA
    "#E7A74B",  # ACh
    "#6FB658",  # Glut
    "#CF885F",  # Ser
    "#7F74AD",  # Oct
    "#B6D7E4",  # Dop
]

class_names = ["GABA", "ACh", "Glut", "Ser", "Oct", "Dop"]

# %% [markdown] QuAC scores for the population
# Plotting the QuAC scores for the different attribution methods.


# %% Necessary setup
load_dir = "/nrs/funke/adjavond/projects/quac/20240201"

reports = {
    method: Report(name=method)
    for method in ["discriminative_ig", "discriminative_dl", "discriminative_ingrad"]
}

for method in reports:
    reports[method].load(Path(load_dir) / f"{method}.json")

# %% Plotting
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for method, report in reports.items():
    report.plot_curve(ax)
plt.legend()
plt.show()

# %% Store the report outputs as CSV
import pandas as pd

columns = []
data = []
for method, report in reports.items():
    mean, std = report.get_curve()
    data.append(mean)
    data.append(std)

    columns.append(f"{method}_mean")
    columns.append(f"{method}_std")

# %%
df = pd.DataFrame(np.array(data).T, columns=columns)
df.to_csv("curves.csv", index=False)
# %% [markdown] Load the report from Integrated Gradients
# Choosing the best report to use for the rest of the experiment

# %%
report = reports["discriminative_ig"]
quac_scores = report.compute_scores()

# %%
# Choose samples from class 0 to class 1
# 0 is GABA, 1 is ACh
index = np.where(
    (np.array(report.labels) == 0) * (np.array(report.target_labels) == 1)
)[0]
index = index[np.argsort(np.array(quac_scores)[index])[-10:]][::-1]

# %%
classifier = Vgg2D(input_size=(128, 128), output_classes=6, fmaps=12)
checkpoint = torch.load(
    "/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint"
)["model_state_dict"]
classifier.eval()
classifier.load_state_dict(checkpoint)

evaluator = Evaluator(classifier)

# %%
save_dir = Path("./output")
save_dir.mkdir(exist_ok=True, parents=True)

with open(save_dir / "classifications.csv", "w") as f:
    f.write("Path,GABA,Ach,Glut,Ser,Oct,Dop\n")

with open(save_dir / "quac_scores.csv", "w") as f:
    f.write("Path, Score\n")


def plot_overlay(image, mask, contour_threshold=0.25):
    if image.dtype == np.uint8:
        vmin, vmax = 0, 255
    else:
        vmin, vmax = 0, 1
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    contours = measure.find_contours(mask, contour_threshold)
    alpha = np.clip(mask + 0.6, 0, 1)
    ax.imshow(image.squeeze(), cmap="gray", alpha=alpha, vmin=vmin, vmax=vmax)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="magenta")
    ax.axis("off")
    fig.tight_layout()
    return fig


def normalize(image):
    # TODO check what range the image is in to begin with
    return 2 * (image / 255) - 1


def do_the_thing(report, index, order, evaluator, save_dir, source, target):
    # 1 Get the image
    print("Processing", index)
    image_path = report.paths[index]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #  filename = Path(image_path).stem
    source_class = class_names[source]
    target_class = class_names[target]
    mask_cmap = LinearSegmentedColormap.from_list(
        "mask_cmap", [colors[source], colors[target]]
    )
    filename = f"{source_class}_{target_class}_{order:02d}_{index:03d}"
    # 2 Get the counterfactual
    counterfactual_path = report.target_paths[index]
    counterfactual = cv2.imread(counterfactual_path, cv2.IMREAD_GRAYSCALE)
    # 3 Get the attribution
    print("Creating attribution")
    attribution = report.load_attribution(index)
    # 4 Create the five masks
    thresholds = report.thresholds[index][110:160:10]
    masks = [
        evaluator.create_mask(attribution, threshold)[0].squeeze()
        for threshold in thresholds
    ]
    mask_size = [mask.sum() / mask.size for mask in masks]
    mask_paths = [str(save_dir / f"{filename}_mask_{ms}.png") for ms in mask_size]
    # 5 Create the five hybrids
    print("Creating hybrids")
    hybrids = [counterfactual * mask + image * (1 - mask) for mask in masks]
    hybrid_paths = [(save_dir / f"{filename}_hybrid_{ms}.png") for ms in mask_size]
    # 6 Get the optimal mask
    print("Getting optimal mask")
    optimal_threshold = report.get_optimal_threshold(index)
    optimal_mask, _ = evaluator.create_mask(attribution, optimal_threshold)
    optimal_hybrid = counterfactual * optimal_mask + image * (1 - optimal_mask)
    # 7 Create the overlay for the real image
    real_overlay = plot_overlay(image.squeeze(), optimal_mask.squeeze())
    # 8 Create the overlay for the counterfactual
    counterfactual_overlay = plot_overlay(
        counterfactual.squeeze(), optimal_mask.squeeze()
    )
    # 9 Save the images
    for mask_path, mask in zip(mask_paths, masks):
        print("Saving", mask_path)
        plt.imsave(mask_path, mask, cmap=mask_cmap, vmin=0, vmax=1)
        plt.imsave(
            mask_path.replace(".png", "_gray.png"), mask, cmap="gray", vmin=0, vmax=1
        )
    for hybrid_path, hybrid in zip(hybrid_paths, hybrids):
        print("Saving", hybrid_path)
        plt.imsave(str(hybrid_path), hybrid, cmap="gray", vmin=0, vmax=255)
    plt.imsave(
        str(save_dir / f"{filename}_mask_optimal.png"),
        optimal_mask.squeeze(),
        vmin=0,
        vmax=1,
        cmap=mask_cmap,
    )
    plt.imsave(
        str(save_dir / f"{filename}_mask_optimal_gray.png"),
        optimal_mask.squeeze(),
        vmin=0,
        vmax=1,
        cmap="gray",
    )
    plt.imsave(
        str(save_dir / f"{filename}_hybrid_optimal.png"),
        optimal_hybrid.squeeze(),
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    real_overlay.savefig(
        save_dir / f"{filename}_real_overlay.png", bbox_inches="tight", pad_inches=0
    )
    counterfactual_overlay.savefig(
        save_dir / f"{filename}_counterfactual_overlay.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.imsave(
        str(save_dir / f"{filename}_attribution.png"),
        attribution.squeeze(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.imsave(
        str(save_dir / f"{filename}_residual.png"),
        (image - counterfactual).squeeze(),
        cmap="coolwarm",
    )  # , vmin=-255, vmax=255)
    plt.imsave(
        str(save_dir / f"{filename}_real.png"), image, cmap="gray", vmin=0, vmax=255
    )
    plt.imsave(
        str(save_dir / f"{filename}_counterfactual.png"),
        counterfactual,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    # 10 Get and save the classifications
    with open(save_dir / "classifications.csv", "a") as f:
        # TODO make this a bit more general so that it does not depend on the number of classes
        p_o = report.predictions[index]
        f.write(
            f"{filename}_real,{p_o[0]},{p_o[1]},{p_o[2]},{p_o[3]},{p_o[4]},{p_o[5]}\n"
        )
        p_c = report.target_predictions[index]
        f.write(
            f"{filename}_counterfactual,{p_c[0]},{p_c[1]},{p_c[2]},{p_c[3]},{p_c[4]},{p_c[5]}\n"
        )
        for hybrid_path, hybrid in zip(hybrid_paths, hybrids):
            p_h = evaluator.run_inference(normalize(hybrid))[0]
            f.write(
                f"{hybrid_path.stem},{p_h[0]},{p_h[1]},{p_h[2]},{p_h[3]},{p_h[4]},{p_h[5]}\n"
            )
        p_h = evaluator.run_inference(normalize(optimal_hybrid))[0]
        f.write(
            f"{filename}_hybrid_optimal,{p_h[0]},{p_h[1]},{p_h[2]},{p_h[3]},{p_h[4]},{p_h[5]}\n"
        )
    with open(save_dir / "quac_scores.csv", "a") as f:
        f.write(f"{filename}_real,{quac_scores[index]}\n")
    # Close the figures
    plt.close(real_overlay)
    plt.close(counterfactual_overlay)


# %%
for i, idx in enumerate(index):
    do_the_thing(report, idx, i, evaluator, save_dir, 0, 1)

# %%
