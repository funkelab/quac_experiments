# %% Setup
import yaml
import pandas as pd
import numpy as np
from skimage import measure
from skimage.morphology import binary_dilation

# %%
metadata = yaml.safe_load(open("configs/stargan.yml"))
kind = "latent"
metadata["report_directory"] = metadata["solver"]["root_dir"] + f"/reports/{kind}"
# %% [markdown]
# We will want to get the classification of the hybrid image.
# To do this, we need to load the classifier.
# %%
import torch
from quac.training.classification import ClassifierWrapper
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading classifier")
classifier_checkpoint = Path(metadata["validation_config"]["classifier_checkpoint"])
mean = metadata["validation_config"]["mean"]
std = metadata["validation_config"]["std"]
classifier = ClassifierWrapper(
    classifier_checkpoint,
    mean=mean,
    std=std,
    assume_normalized=True,  # We are going to be putting data between 0 and 1 with a transform
).to(device)
classifier.eval()

# %%
from quac.report import Report

reports = {
    method: Report(name=method)
    for method in [
        "DDeepLift",
        "DIntegratedGradients",
        "VanillaDeepLift",
        "VanillaIntegratedGradients",
    ]
}

for method, report in reports.items():
    report.load(metadata["report_directory"] + "/" + method + "/default.json")

# %% Plot the curves
import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# for method, report in reports.items():
#     report.plot_curve(ax=ax)
# # Add the legend
# plt.legend()
# plt.show()

# %% Save the curve data

# for method, report in tqdm(reports.items(), total=len(reports)):
#     median, p25, p75 = report.get_curve()
#     data = {
#         "median": median,
#         "p25": p25,
#         "p75": p75,
#     }
#     df = pd.DataFrame(data)
#     df.to_csv(f"results/{method}_curve.csv", index=False)

# %% [markdown]
# ## Choosing the best attribution method for each sample
# While one attribution method may be better than another on average, it is possible that the best method for a given example is different.
# Therefore, we will make a list of the best method for each example by comparing the quac scores.
# %%
quac_scores = pd.DataFrame(
    {method: report.quac_scores for method, report in reports.items()}
)
# %%
best_methods = quac_scores.idxmax(axis=1)
best_quac_scores = quac_scores.max(axis=1)
# %%
# TODO fill in here with the best of the methods
# report = reports["ig"]
# %% [markdown]
# ## Choosing the best example
# Next we want to choose the best example, given the best method.
# This is done by ordering the examples by the QuAC score, and then choosing the one with the highest score.
#
# %%
order = best_quac_scores[::-1].argsort()
# %%
idx = 10
# Get the corresponding report
report = reports[best_methods[order[idx]]]
# %% [markdown]
# We will then load that example and its counterfactual from its path, and visualize it.
# We also want to see the classification of both the original and the counterfactual.
# %%
# Transform to apply to the images so they match each other
# loading
from PIL import Image

image_path, cf_path = report.paths[order[idx]], report.target_paths[order[idx]]
image, cf_image = Image.open(image_path), Image.open(cf_path)

prediction = report.predictions[order[idx]]
target_prediction = report.target_predictions[order[idx]]
# %% [markdown]
# ## Loading the attribution
# We next want to load the attribution for the example, and visualize it.
# %%
attribution_path = report.attribution_paths[order[idx]]
attribution = np.load(attribution_path)

# %% [markdown]
# ## Getting the processor
# We want to see the specific mask that was optimal in this case.
# To do this, we will need to get the optimal threshold, and get the processor used for masking.

# %% Getting the mask and hybrid
from quac.evaluation import Processor, UnblurredProcessor

gaussian_kernel_size = 11
struc = 10
thresh = report.optimal_thresholds()[order[idx]]
print(thresh)
processor = Processor(gaussian_kernel_size=gaussian_kernel_size, struc=struc)
processor2 = UnblurredProcessor(struc=struc)

mask, _ = processor.create_mask(attribution, thresh)
simple_mask, _ = processor2.create_mask(attribution, thresh)
rgb_mask = mask.transpose(1, 2, 0)
# zero-out the green and blue channels
rgb_mask[:, :, 1] = 0
rgb_mask[:, :, 2] = 0
simple_rgb_mask = simple_mask.transpose(1, 2, 0).astype(np.float32)
hybrid = np.array(cf_image) / 255 * rgb_mask + np.array(image) / 255 * (1.0 - rgb_mask)


# %%
from scipy.special import softmax

classifier_output = classifier(
    torch.tensor(hybrid).permute(2, 0, 1).float().unsqueeze(0).to(device)
)
hybrid_prediction = softmax(classifier_output[0].detach().cpu().numpy())

# %% [markdown]
# ## Visualizing the results
# %%
fig, axes = plt.subplots(2, 5)
axes[1, 0].imshow(image)
axes[0, 0].bar(np.arange(len(prediction)), prediction)
axes[1, 1].imshow(cf_image)
axes[0, 1].bar(np.arange(len(target_prediction)), target_prediction)
axes[0, 2].bar(np.arange(len(hybrid_prediction)), hybrid_prediction)
axes[1, 2].imshow(hybrid)
axes[1, 3].imshow(rgb_mask)
axes[0, 3].axis("off")
axes[0, 4].axis("off")
axes[1, 4].imshow(simple_rgb_mask)
fig.suptitle(f"QuAC Score: {report.quac_scores[order[idx]]}")
plt.show()

# %% [markdown]
# Let's put all of that together into a function that we can call to get all the parts used in the visualization.
# And another function to plot the results.
# Finally, a function to save the results.


# %%
def get_summary(reports, best_methods, idx, classifier, processor):
    report = reports[best_methods[idx]]
    image_path, cf_path = report.paths[idx], report.target_paths[idx]
    image, cf_image = Image.open(image_path), Image.open(cf_path)

    prediction = report.predictions[idx]
    target_prediction = report.target_predictions[idx]
    # Getting the attribution
    attribution_path = report.attribution_paths[idx]
    attribution = np.load(attribution_path)

    # Getting the mask and hybrid
    thresh = report.get_optimal_threshold(idx)

    mask, _ = processor.create_mask(attribution, thresh)
    simple_mask, _ = processor2.create_mask(attribution, thresh)
    simple_rgb_mask = simple_mask.transpose(1, 2, 0).astype(np.float32)
    rgb_mask = mask.transpose(1, 2, 0)
    hybrid = np.array(cf_image) / 255 * rgb_mask + np.array(image) / 255 * (
        1.0 - rgb_mask
    )

    classifier_output = classifier(
        torch.tensor(hybrid).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )
    hybrid_prediction = softmax(classifier_output[0].detach().cpu().numpy())

    return (
        image,
        cf_image,
        prediction,
        target_prediction,
        simple_rgb_mask,
        hybrid,
        hybrid_prediction,
    )


def plot_summary(
    score,
    image,
    cf_image,
    prediction,
    target_prediction,
    rgb_mask,
    hybrid,
    hybrid_prediction,
):
    fig, axes = plt.subplots(2, 4)
    axes[1, 0].imshow(image)
    axes[0, 0].bar(np.arange(len(prediction)), prediction)
    axes[1, 1].imshow(cf_image)
    axes[0, 1].bar(np.arange(len(target_prediction)), target_prediction)
    axes[0, 2].bar(np.arange(len(hybrid_prediction)), hybrid_prediction)
    axes[1, 2].imshow(hybrid)
    axes[1, 3].imshow(rgb_mask)
    axes[0, 3].axis("off")
    fig.suptitle(f"QuAC Score: {score}")
    plt.show()


def save_mask_contour(mask, filepath, threshold=0.0, linewidth=3):
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


def store_summary(
    image,
    cf_image,
    prediction,
    target_prediction,
    rgb_mask,
    hybrid,
    hybrid_prediction,
    path,
):
    """
    Store the summary results in a folder.
    We store the images as png, and the predictions as a csv.

    image: PIL.Image
    cf_image: PIL.Image
    prediction: np.array
    target_prediction: np.array
    rgb_mask: np.array
    hybrid: np.array
    hybrid_prediction: np.array
    path: Path
    """
    path.mkdir(parents=True, exist_ok=True)
    image.save(path / "image.png")
    cf_image.save(path / "cf_image.png")
    # Mask as three separate images
    for name, channel in zip(["red", "green", "blue"], rgb_mask.transpose(2, 0, 1)):
        Image.fromarray((channel * 255).astype(np.uint8)).save(
            path / f"mask_{name}.png"
        )
        # Get a contour as well
        save_mask_contour(channel[None, ...], path / f"contour_{name}.png")

    Image.fromarray((rgb_mask * 255).astype(np.uint8)).save(path / "mask.png")
    Image.fromarray((hybrid * 255).astype(np.uint8)).save(path / "hybrid.png")

    # The format needs to be filename,A,B,C,D,style where A,B,C,D are the probabilities of each class and style is "image" for the image and "counterfactual" for the hybrid
    # We do not save the "cf_image"
    output = pd.DataFrame(
        {
            "filename": ["image.png", "hybrid.png"],
            "A": [prediction[0], hybrid_prediction[0]],
            "B": [prediction[1], hybrid_prediction[1]],
            "C": [prediction[2], hybrid_prediction[2]],
            "D": [prediction[3], hybrid_prediction[3]],
            "style": ["image", "counterfactual"],
        }
    )
    output.to_csv(path / "predictions.csv", index=False)


# %%
# Get the best examples, given the quac score from the best reports!
# We can choose any report to give us the source and target labels and predictions, those are the same for all reports.
report = reports["DIntegratedGradients"]

df = pd.DataFrame(
    {
        "QuAC Score": best_quac_scores.values,
        "Best Method": best_methods.values,
        "Source": report.labels,
        "Target": report.target_labels,
        "Source Pred": np.array(report.predictions).argmax(axis=1),
        "Target Pred": np.array(report.target_predictions).argmax(axis=1),
    }
)
# %%
df = df[(df["Source Pred"] == df["Source"]) & (df["Target Pred"] == df["Target"])]
# %%
performance = df.groupby(["Source", "Target"])["QuAC Score"].mean().unstack()
# %%
import seaborn as sns

sns.heatmap(performance.reindex([0, 2, 1, 3]), annot=True)
# # %%
# sns.heatmap(performance, annot=True)
# %%

# %% [markdown]
# Now we save everything!
# %% First, the performance matrix
from itertools import product

dec_to_bin = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

# For each pair of classes
with open("performance.csv", "w") as fd:
    fd.write("source_x,source_y,target_x,target_y,quac_average\n")
    for source, target in product(performance.index, performance.columns):
        if source != target:
            source_bin, target_bin = dec_to_bin[source], dec_to_bin[target]
            csv_line = "{},{},{},{},{:.4f}\n".format(
                *source_bin, *target_bin, performance[source][target]
            )
            fd.write(csv_line)

# %% [markdown]
# The next thing we want to do is get the best example for each pair of classes.
# %%
best_examples = df.groupby(["Source", "Target"])["QuAC Score"].idxmax()
# %%
best_examples_df = df.loc[best_examples]
# %%
best_examples_df

# %% [markdown]
# Finally, let's plot the best examples.
# %%
for idx, row in best_examples_df.iterrows():
    print(f"Source: {row['Source']}, Target: {row['Target']}")
    (
        image,
        cf_image,
        prediction,
        target_prediction,
        rgb_mask,
        hybrid,
        hybrid_prediction,
    ) = get_summary(reports, best_methods, idx, classifier, processor)
    # plot_summary(
    #     row["QuAC Score"],
    #     image,
    #     cf_image,
    #     prediction,
    #     target_prediction,
    #     rgb_mask,
    #     hybrid,
    #     hybrid_prediction,
    # )
    path = Path("results/best_examples") / f"{int(row['Source'])}_{int(row['Target'])}"
    store_summary(
        image,
        cf_image,
        prediction,
        target_prediction,
        rgb_mask,
        hybrid,
        hybrid_prediction,
        path,
    )
# %%
