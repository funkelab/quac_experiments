# %% Setup
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import measure
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt

# %%
metadata = yaml.safe_load(open("configs/stargan.yml"))
subdir = "Day2/val"
kind = "latent"
metadata["report_directory"] = (
    metadata["solver"]["root_dir"] + f"/reports/{kind}/{subdir}"
)
# %% [markdown]
# We will want to get the classification of the hybrid image.
# To do this, we need to load the classifier.
# %%
import torch
from quac.training.classification import ClassifierWrapper
from pathlib import Path
from quac.evaluation import FinalReport
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading classifier")
classifier_checkpoint = Path(metadata["validation_config"]["classifier_checkpoint"])
classifier = ClassifierWrapper(classifier_checkpoint, do_nothing=True).to(device)
classifier.eval()


size = metadata["data"]["img_size"]
transform = transforms.Resize((size, size))

report = FinalReport.from_directory(
    metadata["report_directory"], transform=transform, classifier=classifier
)
output = report[0]

# %%
plt.imshow(output["counterfactual"].permute(1, 2, 0))
plt.imshow(output["mask"].transpose(1, 2, 0), alpha=0.5)
print(output["query_prediction"])
print(output["counterfactual_prediction"])
print(output["source_class"], output["target_class"])

# %% Plot the curves

fig, ax = plt.subplots()
for method, rep in report._reports.items():
    rep.plot_curve(ax=ax)
# Add the legend
plt.legend()
plt.show()
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
order = best_quac_scores.argsort()[::-1].values
# Just order them as they are:

order = np.arange(len(best_quac_scores))
# %%
idx = 0
# Get the corresponding report
report = reports[best_methods[order[idx]]]
# report = reports["DDeepLift"]
# st_quac_scores = report.quac_scores
# %% [markdown]
# We will then load that example and its counterfactual from its path, and visualize it.
# We also want to see the classification of both the original and the counterfactual.
# %%
# Transform to apply to the images so they match each other
# loading
from PIL import Image

image_path, cf_path = report.paths[order[idx]], report.target_paths[order[idx]]
image, cf_image = Image.open(image_path), Image.open(cf_path)
# Center crop the image to 224x224
image = image.resize((224, 224))
# Make it RGB
image = image.convert("RGB")


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
processor = Processor(
    gaussian_kernel_size=gaussian_kernel_size,
    struc=struc,
    channel_wise=False,
    name="aname",
)
processor2 = UnblurredProcessor(struc=struc, channel_wise=False, name="aname")

mask, _ = processor.create_mask(attribution, thresh)
simple_mask, _ = processor2.create_mask(attribution, thresh)
rgb_mask = mask.transpose(1, 2, 0)
# zero-out the green and blue channels
# rgb_mask[:, :, 1] = 0
# rgb_mask[:, :, 2] = 0
simple_rgb_mask = simple_mask.transpose(1, 2, 0).astype(np.float32)
cf_part = np.array(cf_image) / 255 * simple_rgb_mask
image_part = np.array(image) / 255 * (1.0 - simple_rgb_mask)
hybrid = cf_part + image_part


# %%
from scipy.special import softmax

classifier_output = classifier(
    torch.tensor(hybrid).permute(2, 0, 1).float().unsqueeze(0).to(device)
)
hybrid_prediction = softmax(classifier_output[0].detach().cpu().numpy())

# %% [markdown]
# ## Visualizing the results
# %%
fig, axes = plt.subplots(2, 5, figsize=(20, 10), gridspec_kw={"height_ratios": [1, 2]})
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
# %%
plt.imshow(image)
plt.imshow(simple_rgb_mask, alpha=0.1)

# %% [markdown]
# Let's put all of that together into a function that we can call to get all the parts used in the visualization.
# And another function to plot the results.
# Finally, a function to save the results.


# %%
def get_summary(reports, best_methods, idx, classifier, processor):
    report = reports[best_methods[idx]]
    image_path, cf_path = report.paths[idx], report.target_paths[idx]
    image, cf_image = Image.open(image_path), Image.open(cf_path)
    # Center crop the image to 224x224
    image = image.resize((224, 224))
    # Make it RGB
    image = image.convert("RGB")

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

    cf_part = np.array(cf_image) / 255 * rgb_mask
    image_part = np.array(image) / 255 * (1.0 - rgb_mask)
    hybrid = cf_part + image_part

    classifier_output = classifier(
        (torch.tensor(hybrid).permute(2, 0, 1).float().unsqueeze(0).to(device) - 0.5)
        / 0.5
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
    fig, axes = plt.subplots(
        2, 4, figsize=(20, 10), gridspec_kw={"height_ratios": [1, 2]}
    )
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Image")
    axes[0, 0].bar(np.arange(len(prediction)), prediction)
    axes[1, 1].imshow(cf_image)
    axes[1, 1].set_title("Counterfactual")
    axes[0, 1].bar(np.arange(len(target_prediction)), target_prediction)
    axes[0, 2].bar(np.arange(len(hybrid_prediction)), hybrid_prediction)
    axes[1, 2].imshow(hybrid)
    axes[1, 2].set_title("Hybrid")
    axes[1, 3].imshow(image)
    axes[1, 3].imshow(rgb_mask, alpha=0.5)
    axes[1, 3].set_title("Mask")
    axes[0, 3].axis("off")
    # remove the axes
    for ax in axes[1]:
        ax.axis("off")
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


def change_binary_colors(
    binary_mask, foreground_color="#f12345", background_color="#f54321"
):
    """Take a binary mask and make it into an RGB image with the given colors."""
    foreground = np.array(Image.new("RGB", binary_mask.shape[::-1], foreground_color))
    background = np.array(Image.new("RGB", binary_mask.shape[::-1], background_color))
    return Image.fromarray(np.where(binary_mask[..., None], foreground, background))


def store_summary(
    image,
    cf_image,
    prediction,
    target_prediction,
    rgb_mask,
    hybrid,
    hybrid_prediction,
    path,
    rgb=False,
    mask_colors=None,
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
    if rgb:
        for name, channel in zip(["red", "green", "blue"], rgb_mask.transpose(2, 0, 1)):
            Image.fromarray((channel * 255).astype(np.uint8)).save(
                path / f"mask_{name}.png"
            )
            # Get a contour as well
            save_mask_contour(channel[None, ...], path / f"contour_{name}.png")
    else:
        # Save the mask as a single image
        mask = Image.fromarray((rgb_mask * 255).astype(np.uint8))
        if mask_colors is not None:
            print(mask_colors)
            mask = change_binary_colors(rgb_mask.sum(axis=-1), *mask_colors)
        mask.save(path / "mask.png")
        # Get a contour as well
        # sum the channels to get a single channel mask
        single_channel_mask = rgb_mask.sum(axis=2)
        save_mask_contour(single_channel_mask[None, ...], path / "contour.png")

    # Image.fromarray((rgb_mask * 255).astype(np.uint8)).save(path / "mask.png")
    Image.fromarray((hybrid * 255).astype(np.uint8)).save(path / "hybrid.png")

    # The format needs to be filename,A,B,C,D,style where A,B,C,D are the probabilities of each class and style is "image" for the image and "counterfactual" for the hybrid
    # We do not save the "cf_image"
    output = pd.DataFrame(
        {
            "filename": ["image.png", "hybrid.png"],
            "A": [prediction[0], hybrid_prediction[0]],
            "B": [prediction[1], hybrid_prediction[1]],
            "C": [prediction[2], hybrid_prediction[2]],
            "style": ["image", "counterfactual"],
        }
    )
    output.to_csv(path / "predictions.csv", index=False)


# %%
# Get the best examples, given the quac score from the best reports!
# We can choose any report to give us the source and target labels and predictions, those are the same for all reports.
report = reports["VanillaIntegratedGradients"]

df = pd.DataFrame(
    {
        "QuAC Score": best_quac_scores.values,
        "Best Method": best_methods.values,
        "Source": report.labels,
        "Target": report.target_labels,
        "Source Pred": np.array(report.predictions).argmax(axis=1),
        "Target Pred": np.array(report.target_predictions).argmax(axis=1),
        "Path": report.paths,
        "CF Path": report.target_paths,
        "Attribution Path": report.attribution_paths,
    }
)
# %%
# df.head()
# print(len(df))
# # Drop samples in df that have the same path, cf_path, and attribution path
# df = df.drop_duplicates(subset=["Path", "CF Path", "Attribution Path"])
# print(len(df))
# %%
df = df[(df["Source Pred"] == df["Source"]) & (df["Target Pred"] == df["Target"])]
# %% [markdown]
# The next thing we want to do is get the best example for each pair of classes.
# %%
n = 1  # Number of best examples to get
best_examples_df = df.groupby(["Source", "Target"]).apply(
    lambda x: x.nlargest(n, "QuAC Score")
)
# %%
best_examples_df.reset_index(level=[0, 1], drop=True, inplace=True)
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
    path = Path("results/best_examples") / f"{int(row['Source'])}_{int(row['Target'])}"
    # store_summary(
    #     image,
    #     cf_image,
    #     prediction,
    #     target_prediction,
    #     rgb_mask,
    #     hybrid,
    #     hybrid_prediction,
    #     path,
    # )
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

# %%
df[df.Path.str.endswith("dir[1]_3_13029.png")]

# %%
for idx, row in df[df.Path.str.endswith("dir[1]_3_13029.png")].iterrows():
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
    plot_summary(
        row["QuAC Score"],
        image,
        cf_image,
        prediction,
        target_prediction,
        rgb_mask,
        hybrid,
        hybrid_prediction,
    )

# %%
# Get some random good examples
n = 3  # Number of good examples to get
good_samples_df = df[df["QuAC Score"] > 0.991]
# get a random sample of n for each source, target pair
good_examples_df = good_samples_df.groupby(["Source", "Target"]).apply(
    lambda x: x.sample(n, random_state=42)
)
# %%
good_examples_df.reset_index(level=[0, 1], drop=True, inplace=True)
good_examples_df

# %% [markdown]
# Finally, let's plot the good examples.
# %%
green = "#7ABE6F"
blue = "#5D80B9"
for (source, target), examples in good_examples_df.groupby(["Source", "Target"]):
    print(f"Source: {row['Source']}, Target: {row['Target']}")
    for idx, row in examples.iterrows():
        (
            image,
            cf_image,
            prediction,
            target_prediction,
            rgb_mask,
            hybrid,
            hybrid_prediction,
        ) = get_summary(reports, best_methods, idx, classifier, processor)
        path = (
            Path("results/good_examples")
            / f"{int(row['Source'])}_{int(row['Target'])}_{idx}"
        )
        store_summary(
            image,
            cf_image,
            prediction,
            target_prediction,
            rgb_mask,
            hybrid,
            hybrid_prediction,
            path,
            mask_colors=[blue, green],
        )

# %%
# %%
from skimage.data import binary_blobs

binary_mask = binary_blobs(length=128, seed=42)
# Blue foreground, green background
rgb_mask = change_binary_colors(
    binary_mask, foreground_color=blue, background_color=green
)
img = Image.fromarray(rgb_mask)
# %%
img

# %%
change_binary_colors(mask.sum(axis=0), foreground_color=blue, background_color=green)
# %%
best_method = df.loc[3361]["Best Method"]

image = Image.open(df.loc[3361]["Path"])
image

# %%
# Optimal threshold
thresh = reports[best_method].get_optimal_threshold(3361)
blah = processor2.create_mask(attribution, thresh)
blah = change_binary_colors(
    blah[0].sum(axis=0), foreground_color=blue, background_color=green
)
blah

# %%
# Get example at 3361
# get the attribution
attribution = np.load(df.loc[1245]["Attribution Path"])

# %%
# Do a bunch of random non-optimal thresholds
thresholds = np.linspace(0.01, 0.4, 10)

# Get the mask for each threshold
masks = [processor2.create_mask(attribution, thresh)[0] for thresh in thresholds]
# Get a binary, colored mask for each threshold
rgb_masks = [
    change_binary_colors(
        mask.sum(axis=0), foreground_color=blue, background_color=green
    )
    for mask in masks
]

# %%
rgb_masks[0]

# %%
rgb_masks[9]

# %%
save_location = "results/masks"
Path(save_location).mkdir(parents=True, exist_ok=True)
for i, mask in enumerate(rgb_masks):
    mask.save(f"{save_location}/mask_{i + 10}.png")

# %%
