# %% Setup
import yaml

metadata = yaml.safe_load(open("configs/stargan.yml"))
kind = "latent"
metadata["report_directory"] = metadata["solver"]["root_dir"] + f"/reports/{kind}"

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
from tqdm import tqdm
import pandas as pd

for method, report in tqdm(reports.items(), total=len(reports)):
    median, p25, p75 = report.get_curve()
    data = {
        "median": median,
        "p25": p25,
        "p75": p75,
    }
    df = pd.DataFrame(data)
    df.to_csv(f"results/{method}_curve.csv", index=False)


# %%
# TODO fill in here with the best of the methods
report = reports["ig"]
# %% [markdown]
# ## Choosing the best example
# Next we want to choose the best example, given the best method.
# This is done by ordering the examples by the QuAC score, and then choosing the one with the highest score.
#
# %%
import numpy as np

order = np.argsort(report.quac_scores)[::-1]
# %%
idx = 200
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
from quac.evaluation import Processor

thresh = report.get_optimal_threshold(order[idx])
processor = Processor()

mask, _ = processor.create_mask(attribution, thresh)
rgb_mask = mask.transpose(1, 2, 0)
hybrid = np.array(cf_image) / 255 * rgb_mask + np.array(image) / 255 * (1.0 - rgb_mask)
# %% [markdown]
# The final missing point is to get the classification of the hybrid image.
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
from scipy.special import softmax

classifier_output = classifier(
    torch.tensor(hybrid).permute(2, 0, 1).float().unsqueeze(0).to(device)
)
hybrid_prediction = softmax(classifier_output[0].detach().cpu().numpy())

# %% [markdown]
# ## Visualizing the results
# %%
fig, axes = plt.subplots(2, 4)
axes[1, 0].imshow(image)
axes[0, 0].bar(np.arange(len(prediction)), prediction)
axes[1, 1].imshow(cf_image)
axes[0, 1].bar(np.arange(len(target_prediction)), target_prediction)
axes[0, 2].bar(np.arange(len(hybrid_prediction)), hybrid_prediction)
axes[1, 2].imshow(hybrid)
axes[1, 3].imshow(rgb_mask)
axes[0, 3].axis("off")
fig.suptitle(f"QuAC Score: {report.quac_scores[order[idx]]}")
plt.show()

# %% [markdown]
# Let's put all of that together into a function that we can call to get all the parts used in the visualization.
# And another function to plot the results.
# Finally, a function to save the results.


# %%
def get_summary(report, idx, classifier, processor):
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
        rgb_mask,
        hybrid,
        hybrid_prediction,
    )


def plot_summary(
    image, cf_image, prediction, target_prediction, rgb_mask, hybrid, hybrid_prediction
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
    fig.suptitle(f"QuAC Score: {report.quac_scores[order[idx]]}")
    plt.show()


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
    Image.fromarray((rgb_mask * 255).astype(np.uint8)).save(path / "mask.png")
    Image.fromarray((hybrid * 255).astype(np.uint8)).save(path / "hybrid.png")

    pd.DataFrame(
        {
            "image.png": prediction,
            "cf_image": target_prediction,
            "hybrid.png": hybrid_prediction,
        }
    ).to_csv(path / "predictions.csv")


# %%
import pandas as pd

df = pd.DataFrame(
    {
        "QuAC Score": report.quac_scores,
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
# %%
sns.heatmap(performance, annot=True)
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
    # print(f"Source: {row['Source']}, Target: {row['Target']}")
    (
        image,
        cf_image,
        prediction,
        target_prediction,
        rgb_mask,
        hybrid,
        hybrid_prediction,
    ) = get_summary(report, idx, classifier, processor)
    # plot_summary(
    #     image,
    #     cf_image,
    #     prediction,
    #     target_prediction,
    #     rgb_mask,
    #     hybrid,
    #     hybrid_prediction,
    # )
    path = Path("best_examples") / f"{int(row['Source'])}_{int(row['Target'])}"
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
