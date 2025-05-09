# %% Setup
import yaml
import matplotlib.pyplot as plt
import numpy as np
from quac.report import Report
from sklearn.metrics import confusion_matrix

# %%
metadata = yaml.safe_load(open("configs/stargan.yml"))
subdir = "Day2/val"
kind = "latent"

metadata["report_directory"] = (
    metadata["solver"]["root_dir"] + f"/reports/{kind}/{subdir}"
)
# %% Final version of the report
# Load the results
report = Report.from_directory(metadata["report_directory"], name="final_report")

# %% Get counterfactual confusion matrix from the report!
predictions = [
    np.argmax(explanation.counterfactual_prediction) for explanation in report
]
targets = [explanation.target_class for explanation in report]
cm = confusion_matrix(targets, predictions)
# Save the confusion matrix
np.savetxt("results/cf_confusion_matrix.csv", cm, delimiter=",")

# %% Select a subset of the report
filtered_report = report.from_source(2).to_target(0).score_threshold(0.95)

print(len(filtered_report))
# %% Choose a sample, they are ordered by QuAC score.
output = filtered_report[10]


# Plotting
def gray_to_rgb(img):
    """Convert a grayscale image to RGB by repeating the channels."""
    return img.repeat(3, 1, 1)


def plot_example(output, show_mask=True):
    fig, ((ax1, bax1), (ax2, bax2)) = plt.subplots(
        2, 2, gridspec_kw={"width_ratios": [1, 0.2], "wspace": -0.4}
    )
    ax1.imshow(gray_to_rgb(output.query).permute(1, 2, 0))
    ax1.axis("off")
    ax1.set_title(f"Query: {output.source_class}")
    bax1.bar(
        np.arange(len(output.query_prediction)), output.query_prediction, color="gray"
    )

    ax2.imshow(output.counterfactual.permute(1, 2, 0))
    ax2.set_title(f"Counterfactual: {output.target_class}")
    ax2.axis("off")
    bax2.bar(
        np.arange(len(output.counterfactual_prediction)),
        output.counterfactual_prediction,
        color="gray",
    )

    if show_mask:
        ax1.imshow(output.mask.sum(0), alpha=0.3, cmap="coolwarm")
        ax2.imshow(output.mask.sum(0), alpha=0.3, cmap="coolwarm")

    fig.tight_layout()
    fig.suptitle(f"Score: {output.score:.2f}")
    return


# %%
plot_example(filtered_report[50], show_mask=True)

# %%
