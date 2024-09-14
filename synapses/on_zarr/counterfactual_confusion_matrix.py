# %%
import zarr
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%
results = zarr.open(
    "/nrs/funke/adjavond/projects/quac/synapses_onzarr/20240808_results_test.zarr", "r"
)
labels = results["labels"][:]

classes = [
    "0_gaba",
    "1_acetylcholine",
    "2_glutamate",
    "3_serotonin",
    "4_octopamine",
    "5_dopamine",
]

# Create an overall confusion matrix
overall_cm = np.zeros((6, 6))

for t, name in enumerate(classes):
    cf_vals = results[f"counterfactual_predictions/{name}"][:]
    cf_preds = cf_vals.argmax(axis=1)
    pred_vals = results["predictions"][:]
    target = t * np.ones_like(labels)
    cm = confusion_matrix(labels, cf_preds, normalize="true")
    acc = balanced_accuracy_score(target, cf_preds)
    acc_wo = balanced_accuracy_score(
        target[labels != target], cf_preds[labels != target]
    )
    ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Source")
    plt.title(f"{name}: {acc_wo:.2f} / {acc:.2f}")
    plt.show()

    indices = np.where((labels != t))[0]
    cf_images = results[f"counterfactuals/{name}"][indices[:4]]
    images = results["images"][indices[:4]]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f"{name}: Real/Generated")
    for i, ax in enumerate(axes[0]):
        ax.imshow(images[i, 0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        ax.set_title(
            f"{pred_vals[indices[i], labels[indices[i]]]:.2f} | {pred_vals[indices[i], t]:.2f}"
        )
    for i, ax in enumerate(axes[1]):
        ax.imshow(cf_images[i, 0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        ax.set_title(
            f"{cf_vals[indices[i], labels[indices[i]]]:.2f} | {cf_vals[indices[i], t]:.2f}"
        )
    plt.show()

    overall_cm += confusion_matrix(target[labels != target], cf_preds[labels != t])

# %%
# Normalize the confusion matrix
overall_cm = overall_cm / overall_cm.sum(axis=1)[:, np.newaxis]
# Plot the overall confusion matrix
ax = sns.heatmap(overall_cm, annot=True, xticklabels=classes, yticklabels=classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

# %% Source confusion matrix
predictions = results["predictions"][:]
source_cm = confusion_matrix(labels, predictions.argmax(axis=1), normalize="true")

# Plot the source confusion matrix
ax = sns.heatmap(source_cm, annot=True, xticklabels=classes, yticklabels=classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# %% Store the overall confusion matrix in the format used in the paper
# This is a CSV file with three columns: label,prediction,value
# The value is the percentage of the confusion matrix
with open("paper_results/cf_confusion_matrix.csv", "w") as f:
    f.write("label,prediction,value\n")
    for pred in range(6):
        for lbl in range(6):
            f.write(f"{lbl},{pred},{overall_cm[lbl, pred]}\n")

# %%
