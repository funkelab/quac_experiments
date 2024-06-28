# %% Defining processors and evaluators
from quac.evaluation import Processor, Evaluator
from sklearn.metrics import ConfusionMatrixDisplay
from yaml import safe_load
import torch
from quac.training.classification import ClassifierWrapper
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt

# %% Load metadata
print("Loading metadata")
with open("configs/stargan.yml", "r") as f:
    metadata = safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO change this between latent and reference
kind = "latent"
# %% Load the classifier
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
method = "VanillaIntegratedGradients"
source_dir = metadata["validation_data"]["source"]
counterfactual_dir = Path(metadata["solver"]["root_dir"]) / f"counterfactuals/{kind}/"
attribution_dir = Path(metadata["solver"]["root_dir"]) / f"attributions/{kind}/{method}"
# This transform will be applied to all images before running attributions
transform = transforms.Compose([transforms.ToTensor()])

# %% Setting up the evaluation
evaluator = Evaluator(
    classifier,
    source_directory=source_dir,
    counterfactual_directory=counterfactual_dir,
    attribution_directory=attribution_dir,
    transform=transform,
)

# %% Check the confusion matrix for the counterfactuals
# cf_confusion_matrix = evaluator.classification_report(
#     data="counterfactuals",  # this is the default
#     return_classification=False,
#     print_report=True,
# )

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cf_confusion_matrix,
# ).plot()
# plt.show()

# # %% Check the confusion matrix for the source
# source_confusion_matrix = evaluator.classification_report(
#     data="source",
#     return_classification=False,
#     print_report=True,
# )
# disp = ConfusionMatrixDisplay(confusion_matrix=source_confusion_matrix).plot()
# plt.show()
# %% Run QuAC evaluation on your attribution and store a report
report = evaluator.quantify(processor=Processor())
# %%
# The report will be stored based on the processor's name, which is "default" by default
report_dir = Path(metadata["solver"]["root_dir"]) / f"reports/{kind}/{method}"
report.store(report_dir)

# %%
report.plot_curve()
# %% Store it for later
report_dl = report

# %% [markdown]
# Store the confusion matrices for later
# %%
import numpy as np

np.savetxt(
    "source_confusion_matrix.csv", source_confusion_matrix, delimiter=",", fmt="%.4f"
)
np.savetxt("cf_confusion_matrix.csv", cf_confusion_matrix, delimiter=",", fmt="%.4f")
# %%
