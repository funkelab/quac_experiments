# %% Defining processors and evaluators
from quac.evaluation import Processor, Evaluator
from sklearn.metrics import ConfusionMatrixDisplay
from yaml import safe_load
import torch
from quac.training.classification import ClassifierWrapper
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# %% Load metadata
# print("Loading metadata")
# with open("configs/stargan.yml", "r") as f:
#     metadata = safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO Which is it??
# kind = "latent"
# %% Load the classifier
classifier_checkpoint = (
    "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
)
mean = 0.5
std = 0.5
classifier = ClassifierWrapper(
    classifier_checkpoint,
    mean=mean,
    std=std,
    assume_normalized=True,  # We are going to be putting data between 0 and 1 with a transform
).to(device)
classifier.eval()

# %%
method = "discriminative_ig"
# source_dir = metadata["validation_data"]["source"]
# counterfactual_dir = Path(metadata["solver"]["root_dir"]) / f"counterfactuals/{kind}/"
# attribution_dir = Path(metadata["solver"]["root_dir"]) / f"attributions/{kind}/{method}"

source_dir = Path("/nrs/funke/adjavond/data/synapses/test")
counterfactual_dir = Path(
    "/nrs/funke/adjavond/data/synapses/counterfactuals/stargan_invariance_v0/test"
)
attribution_dir = Path(
    f"/nrs/funke/adjavond/projects/quac/synapses_v1/attributions/{method}"
)
# This transform will be applied to all images before running attributions
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

# %% Setting up the evaluation
evaluator = Evaluator(
    classifier,
    source_directory=source_dir,
    counterfactual_directory=counterfactual_dir,
    attribution_directory=attribution_dir,
    transform=transform,
)

# %% Check the confusion matrix for the counterfactuals
cf_confusion_matrix = evaluator.classification_report(
    data="counterfactuals",  # this is the default
    return_classification=False,
    print_report=True,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cf_confusion_matrix,
).plot()
plt.show()

# %% Check the confusion matrix for the source
source_confusion_matrix = evaluator.classification_report(
    data="source",
    return_classification=False,
    print_report=True,
)
disp = ConfusionMatrixDisplay(confusion_matrix=source_confusion_matrix).plot()
plt.show()
# %% [markdown]
# Store the confusion matrices for later
# %%
np.savetxt(
    "source_confusion_matrix.csv", source_confusion_matrix, delimiter=",", fmt="%.4f"
)
np.savetxt("cf_confusion_matrix.csv", cf_confusion_matrix, delimiter=",", fmt="%.4f")
# %%
# %% Run QuAC evaluation on your attribution and store a report
report = evaluator.quantify(processor=Processor())
# %%
# The report will be stored based on the processor's name, which is "default" by default
# report_dir = Path(metadata["solver"]["root_dir"]) / f"reports/{kind}/{method}"
report_dir = Path(f"/nrs/funke/adjavond/projects/quac/synapses_v1/reports/{method}")
report.store(report_dir)

# %%
report.plot_curve()
# %% Store it for later
report_dl = report
