# %% Setup
from quac.attribution import DDeepLift, DIntegratedGradients, AttributionIO
from quac.training.classification import ClassifierWrapper
from pathlib import Path
import torch
from torchvision import transforms
from yaml import safe_load

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

# %% Setting up the attribution
attribution_directory = Path(metadata["solver"]["root_dir"]) / f"attributions/{kind}"
# Create the Attribution objects
attributor = AttributionIO(
    attributions={
        "deeplift": DDeepLift(classifier),
        "ig": DIntegratedGradients(classifier),
    },
    output_directory=attribution_directory,
)


# %% Load the data
source_dir = metadata["validation_data"]["source"]
counterfactual_dir = Path(metadata["solver"]["root_dir"]) / f"counterfactuals/{kind}/"
# This transform will be applied to all images before running attributions
transform = transforms.Compose([transforms.ToTensor()])


# %% This will run attributions and store all of the results in the output_directory
# Shows a progress bar
attributor.run(
    source_directory=source_dir,
    counterfactual_directory=counterfactual_dir,
    transform=transform,
)

# %%
