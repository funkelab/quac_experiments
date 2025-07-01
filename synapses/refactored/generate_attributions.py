# %% Setup
from quac.attribution import (
    DDeepLift,
    DIntegratedGradients,
    AttributionIO,
    VanillaDeepLift,
    VanillaIntegratedGradients,
)
from quac.training.classification import ClassifierWrapper
from pathlib import Path
import torch
from torchvision import transforms
from yaml import safe_load

if __name__ == "__main__":
    #  Load metadata
    # TODO metadata
    # print("Loading metadata")
    # with open("configs/stargan.yml", "r") as f:
    #     metadata = safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO which is it???
    # kind = "latent"

    # Load the classifier
    print("Loading classifier")
    # classifier_checkpoint = Path(metadata["validation_config"]["classifier_checkpoint"])
    # mean = metadata["validation_config"]["mean"]
    # std = metadata["validation_config"]["std"]
    classifier_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
    )
    mean = 0.5
    std = 0.5
    assume_normalized = True
    classifier = ClassifierWrapper(
        classifier_checkpoint, mean=mean, std=std, assume_normalized=assume_normalized
    )
    classifier.to(device)
    classifier.eval()

    #  Setting up the attribution
    # attribution_directory = Path(metadata["solver"]["root_dir"]) / f"attributions/{kind}"
    attribution_directory = Path(
        "/nrs/funke/adjavond/projects/quac/synapses_v1/attributions"
    )
    # Create the Attribution objects
    attributor = AttributionIO(
        attributions={
            # "DDeepLift": DDeepLift(classifier),
            # "DIntegratedGradients": DIntegratedGradients(classifier),
            "VanillaDeepLift": VanillaDeepLift(classifier),
            "VanillaIntegratedGradients": VanillaIntegratedGradients(classifier),
        },
        output_directory=attribution_directory,
    )

    #  Load the data
    # TODO metadata yaml
    # source_dir = metadata["validation_data"]["source"]
    # counterfactual_dir = Path(metadata["solver"]["root_dir"]) / f"counterfactuals/{kind}/"
    source_dir = Path("/nrs/funke/adjavond/data/synapses/test")
    counterfactual_dir = Path(
        "/nrs/funke/adjavond/data/synapses/counterfactuals/stargan_invariance_v0/test"
    )
    # This transform will be applied to all images before running attributions
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    #  This will run attributions and store all of the results in the output_directory
    # Shows a progress bar
    attributor.run(
        source_directory=source_dir,
        counterfactual_directory=counterfactual_dir,
        transform=transform,
    )

#
