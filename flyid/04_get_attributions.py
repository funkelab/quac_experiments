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


def main(
    subdir="Day2/val",
    config_path: str = "configs/stargan.yml",
    kind="latent",
    mean=0.5,
    std=0.5,
):
    print("Loading metadata")
    with open(config_path, "r") as f:
        metadata = safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading classifier")
    classifier_checkpoint = Path(metadata["validation_config"]["classifier_checkpoint"])
    classifier = ClassifierWrapper(
        # TODO use all of the details from the config
        classifier_checkpoint,
        do_nothing=True,
    ).to(device)
    classifier.eval()

    print("Setting up attributions")
    attribution_directory = (
        Path(metadata["solver"]["root_dir"]) / f"attributions/{kind}/{subdir}"
    )
    # Create the Attribution objects
    attributor = AttributionIO(
        attributions={
            "DDeepLift": DDeepLift(classifier),
            "DIntegratedGradients": DIntegratedGradients(classifier),
            "VanillaDeepLift": VanillaDeepLift(classifier),
            "VanillaIntegratedGradients": VanillaIntegratedGradients(classifier),
        },
        output_directory=attribution_directory,
    )

    print("Running attributions")
    source_dir = metadata["test_data"]["source"]
    counterfactual_dir = (
        Path(metadata["solver"]["root_dir"]) / f"counterfactuals/{kind}/{subdir}"
    )
    # This transform will be applied to all images before running attributions
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    # Shows a progress bar
    attributor.run(
        source_directory=source_dir,
        counterfactual_directory=counterfactual_dir,
        transform=transform,
    )


if __name__ == "__main__":
    main(
        subdir="Day2/val",
        config_path="configs/stargan.yml",
        kind="latent",
        mean=0.5,
        std=0.5,
    )
