from quac.evaluation import Processor, Evaluator
from yaml import safe_load
import torch
from quac.training.classification import ClassifierWrapper
from torchvision import transforms
from pathlib import Path
import sys

# TODO YAML file
metadata = {
    "validation_config": {
        "classifier_checkpoint": "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt",
        "mean": 0.5,
        "std": 0.5,
    },
    "validation_data": {
        "source": "/nrs/funke/adjavond/data/synapses/test",
    },
    "solver": {
        "root_dir": "/nrs/funke/adjavond/projects/quac/synapses_v1",
        "counterfactual_dir": "/nrs/funke/adjavond/data/synapses/counterfactuals/stargan_invariance_v0/test",
    },
}


def run_worker(method):
    # print("Loading metadata")
    # with open("configs/stargan.yml", "r") as f:
    #     metadata = safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO, make kind available
    # kind = "latent"
    # Load the classifier
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

    #
    # method = "VanillaIntegratedGradients"
    source_dir = metadata["validation_data"]["source"]
    # counterfactual_dir = (
    #     # Path(metadata["solver"]["root_dir"]) / f"counterfactuals/{kind}/"
    #     Path(metadata["solver"]["root_dir"])
    #     / f"counterfactuals/"
    # )
    counterfactual_dir = Path(metadata["solver"]["counterfactual_dir"])
    attribution_dir = (
        # Path(metadata["solver"]["root_dir"]) / f"attributions/{kind}/{method}"
        Path(metadata["solver"]["root_dir"])
        / f"attributions/{method}"
    )
    # This transform will be applied to all images before running attributions
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    # Setting up the evaluation
    evaluator = Evaluator(
        classifier,
        source_directory=source_dir,
        counterfactual_directory=counterfactual_dir,
        attribution_directory=attribution_dir,
        transform=transform,
    )
    print("Quantifying attributions")
    report = evaluator.quantify(processor=Processor())
    # The report will be stored based on the processor's name, which is "default" by default
    print("Storing report.")
    # report_dir = Path(metadata["solver"]["root_dir"]) / f"reports/{kind}/{method}"
    report_dir = Path(metadata["solver"]["root_dir"]) / f"reports/{method}"
    report.store(report_dir)


if __name__ == "__main__":
    method = sys.argv[1]
    run_worker(method)
