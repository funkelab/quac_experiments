from quac.evaluation import Processor, Evaluator
from yaml import safe_load
import torch
from quac.training.classification import ClassifierWrapper
from torchvision import transforms
from pathlib import Path
import sys


def run_worker(
    method,
    subdir="Day2/val",
    config_path="configs/stargan.yml",
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
        classifier_checkpoint,
        do_nothing=True,
    ).to(device)
    classifier.eval()

    source_dir = metadata["test_data"]["source"]
    counterfactual_dir = (
        Path(metadata["solver"]["root_dir"]) / f"counterfactuals/{kind}/{subdir}"
    )
    print(f"Counterfactuals: {counterfactual_dir}")
    attribution_dir = (
        Path(metadata["solver"]["root_dir"]) / f"attributions/{kind}/{subdir}/{method}"
    )
    print(f"Attributions: {attribution_dir}")
    # This transform will be applied to all images before running attributions
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

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
    report_dir = (
        Path(metadata["solver"]["root_dir"]) / f"reports/{kind}/{subdir}/{method}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report.store(report_dir)


if __name__ == "__main__":
    method = sys.argv[1]
    print(f"Running evaluation for {method}")
    run_worker(
        method, subdir="Day2/val", config_path="configs/stargan.yml", kind="latent"
    )
