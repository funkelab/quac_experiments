from pathlib import Path
from quac.generate import load_data, load_stargan, get_counterfactual
from quac.training.classification import ClassifierWrapper
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from yaml import safe_load


# Load metadata
print("Loading metadata")
with open("configs/stargan.yml", "r") as f:
    metadata = safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kind = "latent"
checkpoint_iter = 100000


# Load the classifier
print("Loading classifier")
classifier_checkpoint = Path(metadata["validation_config"]["classifier_checkpoint"])
mean = metadata["validation_config"]["mean"]
std = metadata["validation_config"]["std"]
assume_normalized = metadata["validation_config"]["assume_normalized"]
classifier = ClassifierWrapper(
    classifier_checkpoint,
    mean=mean,
    std=std,
    assume_normalized=assume_normalized,
)
classifier.to(device)
classifier.eval()

# Load the model
print("Loading model")
latent_model_checkpoint_dir = Path(metadata["solver"]["root_dir"]) / "checkpoints"
inference_model = load_stargan(
    latent_model_checkpoint_dir,
    checkpoint_iter=100000,
    kind=kind,
    **metadata["model"],
)

# Load the data
soure_dir = metadata["validation_data"]["source"]
available_sources = [d.name for d in Path(soure_dir).iterdir() if d.is_dir()]
img_size = metadata["validation_data"]["img_size"]

# Generate counterfactuals
for source, source_name in enumerate(available_sources):
    for target, target_name in enumerate(available_sources):
        if source == target:
            continue
        print("Running for source", source_name, "and target", target_name)

        data_directory = Path(metadata["validation_data"]["source"]) / source_name
        dataset = load_data(data_directory, img_size, grayscale=False)

        output_directory = (
            Path(metadata["solver"]["root_dir"])
            / f"counterfactuals/{kind}/{source_name}/{target_name}"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        print("Counterfactuals will be saved in:", str(output_directory))

        for x, name in tqdm(dataset):
            xcf = get_counterfactual(
                classifier,
                inference_model,
                x,
                target=target,
                kind=kind,
                device=device,
                max_tries=10,
                batch_size=10,
            )
            assert (
                xcf.min() >= 0 and xcf.max() <= 1
            ), f"Counterfactuals are not in [0, 1]: {xcf.min()}, {xcf.max()}"
            # For example, you can save the images here
            save_image(xcf, output_directory / name)
