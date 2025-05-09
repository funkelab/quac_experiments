from pathlib import Path
from quac.generate import load_data, load_stargan, get_counterfactual
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from yaml import safe_load


def load_classifier(checkpoint):
    """Load the classifier."""
    model = torch.jit.load(checkpoint)
    model = model.cuda()
    model = model.eval()
    return model


@torch.no_grad()
def main(
    config_path: str = "configs/stargan.yml",
    checkpoint_iter: int = 100000,
    kind: str = "latent",
    subdir: str = "Day2/val",
):
    # Load metadata
    print("Loading metadata")
    with open(config_path, "r") as f:
        metadata = safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the classifier
    print("Loading classifier")
    classifier = load_classifier(metadata["validation_config"]["classifier_checkpoint"])

    # Load the model
    print("Loading model")
    latent_model_checkpoint_dir = Path(metadata["solver"]["root_dir"]) / "checkpoints"
    inference_model = load_stargan(
        latent_model_checkpoint_dir,
        checkpoint_iter=checkpoint_iter,
        kind=kind,
        **metadata["model"],
    )

    # Load the data
    soure_dir = metadata["test_data"]["source"]
    available_sources = [d.name for d in Path(soure_dir).iterdir() if d.is_dir()]
    img_size = metadata["test_data"]["img_size"]

    # Generate counterfactuals
    for source, source_name in enumerate(available_sources):
        for target, target_name in enumerate(available_sources):
            if source == target:
                continue
            print("Running for source", source_name, "and target", target_name)

            data_directory = Path(metadata["test_data"]["source"]) / source_name
            dataset = load_data(data_directory, img_size, grayscale=False)

            output_directory = (
                Path(metadata["solver"]["root_dir"])
                / f"generated_images/{kind}/{source_name}/{target_name}/{subdir}"
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            print("Generated images will be saved in:", str(output_directory))

            for x, name in tqdm(dataset):
                # Repeat the image 3 times to get the 3 channels
                x = x.repeat(1, 3, 1, 1).to(device).squeeze()
                xcf = get_counterfactual(
                    classifier,
                    inference_model,
                    x,
                    target=target,
                    kind=kind,
                    device=device,
                    max_tries=1,
                    batch_size=10,
                )
                xcf = xcf * 0.5 + 0.5
                assert (
                    xcf.min() >= 0 and xcf.max() <= 1
                ), f"Generated images are not in [0, 1]: {xcf.min()}, {xcf.max()}"
                # For example, you can save the images here
                save_image(xcf, output_directory / name)


if __name__ == "__main__":
    main(
        checkpoint_iter=35000,
        kind="latent",
    )
