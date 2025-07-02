import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
import typer
from pathlib import Path
from quac.generate import load_data, load_classifier, load_stargan, get_counterfactual


def unnormalize(x):
    return (x + 1) / 2


def main(
    root_directory: str = "/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/DR_grading_processed",
    split: str = "test",
    source: int = 0,
    target: int = 1,
    output_directory: str = "/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/counterfactuals",
    classifier_checkpoint: str = "/nrs/funke/adjavond/projects/quac/retina-ddr-classification/final_model.pt",
    # TODO merge latent_model_root_dir with run_name, somehow
    latent_model_root_dir: str = "/nrs/funke/adjavond/projects/quac/",
    run_name: str = "retina_stargan",
    source_name: str = None,
    target_name: str = None,
    max_tries: int = 100,
    device: str = None,
    # STARGAN parameters
    img_size: int = 224,
    input_dim: int = 3,
    style_dim: int = 64,
    latent_dim: int = 16,
    num_domains: int = 5,
    checkpoint_iter: int = 90000,
    kind: str = "latent",
    single_output_encoder: bool = False,
    # ImageNet normalization
):

    class_names = {
        0: "0_No_DR",
        1: "1_Mild",
        2: "2_Moderate",
        3: "3_Severe",
        4: "4_Proliferative_DR",
    }
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    data_directory = Path(root_directory) / f"{split}/{class_names[source]}"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grayscale = input_dim == 1
    dataset = load_data(data_directory, img_size, grayscale=grayscale)

    if kind == "reference":
        reference_data_directory = (
            Path(root_directory) / f"{split}/{class_names[target]}"
        )
        reference_dataset = load_data(
            reference_data_directory, img_size, grayscale=grayscale
        )
    else:
        reference_dataset = None

    # Use ImageNet normalization
    classifier = load_classifier(
        classifier_checkpoint, mean=mean, std=std, eval=True, device=device
    )
    latent_model_checkpoint_dir = (
        Path(latent_model_root_dir) / f"{run_name}/checkpoints"
    )
    inference_model = load_stargan(
        latent_model_checkpoint_dir,
        img_size=img_size,
        input_dim=input_dim,
        style_dim=style_dim,
        latent_dim=latent_dim,
        num_domains=num_domains,
        checkpoint_iter=checkpoint_iter,
        kind=kind,
        single_output_encoder=single_output_encoder,
    )

    if target_name is None:
        target_name = class_names[target]
    if source_name is None:
        source_name = class_names[source]

    # Add source and target to the output directory
    output_directory = (
        Path(output_directory)
        / f"{run_name}/{kind}/{split}/{source_name}/{target_name}"
    )
    print("Output directory:", output_directory)
    # Make directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    for x, name in tqdm(dataset):
        xcf = get_counterfactual(
            classifier,
            inference_model,
            x,
            target,
            kind,
            reference_dataset,
            device=device,
            max_tries=max_tries,
        )
        # TODO save in grayscale and with correct normalization?
        if grayscale:
            xcf = unnormalize(xcf).squeeze()
        else:
            xcf = np.transpose(unnormalize(xcf).squeeze(), (1, 2, 0))
        plt.imsave(
            os.path.join(output_directory, f"{name}.png"),
            xcf,
            cmap="gray",
            vmin=0,
            vmax=1,
        )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    typer.run(main)
