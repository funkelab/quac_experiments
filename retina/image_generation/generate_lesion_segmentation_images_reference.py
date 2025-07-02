import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import transforms
from tqdm import tqdm
import typer
from pathlib import Path
from quac.generate.data import LabelFreePngFolder
from quac.generate import load_classifier, load_stargan, get_counterfactual


def unnormalize(x):
    return (x + 1) / 2


def main(
    root_directory: str = "/nrs/funke/adjavond/data/retina/DDR-dataset/lesion_segmentation/train/image/",
    reference_directory: str = "/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/DR_grading_processed/val/0_No_DR",
    target: int = 0,
    output_directory: str = "/nrs/funke/adjavond/data/retina/DDR-dataset/lesion_segmentation/train/counterfactual",
    classifier_checkpoint: str = "/nrs/funke/adjavond/projects/quac/retina-ddr-classification/final_model.pt",
    # TODO merge latent_model_root_dir with run_name, somehow
    latent_model_root_dir: str = "/nrs/funke/adjavond/projects/quac/",
    run_name: str = "retina_stargan",
    max_tries: int = 100,
    device: str = None,
    # STARGAN parameters
    img_size: int = 224,
    input_dim: int = 3,
    style_dim: int = 64,
    latent_dim: int = 16,
    num_domains: int = 5,
    checkpoint_iter: int = 90000,
    kind: str = "reference",
    single_output_encoder: bool = False,
    # ImageNet normalization
):
    """
    For this particular generation, we do not have the source annotated, and we also only ever use target=0.
    """

    class_names = {
        0: "0_No_DR",
        1: "1_Mild",
        2: "2_Moderate",
        3: "3_Severe",
        4: "4_Proliferative_DR",
    }
    mean_classifier = (0.485, 0.456, 0.406)
    std_classifier = (0.229, 0.224, 0.225)

    mean_generator = 0.5
    std_generator = 0.5

    data_directory = Path(root_directory)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LabelFreePngFolder(
        root=data_directory,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_generator, std=std_generator),
            ]
        ),
    )
    # TODO get a location for the reference dataset
    reference_dataset = LabelFreePngFolder(
        root=Path(reference_directory),
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_generator, std=std_generator),
            ]
        ),
    )

    # Use ImageNet normalization in the classifier
    classifier = load_classifier(
        classifier_checkpoint,
        mean=mean_classifier,
        std=std_classifier,
        eval=True,
        device=device,
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

    # Add source and target to the output directory
    output_directory = Path(output_directory) / f"{kind}/{class_names[target]}"

    print("Output directory:", output_directory)
    # Make directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    for x, name in tqdm(dataset):
        xcf, source_path = get_counterfactual(
            classifier,
            inference_model,
            x,
            target,
            kind,
            dataset_ref=reference_dataset,
            device=device,
            max_tries=max_tries,
            return_path=True,
        )
        xcf = np.transpose(unnormalize(xcf).squeeze(), (1, 2, 0))
        plt.imsave(
            os.path.join(output_directory, name),
            xcf,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        # Save the source image name as well
        with open(os.path.join(output_directory, "source.txt"), "a") as f:
            f.write(f"{name} {source_path}\n")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    typer.run(main)
