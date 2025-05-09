"""
Run StarGAN-based conversion on the validation set post-hoc, because the classifier needs weird cropping etc.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from quac.generate import load_stargan
import torch
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(checkpoint):
    """Load the classifier."""
    model = torch.jit.load(checkpoint)
    model = model.cuda()
    model = model.eval()
    return model


@torch.no_grad()
def main(
    data_directory: Path,
    output_directory: Path,
    classifier_checkpoint: Path,
    latent_model_checkpoint_dir: Path,
    source_class: int,
    target_class: int,
    img_size: int = 224,
    input_dim: int = 3,
    style_dim: int = 64,
    latent_dim: int = 10,
    num_domains: int = 3,
    checkpoint_iter: int = 100000,
    batch_size: int = 16,
    num_workers: int = 12,
    mu: float = 0.5,
    sigma: float = 0.5,
):
    """Run the generation."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print("Output directory:", output_directory)

    # transformations for testing
    trans_test = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mu, std=sigma),
        ]
    )
    dataset = ImageFolderWithPaths(data_directory, trans_test)
    # Subset only
    is_source = np.where(np.array(dataset.targets) == source_class)[0]
    dataset = torch.utils.data.Subset(dataset, is_source)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    classifier = load_classifier(classifier_checkpoint)

    latent_model_checkpoint_dir = Path(latent_model_checkpoint_dir)

    inference_model = load_stargan(
        latent_model_checkpoint_dir,
        img_size=img_size,
        input_dim=input_dim,
        style_dim=style_dim,
        latent_dim=latent_dim,
        num_domains=num_domains,
        checkpoint_iter=checkpoint_iter,
        kind="latent",
    )

    inference_model = inference_model.to(device)
    inference_model = inference_model.eval()

    correct = 0
    correct_real = 0
    predicted = []
    for x, y, paths in tqdm(dataloader):
        x = x.to(device)
        target = torch.tensor([target_class] * len(x)).to(device)
        xcf = inference_model(x, target)

        output = classifier(xcf)
        output_real = classifier(x)
        # Save images here
        for path, xcf_i in zip(paths, xcf):
            name = Path(path).name
            image = (xcf_i + 1) / 2  # unnormalize
            image = transforms.ToPILImage()(image)
            image.save(output_directory / name)

        # Accuracy stuff
        # Only take the first three classes
        pred_real = torch.argmax(output_real, dim=1)
        pred = torch.argmax(output, dim=1)
        correct += (pred == target_class).sum().item()
        correct_real += (pred_real == source_class).sum().item()
        predicted.extend(pred.cpu().numpy())

    print(f"Accuracy: {correct / len(dataset)}")
    # Print counts of each predicted class
    print(f"Accuracy real: {correct_real / len(dataset)}")


if __name__ == "__main__":
    class_names = ["01", "02", "03"]
    source_subdir = "Day2/val"

    for source_class, source_class_name in enumerate(class_names):
        for target_class, target_class_name in enumerate(class_names):
            if source_class == target_class:
                continue
            print(
                "Running for source", source_class_name, "and target", target_class_name
            )
            main(
                data_directory=f"/nrs/funke/adjavond/data/flyid/week1_limited/{source_subdir}",
                output_directory=f"/nrs/funke/adjavond/projects/quac/flyid/stargan_limited_2/counterfactuals/latent/{source_subdir}/{source_class_name}/{target_class_name}/",
                classifier_checkpoint="/nrs/funke/adjavond/projects/quac/flyid/final_classifier_jit_retry.pth",
                latent_model_checkpoint_dir="/nrs/funke/adjavond/projects/quac/flyid/stargan_limited_2/checkpoints",
                source_class=source_class,
                target_class=target_class,
                checkpoint_iter=15000,
            )
