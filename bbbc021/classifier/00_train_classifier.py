# Train a classifier on BBBC021
# In this notebook I train a ResNet classifier on the BBC021 dataset, to predict mechanism of action.
import torch.nn as nn
from torch.utils.data import DataLoader
from quac.training.data_loader import LabelledDataset
from quac.data import create_transform
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.optim import RAdam
import torch
import timm
import accelerate
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from accelerate.logging import get_logger
from accelerate.utils import set_seed, tqdm

from classification_utils import ConfusionMatrix, plot_confusion_matrix


class GaussianNoise:
    """
    Add Gaussian noise to the input image.
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return img + torch.randn_like(img) * self.std + self.mean


def create_augmentation(transform):
    """
    Create the augmentation pipeline for the training dataset.
    The augmentation includes random horizontal and vertical flips,
    Gaussian blur, random rotation, and Gaussian noise.

    Parameters
    ----------
    transform : Callable
        The base transform to apply to the images.
    """
    # Augmentation for the training dataset
    augmentation = transforms.Compose(
        [
            transform,  # Apply the base transform first
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomRotation(30),
            GaussianNoise(mean=0.0, std=0.01),  # Add Gaussian noise
        ]
    )
    return augmentation


def main(
    data_dir: str = "/nrs/funke/adjavond/data/bbbc021_processed/train",
    val_data_dir: str = "/nrs/funke/adjavond/data/bbbc021_processed/val",
    project_dir: str = "/nrs/funke/adjavond/projects/quac/bbbc021/classifier",
    model: str = "resnet34",
    img_size: int = 80,
    grayscale: bool = False,
    rgb: bool = True,
    scale: int = 2,
    shift: int = -1,
    num_classes: int = 13,
    num_workers: int = 64,
    pin_memory: bool = True,
    batch_size: int = 48,
    lr: float = 1e-3,
    eta_min: float = 1e-6,
    total_epochs: int = 10,
):
    accelerator = accelerate.Accelerator(
        log_with="wandb", step_scheduler_with_optimizer=False
    )

    # Setup the logger for distributed setup
    logger = get_logger(__name__, log_level="INFO")
    logger.info("Using accelerate to train the model, logger to wandb")

    logger.info(f"Using {accelerator.num_processes} GPUs, updating learning rate.")
    # Update the learning rates based on number of GPUs
    lr *= accelerator.num_processes
    eta_min *= accelerator.num_processes

    logger.info(f"Effective batch size: {batch_size * accelerator.num_processes}")
    logger.info(f"Effective learning rate: {lr}")
    logger.info(f"Effective eta_min: {eta_min}")

    transform = create_transform(
        img_size=img_size, grayscale=grayscale, rgb=rgb, scale=scale, shift=shift
    )
    augmentation = create_augmentation(transform)

    logger.info(f"Loading dataset from {data_dir}")
    dataset = LabelledDataset(data_dir, transform=augmentation)
    logger.info(f"Loading validation dataset from {val_data_dir}")
    val_dataset = LabelledDataset(val_data_dir, transform=transform)

    logger.info("Creating model")
    classes = dataset.classes
    assert (
        len(classes) == num_classes
    ), f"Number of classes in dataset {len(classes)} != {num_classes}"
    model = timm.create_model(model, pretrained=True, num_classes=num_classes)
    logger.info("Creating optimizer and loss function")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    # Create a sampler to balance the training dataset
    # Compute the class weights based on frequency in "targets"
    # TODO this is a repeat of something already in QuAC.
    class_counts = np.bincount(dataset.targets)
    assert np.all(class_counts > 0), f"Some of the classes are empty. {class_counts}"
    class_weights = 1.0 / class_counts
    weights = class_weights[dataset.targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(dataset), replacement=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Accuracy metrics using huggingface evaluate
    clf_metrics = evaluate.load("accuracy")
    confusion_matrix = ConfusionMatrix(num_classes)

    # Use `accelerate` to use the GPU if available.
    # Accelerate will also take care of logger to wandb.

    model, optimizer, scheduler, dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader, val_dataloader
    )
    accelerator.init_trackers(
        "bbbc021_classifier",
        config={
            "batch_size": batch_size,
            "learning_rate": lr,
            "total_epochs": total_epochs,
            "img_size": img_size,
            "num_classes": num_classes,
            "eta_min": eta_min,
        },
    )

    # Create the run directory
    if accelerator.is_main_process:
        # Only the main process will create the directory
        logger.info("Creating run directory")
        # Create the directory
        run_dir = Path(project_dir) / accelerator.get_tracker("wandb").run.name
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")

    logger.info("Starting training")
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(dataloader, total=len(dataloader)):

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
        # Step the scheduler once per epoch
        scheduler.step()
        # Write out the loss... maybe not necessary
        running_loss /= len(dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

        # Validation
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_dataloader, total=len(val_dataloader)):
                # Get predictions
                predictions = model(x).argmax(dim=1)
                all_predictions, all_labels = accelerator.gather_for_metrics(
                    (predictions, y)
                )
                clf_metrics.add_batch(
                    predictions=all_predictions,
                    references=all_labels,
                )
                confusion_matrix.update(
                    all_predictions.cpu().numpy(), all_labels.cpu().numpy()
                )
        if accelerator.is_main_process:
            # Do all the logger and plotting on the main process
            metrics = clf_metrics.compute()
            cm = confusion_matrix.compute()
            # Reset the metrics, so that it doesn't accumulate
            clf_metrics = evaluate.load("accuracy")
            confusion_matrix.reset()
            # plot for the confusion matrix
            cm_plot = plot_confusion_matrix(cm, classes)
            # Log the loss, learning rate, and metrics
            accelerator.log(
                {
                    "loss": running_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    **metrics,
                    # Confusion matrix
                    "confusion_matrix": wandb.Image(cm_plot),
                },
                step=epoch,
            )
            # Save the model
            model_path = run_dir / f"model_epoch_{epoch}.pth"
            torch.save(accelerator.unwrap_model(model).state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            # Close the figure
            plt.close(cm_plot)

    accelerator.end_training()
    logger.info("Training finished")


if __name__ == "__main__":
    # Reproducibility
    set_seed(42)
    # configuration information
    data_dir = "/nrs/funke/adjavond/data/bbbc021_processed/train"
    val_data_dir = "/nrs/funke/adjavond/data/bbbc021_processed/val"
    project_dir = "/nrs/funke/adjavond/projects/quac/bbbc021/classifier"

    config = {
        "model": "resnet34",
        "batch_size": 48,
        "total_epochs": 50,
        "img_size": 80,
        "grayscale": False,
        "rgb": True,
        "scale": 2,
        "shift": -1,
        "num_classes": 13,
        "num_workers": 12,
        "pin_memory": True,
        "lr": 1e-3,
        "eta_min": 1e-6,
    }

    # Call the main function
    main(
        data_dir=data_dir,
        val_data_dir=val_data_dir,
        **config,
    )
