from config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from funlib.learn.torch.models import Vgg2D
import matplotlib.pyplot as plt
from noise_augment import AddGaussianNoise
import numpy as np
from pathlib import Path
from plot_confusion import plot_confusion_matrix
from resnet import ResNet2D
import tifffile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
import torchmetrics
from tqdm import tqdm
import typer
import wandb
import yaml
# TODO Make a Training Class


def imread(path):
    data = tifffile.imread(path)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    return torch.from_numpy(data / 255.0).float()


def initialize_model(config: ModelConfig):
    """
    type: str = "vgg", img_size: int = 128, num_classes: int = 2, device: str = "cuda"
    """
    if config.type == "vgg":
        model = Vgg2D(
            (config.image_size, config.image_size),
            output_classes=config.num_classes,
            input_fmaps=config.input_fmaps,
            fmaps=config.fmaps,
        )
    elif config.type == "resnet18":
        model = ResNet2D(
            config.num_classes,
            input_channels=config.input_fmaps,
            start_channels=config.fmaps,
            layers=[2, 2, 2, 2],
        )
    elif config.type == "resnet34":
        model = ResNet2D(
            config.num_classes,
            input_channels=config.input_fmaps,
            start_channels=config.fmaps,
            layers=[3, 4, 6, 3],
        )
    else:
        raise ValueError(f"Unknown model type: {type}")
    return model


def initialize_dataloader(config: DataConfig):
    """
    data_path: str, batch_size: int = 32, augment: bool = True
    """
    if config.augment:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # Translations
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.RandomRotation(90),
                # Noise
                AddGaussianNoise(mean=0.0, std=0.1, clip=True),
            ]
        )
    else:
        transform =  None
    dataset = ImageFolder(root=config.data_path, transform=transform, loader=imread)
    sampler = None
    if config.balance:
        # Balance samples by inverse class frequency
        _, count = np.unique(dataset.targets, return_counts=True)
        sample_counts = np.array([count[i] for i in dataset.targets])
        weights = 1 / sample_counts
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(dataset), replacement=True
        )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    return dataloader


def save_checkpoint(checkpoint_dir, i, model, optimizer, avg_loss, acc, val_acc=None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint = {
        "epoch": i,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "accuracy": acc,
    }
    if val_acc is not None:
        checkpoint["val_accuracy"] = val_acc
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_{i}.pt")


def train_classifier(
    config: ExperimentConfig,
):
    """
    data_path: str = None,
    type: str = "vgg",
    image_size: int = 128,
    num_classes: int = 2,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
    val_data_path: str = None,
    checkpoint_dir: str = "checkpoints",
    """

    log_config = dict()
    log_config.update(config.model.dict())
    log_config.update(config.training.dict())

    run = wandb.init(
        project=config.project, notes=config.notes, tags=config.tags, config=log_config
    )

    checkpoint_dir = Path(config.training.checkpoint_dir)

    dataloader = initialize_dataloader(config.data)
    validation = False
    if config.val_data is not None:
        validation = True
        val_dataloader = initialize_dataloader(config.val_data)

    model = initialize_model(config.model).to(config.training.device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=config.model.num_classes
    ).to(config.training.device)
    if validation:
        val_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.model.num_classes
        ).to(config.training.device)
        val_confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=config.model.num_classes, normalize="true"
        ).to(config.training.device)
        val_accuracy_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.model.num_classes, average="macro"
        ).to(config.training.device)
        val_accuracy_weighted = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.model.num_classes, average="weighted"
        ).to(config.training.device)

    for i in range(config.training.epochs):
        avg_loss = 0
        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch {i+1}/{config.training.epochs}",
        ):
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs.to(config.training.device))
            # compute loss and update model
            loss = loss_fn(outputs, targets.to(config.training.device))
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            metric.update(outputs, targets.to(config.training.device))
        acc = metric.compute()
        avg_loss /= len(dataloader)
        # Log metrics
        # print(f"Epoch {i+1}/{config.training.epochs} - avg loss: {avg_loss}")
        # print(f"Epoch {i+1}/{config.training.epochs} - accuracy: {acc}")
        metric.reset()

        # Check number of channels
        sample = inputs[: config.log_images].cpu()
        if sample.shape[1] != 1:
            sample = sample[:, 0:1, ...]    
        sample = make_grid(sample) 

        # Log to wandb
        metrics = {
            "avg_loss": avg_loss,
            "accuracy": acc,
            "sample": wandb.Image(sample),
        }

        # Validation
        if validation:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    val_dataloader,
                    total=len(val_dataloader),
                    desc=f"Epoch {i+1}/{config.training.epochs} - validation",
                ):
                    inputs, targets = batch
                    outputs = model(inputs.to(config.training.device))
                    val_metric.update(outputs, targets.to(config.training.device))
                    val_confusion.update(outputs, targets.to(config.training.device))
                    val_accuracy_macro.update(
                        outputs, targets.to(config.training.device)
                    )
                    val_accuracy_weighted.update(
                        outputs, targets.to(config.training.device)
                    )
            model.train()
            val_acc = val_metric.compute()
            val_acc_macro = val_accuracy_macro.compute()
            val_acc_weighted = val_accuracy_weighted.compute()
            val_metric.reset()
            val_accuracy_macro.reset()
            val_accuracy_weighted.reset()
            # print(f"Epoch {i+1}/{config.training.epochs} - val accuracy: {val_acc}")
            metrics["val_accuracy"] = val_acc
            metrics["val_accuracy_macro"] = val_acc_macro
            metrics["val_accuracy_weighted"] = val_acc_weighted
            # Plotting the confusion matrix
            fig, _ = plot_confusion_matrix(
                val_confusion.compute().cpu().numpy(),
                val_dataloader.dataset.classes,
                title=f"Validation Confusion Matrix {i+1}/{config.training.epochs}",
            )
            metrics["validation_confusion"] = wandb.Image(fig)
            val_confusion.reset()
            plt.close(fig)

        else:
            val_acc = None
        # Save checkpoint
        save_checkpoint(checkpoint_dir, i, model, optimizer, avg_loss, acc, val_acc)

        # Log to wandb
        run.log(metrics)
    run.finish()


def main(config: str = "vgg.yml"):
    with open(config, "r") as fd:
        config = yaml.safe_load(fd)
    experiment = ExperimentConfig(**config)
    train_classifier(experiment)


if __name__ == "__main__":
    typer.run(main)
