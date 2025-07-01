import argparse
from torch.utils.data import DataLoader
from quac.training.data_loader import LabelledDataset
from quac.data import create_transform
from classification_utils import ConfusionMatrix, plot_confusion_matrix
import torch
import timm
import accelerate
from pathlib import Path
from accelerate.logging import get_logger
from accelerate.utils import tqdm
import pandas as pd


def main(
    run_name: str,
    checkpoint: int = 50,
    test_data_dir: str = "/nrs/funke/adjavond/data/bbbc021_processed/test",
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
    copy_cm: bool = False,
):
    accelerator = accelerate.Accelerator()
    logger = get_logger(__name__, log_level="INFO")

    logger.info("Loading test data")
    transform = create_transform(
        img_size=img_size, grayscale=grayscale, rgb=rgb, scale=scale, shift=shift
    )

    dataset = LabelledDataset(test_data_dir, transform=transform)

    logger.info("Creating model")
    classes = dataset.classes
    assert (
        len(classes) == num_classes
    ), f"Number of classes in dataset {len(classes)} != {num_classes}"
    model = timm.create_model(model, pretrained=True, num_classes=num_classes)

    logger.info("Loading model checkpoint")
    checkpoint_path = Path(project_dir) / f"{run_name}/model_epoch_{checkpoint}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} does not exist. Please check the path."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    cm = ConfusionMatrix(num_classes)

    model.eval()
    logger.info("Running inference")
    all_predictions = []
    all_labels = []
    names = dataset.samples
    with torch.inference_mode():
        for images, labels in tqdm(
            dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process
        ):
            predictions = model(images).argmax(dim=1)
            gathered_predictions, gathered_labels = accelerator.gather_for_metrics(
                (predictions, labels)
            )
            cm.update(gathered_predictions.cpu().numpy(), gathered_labels.cpu().numpy())
            all_predictions.extend(gathered_predictions.cpu().numpy())
            all_labels.extend(gathered_labels.cpu().numpy())

    accelerator.wait_for_everyone()

    assert len(all_predictions) == len(
        all_labels
    ), f"Number of predictions {len(all_predictions)} != number of labels {len(all_labels)}"
    assert len(all_predictions) == len(
        names
    ), f"Number of predictions {len(all_predictions)} != number of filenames {len(names)}"

    # Make a dataframe of the predictions and labels and accompanying filenames
    df = pd.DataFrame(
        {
            "filename": names,
            "predictions": all_predictions,
            "labels": all_labels,
        }
    )
    # Save the dataframe to a CSV file
    df_path = Path(project_dir) / f"{run_name}/predictions_epoch_{checkpoint}.csv"
    df.to_csv(df_path, index=False)
    if copy_cm:
        # Copy the predictions to the current directory, for easy access
        df_path = Path.cwd() / f"predictions_epoch_{checkpoint}.csv"
        df.to_csv(df_path, index=False)

    logger.info("Computing confusion matrix")
    confusion_matrix = cm.compute()
    cm_plot = plot_confusion_matrix(confusion_matrix, classes)
    # Save confusion matrix plot
    cm_plot_path = (
        Path(project_dir) / f"{run_name}/confusion_matrix_epoch_{checkpoint}.png"
    )
    cm_plot.savefig(cm_plot_path)
    logger.info(f"Confusion matrix saved to {cm_plot_path}")

    if copy_cm:
        # Copy the confusion matrix to the current directory, for easy access
        cm_plot_path = Path.cwd() / f"confusion_matrix_epoch_{checkpoint}.png"
        cm_plot.savefig(cm_plot_path)
        logger.info(f"Confusion matrix copied to {cm_plot_path}")

    # Close the accelerator
    accelerator.end_training()


def parser():
    parser = argparse.ArgumentParser(description="Test classifier")
    parser.add_argument(
        "-r",
        "--run_name",
        type=str,
        help="Name of the run to test",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        default=50,
        help="Checkpoint to test",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="/nrs/funke/adjavond/data/bbbc021_processed/test",
        help="Path to the test data directory",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="/nrs/funke/adjavond/projects/quac/bbbc021/classifier",
        help="Path to the project directory",
    )
    parser.add_argument(
        "--copy_cm",
        action="store_true",
        help="Copy the confusion matrix to the current directory",
    )
    return parser


if __name__ == "__main__":
    # configuration information
    parser = parser()
    args = parser.parse_args()

    main(
        run_name=args.run_name,
        checkpoint=args.checkpoint,
        test_data_dir=args.test_data_dir,
        project_dir=args.project_dir,
        copy_cm=args.copy_cm,
    )
