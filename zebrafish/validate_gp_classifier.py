# Setup
import argparse
from pathlib import Path
from yaml import safe_load
from funlib.learn.torch.models import Vgg2D
import random
import torch
from tqdm import tqdm
from torchmetrics.classification import Accuracy
import wandb
import gunpowder as gp
import numpy as np
from utils import AddLabel


def train(
    run_id,
    seed,
    metadata_file="test_metadata.yaml",
    val_steps=100,
    batch_size=32,
    num_workers=12,
    device=None,
    shape=(1, 512, 512),
):
    # Setup
    metadata = safe_load(open(metadata_file))
    shape = gp.Coordinate(shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = gp.ArrayKey("RAW")
    label = gp.ArrayKey("LABEL")

    # Create the pipeline
    zebrafish_sources = []
    class_names = ["WT", "Mutant"]
    for class_index in [0, 1]:
        for sample, fish in metadata[class_names[class_index]].items():
            base_dir = f"/nrs/funke/adjavond/zebrafish/data/stitched/{sample}/"
            array_roi = gp.Roi(
                gp.Coordinate((fish["zmin"], fish["ymin"], fish["xmin"])),
                gp.Coordinate(
                    (
                        fish["zmax"] - fish["zmin"],
                        fish["ymax"] - fish["ymin"],
                        fish["xmax"] - fish["xmin"],
                    )
                ),
            )
            zarr_source = (
                gp.ZarrSource(
                    base_dir + fish["file"],
                    {raw: "raw/s0"},
                    {
                        raw: gp.ArraySpec(
                            interpolatable=True,
                            voxel_size=(1, 1, 1),
                        )
                    },
                )
                + gp.Crop(raw, array_roi)
                + AddLabel(label, class_index)
                + gp.RandomLocation()  # Randomly sample in space
                + gp.Normalize(raw)
            )
            zebrafish_sources.append(zarr_source)

    pipeline = (
        tuple(zebrafish_sources)
        + gp.RandomProvider()
        + gp.Stack(batch_size)
        + gp.PreCache(num_workers=num_workers)
    )

    # Setup the request
    request = gp.BatchRequest()
    request.add(raw, shape)
    request[label] = gp.ArraySpec(nonspatial=True)

    # Load the model
    model = Vgg2D(input_size=(shape[1], shape[2]), output_classes=1)
    model.to(device)
    loss = torch.nn.BCEWithLogitsLoss()

    run = wandb.init(project="zebrafish-classification", id=run_id, resume="must")
    checkpoint_dir = Path(
        f"/nrs/funke/adjavond/zebrafish/classifiers/checkpoints/{run.name}"
    )
    wandb.define_metric("val_step")
    wandb.define_metric("val_loss", step_metric="val_step")
    wandb.define_metric("val_accuracy@0.5", step_metric="val_step")
    wandb.define_metric("val_accuracy@0.3", step_metric="val_step")
    wandb.define_metric("val_accuracy@0.7", step_metric="val_step")
    wandb.define_metric("false_positives", step_metric="val_step")
    wandb.define_metric("false_negatives", step_metric="val_step")

    metric = Accuracy(task="binary").to(device)
    metric_high = Accuracy(task="binary", threshold=0.7).to(device)
    metric_low = Accuracy(task="binary", threshold=0.3).to(device)
    with gp.build(pipeline):
        # Iterate over checkpoints
        for checkpoint in sorted(checkpoint_dir.glob("*.pth")):
            model.load_state_dict(torch.load(checkpoint))
            model.eval()

            # Seed so it's the same validation each  time
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Get the step from the checkpoint name
            step = int(checkpoint.stem.split("_")[1])

            # Validation loop
            val_loss = 0
            false_positives = 0
            false_negatives = 0
            for _ in tqdm(range(val_steps), desc=f"Validation @ {step}"):
                # Getting a batch, something is weird with dataloaders
                batch = pipeline.request_batch(request)
                X = torch.tensor(batch[raw].data).to(device)
                Y = torch.tensor(batch[label].data).to(device).unsqueeze(1).float()

                with torch.no_grad():
                    # TODO put this in Gunpowder as well?
                    pred = model(X.to(device))
                loss_value = loss(pred, Y)
                val_loss += loss_value.item()
                false_positives += torch.sum((pred > 0.5) & (Y == 0)).item()
                false_negatives += torch.sum((pred < 0.5) & (Y == 1)).item()
                acc = metric(pred, Y)
                acc_high = metric_high(pred, Y)
                acc_low = metric_low(pred, Y)

            val_loss /= val_steps
            accuracy = metric.compute()
            acc_high = metric_high.compute()
            acc_low = metric_low.compute()
            false_positives /= val_steps * batch_size
            false_negatives /= val_steps * batch_size
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_accuracy@0.5": accuracy,
                    "val_accuracy@0.3": acc_low,
                    "val_accuracy@0.7": acc_high,
                    "val_step": step,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                }
            )

            print(f"Validation loss: {val_loss}")
            print(f"Validation accuracy: {accuracy}")
            metric.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.run_id, args.seed)
