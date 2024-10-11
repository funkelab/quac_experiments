# %%
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
from funlib.learn.torch.models import Vgg2D
import torch
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from scipy.special import expit
import wandb
import gunpowder as gp
from utils import AddLabel
import argparse


def train(
    noise_variance,
    intensity_shift,
    iterations,
    resume=0,
    metadata_file="metadata.yaml",
    log_every=100,
    save_every=1000,
    batch_size=16,
    num_workers=12,
    shape=(1, 512, 512),
    lr=0.0001,
):
    # TODO Resuming training?
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
                # Adding augmentations to reduce overfitting
                + gp.SimpleAugment()
                + gp.IntensityAugment(
                    raw,
                    scale_min=0.9,
                    scale_max=1.1,
                    shift_min=-intensity_shift,
                    shift_max=intensity_shift,
                )
                + gp.NoiseAugment(raw, var=noise_variance)
                + gp.DeformAugment(
                    control_point_spacing=(32, 32),
                    jitter_sigma=(3, 3),
                    graph_raster_voxel_size=(1, 1),
                    spatial_dims=2,
                )
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.BCEWithLogitsLoss()

    # Setup wandb
    run = wandb.init(
        project="zebrafish-classification",
        tags=[
            "zebrafish",
            "classification",
            "vgg2d",
            "bcewithlogits",
            "gunpowder",
            "augmentation_sweep",
        ],
        notes="Version with saved checkpoints",
        config={
            "model": "Vgg2D",
            "input_size": shape,
            "output_classes": 1,
            "optimizer": "Adam",
            "lr": lr,
            "loss": "BCEWithLogitsLoss",
            "noise_variance": noise_variance,
            "intensity_shift": intensity_shift,
        },
    )

    # Set up checkpointing
    checkpoint_dir = Path(
        f"/nrs/funke/adjavond/zebrafish/classifiers/checkpoints/{run.name}"
    )
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Train
    log_loss = 0
    metric = Accuracy(task="binary").to(device)
    with gp.build(pipeline):
        for j in tqdm(range(iterations), total=iterations):
            optimizer.zero_grad()
            # Getting a batch, something is weird with dataloaders
            batch = pipeline.request_batch(request)
            X = torch.tensor(batch[raw].data).to(device)
            Y = torch.tensor(batch[label].data).to(device).unsqueeze(1).float()

            # TODO put this in Gunpowder as well?
            pred = model(X.to(device))
            loss_value = loss(pred, Y)
            loss_value.backward()
            optimizer.step()
            log_loss += loss_value.item()
            acc = metric(pred, Y)

            if j % log_every == 0:
                accuracy = metric.compute().item()
                wandb.log({"loss": log_loss / log_every, "accuracy": accuracy}, step=j)
                # log images
                run.log(
                    {
                        "images": [
                            wandb.Image(
                                x,
                                caption=f"True: {class_names[int(y)]}, Pred: {expit(p.item()):.4f}",
                            )
                            for x, y, p in zip(X, Y, pred)
                        ]
                    },
                    step=j,
                )
                log_loss = 0
                metric.reset()

            if j % save_every == 0 and j > 0:  # Don't save the first model
                torch.save(model.state_dict(), checkpoint_dir / f"model_{j:06d}.pth")


# NOTE Can use this to visualize the data
# with gp.build(pipeline):
#     for i in tqdm(range(2)):
#         batch = pipeline.request_batch(request)

# # %%
# x = batch[raw].data.squeeze()
# y = batch[label].data
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# for i, ax in enumerate(axes.ravel()):
#     ax.imshow(x[i], cmap="gray")
#     ax.set_title(class_names[int(y[i])])
#     ax.axis("off")
# fig.tight_layout()
# plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_variance", type=float, default=0.01)
    parser.add_argument("--intensity_shift", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=10001)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--shape", type=int, nargs=3, default=(1, 512, 512))
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    train(
        args.noise_variance,
        args.intensity_shift,
        args.iterations,
        log_every=args.log_every,
        save_every=args.save_every,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shape=args.shape,
        lr=args.lr,
    )


# %%
