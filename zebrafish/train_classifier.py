# %%
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
from funlib.learn.torch.models import Vgg2D
import torch
from tqdm import tqdm
from scipy.special import expit
import wandb

from utils import (
    Compose,
    RandomCrop,
    Rescale,
    IntensityAugmentation,
    ZfishDataset,
    random_rotation,
)

# %%  Setup
metadata = safe_load(open("metadata.yaml"))
# shape = (256, 256)
shape = (512, 512)
iterations = 10000
log_every = 100
save_every = 100
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose(
    [
        RandomCrop(shape),
        Rescale(),
        IntensityAugmentation(factor=0.3),
    ]
)
dataset = ZfishDataset(
    metadata, rotation_transform=random_rotation, transform=transform
)

model = Vgg2D(input_size=shape, output_classes=1)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss = torch.nn.BCEWithLogitsLoss()

# %% Setup wandb

run = wandb.init(
    project="zebrafish-classification",
    tags=["zebrafish", "classification", "vgg2d", "bcewithlogits", "random_rotation"],
    notes="Version with saved checkpoints",
    config={
        "model": "Vgg2D",
        "input_size": shape,
        "output_classes": 1,
        "optimizer": "Adam",
        "lr": 0.0001,
        "loss": "BCEWithLogitsLoss",
    },
)

checkpoint_dir = Path(f"/nrs/funke/adjavond/zebrafish/classifiers/checkpoints/{run.id}")
checkpoint_dir.mkdir(exist_ok=True, parents=True)
# %% Train
log_loss = 0
for j in tqdm(range(iterations), total=iterations):
    optimizer.zero_grad()
    # Getting a batch, something is weird with dataloaders
    X = torch.empty((batch_size, *shape))
    Y = torch.empty(batch_size)

    for i in range(batch_size):
        x, y = next(iter(dataset))
        X[i] = torch.from_numpy(x).float()
        Y[i] = y

    pred = model(X.unsqueeze(1).to(device))
    loss_value = loss(pred, Y.unsqueeze(1).to(device))
    loss_value.backward()
    optimizer.step()
    log_loss += loss_value.item()

    if j % log_every == 0:
        wandb.log({"loss": log_loss / log_every}, step=j)
        # log images
        run.log(
            {
                "images": [
                    wandb.Image(
                        x,
                        caption=f"True: {dataset.categories[int(y)]}, Pred: {expit(p.item())}",
                    )
                    for x, y, p in zip(X, Y, pred)
                ]
            },
            step=j,
        )
        log_loss = 0

    if j % save_every == 0:
        torch.save(model.state_dict(), checkpoint_dir / f"model_{j:06d}.pth")
