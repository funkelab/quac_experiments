# %% Setup
from pathlib import Path
from yaml import safe_load
from funlib.learn.torch.models import Vgg2D
import random
import torch
from tqdm import tqdm
from torchmetrics.classification import Accuracy
import wandb
from utils import (
    Compose,
    RandomCrop,
    Rescale,
    IntensityAugmentation,
    ZfishDataset,
    random_rotation,
)

# %% Metadata
# run_id = "sijhv9eu"
run_id = "b4jbao6z"
metadata = safe_load(open("test_metadata.yaml"))
shape = (512, 512)
val_steps = 100
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Setting up the dataset
transform = Compose(
    [
        RandomCrop(shape),
        Rescale(),
        # IntensityAugmentation(factor=0.3),
    ]
)
dataset = ZfishDataset(
    metadata, rotation_transform=random_rotation, transform=transform
)
# %% Setting up the model
model = Vgg2D(input_size=shape, output_classes=1)
model.to(device)

# %% Setting up the run
run = wandb.init(project="zebrafish-classification", id=run_id, resume="must")
checkpoint_dir = Path(f"/nrs/funke/adjavond/zebrafish/classifiers/checkpoints/{run.id}")
wandb.define_metric("val_step")
wandb.define_metric("val_loss", step_metric="val_step")
wandb.define_metric("val_accuracy@0.5", step_metric="val_step")

# %% Validate
metric = Accuracy(task="binary")

# Iterate over checkpoints
for checkpoint in sorted(checkpoint_dir.glob("*.pth")):
    # Seed the randomness so that we get the same validation set every time
    random.seed(42)
    # Load the model weights
    print(f"Loading checkpoint {checkpoint}")
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # Get the step from the checkpoint name
    step = int(checkpoint.stem.split("_")[1])
    val_loss = 0
    for _ in tqdm(range(val_steps), desc=f"Validation @ {step}"):
        with torch.no_grad():
            # Getting a batch, something is weird with dataloaders
            X = torch.empty((batch_size, *shape))
            Y = torch.empty(batch_size)

            for i in range(batch_size):
                x, y = next(iter(dataset))
                X[i] = torch.from_numpy(x).float()
                Y[i] = y

            outputs = model(X.unsqueeze(1).to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs, Y.unsqueeze(1).to(device)
            )
            val_loss += loss.item()
            acc = metric(outputs.cpu(), Y.unsqueeze(1))

    val_loss /= val_steps
    accuracy = metric.compute()
    wandb.log({"val_loss": val_loss, "val_accuracy@0.5": accuracy, "val_step": step})

    print(f"Validation loss: {val_loss}")
    print(f"Validation accuracy: {accuracy}")
    metric.reset()
