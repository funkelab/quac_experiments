# %%
from torchvision.datasets import ImageFolder
from torchvision import transforms
import wandb
from torchvision.utils import make_grid
from train_classifier import imread


ds = ImageFolder('/nrs/funke/adjavond/data/kidney_cortex_cells/F33/All_3D', loader=imread)
# %%
x, y = ds[10]
print(x.shape)

# %%
import numpy as np
import torch
batch = torch.from_numpy(np.stack([ds[i][0] for i in range(8)])).float()

# %%
batch.shape

# %%
grid = make_grid(batch[:, 0:1])
# %%
grid.shape
# %%
wandb.Image(grid)
# %%
