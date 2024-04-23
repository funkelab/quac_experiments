# %%
from quac.training.data_loader import ValDataset

# %%
source_directory = "/nrs/funke/adjavond/data/fictus/aggregatum/val"
reference_directory = "/nrs/funke/adjavond/data/fictus/aggregatum/val"

dataset = ValDataset(source_directory, reference_directory)
dataset.print_info()

# %%
dataset.set_source("0")
dataset.set_target("2")
# %%
dataset.print_info()

# %%
dataset.set_mode("reference")
dataset.print_info()

# %%
batch = next(iter(dataset.loader_src))

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(np.transpose(batch[0], (1, 2, 0)))
# %%
len(batch), batch.min(), batch.max()
# %%
batch_ref = next(iter(dataset.loader_ref))
# %%
plt.imshow(np.transpose(batch_ref[0], (1, 2, 0)))
# %%
len(batch_ref), batch_ref.min(), batch_ref.max()

# %%
