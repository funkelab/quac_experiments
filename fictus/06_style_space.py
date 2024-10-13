# %% [markdown]
# Looking through the style space
#
# %%
from pathlib import Path
from quac.generate import load_stargan
from yaml import safe_load
import torch

# %%
# Load metadata
print("Loading metadata")
with open("configs/stargan.yml", "r") as f:
    metadata = safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kind = "reference"
checkpoint_iter = 100000

# Load the model
print("Loading encoder model")
latent_model_checkpoint_dir = Path(metadata["solver"]["root_dir"]) / "checkpoints"
infer_model = load_stargan(
    latent_model_checkpoint_dir,
    checkpoint_iter=checkpoint_iter,
    kind=kind,
    **metadata["model"],
)
# Get the encoder
encoder = infer_model.nets.style_encoder
# %% Load the data
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

transform = Compose(
    [
        Resize((128, 128)),
        ToTensor(),
    ]
)

data = ImageFolder(
    metadata["validation_data"]["source"],
    transform=transform,
)

dataloader = torch.utils.data.DataLoader(
    data, batch_size=32, shuffle=False, num_workers=32
)

# %%
from tqdm import tqdm

styles = []
labels = []
with torch.inference_mode():
    for img, y in tqdm(dataloader, total=len(dataloader)):
        img = img.to(device)
        style = encoder(img, y)
        styles.append(style.cpu().detach())
        labels.append(y)
# %%
styles = torch.cat(styles)
labels = torch.cat(labels)

# %% Get principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=64)
X = pca.fit_transform(styles.numpy())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=X[:, 0], y=X[:, 2], hue=labels.numpy(), palette="tab10")

# %%
# Get UMAP
import umap

reducer = umap.UMAP()
X_umap = reducer.fit_transform(styles.numpy())
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels.numpy(), palette="tab10")

# %% Get tSNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(styles.numpy())
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels.numpy(), palette="tab10")

# %%
# Which of the features are most important?
plt.bar(range(len(pca.components_[0])), pca.components_[0], color="red")
plt.bar(range(len(pca.components_[1])), pca.components_[1], color="blue")

# %%
# If I change only component 31, where do the styles go in the space?
new_styles = styles.clone()
# Set it to the maximum value
max_value = styles[:, 31].max()
new_styles[:, 63] = 10 * max_value

# %%
X_new = pca.transform(new_styles.numpy())

# %%
plt.scatter(X[:, 0], X[:, 1], label="Original", color="black", s=1)
plt.scatter(X_new[:, 0], X_new[:, 1], label="Modified", color="red", s=1, marker="x")

# %%
plt.imshow(data[0][0].permute(1, 2, 0))
# %% Generate one with the new style
img = data[0][0].unsqueeze(0).to(device)
style = new_styles[0].unsqueeze(0).to(device)

with torch.inference_mode():
    x_fake = infer_model.nets.generator(img, style)
    x_fake = x_fake.cpu().detach()

# %%
plt.imshow(x_fake[0].permute(1, 2, 0))

# %%
import numpy as np

new_styles2 = styles.clone()
# Find feature most present in second principal component
index = 63
new_styles2[:, index] = -5 * np.sign(styles[:, index])  # Flip the sign

X_new2 = pca.transform(new_styles2.numpy())

plt.scatter(X[:, 0], X[:, 1], label="Original", c=labels.numpy(), s=1)
plt.scatter(X_new2[:, 0], X_new2[:, 1], c=labels.numpy(), s=1, marker="x")
# %% Generate one with the new style2

style2 = new_styles2[0].unsqueeze(0).to(device)
with torch.inference_mode():
    x_fake2 = infer_model.nets.generator(img, style2)
    x_fake2 = x_fake2.cpu().detach()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(data[0][0].permute(1, 2, 0))
ax1.set_title("Original")
ax2.imshow(x_fake[0].permute(1, 2, 0))
ax2.set_title("Modified 1")
ax3.imshow(x_fake2[0].permute(1, 2, 0))
ax3.set_title("Modified 2")

# %%
import pandas as pd

df = pd.DataFrame(
    styles.numpy(), columns=[f"style_{i}" for i in range(styles.shape[1])]
)
df["label"] = labels.numpy()

# %%
sns.boxplot(data=df, x="label", y="style_31")
# %%
sns.boxplot(data=df, x="label", y="style_58")

# %% [markdown]
# Let's try to shift data!
X_goal = X.copy()
X_goal[:, 0] = 0

import numpy as np

plt.scatter(X_goal[:, 0], X_goal[:, 1], c=labels.numpy(), s=1)
# %%
# Get styles back
styles_goal = pca.inverse_transform(X_goal)

# %%
# Generate an image
idx = 1
img, y = data[idx]
img = img.unsqueeze(0).to(device)
style_goal = torch.tensor(styles_goal[0]).unsqueeze(0).to(device)
with torch.inference_mode():
    x_fake_goal = infer_model.nets.generator(img, style_goal)
    x_fake_goal = x_fake_goal.cpu().detach()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img[0].permute(1, 2, 0).cpu())
ax1.set_title("Original")
ax2.imshow(x_fake_goal[0].permute(1, 2, 0))
ax2.set_title("Goal")


# %%
scatter = plt.scatter(styles[:, 31], styles[:, 58], c=labels.numpy(), s=1)
# Get a legend for the colors
plt.legend(*scatter.legend_elements(), title="Classes")
# %%
