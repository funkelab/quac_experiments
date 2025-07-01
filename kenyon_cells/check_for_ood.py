# %% [markdown]
# # Question: Is it the Bojack Horseman effect?
#
# We notice that the Kenyon Cells are mostly classified as Dopaminergic.
# This is a clear mis-classification, as they are cholinergice.
#
# We start with a little experiment to see whether our problem is simply that the
# images of the Kenyon cells are out-of-distribution for the classifier.
# %% Setup
import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
from funlib.learn.torch.models import Vgg2D
from tqdm import tqdm

# %% [markdown]
# Let's start by looking at the counts of the different classes in the predictions.
# We look at 10000 predictions to get an estimate of the distribution.
# %% Loading the results
results = zarr.open("/nrs/funke/adjavond/projects/quac/kenyon_cells/results.zarr", "r")

images = results["images"]
predictions = results["predictions"]

current_preds = predictions[:10000]
# %%
classes = ["GABA", "Acetylcholine", "Glutamate", "Serotonin", "Octopamine", "Dopamine"]
current_classes = np.argmax(current_preds, axis=1)
# %% Check the distribution of the classes

for i, name in enumerate(classes):
    print(f"{name}: {np.sum(np.array(current_classes) == i)/ 10000}")

# %% [markdown]
# Next, we want to compare to the distribution of a random sample of images.
# We start with a random sample of images from the range [-1, 1].
# %% Checking whether this distribution can be attributed to a "Bojack Horseman" effect
model = Vgg2D(input_size=(128, 128), fmaps=12)
model.load_state_dict(
    torch.load("/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint")[
        "model_state_dict"
    ]
)
model.eval()

# %%
random_classes = []
confidences = []
for i in tqdm(range(10000)):
    random = 2 * torch.rand(1, 1, 128, 128) - 1
    random_prediction = torch.nn.functional.softmax(model(random), dim=1)
    random_class = torch.argmax(random_prediction).item()
    random_classes.append(random_class)
    confidences.append(random_prediction[0, random_class].item())

# %%
for i, name in enumerate(classes):
    print(f"{name}: {np.sum(np.array(random_classes) == i)/ 10000}")

# %% [markdown]
# There could be multiple levels of "out-of-distribution".
# Here, we check whether images that *do* look like EM but are *not* synapses are
# classified with the same kind of distribution as the Kenyon cells were.
# %% Same check, but randomly sampled from the volume instead
import gunpowder as gp

voxel_size = gp.Coordinate((40, 4, 4))
size = gp.Coordinate((1, 128, 128))
size_nm = size * voxel_size
raw = gp.ArrayKey("RAW")
prediction = gp.ArrayKey("PREDICTION")

pipeline = gp.ZarrSource(
    "/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5",
    {raw: "volumes/raw/s0"},
    {raw: gp.ArraySpec(interpolatable=True)},
)

pipeline += gp.RandomLocation()
pipeline += gp.Normalize(raw)
pipeline += gp.IntensityScaleShift(raw, scale=2, shift=-1)
pipeline += gp.PreCache(cache_size=36, num_workers=12)
pipeline += gp.Stack(1)
pipeline += gp.torch.Predict(model, inputs={"raw": raw}, outputs={0: prediction})

# %% Setting up the gunpowder pipeline
from scipy.special import softmax

request = gp.BatchRequest()
request[raw] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), size_nm))
request[prediction] = gp.ArraySpec(nonspatial=True)
random_em_classes = []
confidences_em = []

with gp.build(pipeline):
    for i in tqdm(range(10000)):
        batch = pipeline.request_batch(request)
        random_prediction = softmax(batch[prediction].data, axis=1)
        random_class = np.argmax(random_prediction)
        random_em_classes.append(random_class)
        confidences_em.append(random_prediction[0, random_class].item())

# %%
for i, name in enumerate(classes):
    print(f"{name}: {np.sum(np.array(random_em_classes) == i)/ 10000}")
# %%
