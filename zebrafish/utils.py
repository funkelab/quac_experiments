import dask.array as da
import random
import numpy as np
import torch
import gunpowder as gp


# Gunpowder pipeline
class PrintBatch(gp.BatchFilter):

    def process(self, batch, request):
        for key, array in batch.items():
            print(key, array)


class AddLabel(gp.BatchFilter):

    def __init__(self, array_key, label):
        self.array_key = array_key
        self.label = label

    def setup(self):
        self.provides(self.array_key, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        pass

    def process(self, batch, request):

        array = gp.Array(np.array(self.label), spec=gp.ArraySpec(nonspatial=True))
        batch = gp.Batch()
        batch[self.array_key] = array
        return batch


# Non gunpowder
def random_rotation(data):
    p = random.random()
    if p < 1 / 3:
        return data.transpose(1, 2, 0)
    elif p < 2 / 3:
        return data.transpose(2, 0, 1)
    else:
        return data


class RandomCrop:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, arr):
        """Randomly crop the array to the specified size."""
        x = random.randint(0, arr.shape[0] - self.size[0])
        y = random.randint(0, arr.shape[1] - self.size[1])
        return arr[x : x + self.size[0], y : y + self.size[1]]


class Rescale:
    """Puts data between 0 and 1."""

    def __call__(self, arr):
        dtype = arr.dtype
        max_val = np.iinfo(dtype).max
        return arr / max_val


class IntensityAugmentation:
    """Randomly change the intensity of the image."""

    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, arr):
        return arr * (1 + self.factor * (2 * random.random() - 1))


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, arr):
        for transform in self.transforms:
            arr = transform(arr)
        return arr


class ZfishDataset(torch.utils.data.IterableDataset):
    def __init__(self, metadata, rotation_transform=None, transform=None):
        self.metadata = metadata
        self.categories = list(metadata.keys())
        self.samples = {
            category: list(metadata[category].keys()) for category in self.categories
        }
        self.arrays = {}
        self._init_arrays()
        self.rotation_transform = rotation_transform
        self.transform = transform

    def _init_arrays(self):
        for category in self.categories:
            cm = self.metadata[category]
            for sample in cm.keys():
                sm = cm[sample]
                base_dir = f"/nrs/funke/adjavond/zebrafish/data/stitched/{sample}/"
                file_location = base_dir + sm["file"]

                data = da.from_zarr(file_location, component="raw/s0")
                data = data[
                    sm["zmin"] : sm["zmax"],
                    sm["ymin"] : sm["ymax"],
                    sm["xmin"] : sm["xmax"],
                ]
                self.arrays[sample] = data

    def __iter__(self):
        # Randomly sample a category
        category = random.choice(self.categories)
        # Randomly sample a sample from the category
        sample = random.choice(self.samples[category])
        if self.rotation_transform is not None:
            data = self.rotation_transform(self.arrays[sample])
        else:
            data = self.arrays[sample]
        # Randomly choose a z-slice
        z = random.randint(0, data.shape[0] - 1)
        image = data[z]
        # Apply the transform
        if self.transform is not None:
            image = self.transform(image)
        image = image.compute()
        yield image, self.categories.index(category)
