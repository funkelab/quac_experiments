"""Colored MNIST Data set."""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

download_path = Path(__file__).parent / 'downloads'


def colorize(image, color):
    """Turn a grayscale image into a single-colored image."""
    image = torch.stack(tuple(image * x for x in color), dim=1).squeeze()
    return image


def get_color(condition_labels, condition, sample):
    """Get matplotlib color based on condition and random sample.

    Parameters
    ----------
    condition_labels: List[str]
        List of available conditions; i.e. `matplotlib` colormaps.
    condition: int
        The index of the condition
    sample: float
        Sampling value for the colormap, must be between 0 and 1.

    Returns
    -------
    color: np.array
        (3,) array of RGB values
    """
    color = plt.cm.get_cmap(condition_labels[condition])(sample)[:-1]
    return color


class ColoredMNIST(torchvision.datasets.MNIST):
    """MNIST with added color.

    The original MNIST images make up the content of the data set.
    They are styled with colors sampled from `matplotlib` colormaps.
    The colormaps correspond to the data's condition.
    """
    def __init__(self, root, classes=None, c_transform=None, train=True,
                 download=False, transform=None):
        """
        Parameters
        ----------
        root: Union[str, pathlib.Path]
            Data root for download; defaults to ./downloads
        classes: List[str]
            The names of the `matplotlib` colormaps to use; defaults to the
            conditions: `['spring', 'summer', 'autumn', 'winter']`.
        c_transform: Callable
            A torchvision transform to apply to the images after colorization.
            Defaults to `None`. Must be a callable that takes a torch.Tensor
        train: bool
            Passed to `torchvision.datasets.MNIST`; default is True
        download: bool
            Passed to `torchvision.datasets.MNIST`; default is True
        transform: Callable
            Passed to `torchvision.datasets.MNIST`; default is None
            Used to apply a transform to the MNIST images before colorization.
            This transform should take a PIL image.
            It can be used for augmenting the data.
        """
        super().__init__(root, train=train, download=download, transform=transform)
        if classes is None:
            self.classes = ['spring', 'summer', 'autumn', 'winter']
        else:
            self.classes = classes
        # Initialize a transform that is run after colorization
        self.c_transform = c_transform
        # Initialise a random set of conditions, of the same length as the data
        seed = 0 if train else 42  # Separate seed for train and test
        torch.manual_seed(seed)
        self.conditions = torch.randint(len(self.classes),
                                        (len(self),))
        # Initialise a set of style values, the actual color will be dependent
        # on the condition
        self.style_values = torch.rand((len(self),))
        self.colors = [get_color(self.classes, condition, sample)
                       for condition, sample
                       in zip(self.conditions.numpy(),
                              self.style_values.numpy())]
        # Initialize all existing prior knowledge
        self.knowledge = dict(conditions=self.conditions.numpy(),
                              numbers=self.targets)

    def preprocess(self, image):
        if not isinstance(image, torch.Tensor):
            # Could have already be transformed by the transforms on MNIST
            image = torchvision.transforms.functional.pil_to_tensor(image)
        if image.max() > 1:
            image = image / 255.
        return image

    def __getitem__(self, item):
        image, label = super().__getitem__(item)
        image = self.preprocess(image)
        color = torch.Tensor(self.colors[item])
        condition = self.conditions[item]
        label = torch.tensor(label)
        image = colorize(image, color)
        if self.c_transform is not None:
            image = self.c_transform(image)
        return image, condition

    def get_counterfactual(self, item, condition):
        """
        Get the counterfactual image for a given item and condition.
        """
        image, _ = super().__getitem__(item)
        image = self.preprocess(image)
        # Get the counterfactual color
        counterfactual_color = get_color(self.classes,
                                         condition,
                                         self.style_values[item])
        # Get the counterfactual image
        counterfactual_image = colorize(image, counterfactual_color)
        return counterfactual_image, condition
