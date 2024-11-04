"""
Test that kornia operations match openCV operations
Conclusion: barring differences most likely due to floating point precision, the results are the same!
AND the kornia operations are faster and can be run on the GPU
SO we should use kornia for these operations
"""

import cv2
import kornia
import numpy as np
import torch
from skimage.data import astronaut
import pytest


@pytest.fixture
def image():
    return (torch.tensor(astronaut()).float() / 255).unsqueeze(0).permute(0, 3, 1, 2)


@pytest.fixture
def numpy_image(image):
    return image.numpy().squeeze().transpose(1, 2, 0)


@pytest.fixture
def kernel():
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


@pytest.fixture
def torch_kernel(kernel):
    return torch.tensor(kernel)


def test_cv2_closing(numpy_image, kernel):
    """
    Check that morphological closing is the same channel by channel as it is on the whole image
    """
    cv2_closing1 = cv2.morphologyEx(numpy_image, cv2.MORPH_CLOSE, kernel)
    cv2_closing2 = []
    for c in range(3):
        cv2_closing2.append(
            cv2.morphologyEx(numpy_image[:, :, c], cv2.MORPH_CLOSE, kernel)
        )
    cv2_closing2 = np.stack(cv2_closing2, axis=-1)
    assert np.all(cv2_closing1 == cv2_closing2)


def test_kornia_closing(image, torch_kernel):
    """
    Check that morphological closing is the same in kornia as in openCV
    """
    kornia_closing1 = kornia.morphology.closing(image, torch_kernel.float())
    cv2_closing1 = cv2.morphologyEx(numpy_image, cv2.MORPH_CLOSE, kernel)
    assert np.all(cv2_closing1 == kornia_closing1.numpy().squeeze().transpose(1, 2, 0))


def test_binary_kornia_closing(image, torch_kernel):
    """
    Check that morphological closing and thresholding can be interchanged
    """
    image_binary = (image > 0.5).float()
    kornia_closing1 = kornia.morphology.closing(image, torch_kernel.float())
    kornia_closing2 = kornia.morphology.closing(image_binary, torch_kernel.float())
    assert np.all((kornia_closing1 > 0.5).float().numpy() == kornia_closing2.numpy())


def test_kornia_blur(image):
    """
    Check that Gaussian blur is the same in kornia as in openCV

    NOTE: We use allclose instead of all because of floating point precision
    """
    cv2_blur1 = cv2.GaussianBlur(numpy_image, (11, 11), 0)
    sigma = 0.3 * ((11 - 1) * 0.5 - 1) + 0.8
    kornia_blur1 = kornia.filters.gaussian_blur2d(image, (11, 11), (sigma, sigma))
    assert np.allclose(
        cv2_blur1, kornia_blur1.numpy().squeeze().transpose(1, 2, 0), atol=1e-5
    )
