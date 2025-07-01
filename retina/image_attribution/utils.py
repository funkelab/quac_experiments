import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


def load_label(image_path, label_type, img_size=224):
    label_transform = transforms.Compose(
        [
            transforms.Resize(img_size),  # ), interpolation=Image.NEAREST),
            transforms.CenterCrop(img_size),
        ]
    )
    image_path = Path(image_path)
    label_directory = image_path.parent.parent.parent / "label"
    label_path = label_directory / f"{label_type}/{image_path.stem}.tif"
    label = Image.open(label_path)
    label = label_transform(label)

    label = np.array(label) / 255
    return label


def tp_fp_overlay(mask, label_image, thresh=0.1):
    average_mask = mask.mean(axis=0)
    true_positives = np.logical_and(average_mask > thresh, label_image > 0)
    false_negatives = np.logical_and(average_mask < thresh, label_image > 0)

    rgb = np.repeat((255 * average_mask).astype(np.uint8)[..., None], 3, axis=-1)
    rgb[true_positives] = [0, 255, 0]
    rgb[false_negatives] = [255, 0, 0]
    return rgb


def mask_label_overlay(
    mask,
    label_image,
    thresh=0.1,
    color_one=(91, 192, 235),
    color_two=(253, 231, 76),
    color_both=(155, 197, 61),
):
    average_mask = mask.mean(axis=0)
    true_positives = np.logical_and(average_mask > thresh, label_image > 0)
    false_negatives = np.logical_and(average_mask < thresh, label_image > 0)

    rgb = np.stack(
        [   # color_one
            (c * average_mask).astype(np.uint8) 
            for c in color_one
        ],
        axis=-1,
    )
    rgb[true_positives] = color_both
    rgb[false_negatives] = color_two
    return rgb


def recall(mask, label_image, thresh=0.1):
    average_mask = mask.mean(axis=0)
    true_positives = np.logical_and(average_mask > thresh, label_image > 0)
    false_negatives = np.logical_and(average_mask < thresh, label_image > 0)
    # recall = TP / (TP + FN)
    recall = true_positives.sum() / max(true_positives.sum() + false_negatives.sum(), 1)
    return recall


def precision(mask, label_image, thresh=0.1):
    average_mask = mask.mean(axis=0)
    true_positives = np.logical_and(average_mask > thresh, label_image > 0)
    false_positives = np.logical_and(average_mask > thresh, label_image == 0)
    # precision = TP / (TP + FP)
    precision = true_positives.sum() / max(
        true_positives.sum() + false_positives.sum(), 1
    )
    return precision


def iou(mask, label_image, thresh=0.1):
    average_mask = mask.mean(axis=0)
    intserection = np.logical_and(average_mask > thresh, label_image > 0)
    union = np.logical_or(average_mask > thresh, label_image > 0)
    iou = intserection.sum() / max(union.sum(), 1)
    return iou
