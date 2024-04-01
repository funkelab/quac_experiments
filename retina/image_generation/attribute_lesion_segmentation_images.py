# %%
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from quac.generate import load_classifier 
from quac.attribution import DIntegratedGradients, DDeepLift, DInGrad
from quac.evaluation import Evaluator
from quac.report import Report
from tqdm import tqdm
import torch
from torchvision import transforms
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# %%
data_directory = Path("/nrs/funke/adjavond/data/retina/DDR-dataset/lesion_segmentation/train/")
image_directory = data_directory / "image"
counterfactual_directory = data_directory / "counterfactual/reference/0_No_DR"
label_directory = data_directory / "label/MA"

save_dir = data_directory / "reports"
save_dir.mkdir(exist_ok=True)

img_size = 224
mean_generator = 0.5
std_generator = 0.5

def to_numpy(image):
    img = np.array(image)
    return np.transpose(img, (2, 0, 1))

transform=transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_generator, std=std_generator),
    ]
)

# %%
classifier_checkpoint = "/nrs/funke/adjavond/projects/quac/retina-ddr-classification/final_model.pt"
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
device = "cpu"

classifier = load_classifier(
    classifier_checkpoint, mean=mean, std=std, eval=True, device=device
)

evaluator = Evaluator(classifier)
# %%
image_paths = list(image_directory.glob("*.jpg"))

# %% Start by classifying the images, to make sure that nothing went wrong in the generation

classes_og = []
classes_cf = []
for path in tqdm(image_paths):
    cf_path = counterfactual_directory / path.name
    label_path = label_directory / f"{path.stem}.tif"
    assert cf_path.exists(), f"Counterfactual image {cf_path} does not exist"
    assert label_path.exists(), f"Label image {label_path} does not exist"
    image = Image.open(path)
    cf_image = Image.open(cf_path)
    label_image = Image.open(label_path)

    x = transform(image)
    x_t = transform(cf_image)

    with torch.no_grad():
        y = evaluator.run_inference(x)[0].argmax().item()
        y_t = evaluator.run_inference(x_t)[0].argmax().item()

    classes_og.append(y)
    classes_cf.append(y_t)

# %%
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(np.zeros_like(classes_cf), classes_cf, normalize="true")
sns.heatmap(cm, annot=True)

# %% Next, we'll do attribution

methods = {
	#"discriminative_ig": DIntegratedGradients(classifier),
    "discriminative_dl": DDeepLift(classifier),
    # "discriminative_ingrad": DInGrad(classifier)
}
reports = {name: Report(save_dir, name=name) for name in methods}

# %%
# idx = 2

# im = Image.open(image_paths[idx])
# cf = Image.open(counterfactual_directory / image_paths[idx].name)
# label = Image.open(label_directory / f"{image_paths[idx].stem}.tif")

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(im)
# axes[0].set_title("Image")
# axes[1].imshow(cf)
# axes[1].set_title("Counterfactual")
# axes[2].imshow(label)
# axes[2].set_title("Label")
# plt.show()
# %%
for path in tqdm(image_paths):
    cf_path = counterfactual_directory / path.name
    label_path = label_directory / f"{path.stem}.tif"
    assert cf_path.exists(), f"Counterfactual image {cf_path} does not exist"
    assert label_path.exists(), f"Label image {label_path} does not exist"
    image = Image.open(path)
    cf_image = Image.open(cf_path)
    label_image = Image.open(label_path)

    x = transform(image)
    x_t = transform(cf_image)

    predictions = {
        "original": evaluator.run_inference(x)[0],
        "counterfactual": evaluator.run_inference(x_t)[0]
    }

    y = predictions["original"].argmax().item()
    y_t = 0

    for method, attributor in methods.items():
        attribution = attributor.attribute(x, x_t, y, y_t)
        results = evaluator.evaluate(x, x_t, y, y_t, attribution, predictions)
        reports[method].accumulate(
            {
                "sample_path": path,
                "target_path": cf_path,
                "sample": x,
                "target_sample": x_t,
                "class_index": y,
                "target_class_index": y_t,
            },
            predictions,
            attribution,
            results,
            save_intermediates=False
        )

# %%
# Plot results
n = len(reports)
if n > 1:
    fig, axes = plt.subplots(1, n, figsize=(n*5, 5))
    for ax, (method, report) in zip(axes, reports.items()):
        report.plot_curve(ax)
        ax.set_title(method)
else: 
    for method, report in reports.items():
        report.plot_curve()
        plt.title(method)
plt.show()
# %%
report = reports["discriminative_dl"]
# %%
quac_scores = report.compute_scores()
plt.hist(quac_scores)

# %%
viz_transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ]
)

def load_labels(image_path, img_size=224, label_types=["EX", "HE", "MA", "SE"]):
    label_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
        ]
    )
    all_labels = np.zeros((img_size, img_size))
    for i, label_type in enumerate(label_types):
        label_path = str(image_path).replace("image", f"label/{label_type}").replace("jpg", "tif")
        label = Image.open(label_path)
        label = label_transform(label)

        label = np.array(label) / 255
        all_labels += label * (i+1)
    return all_labels

def mask_label_overlay(mask, label_image, thresh=0.1):
    true_positives = np.logical_and(mask[..., 0] > thresh, label_image > 0)
    false_negatives = np.logical_and(mask[..., 0] < thresh, label_image > 0)

    others = np.logical_and(mask[..., 0] > 0.1, label_image == 0)

    # rgb = np.zeros((label_image.shape[0], label_image.shape[1], 3))
    rgb = (255 * mask).astype(np.uint8)
    rgb[true_positives] = [0, 255, 0]
    rgb[false_negatives] = [255, 0, 0]
    # rgb[others] = np.stack([mask_image[others], mask_image[others], mask_image[others]], axis=-1)
    
    return rgb
# %%
with PdfPages("lesion_segmentation_reference.pdf") as pdf:
    for idx in tqdm(np.argsort(quac_scores)[::-1][:50]):
        attr = report.load_attribution(idx)
        image = Image.open(report.paths[idx])
        cf_image = Image.open(report.target_paths[idx])
        # TODO get the label image
        label_image = load_labels(report.paths[idx])
        x = viz_transform(image)
        x_t = viz_transform(cf_image)

        thr = report.get_optimal_threshold(idx)
        mask, mask_size = evaluator.create_mask(attr, thr)
        
        mask = np.transpose(mask, (1, 2, 0))
        hybrid = mask * np.array(x_t) / 255  + (1 - mask) * np.array(x) / 255
        normalized_hybrid = torch.from_numpy(2 * np.transpose(hybrid, (2, 0, 1)) - 1)

        fig, (headers, axes) = plt.subplots(2, 4, figsize=(20, np.ceil(1.3 * 5)), gridspec_kw={"height_ratios": [0.2, 1], "hspace": 0.})
        # axes[0].imshow(x.cpu().numpy().transpose(1, 2, 0))
        # axes[1].imshow(x_t.cpu().numpy().transpose(1, 2, 0))
        axes[0].imshow(x)
        axes[1].imshow(x_t)
        axes[2].imshow((255 * hybrid).astype(np.uint8))
        pred = report.predictions[idx]
        target_pred = report.target_predictions[idx]
        hybrid_pred = classifier(normalized_hybrid.unsqueeze(0)).softmax(1).detach().cpu().numpy().squeeze()

        
        headers[0].barh(range(len(pred)), pred)
        headers[0].set_xlim(0, 1)
        headers[0].set_yticks(range(len(pred)))
        headers[0].set_title("Original")
        headers[1].barh(range(len(target_pred)), target_pred)
        headers[1].set_xlim(0, 1)
        headers[1].set_yticks(range(len(pred)))
        headers[1].set_title("Counterfactual")
        headers[2].barh(range(len(hybrid_pred)), hybrid_pred)
        headers[2].set_xlim(0, 1)
        headers[2].set_yticks(range(len(pred)))
        headers[2].set_title("Hybrid")

        # Masks etc
        axes[3].imshow(mask_label_overlay(mask, label_image)) 
        axes[3].set_title(f"Score: {quac_scores[idx]:.2f}")

        for ax in axes:
            ax.set_axis_off()

        headers[3].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.clf()

    # %%
