# %%
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from quac.report import Report
import cv2
from funlib.learn.torch.models import Vgg2D
import torch
from quac.data import PairedImageFolders
from torchvision import transforms
from skimage import measure

# %%
methods = [
    "discriminative_ig",
    "discriminative_dl",
    "discriminative_ingrad"
]
# %%
classifier = Vgg2D(input_size=(128, 128), output_classes=6, fmaps=12)
checkpoint = torch.load("/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint")["model_state_dict"]
classifier.eval()
classifier.load_state_dict(checkpoint)

# %%
# Singular for all attribution types
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = PairedImageFolders(
	"/nrs/funke/adjavond/data/synapses/test/", 
	"/nrs/funke/adjavond/data/synapses/counterfactuals/stargan_invariance_v0/test/",
    transform=transform
)
# %% Specific to each method
load_dir = "/nrs/funke/adjavond/projects/quac/20240201"

loaded_reports = {name: Report(load_dir, name=name) for name in methods}
fig, axes = plt.subplots(1, len(loaded_reports), figsize=(15, 5))
for ax, (method, report) in zip(axes, loaded_reports.items()):
    report.load( Path(load_dir) / f"{method}.json")
    report.plot_curve(ax)
# %% Let's look at IG specifically
report = loaded_reports["discriminative_ig"]
# %% Get the quac scores
quac_scores = report.compute_scores()
# %%
df = pd.DataFrame({
    "paths": report.paths,
    "target_paths": report.target_paths,
    "quac_scores": quac_scores,
    "labels": report.labels,
    "target_labels": report.target_labels,
    "pred_labels": np.argmax(report.predictions, axis=1),
    "pred_target_labels": np.argmax(report.target_predictions, axis=1),
})
# %%
df.sort_values("quac_scores", ascending=False, inplace=True)
# %% Plotting the quac scores
sns.displot(
    df, x="quac_scores", col="target_labels", row="labels", kde=True
)
# %% Confusion matrices
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(df["labels"], df["pred_labels"], normalize='true'), annot=True)
# %%
for name, sub_df in df.groupby("labels"):
    sns.heatmap(confusion_matrix(sub_df["target_labels"], sub_df["pred_target_labels"], normalize='true'), annot=True)
    plt.title(name)
    plt.show()
# %%
def get_shift(predictions, labels):
    pred_at_label = predictions[np.arange(predictions.shape[0]), labels]
    new_preds = predictions.copy()
    new_preds[np.arange(predictions.shape[0]), labels] = -np.inf
    return pred_at_label - np.max(new_preds, axis=1)

shift_og = get_shift(np.array(report.predictions), report.labels)
shift_target = get_shift(np.array(report.target_predictions), report.target_labels)

# %%
plt.hist([shift_og, shift_target], bins=20, label=["original", "target"])
plt.legend()
# %%
def ascore(m_s, m_n):
    return m_n**2  + (1 - m_s)**2

def get_optimal_mask(thresholds, mask_sizes, mask_scores):

    ascores = ascore(np.array(mask_scores), np.array(mask_sizes))

    thr_idx = np.argmin(ascores)
    thr = thresholds[thr_idx]
    mask_size = mask_sizes[thr_idx]
    mask_score = mask_scores[thr_idx]

    return thr_idx, thr, mask_size, mask_score

# %%
from quac.evaluation import Evaluator
evaluator = Evaluator(classifier)

# %%
def fancify(ax, color):
    ax.patch.set_edgecolor(color)
    ax.patch.set_linewidth(10)
    # Turn off all ticks
    ax.set_xticks([])
    ax.set_yticks([])

colors = {
    0:  "#1F77B4",
    1: "#FF7F0E",
    2: "#4AAD4A",
    3: "#D62728",
    4: "#9265BA",
    5: "#8C564C"
}
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_img_classification(ax, img, p):
    ax.imshow(img.squeeze(), cmap='gray', vmin=-1, vmax=1)
    fancify(ax, colors[np.argmax(p)])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="30%", pad=0.1)
    cax.bar(np.arange(len(p)), p, color=colors.values())
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([0, 1])
    # Remove top and right spines
    cax.spines['top'].set_visible(False)
    cax.spines['right'].set_visible(False)

def make_plot(idx):
    input = dataset[idx]
    x = input["sample"]
    x_t = input["target_sample"]
    y = input["class_index"]
    y_t = input["target_class_index"]
    # Attribution
    attr = report.load_attribution(idx)
    
    interp_score_changes = report.interpolate_score_values(
        report.normalized_mask_sizes[idx], report.score_changes[idx]
    )
    _, thr, mask_size, mask_score = get_optimal_mask(
                                            report.thresholds[idx], 
                                            report.normalized_mask_sizes[idx],
                                            report.score_changes[idx]
                                        )
    mask, _ = evaluator.create_mask(attr, thr)

    # hybrid = real parts copied to fake
    hybrid = x_t * mask + x * (1.0 - mask)

    p_h = evaluator.run_inference(hybrid)[0]
    p_og = report.predictions[idx]
    p_cf = report.target_predictions[idx]
    # with torch.no_grad():
    #     p_h = softmax(classifier(torch.from_numpy(normalize(hybrid[None]))), dim=1).squeeze()
    y_h = np.argmax(p_h)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_img_classification(axes[0], x, p_og)
    plot_img_classification(axes[1], x_t, p_cf)
    plot_img_classification(axes[2], hybrid, p_h)
    fig.tight_layout()
    plt.show()
    
    # bbox_mask = (mask.squeeze() > 0.01).astype(np.uint8)
    # xmin, ymin, xmax, ymax = measure.regionprops(measure.label(bbox_mask))[0].bbox
    xmin, ymin, xmax, ymax = 0, 0, 128, 128
    sub_mask = mask.squeeze()[xmin:xmax, ymin:ymax]
    contours = measure.find_contours(sub_mask, 0.2)
    alpha = np.clip(sub_mask + 0.6, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(x.squeeze()[xmin:xmax, ymin:ymax], alpha=alpha, cmap='gray', vmin=-1, vmax=1)
    axes[1].imshow(x_t.squeeze()[xmin:xmax, ymin:ymax], alpha=alpha, cmap='gray', vmin=-1, vmax=1)
    for contour in contours:
        axes[0].plot(contour[:, 1], contour[:, 0], c='pink', lw=3)
        axes[1].plot(contour[:, 1], contour[:, 0], c='pink', lw=3)
    for ax in axes:
        ax.axis('off')
    # # axes[0, 3].plot(report.normalized_mask_sizes[idx], report.score_changes[idx])
    # axes[3].set_title(f"quac score: {df.loc[idx]['quac_scores']:.2f}")
    plt.show()
    fig = plt.figure(dpi=300)
    plt.plot(report.interp_mask_values, interp_score_changes, color='k')
    plt.scatter([mask_size], [mask_score], c='r', marker='x')
    # plt.xlabel("Mask size")
    # plt.ylabel("Score change")
    # Hide spines on the right and top
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(0)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    plt.imshow(attr.squeeze(), cmap='seismic', vmax=1, vmin=-1)
    plt.gca().axis('off')
    plt.show()
    for thr in report.thresholds[idx][100::20]:
        mask, _ = evaluator.create_mask(attr, thr)
        hybrid = x_t * mask + x * (1.0 - mask)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.gca().axis('off')
        plt.show()
        plt.imshow(hybrid.squeeze(), cmap='gray', vmin=-1, vmax=1)
        plt.gca().axis('off')
        plt.show()



# %%
sub_df = df[(df["labels"] == 0) & (df["target_labels"] == 1)]
for i in sub_df.index[:1]:
    make_plot(i)
    plt.show()
    print(sub_df.loc[i]["quac_scores"])

# %%
for (label, target_label), sub_df in df.groupby(["labels", "target_labels"]):
    if target_label == 1:
        make_plot(sub_df.index[0])
# %%
sub_df = df[(df["labels"] == 4) & (df["target_labels"] == 5)]
for i in sub_df.index[0:5]:
    make_plot(i)
    plt.show()
    print(sub_df.loc[i]["quac_scores"])

# %%
blah = np.zeros((128, 128))
# blah = np.ones((128, 128))
blah[5, 5] = 1
blah[127, 127] = 1
# %%
# Gaussian Blur
blurred = cv2.GaussianBlur(blah, (11, 11), 0)
blurred.sum()

# %%
from skimage.filters import gaussian 

sk_blurred = gaussian(blah, sigma=11)
sk_blurred.sum()
# %%
