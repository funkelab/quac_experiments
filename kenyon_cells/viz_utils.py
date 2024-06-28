from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.colors import rgb2hex
import numpy as np
import seaborn as sns

colors = sns.color_palette("tab10", n_colors=6)
classes = [
    "GABA",
    "ACh",
    "Glu",
    "Ser",
    "Oct",
    "Dop",
]


def plot_with_bar(image, prediction, axes, title="", mask=None, threshold=0.01):
    pred_ax, im_ax = axes
    pred_ax.bar(np.arange(6), prediction, color=colors, alpha=0.6, edgecolor="gray")
    class_idx = np.argmax(prediction)
    pred_ax.set_ylim(0, 1)
    # pred_ax.grid(axis="y")
    pred_ax.set_yticks([0, 1], labels=["0", "1"], color="gray")
    pred_ax.tick_params(axis="y", colors="gray")
    pred_ax.set_xticks([])
    # add the name on top of the bar
    pred_ax.text(
        class_idx,
        prediction[class_idx],
        classes[class_idx],
        ha="center",
        va="bottom",
        fontsize=12,
        color=colors[class_idx],
        weight="bold",
    )
    sns.despine(ax=pred_ax)  # , left=True)
    pred_ax.spines["left"].set_color("gray")
    pred_ax.spines["bottom"].set_color("gray")
    im_ax.imshow(image.squeeze(), cmap="gray")
    im_ax.imshow(image.squeeze(), cmap="gray")
    im_ax.set_title(title, fontsize=14, y=-0.1)
    im_ax.axis("off")
    color_idx = np.argmax(prediction)
    if mask is not None:
        im_ax.contour(
            mask.squeeze(),
            levels=[threshold],
            colors=rgb2hex(colors[color_idx]),
            linewidths=4,
        )
        # Plot the mask on top of the hybrid
        plot_mask = mask.squeeze().copy()
        plot_mask = np.ma.masked_where(
            mask.squeeze() > threshold, np.ones_like(plot_mask)
        )
        im_ax.imshow(plot_mask, alpha=0.6, cmap="gray_r", vmin=0, vmax=1)


def plot_with_text(
    image, prediction, im_ax, target, source=None, mask=None, threshold=0.01, title=""
):
    kw = {"fontsize": 14}
    # fontsize = 14
    anchorpad = 0
    # Create the title
    if source is None:
        source = np.argmax(prediction)
    list_of_strings = [
        classes[source],
        f" {prediction[source]:.2f}",
        # "→",
        "|",
        classes[target],
        f" {prediction[target]:.2f}",
    ]
    list_of_colors = [colors[source], "black", "black", colors[target], "black"]
    boxes = [
        TextArea(text, textprops=dict(color=color, ha="left", va="bottom", **kw))
        for text, color in zip(list_of_strings, list_of_colors)
    ]
    xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
    anchored_xbox = AnchoredOffsetbox(
        loc=3,
        child=xbox,
        pad=anchorpad,
        frameon=False,
        bbox_to_anchor=(0.07, 1.0),
        bbox_transform=im_ax.transAxes,
        borderpad=0.0,
    )
    im_ax.add_artist(anchored_xbox)

    im_ax.imshow(image.squeeze(), cmap="gray")
    im_ax.set_title(title, fontsize=14, y=-0.1)
    im_ax.axis("off")
    color_idx = np.argmax(prediction)
    if mask is not None:
        im_ax.contour(
            mask.squeeze(),
            levels=[threshold],
            colors=rgb2hex(colors[color_idx]),
            linewidths=4,
        )
        # Plot the mask on top of the hybrid
        plot_mask = mask.squeeze().copy()
        plot_mask = np.ma.masked_where(
            mask.squeeze() > threshold, np.ones_like(plot_mask)
        )
        im_ax.imshow(plot_mask, alpha=0.6, cmap="gray_r", vmin=0, vmax=1)
