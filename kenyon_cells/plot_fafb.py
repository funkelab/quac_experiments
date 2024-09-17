# %%
import navis
import flybrains
from fafbseg import flywire
from tqdm import tqdm
import time
import pandas as pd

# %%
syn_class_confidences = pd.read_csv("syn_class_with_conf_nt.csv")
# Get a subset/sample of the root_ids
syn_class_confidences = syn_class_confidences.sample(500)
# Get skeletons from flywire
sk = flywire.get_skeletons(
    flywire.NeuronCriteria(
        root_id=syn_class_confidences.pre.to_list(),
        materialization=783,
    )
)
# Only keep the ones that are in both the skeletons and the synapse data
in_both = [neuron for neuron in sk if neuron.id in syn_class_confidences["pre"].values]
# Set the "pre" column as the index
sy = syn_class_confidences.set_index("pre")

# %%
classes = [
    "gaba",
    "acetylcholine",
    "glutamate",
    "serotonin",
    "octopamine",
    "dopamine",
    "unknown",
]

# Use the color palette from the paper
color_palette = [
    "#834D9D",  # funkey_color_1 purple
    "#F2A431",  # funkey_color_2 orange
    "#55B849",  # funkey_color_3 green
    "#DB8457",  # funkey_color_4 peach
    "#8174B1",  # funkey_color_5 lavender
    "#ADD8E6",  # funkey_color_6 aquamarine
    "#666666",  # funkey_gray gray
]

neurons = []
colors = []
n_samples = len(in_both)
for neuron in tqdm(in_both[:n_samples], total=n_samples):
    neurons.append(neuron)
    colors.append(color_palette[classes.index(sy.loc[neuron.id, "conf_nt"])])


neurons.append(flybrains.FLYWIRE)
colors.append((0.3, 0.3, 0.3, 0.1))
# %% Plot
t0 = time.perf_counter()
fig, ax = navis.plot2d(
    neurons,
    colors=colors,
    view=("x", "-y"),
    method="2d",
)
# ax.azim, ax.elev = 90, 270
ax.dist = 1
# legend_patches = [
#     mpatches.Patch(color=color, label=class_) for class_, color in zip(classes, palette)
# ]
# plt.legend(
#     handles=legend_patches,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.25),
#     ncol=3,
#     frameon=False,
#     prop={"family": "Palatino", "size": 14},
# )
fig.tight_layout()
fig.savefig("kenyon_cells.png", dpi=300)

print(time.perf_counter() - t0)

# %%
