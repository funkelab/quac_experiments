# %%
from matplotlib import pyplot as plt
from quac.experiments.cmnist import ColoredMNIST
from pathlib import Path

# %%
dataset = ColoredMNIST(root="/nrs/funke/adjavond/data/ColoredMNIST", 
                       train=True, download=True,
                       classes=["spring", "winter"])
# %%
save_dir = Path("example")
for source in dataset.classes:
    (save_dir / "originals" / source).mkdir(exist_ok=True, parents=True)
    for target in dataset.classes:
        if source == target:
            continue
        (save_dir / "counterfactuals"/ source / target).mkdir(exist_ok=True, parents=True)

# %%
for i in range(500):
    x, y = dataset[i]
    condition = (y + 1) % 2
    cf, ycf = dataset.get_counterfactual(i, condition)

    assert y != ycf
    source = dataset.classes[y]
    target = dataset.classes[ycf]

    plt.imsave(save_dir / f"originals/{source}/{i}.png", x.permute(1, 2, 0).cpu().numpy())
    plt.imsave(save_dir / f"counterfactuals/{source}/{target}/{i}.png", cf.permute(1, 2, 0).cpu().numpy())

# %%
