# %%
from quac.training.classification import ClassifierWrapper
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

save_dir = Path("data_description")

# %%
# Load the classifier
classifier = ClassifierWrapper(
    "/nrs/funke/adjavond/projects/quac/retina-ddr-classification/final_model.pt",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
# %%
root_folder = Path(
    "/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/counterfactuals/retina_stargan/reference/val"
)

classes = {
    "0_No_DR": 0,
    "1_Mild": 1,
    "2_Moderate": 2,
    "3_Severe": 3,
    "4_Proliferative_DR": 4,
}

values_for_zero = {
    "0_No_DR": 0,
    "1_Mild": 0,
    "2_Moderate": 0,
    "3_Severe": 0,
    "4_Proliferative_DR": 0,
}

for source in classes:
    source_folder = root_folder / source

    # source_folder = Path("/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/DR_grading_processed/val")

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0.5, std=0.5
            ),  # because of the assumption of the ClassifierWrapper
        ]
    )

    dataset = ImageFolder(source_folder, transform=transform)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=False
    )

    preds = []
    true = []
    for x, y in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            logits = classifier(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        actual_y = np.array([classes[idx_to_class[i.item()]] for i in y.cpu()])
        true.append(actual_y)

    preds = np.concatenate(preds)
    true = np.concatenate(true)

    print(classification_report(true, preds))
    cm = confusion_matrix(true, preds, normalize="true", labels=[0, 1, 2, 3, 4])
    ax = sns.heatmap(
        cm,
        annot=True,
    )
    source_index = classes[source]
    values_for_zero[source] = cm[0, 0]  # fraction properly classified as 0
    ax.set_xticklabels(list(classes.keys()))
    ax.set_yticklabels(list(classes.keys()), rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(source)
    plt.show()

# Store the values for the zero class
with open(save_dir / "values_for_zero.csv", "w") as f:
    f.write("source,value\n")
    for source, value in values_for_zero.items():
        f.write(f"{source},{value}\n")
    # Remove the last newline
    f.seek(f.tell() - 1)
    f.truncate()

# %% Also just run basic classification on the original dataset
root_folder = Path(
    "/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/DR_grading_processed/val"
)
dataset = ImageFolder(
    root_folder,
    transform=transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0.5, std=0.5
            ),  # because of the assumption of the ClassifierWrapper
        ]
    ),
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=False, drop_last=False
)

preds = []
true = []
for x, y in tqdm(dataloader):
    with torch.no_grad():
        logits = classifier(x)
    preds.append(logits.argmax(dim=1).cpu().numpy())
    true.append(y.cpu().numpy())

preds = np.concatenate(preds)
true = np.concatenate(true)

print(classification_report(true, preds))
cm = confusion_matrix(true, preds, normalize="true", labels=[0, 1, 2, 3, 4])
ax = sns.heatmap(cm, annot=True)
# ax.set_xticklabels(list(classes.keys()))
# ax.set_yticklabels(list(classes.keys()), rotation=0)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
# Save the confusion matrix for source data
with open(save_dir / "confusion_matrix_source.csv", "w") as f:
    f.write("label,prediction,value\n")
    for target in range(5):
        for source in range(5):
            f.write(f"{source},{target},{cm[source, target]}\n")
    # Remove the last newline
    f.seek(f.tell() - 1)
    f.truncate()
# %% Plot an example of a class 0 image
import matplotlib.pyplot as plt
import numpy as np

# Get a dataset, no normalization
dataset = ImageFolder(
    root_folder,
    transform=transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ]
    ),
)

img, label = dataset[0]
assert label == 0
plt.imshow(img)

# %% Save image
save_dir.mkdir(exist_ok=True)
img.save(save_dir / "class_0_example.png")

# %%
