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


# %%
# Load the classifier
classifier = ClassifierWrapper(
    "/nrs/funke/adjavond/projects/quac/retina-ddr-classification/final_model.pt",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
# %%
root_folder = Path("/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/counterfactuals/retina_stargan/reference/val")

classes = {
    "0_No_DR": 0,
    "1_Mild": 1,
    "2_Moderate": 2,
    "3_Severe": 3,
    "4_Proliferative_DR": 4
}

for source in classes:
    source_folder = root_folder / source

    # source_folder = Path("/nrs/funke/adjavond/data/retina/ddrdataset/DR_grading/DR_grading_processed/val")


    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)  # because of the assumption of the ClassifierWrapper
    ])

    dataset = ImageFolder(source_folder, transform=transform)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

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
    ax = sns.heatmap(confusion_matrix(true, preds, normalize='true', labels=[0, 1, 2, 3, 4]), annot=True)
    ax.set_xticklabels(list(classes.keys()))
    ax.set_yticklabels(list(classes.keys()), rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(source)
    plt.show()

# %%

# %%
