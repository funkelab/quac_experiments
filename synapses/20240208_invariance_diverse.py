# %%
from funlib.learn.torch.models import Vgg2D
import torch
from tqdm import tqdm
from quac.data import PairedImageDataset
from torchvision import transforms

# %% Classifier
classifier = Vgg2D(input_size=(128, 128), output_classes=6, fmaps=12)
checkpoint = torch.load(
    "/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint"
)["model_state_dict"]
classifier.eval()
classifier.load_state_dict(checkpoint)

# %% Data
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
dataset = PairedImageDataset(
    "/nrs/funke/adjavond/data/synapses/val/",
    "/nrs/funke/adjavond/data/synapses/counterfactuals/stargan_invariance_v0_diverse/val/",
    transform=transform,
)

# %% Run classification

correct = 0
for input in tqdm(dataset):
    x_t = input["target_sample"]
    y_t = input["target_class_index"]

    pred_t = classifier(x_t.unsqueeze(0))
    y_pred = torch.argmax(pred_t, dim=1)
    correct += int(y_pred == y_t)
print(f"Accuracy: {correct / len(dataset)}")

# %%
