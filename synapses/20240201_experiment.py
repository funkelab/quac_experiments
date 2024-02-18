# %%
from funlib.learn.torch.models import Vgg2D
from matplotlib import pyplot as plt
from quac.data import PairedImageFolders
from quac.attribution import DIntegratedGradients, DDeepLift, DInGrad
from quac.evaluation import Evaluator
from quac.report import Report
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Subset
# %%
# Set up device, only GPU 1 is worth using
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# Where to store results
save_dir = "/nrs/funke/adjavond/projects/quac/20240201"

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
# TODO Remove this line
# dataset = Subset(dataset, range(10))
#
classifier = Vgg2D(input_size=(128, 128), output_classes=6, fmaps=12)
checkpoint = torch.load("/nrs/funke/adjavond/checkpoints/synapses/classifier/vgg_checkpoint")["model_state_dict"]
classifier.eval()
classifier.load_state_dict(checkpoint)
evaluator = Evaluator(classifier)

# %%
# Specific to each attribution
methods = {
	"discriminative_ig": DIntegratedGradients(classifier),
    "discriminative_dl": DDeepLift(classifier),
    "discriminative_ingrad": DInGrad(classifier)
}
reports = {name: Report(save_dir, name=name) for name in methods}

# %% Run attribution and evaluate
for input in tqdm(dataset):
    x = input["sample"]
    y = input["class_index"]
    x_t = input["target_sample"]
    y_t = input["target_class_index"]
    # 
    for method, attributor in methods.items():
        # Get the classification of the original and the counterfactual
        predictions = {
            "original": evaluator.run_inference(x)[0],
            "counterfactual": evaluator.run_inference(x_t)[0]
        }
		# Run attribution
        attribution = attributor.attribute(x, x_t, y, y_t)
        # Evaluate attribution
        results = evaluator.evaluate(x, x_t, y, y_t, attribution, predictions)
        # Store results
        reports[method].accumulate(input, predictions, attribution, results, save_intermediates=False)


# %%
# Plot results
fig, axes = plt.subplots(1, len(reports), figsize=(15, 5))
for ax, (method, report) in zip(axes, reports.items()):
    report.plot_curve(ax)
    ax.set_title(method)
plt.show()
# %%
# Save results
for method, report in reports.items(): 
	report.store()

# %% 
# Try loading the reports
loaded_reports = {name: Report(save_dir, name=name) for name in methods}
fig, axes = plt.subplots(1, len(loaded_reports), figsize=(15, 5))
for ax, (method, report) in zip(axes, loaded_reports.items()):
    report.load( Path(save_dir) / f"{method}.json")
    report.plot_curve(ax)
# %%
a = report.load_attribution(0)
x = dataset[0]["sample"]

plt.imshow(x.squeeze(), cmap="gray")
plt.imshow(a.squeeze(), alpha=0.5, cmap="seismic", vmax=1, vmin=-1)

# %%
