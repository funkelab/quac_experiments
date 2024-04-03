# %% [markdown]
# # Evaluate the attributions 
# In this notebook we will evaluate the attributions that we got in [[attribute_lesion_segmentation.py]]
# %% [markdown]
# ## Getting the metadata
# Once again, we use the YAML file to get the basics
# %% Getting the metadata
from yaml import safe_load

metadata = safe_load(open("lesion_segmentation.yaml"))
# %% [markdown]
# We will be running on the DeepLift attributions, so we will need to change the metadata.
# To run this on the other attributions we generated, just change the "attribution_method" below.

# %% Choose the attribution methoe
attribution_method = "discriminative_deeplift"
metadata["attribution_directory"] = metadata["attribution_directory"] + "/" + attribution_method

# %% [markdown]
# ## Loading the classifier
# We will need the same classifier as we were using to run attributions. 
# As a reminder: the classifier is a ResNet50 that has been trained on the retina DDR dataset.
# It was pre-trained with ImageNet weights, so it expects images to be normalized with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).
from quac.generate import load_classifier

classifier = load_classifier(
    metadata["classifier_checkpoint"], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)
# %% [markdown]
# Once again, we will need to define the transform that we will use to process the images.
# This transform will be used on the counterfactuals and the source images. 
# Make sure that there is no randomness in the transform, or the counterfactuals will not match the source images.
# %%
# Define the transform
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)
# %% [markdown]
# ## Defining the evaluator
# The evaluator is the main class that will be used to evaluate the attributions.
# It needs to be pointed to all of the different data directories we will be using.

# %% Define the evaluator
from quac.evaluation import Evaluator
evaluator = Evaluator(
    classifier,
    source_directory=metadata["source_directory"],
    counterfactual_directory=metadata["counterfactual_directory"],
    attribution_directory=metadata["attribution_directory"],
    transform=transform
)

# %% [markdown]
# ## Running classification of the counterfactuals
# One useful function of the evaluator is that it lets us check how good the counterfactuals are.
# 
# Note: If you a re-running this notebook on a new attribution method, comment the following cell out to save time! 
# %%
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

cf_confusion_matrix = evaluator.classification_report(
                        data="counterfactuals",  # this is the default
                        return_classification=False,
                        print_report=False,
                    )

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cf_confusion_matrix, 
)
disp.plot()
plt.show()

# %% [markdown]
# ## Defining the processor and quantifying the attributions
# In order to get masks from attributions, we need to process them a little. 
# Here we define a processor for the evaluator to use. 
# You can make a custom `Processor` class if your attribution requires more or less processing than is default.
# 
# Given a processor, the evaluator runs quantification and returns a `Report`.
# %%
from quac.evaluation import Processor

report = evaluator.quantify(processor=Processor())
# %% [markdown]
# ## Plotting the report
# The report can be used to plot the QuAC curve. 
# %%
report.plot_curve()
# %% [markdown]
# ## Storing the report
# Finally, the report can be stored for later use.
# The report will be stored based on the processor's name, which is "default" by default
# %%
report.store(metadata["report_directory"])

# %% [markdown]
# That's it! Feel free to run this again with a different attribution method by changing the `attribution_method` variable above.