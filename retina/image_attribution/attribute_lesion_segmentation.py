# %% [markdown]
# # Image Attribution with QuAC
# In this notebook, we will be using the discriminative attribution methods that exist in QuAC to run attribution on some retina DDR data where we have lesion segmentations available.
#
# To simplify this, we have a YAML file that contains the configuration information.
# This is mostly a listing of where the data is for the source images, counterfactual images, and where to put attribution data.
#
# %% Getting the metadata
from yaml import safe_load

metadata = safe_load(open("lesion_segmentation.yaml"))

# %% [markdown]
# ## Getting the clasifier
# The first step is to get the classifier that we will be using for attribution.
# In this case, the classifier is a ResNet50 that has been trained on the retina DDR dataset.
# It was pre-trained with ImageNet weights, so it expects images to be normalized with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).
# %%
from quac.generate import load_classifier

classifier = load_classifier(
    metadata["classifier_checkpoint"], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)
# %% [markdown]
# ## Preparing the attribution parts
# Attributions in QuAC are found in `quac.attribution`.
# There is also a convenience class `AttributionIO` that will handle the running of the attributions and storing the results.
# `AttributionIO` takes in a dictionary of attributions and an output directory.
# %% Creating the attributions
# Defining attributions
from quac.attribution import (
    DDeepLift, 
    DIntegratedGradients, 
    AttributionIO, 
    VanillaDeepLift, 
    VanillaIntegratedGradient
)
from torchvision import transforms

attributor = AttributionIO(
    attributions={
        "discriminative_deeplift": DDeepLift(classifier),
        "discriminative_ig": DIntegratedGradients(classifier),
        "vanilla_deeplift": VanillaDeepLift(classifier),
        "vanilla_ig": VanillaIntegratedGradient(classifier),
    },
    output_directory=metadata["attribution_directory"],
)
# %% [markdown]
# ## Running the attributions
# To run the attributions, we need to make sure that the data is in the format that is expected.
# In general in QuAC, we assume image data will be normalized with mean 0.5 and standard deviation 0.5.
# The classifier wrapper generated with `load_classifier` will handle re-normalization, if it is necessary.
# In this case, we will also resize the images to 224x224, because that is what is expected of the retina classifier.
# %%
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)
# %% [markdown]
# The next cell will run attributions and store all of the results in the output_directory
# It also shows a progress bar, so you can see how far along it is.
# Note that the more attribution methods you have, the longer this will take.
# This is because most of the attribution methods go through a full backpropgation pass for each image.

# %%
attributor.run(
    source_directory=metadata["source_directory"],
    counterfactual_directory=metadata["counterfactual_directory"],
    transform=transform,
)

# %% [markdown]
# After this, you're done! Your attributions should be in the directory specified in the metadata file.
!ls -al {metadata["attribution_directory"]}
# %%
