# BBBC021 Dataset

## Data
113 compounds
8 different concentrations
12(?) mechanisms of action
> The differences between phenotypes were in some cases very subtle: we were only able to identify 6 of the 12 mechanisms visually;

> NOTE: When evaluating accuracy of MOA classification, it is critical to ensure that the cross-validation is set up correctly. MOA classification is the task of classifying the MOA of an unseen compound. Therefore, the evaluation should be a leave-one-compound-out cross validation: in each iteration, hold out one compound (all replicates and at all concentrations), train on the remaining, and test on the held out compound.

1. train a classifier on the 12
2. pick 3 or 4 -- classifier does good

## Classifier

I start by loading the dataset. To do this, I use the `LabelledDataset` class from `quac.training.data_loader`.
This is to make sure that the data is loaded and transformed in the same way as it will be during the training of the StarGAN model later.
Next, I use `timm` to create a model. I use a ResNet34 model to begin with.  The model is pre-trained on ImageNet, which might be a good starting point. I set the number of classes to 13. Although there are 12 MoAs, I also include DMSO as a control.

I use a simple training loop, using:
- Cross entropy loss
- Adam optimizer
- 10 epochs
- learning rate schedule: cosine annealing