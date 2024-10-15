# Fictus
The `fictus.aggregatum` dataset is a synthetic dataset built to simulate light microscopy. 
We created this dataset to have an example where we know the features of variation exactly, so that we could evaluate the QuAC paradigm.

## Classifier
We first train a classifier on the dataset (`00_train_classifier.py`). 
The configuration used for training can be found in `configs/resnet18_20241013.yml`.
We used a ResNet18 from `timm` - we load weights pre-trained on ImageNet.
Images have values from -1 to 1, and we apply random flips, rotations, and a small amount of gaussian noise as augmentations.
The model is trained for 10 epochs on an NVIDIA TITAN Xp in one hour. 
The best iteration of the model is epoch 10 with validation accuracy 98.34%.

We use `torchscript` to obtain a JIT-ed version of the model so that it can be more easily loaded. 
We verify that this does not affect accuracy of the best model(`01_eval_classifier.py`).
The checkpoints, and the final model can be found in `/nrs/funke/adjavond/projects/quac/fictus/classification_20241013`.

## StarGANs
We next train a StarGANs to convert between image types (`02_train_stargan.py`).

### Default StarGAN - no diversity
The configuration used for training can be found in `configs/stargan_20241013.yml`.
The images values range from -1 to 1; we do not apply any augmentations.
We do not use the diversity loss (`lambda_ds=0`).
The model is trained for 10000 iterations on an NVIDI A100 GPU. 
Every 1000 iterations, translation rate and conversion rate are evaluated using the previously trained classifier, the validation dataset, and 10 tries per sample. 
The model weights are also checkpointed at this time.
The conversion rate reaches 1 for all conversions, so we use the translation rates to choose the iterations to keep. 
The iteration with the highest average translation rates is iteration 9000 for both latent-based (0.9852) and reference-based (1.) generation.
We use this iteration checkpoints to generate counterfactuals. 

### Second Stargan - some diversity
The configuration used for training can be found in `configs/stargan_20241014.yml`.
The images values range from -1 to 1; we do not apply any augmentations.
We use the diversity loss (`lambda_ds=0.5`), and keep it relatively high throughout the run (`ds_iter=100000`).
The model is trained for 10000 iterations on an NVIDIA A100 GPU. 
Every 1000 iterations, translation rate and conversion rate are evaluated using the previously trained classifier, the validation dataset, and 10 tries per sample. 
The model weights are also checkpointed at this time.
The conversion rate reaches 1 for all conversions, so we use the translation rates to choose the iterations to keep. 
The iteration with the highest average translation rate is iteration 8000 (latent-based 0.96667, reference-based 0.99792; we averaged over both latent and reference). 
We use this iteration checkpoints to generate counterfactuals. 

### Third Stargan - more diversity
The configuration used for training can be found in `configs/stargan_20241014.yml`.
The images values range from -1 to 1; we do not apply any augmentations.
We use the diversity loss (`lambda_ds=0.9`), and keep it relatively high throughout the run (`ds_iter=100000`).
The model is trained for 10000 iterations on an NVIDI A100 GPU. 
Every 1000 iterations, translation rate and conversion rate are evaluated using the previously trained classifier, the validation dataset, and 10 tries per sample. 
The model weights are also checkpointed at this time.
The conversion rate reaches 1 for all conversions, so we use the translation rates to choose the iterations to keep. 
The iteration with the highest average translation rate is iteration 6000 (latent-based 0.9895, reference-based 1.0; we averaged over both latent and reference). 
We use this iteration checkpoints to generate counterfactuals. 

## Image Conversion
Converted images are generated using both latents and references (`03_generate.py`).
Since the translation rate is close to one for this experiment, we generate 2 images per input.
To avoid issues related to changing data types, and storing to and from image file formats, we store the generated images directly as output by the model, in a Zarr file.
The generated images are grouped by their source-target pair -- the arrays are named `{source}_{target}`, where `source` and `target` correpond to class indices. 
These class indices are the same as the class indices in the classifier output. 
They are stored as `float32`, with dimensions `(index, batch, channel, y, x)` where `index` is the index of the sample within a class, `batch` is the set of generated images per input, `channel, y, x` should match those of the real images.
We simultaneously classify the generated images.
The latent-based generation reaches an average conversion rate of [Cl], with a range of [Cl1, Cl2]. 
The reference-based generation reaches an average conversion rate of [Cr], with a range of [Cr1, Cr2].
The predictions are stored in the same Zarr file.

## Ideation
Experiment/generation-type/run: 
- images
    - 0
    - 1
    - 2
- predictions
    - 0
    - 1
    - 2
- Standard StarGAN encoder - 2024-10-13
    - latent
        - Generated
            - 0_1 (index, batch, channel, y, x)
            - 0_2
            - ...
        - Predictions
            - 0_1 (index, batch, num_classes)
            - 0_2
            - ...
        - Attributions
            - Method 1
                - attributions
                    - 0_1 (index, batch, channel, y, x)
                    - 0_2
                - threshold, mask-size, delta-f (index, batch, evaluation-length, 3)
                    - 0_1
                    - 0_2 
                - scores (index, batch)
                    - 0_1
                    - 0_1
                - optimal_mask (index, batch, channel, y, x) -- unsmoothed mask
            - Method 2
            - Method 3
            - Method 4
    - reference
        - ...
