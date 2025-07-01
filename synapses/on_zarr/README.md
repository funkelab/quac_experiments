# On zarr experiment

Experiment started 2024-08-08. 
Experiment finished ???.
Experiment archived ???.

Background for this experiment - we noticed that the conversion rate obtained from the evaluation in the refactored code did not match the conversion rate that was expected, given what was computed during training. The current hypothesis for this is that the data being stored (as png files) is being clipped/normalized in an odd way, such that the classifier is being fed out-of-distribution data. In order to avoid this problem, I am repeating the prediction, generation, and attribution pipeline in a similar way to what was done for Kenyon cells -- storing things into Zarr rather than trying to use PNGs. 

Steps: 
- [x] Load the existing images, normalize them (-1, 1), run prediction, and store images, labels and predictions as zarr
- [x] Verify that the confusion matrix looks as expected
- [ ] Run generation and store counterfactuals and their corresponding predictions as zarr
- [ ] Verify that the counterfactual confusion matrix looks as expected
