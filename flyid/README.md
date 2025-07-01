## Flies

2025-03-12 Wed: The confusion matrix for the classifier that I trained looks like this:

![[figures/full_confusion_matrix.png]]

Running a 20 by 20 conversion would be very expensive, so we decided to reduce the number of flies to run QuAC on to 3, as a proof-of-concept. 
The naive choice would be to run QuAC on flies 1, 2, 3.  Unfortunately, the 20-part model isn't actually that good at fly 2. 
I have therefore decided to run things on different flies. I will use flies that get 100% test accuracy: 11, 15, 17.

Therefore I have:
- re-organized my subset of the data so that it holds only those flies
- modified the classifier so that it maps {11: 0, 15: 1, 17: 2}
- run the StarGAN on these three flies
- run attribution correspondingly