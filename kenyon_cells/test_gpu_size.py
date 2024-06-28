import torch
from quac.training.classification import ClassifierWrapper

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/synapse-classification/final_model.pt"
    )
    mean = 0.5
    std = 0.5

    assume_normalized = False
    classifier = ClassifierWrapper(
        classifier_checkpoint,
        mean=mean,
        std=std,
        assume_normalized=assume_normalized,
    )
    classifier.to(device)
    classifier.eval()

    # compute the GPU memory usage here
    dummy_input = torch.randn(3000, 1, 128, 128).to(device)
    print(f"Memory allocated before: {torch.cuda.memory_allocated(device)/1024**2} MB")
    dummy_output = classifier(dummy_input)
    print(f"Memory allocated after: {torch.cuda.memory_allocated(device)/1024**2} MB")
    print(dummy_output.shape)
