from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    suffix = "_selected_weights"
    mu = 0.5
    sigma = 0.5
    num_classes = 3

    # suffix = ""
    # mu = 0.73
    # sigma = 0.18

    test_dir = "/nrs/funke/adjavond/data/flyid/week1_limited/Day2/val"
    checkpoint = (
        f"/nrs/funke/adjavond/projects/quac/flyid/final_classifier_jit{suffix}.pth"
    )
    batch_size = 64
    workers = 12

    # Data loading code
    print("loading test data....")

    # transformations for testing
    trans_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[mu, mu, mu], std=[sigma, sigma, sigma]),
        ]
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, trans_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    print("Loading model...")
    model = torch.jit.load(checkpoint)
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        confusion_matrix = torch.zeros(num_classes, num_classes)
        for input, target in tqdm(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            # print(output.shape)
            # print(output[0])
            preds = torch.argmax(output, dim=1)
            # print(preds[0])
            # Add to confusion matrix
            for t, p in zip(target, preds):
                confusion_matrix[t.long(), p.long()] += 1
    print(sum(confusion_matrix.diag()) / confusion_matrix.sum())

    # # store the confusion matrix as CSV
    np.savetxt(f"confusion_matrix{suffix}.csv", confusion_matrix.numpy(), delimiter=",")


if __name__ == "__main__":
    main()
