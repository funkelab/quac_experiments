import torch
import torchvision.models as models

num_classes = 20
select_classes = [
    11,
    15,
    17,
]  # Note, these are 1-indexed, classifier will be 0-indexed


class SelectionLayer(torch.nn.Module):
    def __init__(self, select_classes):
        super(SelectionLayer, self).__init__()
        self.selected = select_classes

    def forward(self, x):
        return x[:, self.selected]


class GrayscaleToRGBLayer(torch.nn.Module):
    """
    A layer to convert grayscale images (1 channel) to RGB (3 channels).
    This layer assumes the input is of shape (N, 1, H, W) or (N, H, W) for grayscale images,
    and outputs a tensor of shape (N, 3, H, W) by repeating the single channel three times.
    If the input is already in RGB format (3 channels), it simply returns the input unchanged.

    This is useful for pre-trained models that expect RGB input but the data is in grayscale.
    """

    def __init__(self):
        super(GrayscaleToRGBLayer, self).__init__()

    def forward(self, x):
        # If the input is of shape (N, H, W), add a channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Now, assuming x is of shape (N, 1, H, W)
        if x.size(1) == 1:
            return x.repeat(
                1, 3, 1, 1
            )  # Repeat the single channel to create 3 channels
        return x  # If already RGB, return as is


if __name__ == "__main__":
    input_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/flyid/checkpoints/retry_model_best.pth.tar"
    )
    output_checkpoint = "/nrs/funke/adjavond/projects/quac/flyid/final_classifier_jit_selected_classes_grayscale.pth"

    checkpoint = torch.load(input_checkpoint)
    arch = checkpoint["arch"]
    print(f"Architecture: {arch}")
    print(f"Best prec@1: {checkpoint['best_prec1']:.3f}")
    print(f"This is epoch {checkpoint['epoch']}")

    print("Loading model...")
    model = models.__dict__[arch]()
    if arch == "resnet18" or arch == "resnet34":
        model.fc = torch.nn.Linear(512, num_classes)
    elif arch == "resnet50":
        model.fc = torch.nn.Linear(2048, num_classes)
        print("Last FC layer changed: ")
        print(model.fc)

    # model was DDP, so need to make it as such
    assert torch.cuda.is_available(), "CUDA is not available"

    # Just get the module from the DataParallel
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint["state_dict"])

    # Wrap the module so that it only outputs the selected classes
    model.module = torch.nn.Sequential(
        GrayscaleToRGBLayer(),
        model.module,
        # add the selection
        SelectionLayer(torch.tensor(select_classes) - 1),
    )

    # JIT
    torch.jit.save(torch.jit.script(model.module), output_checkpoint)
