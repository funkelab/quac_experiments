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


if __name__ == "__main__":
    input_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/flyid/checkpoints/retry_model_best.pth.tar"
    )
    output_checkpoint = "/nrs/funke/adjavond/projects/quac/flyid/final_classifier_jit_selected_classes.pth"

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
        model.module,
        # add the selection
        SelectionLayer(torch.tensor(select_classes) - 1),
    )

    # JIT
    torch.jit.save(torch.jit.script(model.module), output_checkpoint)
