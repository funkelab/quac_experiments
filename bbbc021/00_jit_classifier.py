"""
This script loads a PyTorch model checkpoint, selects specific classes from the model's output,
and saves the modified model as a JIT script. The selected classes are specified in a JSON file.
"""

import json
import logging
import timm
import torch


class SelectionLayer(torch.nn.Module):
    def __init__(self, select_classes):
        super(SelectionLayer, self).__init__()
        self.selected = select_classes

    def forward(self, x):
        return x[:, self.selected]


def main(
    input_checkpoint: str,
    output_checkpoint: str,
    selected_class_file: str,
    model_name: str = "resnet34",
    num_classes: int = 13,
):
    print(f"Loading checkpoint from {input_checkpoint}")
    backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    # Load the checkpoint
    checkpoint = torch.load(input_checkpoint, map_location="cpu")
    backbone.load_state_dict(checkpoint)

    print("Selecting classes.")
    # Read selected classes from a file
    with open(selected_class_file, "r") as f:
        selected_class_dict = json.load(f)

    selected_classes = list(selected_class_dict.values())

    print("Creating selection layer.")
    # Create the model with selection layer
    model = torch.nn.Sequential(
        backbone,
        # add the selection
        SelectionLayer(torch.tensor(selected_classes)),
    )

    print("Saving model.")
    # JIT
    torch.jit.save(torch.jit.script(model), output_checkpoint)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_checkpoint = "/nrs/funke/adjavond/projects/quac/bbbc021/classifier/eternal-shadow-25/model_epoch_49.pth"
    output_checkpoint = (
        "/nrs/funke/adjavond/projects/quac/bbbc021/classifier/selected_classes.pt"
    )
    selected_class_file = "classifier/class_subset.json"

    main(
        input_checkpoint=input_checkpoint,
        output_checkpoint=output_checkpoint,
        selected_class_file=selected_class_file,
        model_name="resnet34",
        num_classes=13,
    )
