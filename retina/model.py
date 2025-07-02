import copy
from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights
from torch import nn
import torch


class PretrainedVGG16(nn.Module):
    def __init__(self, num_classes=5, weights=VGG16_Weights.IMAGENET1K_FEATURES):
        super().__init__()
        model = vgg16(weights=weights)
        self.features = copy.deepcopy(model.features)
        self.avgpool = copy.deepcopy(model.avgpool)
        self.classifier = copy.deepcopy(model.classifier)
        self.classifier[6] = nn.Linear(4096, num_classes)
        # Initialize the weights of the last layer with normal distribution
        # nn.init.normal_(self.classifier[6].weight, 0, 0.01)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PretrainedResnet18(nn.Module):
    def __init__(self, num_classes=5, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        model = resnet18(weights=weights)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        fmaps = model.fc.in_features
        self.fc = nn.Linear(fmaps, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
