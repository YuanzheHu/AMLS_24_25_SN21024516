import torch.nn as nn
from torchvision.models import resnet18

def get_resnet_model(num_classes=8):
    """
    Define a ResNet18 model with modifications for BloodMNIST.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Modified ResNet18 model.
    """
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify output layer
    return model