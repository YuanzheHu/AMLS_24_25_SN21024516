import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet_model(num_classes=8):
    """
    Define a ResNet18 model with modifications for BloodMNIST.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Modified ResNet18 model.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify output layer
    return model
