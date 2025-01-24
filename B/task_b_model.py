import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

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

class EarlyStopping:
    """ 
    Early stopping utility to stop training when validation loss stops improving.
    """
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def get_vit_model(num_classes=8):
    """
    Define a Vision Transformer (ViT) model with modifications for BloodMNIST.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Modified ViT model.
    """
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)  # Modify output layer
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the ResNet model
    resnet_model = get_resnet_model().to(device)

    # Show the architecture of the ResNet model using torchinfo
    resnet_summary = summary(resnet_model, input_size=(1, 3, 28, 28))

    # Write the ResNet summary to a file
    with open('models/resnet_summary.txt', 'w') as f:
        f.write(str(resnet_summary))

    # Create an instance of the ViT model
    vit_model = get_vit_model().to(device)

    # Show the architecture of the ViT model using torchinfo
    vit_summary = summary(vit_model, input_size=(1, 3, 224, 224))

    # Write the ViT summary to a file
    with open('models/vit_summary.txt', 'w') as f:
        f.write(str(vit_summary))