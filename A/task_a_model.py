import torch
import torch.nn as nn
import torch.nn.functional as F

class BreastMNISTCNN(nn.Module):
    """
    CNN model for BreastMNIST binary classification.
    """

    def __init__(self, input_channels=1, num_classes=2):
        """
        Initialize the CNN model.

        Args:
            input_channels (int): Number of input channels (default: 1 for grayscale images).
            num_classes (int): Number of output classes (default: 2 for binary classification).
        """
        super(BreastMNISTCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 128)  # Assuming input image size is 28x28
        self.fc2 = nn.Linear(128, num_classes)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Define the forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test the model
    model = BreastMNISTCNN()
    print(model)
    dummy_input = torch.randn(4, 1, 28, 28)  # Batch size 4, grayscale image 28x28
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")