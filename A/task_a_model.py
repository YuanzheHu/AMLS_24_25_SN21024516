import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC

class BreastMNISTSVM:
    """
    Support Vector Machine (SVM) for BreastMNIST binary classification.
    """
    def __init__(self, C=1.0, kernel='linear'):
        """
        Initialize the SVM model.

        Args:
            C (float): Regularization parameter.
            kernel (str): Kernel type ('linear', 'rbf', etc.).
        """
        self.model = SVC(C=C, kernel=kernel, probability=True)

    def fit(self, X_train, y_train):
        """
        Train the SVM model.

        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict labels using the trained SVM model.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the trained SVM model.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted probabilities.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Calculate the accuracy of the model.

        Args:
            X (numpy.ndarray): Test features.
            y (numpy.ndarray): True labels.

        Returns:
            float: Accuracy score.
        """
        return self.model.score(X, y)

class BreastMNISTCNN(nn.Module):
    """
    CNN model for BreastMNIST binary classification with dropout regularization.
    """

    def __init__(self, input_channels=1, num_classes=2, hidden_units=128, dropout=0.5):
        """
        Initialize the CNN model.

        Args:
            input_channels (int): Number of input channels (default: 1 for grayscale images).
            num_classes (int): Number of output classes (default: 2 for binary classification).
            hidden_units (int): Number of units in the first fully connected layer.
            dropout (float): Dropout probability for regularization (default: 0.5).
        """
        super(BreastMNISTCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, hidden_units)  # Adjust according to hidden_units
        self.fc2 = nn.Linear(hidden_units, num_classes)

        # Pooling, dropout, and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)

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
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
