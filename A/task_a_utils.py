import os
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import BreastMNIST
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

    
def load_breastmnist(batch_size=32, download=True, data_dir="data"):
    """
    Load the BreastMNIST dataset without data augmentation and save to a specified directory.

    Args:
        batch_size (int): Batch size for DataLoader.
        download (bool): Whether to download the dataset.
        data_dir (str): Directory to save the dataset.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Simple normalization for all datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_dataset = BreastMNIST(split='train', transform=transform, download=download, root=data_dir)
    val_dataset = BreastMNIST(split='val', transform=transform, download=download, root=data_dir)
    test_dataset = BreastMNIST(split='test', transform=transform, download=download, root=data_dir)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load_breastmnist_flat(batch_size=32, download=True):
    """
    Load the BreastMNIST dataset and flatten features for traditional ML models.

    Args:
        batch_size (int): Number of samples per batch.
        download (bool): Whether to download the dataset.

    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    train_loader, val_loader, test_loader = load_breastmnist(batch_size, download)

    def flatten(loader):
        """
        Flatten all batches of data in a DataLoader.

        Args:
            loader (DataLoader): PyTorch DataLoader object.

        Returns:
            tuple: (flattened_features, labels)
        """
        X, y = [], []
        for inputs, labels in loader:
            # Flatten inputs and convert to NumPy
            flattened_inputs = inputs.view(inputs.size(0), -1).cpu().numpy()
            labels_np = labels.cpu().numpy().ravel()  # Ensure y is 1D
            X.append(flattened_inputs)
            y.append(labels_np)

        # Concatenate along the first dimension
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    X_train, y_train = flatten(train_loader)
    X_val, y_val = flatten(val_loader)
    X_test, y_test = flatten(test_loader)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_data(X_train, X_val, X_test, n_components=20):
    """
    Standardize the data and reduce dimensionality using PCA.

    Args:
        X_train (numpy.ndarray): Training features.
        X_val (numpy.ndarray): Validation features.
        X_test (numpy.ndarray): Test features.
        n_components (int): Number of PCA components.

    Returns:
        tuple: Transformed (X_train, X_val, X_test), scaler and PCA objects.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reduce dimensions with PCA
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    return X_train, X_val, X_test, scaler, pca

def save_svm_log(timestamp, task, model, best_val_acc, best_params, classification_report, log_path="A/log/log.txt"):
    """
    Save SVM training log to a specified file.

    Args:
        timestamp (str): Timestamp of the log entry.
        task (str): Task identifier (e.g., "A").
        model (str): Model name (e.g., "SVM").
        best_val_acc (float): Best validation accuracy achieved.
        best_params (dict): Best hyperparameters for the model.
        classification_report (str): Classification report text.
        log_path (str): Path to the log file (default: "A/log/log.txt").
    """
    log_entry = f"""
Timestamp: {timestamp}
Task: {task}
Model: {model}
Best Validation Accuracy: {best_val_acc:.4f}
Best Parameters: {best_params}
Classification Report:
{classification_report}
"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(log_entry)
        log_file.write("\n" + "-" * 80 + "\n")
    print(f"Training log saved to {log_path}")

def save_cnn_log(timestamp, model, best_val_acc, best_params, log_path="A/log/log.txt", classification_report=None):
    """
    Save CNN training log to a specified file.

    Args:
        timestamp (str): Timestamp of the log entry.
        model (str): Model name (e.g., "CNN").
        best_val_acc (float or None): Best validation accuracy achieved.
        best_params (dict): Best hyperparameters for the model.
        log_path (str): Path to the log file (default: "A/log/cnn_log.txt").
        classification_report (str): Classification report text.
    """
    # Format best_val_acc if it's not None
    val_acc_str = f"{best_val_acc:.4f}" if best_val_acc is not None else "N/A"

    # Format log entry
    log_entry = f"""
Timestamp: {timestamp}
Model: {model}
Best Validation Accuracy: {val_acc_str}
Best Parameters: {best_params}
Classification Report:
{classification_report if classification_report else "N/A"}
"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(log_entry)
        log_file.write("\n" + "-" * 80 + "\n")
    print(f"Training log saved to {log_path}")
