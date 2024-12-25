import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import numpy as np
import json
from datetime import datetime

def load_breastmnist(batch_size=32, download=True):
    """
    Load the BreastMNIST dataset without data augmentation.

    Args:
        batch_size (int): Batch size for DataLoader.
        download (bool): Whether to download the dataset.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    dataset_name = 'breastmnist'
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # Simple normalization for all datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_dataset = DataClass(split='train', transform=transform, download=download)
    val_dataset = DataClass(split='val', transform=transform, download=download)
    test_dataset = DataClass(split='test', transform=transform, download=download)

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

def save_training_log(params, report, save_dir="logs", file_name="training_log.json"):
    """
    Save training logs including hyperparameters and classification report.

    Args:
        params (dict): Hyperparameters and configuration details.
        report (dict): Classification report or other evaluation metrics.
        save_dir (str): Directory to save the log file.
        file_name (str): File name for the log file.
    """
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Add timestamp to the log
    params["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params["classification_report"] = report

    # Log file path
    log_path = os.path.join(save_dir, file_name)

    # Append to log file if it exists
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(params)

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)
    
    print(f"Training log saved to {log_path}")