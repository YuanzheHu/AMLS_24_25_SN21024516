import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO,BreastMNIST
import numpy as np
from datetime import datetime

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
    # Ensure data directory exists
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

def load_breastmnist_augmented(batch_size=32, download=True, data_dir="data"):
    """
    Load the BreastMNIST dataset with data augmentation and save to a specified directory.

    Args:
        batch_size (int): Batch size for DataLoader.
        download (bool): Whether to download the dataset.
        data_dir (str): Directory to save the dataset.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Data augmentation for training set
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Simple normalization for validation and test sets
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_dataset = BreastMNIST(split='train', transform=transform_train, download=download, root=data_dir)
    val_dataset = BreastMNIST(split='val', transform=transform_val_test, download=download, root=data_dir)
    test_dataset = BreastMNIST(split='test', transform=transform_val_test, download=download, root=data_dir)

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

def save_training_log_plaintext(hyperparameters, classification_report, augmentation=False, save_dir="A/log", filename="training_log.txt"):
    """
    Save training details and classification report to a plain text file.

    Args:
        hyperparameters (dict): Hyperparameters and other training details.
        classification_report (str): Classification report in plain text format.
        augmentation (bool): Whether data augmentation was used.
        save_dir (str): Directory to save the log file.
        filename (str): Name of the log file to save the report.
    """
    import os
    from datetime import datetime

    # Ensure the log directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the log file path
    log_file = os.path.join(save_dir, filename)

    # Prepare log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
Timestamp: {timestamp}
Data Augmentation: {'Yes' if augmentation else 'No'}
Task: {hyperparameters.get('task', 'N/A')}
Model: {hyperparameters.get('model', 'N/A')}
Kernel: {hyperparameters.get('kernel', 'N/A')}
Regularization Param: {hyperparameters.get('regularization_param', 'N/A')}
Num Epochs: {hyperparameters.get('num_epochs', 'N/A')}
Batch Size: {hyperparameters.get('batch_size', 'N/A')}
Learning Rate: {hyperparameters.get('learning_rate', 'N/A')}
Hidden Units: {hyperparameters.get('hidden_units', 'N/A')}

Classification Report:
{classification_report}
"""
    # Write to the log file
    with open(log_file, "a") as f:
        f.write(log_entry)
        f.write("\n" + "-" * 80 + "\n")  # Add a separator for readability

    print(f"Training log saved to {log_file}")