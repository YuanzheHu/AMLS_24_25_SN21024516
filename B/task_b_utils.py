import os
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import BloodMNIST
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_bloodmnist(batch_size=32, data_dir="B/data"):
    """
    Load the BloodMNIST dataset and return data loaders.

    Args:
        batch_size (int): Batch size for DataLoader.
        data_dir (str): Directory to store dataset.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    train_dataset = BloodMNIST(split='train', transform=transform, download=True, root=data_dir)
    val_dataset = BloodMNIST(split='val', transform=transform, download=True, root=data_dir)
    test_dataset = BloodMNIST(split='test', transform=transform, download=True, root=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def log_message(message, log_file="B/log/training_log.txt"):
    """
    Append a log message to a log file.

    Args:
        message (str): The message to log.
        log_file (str): Path to the log file.
    """
    with open(log_file, "a") as f:
        f.write(f"{message}\n")

def generate_classification_report(y_true, y_pred, target_names, log_file):
    """
    Generate, print, log, and optionally save a classification report.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        target_names (list): Class names for the report.
        log_file (str): Path to the log file for saving the report.
        save_path (str): Optional path to save the classification report as a file.
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\nClassification Report:")
    print(report)
    log_message("\nClassification Report:", log_file)
    log_message(report, log_file)

def plot_and_save_confusion_matrix(y_true, y_pred, target_names, save_path, log_file=None):
    """
    Plot, save, and optionally log the confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        target_names (list): Class names for the confusion matrix.
        save_path (str): Path to save the confusion matrix image.
        log_file (str): Optional path to the log file for logging the save operation.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    if log_file:
        log_message(f"Confusion matrix saved to {save_path}", log_file)
