import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

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

if __name__ == "__main__":
    # Test the data loading
    train_loader, val_loader, test_loader = load_breastmnist()
    print(f"Train Loader Size: {len(train_loader)}")
    print(f"Validation Loader Size: {len(val_loader)}")
    print(f"Test Loader Size: {len(test_loader)}")