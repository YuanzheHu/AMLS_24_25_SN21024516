import os
from medmnist import INFO
from medmnist.dataset import BloodMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

def load_bloodmnist(batch_size=32, data_dir="B/data"):
    """
    Load the BloodMNIST dataset.

    Args:
        batch_size (int): Batch size for DataLoader.
        data_dir (str): Directory to save the dataset.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    dataset_name = 'bloodmnist'
    info = INFO[dataset_name]
    DataClass = BloodMNIST

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_dataset = DataClass(split='train', transform=transform, download=True, root=data_dir)
    val_dataset = DataClass(split='val', transform=transform, download=True, root=data_dir)
    test_dataset = DataClass(split='test', transform=transform, download=True, root=data_dir)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader