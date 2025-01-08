import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from B.task_b_model import EarlyStopping

def plot_training_curve(history, save_dir="B/figure", filename= "training_curve.png"):
    """
    Plot training and validation curves for loss and accuracy.

    Args:
        history (dict): Training history.
        save_dir (str): Directory to save the plot.
        filename (str): Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    print(f"Training curve saved to {os.path.join(save_dir, filename)}")

def plot_learning_rate(scheduler, num_epochs, save_path="B/figure", filename="learning_rate_curve.png"):
    """
    Plot the learning rate changes during training.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs for training.
        save_path (str): Directory to save the learning rate plot.
        filename (str): Custom file name for saving the plot.
    """
    # Record learning rates for each epoch
    learning_rates = []
    for epoch in range(num_epochs):
        # Simulate a scheduler step to capture the learning rate
        learning_rates.append(scheduler.optimizer.param_groups[0]['lr'])
        scheduler.step(0)  # Provide a dummy validation loss for the scheduler

    # Plot the learning rate curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), learning_rates, marker='o')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid()

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Construct the full path with the custom filename
    full_save_path = os.path.join(save_path, filename)

    # Save the plot
    plt.savefig(full_save_path)

    print(f"Learning rate curve saved to {full_save_path}")
    
def train_resnet_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_dir="B/models"):
    """
    Train the ResNet model with early stopping and dynamic learning rate adjustment.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Training device.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        save_dir (str): Directory to save the model.

    Returns:
        nn.Module: Trained model.
        dict: Training history.
        list: Learning rate history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    lr_history = []  # Record learning rate changes

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze()  # Ensure labels are 1D
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        scheduler.step(val_loss)

        # Record current learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/resnet_model.pth")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    plot_training_curve(history, save_dir="B/figure", filename="training_curve_resnet.png")
    plot_learning_rate(scheduler, epochs, filename="learning_rate_curve_resnet.png")
    return model, history, lr_history

def test_resnet_model(model, test_loader, device):
    """
    Test the model on the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to test the model on.

    Returns:
        tuple: True labels, predicted labels.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Fix label shape to (batch_size,)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred

def train_vit_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_dir="B/models"):
    """
    Train the Vision Transformer (ViT) model with early stopping and dynamic learning rate adjustment.

    Args:
        model (nn.Module): The ViT model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Training device.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        save_dir (str): Directory to save the model.

    Returns:
        nn.Module: Trained ViT model.
        dict: Training history.
        list: Learning rate history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # Using AdamW optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    lr_history = []  # Record learning rate changes

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze()  # Ensure labels are 1D
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        scheduler.step(val_loss)

        # Record current learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/vit_model.pth")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    plot_training_curve(history, save_dir="B/figure", filename="training_curve_vit.png")
    plot_learning_rate(scheduler, epochs, filename="earning_rate_curve_vit.png")

    return model, history, lr_history

def test_vit_model(model, test_loader, device):
    """
    Test the Vision Transformer (ViT) model on the test set.

    Args:
        model (nn.Module): Trained ViT model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to test the model on.

    Returns:
        tuple: True labels, predicted labels.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Fix label shape to (batch_size,)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred