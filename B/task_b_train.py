import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_dir="B/models"):
    """
    Train the ResNet model.

    Args:
        model (nn.Module): ResNet model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to train the model on.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        save_dir (str): Directory to save the model.

    Returns:
        tuple: Trained model, training history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Fix label shape to (batch_size,)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze()  # Fix label shape to (batch_size,)
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

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/resnet_model.pth")

    return model, history


def test_model(model, test_loader, device):
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

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(8)]))

    return y_true, y_pred