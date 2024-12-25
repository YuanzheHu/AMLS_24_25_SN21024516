import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from task_a_utils import load_breastmnist
from task_a_model import BreastMNISTCNN

from tqdm.auto import tqdm
from timeit import default_timer as timer

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_dir="models"):
    """
    Train the CNN model with progress bar, time measurement, and model saving.

    Args:
        model (nn.Module): The CNN model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to train the model on.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        save_dir (str): Directory to save the model.

    Returns:
        model: Trained model.
        dict: Training history containing loss and accuracy.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Create save directory if not exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_val_acc = 0.0  # Track the best validation accuracy

    # Start training timer
    start_time = timer()

    for epoch in range(epochs):
        # Training loop
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        print(f"Epoch {epoch + 1}/{epochs}:")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({"Loss": loss.item()})

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation loop
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze()
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

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # End training timer
    end_time = timer()
    total_time = end_time - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    return model, history

def plot_history(history, save_dir="figure", file_name="training_history.png"):
    """
    Plot training and validation loss and accuracy and save the plot.

    Args:
        history (dict): Training history containing loss and accuracy.
        save_dir (str): Directory to save the figure.
        file_name (str): File name for the saved figure.
    """
    # Create save directory if it does not exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Plot training history
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()

    # Save the figure
    figure_path = os.path.join(save_path, file_name)
    plt.savefig(figure_path)
    print(f"Figure saved to {figure_path}")

def load_model(model, model_path, device):
    """
    Load a trained model from file.

    Args:
        model (nn.Module): The CNN model to load weights into.
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model on.

    Returns:
        model: Model with loaded weights.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    return model

def test_model(model, test_loader, device, save_dir="figure", file_name="confusion_matrix.png"):
    """
    Test the CNN model and save the confusion matrix.

    Args:
        model (nn.Module): Trained CNN model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to test the model on.
        save_dir (str): Directory to save the confusion matrix image.
        file_name (str): File name for the confusion matrix image.
    """
    model.eval()
    y_true, y_pred = [], []

    # Create save directory if it does not exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    disp.plot(cmap=plt.cm.Blues)

    # Save the confusion matrix figure
    figure_path = os.path.join(save_path, file_name)
    plt.savefig(figure_path)
    print(f"Confusion matrix saved to {figure_path}")

    # Optional: Show the confusion matrix
    plt.show(block=False)

if __name__ == "__main__":
    # Load data with data augmentation and normalization
    train_loader, val_loader, test_loader = load_breastmnist(batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = BreastMNISTCNN()

    # Load or train model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.pth")
    if os.path.exists(model_path):
        model = load_model(model, model_path, device)
    else:
        model, history = train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001)
        plot_history(history, save_dir="figure", file_name="training_history.png")

    # Test the model
    test_model(model, test_loader, device, save_dir="figure", file_name="confusion_matrix.png")