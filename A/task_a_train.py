import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from A.task_a_utils import load_breastmnist
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from timeit import default_timer as timer

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import joblib

from A.task_a_model import BreastMNISTSVM

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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    return model


def test_model(model, test_loader, device):
    """
    Test the CNN model and generate a classification report.

    Args:
        model (nn.Module): Trained CNN model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to test the model on.

    Returns:
        tuple: (y_true, y_pred)
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Generate and print classification report
    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred, target_names=["Benign", "Malignant"]
    )
    print(report)

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, save_dir="figure", file_name="confusion_matrix.png", class_names=["Benign", "Malignant"]):
    """
    Plot and save the confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        save_dir (str): Directory to save the confusion matrix image.
        file_name (str): File name for the confusion matrix image.
        class_names (list): List of class names for the confusion matrix.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)

    # Ensure save directory exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the confusion matrix figure
    figure_path = os.path.join(save_path, file_name)
    plt.savefig(figure_path)
    print(f"Confusion matrix saved to {figure_path}")

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


def train_svm_with_grid_search(X_train, y_train, X_val, y_val, model_path="A/models/best_svm_model.pkl"):
    """
    Train an SVM model using GridSearchCV or load the best model if it exists.

    Args:
        X_train (numpy.ndarray): Preprocessed training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Preprocessed validation features.
        y_val (numpy.ndarray): Validation labels.
        model_path (str): Path to save/load the best model.

    Returns:
        SVC: Best-trained SVM model.
        dict: Best parameters found or loaded.
    """
    if os.path.exists(model_path):
        # Load the existing model
        print(f"Loading saved SVM model from {model_path}...")
        best_model = joblib.load(model_path)
        best_params = best_model.get_params()
        return best_model, best_params

    print("No saved model found. Training a new SVM model...")

    # Define a reduced parameter grid for faster search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Initialize progress bar
    param_combinations = [
        {'C': c, 'kernel': k, 'gamma': g}
        for c in param_grid['C']
        for k in param_grid['kernel']
        for g in param_grid['gamma']
    ]
    total_combinations = len(param_combinations)
    progress_bar = tqdm(total=total_combinations * 5, desc="GridSearchCV Progress")  # 5 是 CV 折数

    # Train using GridSearchCV
    best_score = 0
    best_params = None
    best_model = None
    for params in param_combinations:
        svc = SVC(probability=True, **params)
        grid_search = GridSearchCV(
            svc,
            param_grid={},
            scoring='accuracy',
            cv=5,
            n_jobs=1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        score = grid_search.best_score_

        if score > best_score:
            best_score = score
            best_params = params
            best_model = grid_search.best_estimator_

        progress_bar.update(5)
    progress_bar.close()

    # Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

    # Validation accuracy
    val_accuracy = best_model.score(X_val, y_val)
    print(f"Best Parameters: {best_params}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return best_model, best_params


def test_traditional_model(model, X_test, y_test, class_names=["Benign", "Malignant"]):
    """
    Test a traditional ML model and generate a classification report.

    Args:
        model: Trained traditional ML model (e.g., SVM).
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        class_names (list): Class names for the report.

    Returns:
        tuple: (y_true, y_pred)
    """
    # Predict labels
    y_pred = model.predict(X_test)
    return y_test, y_pred