import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
from tqdm.auto import tqdm
from timeit import default_timer as timer
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from A.task_a_model import BreastMNISTCNN, BreastMNISTSVM

def train_cnn_with_hyperparameters(train_loader, val_loader, device, hyperparam_space, model_path):
    """
    Train CNN with hyperparameter tuning.

    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to train the model on.
        hyperparam_space (dict): Hyperparameter space to search.
        model_path (str): Path to save the best model.

    Returns:
        nn.Module: Best-trained model.
        dict: Best hyperparameter combination.
        dict: Training history of the best model.
        float: Best validation accuracy achieved.
    """
    best_val_acc = 0.0
    best_model = None
    best_params = None
    best_history = None

    # Generate all combinations of hyperparameters
    hyperparam_combinations = [
        {k: v for k, v in zip(hyperparam_space.keys(), values)}
        for values in product(*hyperparam_space.values())
    ]

    for params in hyperparam_combinations:
        print(f"Testing combination: {params}")
        model = BreastMNISTCNN(
            hidden_units=params["hidden_units"],
            dropout=params["dropout"]
        ).to(device)

        # Train the model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=params["num_epochs"],
            lr=params["learning_rate"]
        )

        # Evaluate validation accuracy
        val_acc = max(history["val_acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_params = params
            best_history = history

    # Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model saved to {model_path} with validation accuracy: {best_val_acc:.4f}")

    return best_model, best_params, best_history, best_val_acc

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

def load_best_cnn_model(model_path, device):
    """
    Load the best CNN model from the saved checkpoint with dimension matching.

    Args:
        model_path (str): Path to the saved model.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded CNN model.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Extract dimensions from the state_dict
    fc1_weight_shape = state_dict["fc1.weight"].shape  # e.g., torch.Size([256, 1152])
    fc2_weight_shape = state_dict["fc2.weight"].shape  # e.g., torch.Size([2, 256])

    # Get the number of hidden units from fc1's output size
    hidden_units = fc1_weight_shape[0]

    # Initialize the model with the extracted dimensions
    best_model = BreastMNISTCNN(hidden_units=hidden_units, dropout=0.5).to(device)

    # Load the state_dict into the model
    best_model.load_state_dict(state_dict)
    print(f"Model loaded successfully with hidden_units={hidden_units}")
    return best_model

def train_svm_with_grid_search(X_train, y_train, X_val, y_val, model_path="A/models/best_svm_model.pkl", param_grid=None):
    """
    Train an SVM model using custom parameter grid or load the best model if it exists.

    Args:
        X_train (numpy.ndarray): Preprocessed training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Preprocessed validation features.
        y_val (numpy.ndarray): Validation labels.
        model_path (str): Path to save/load the best model.
        param_grid (dict): Parameter grid for hyperparameter search.

    Returns:
        BreastMNISTSVM: Best-trained SVM model.
        dict: Best parameters found or loaded.
    """
    if os.path.exists(model_path):
        print(f"Pretrained SVM model found. Loading model from {model_path}...")
        best_model = joblib.load(model_path)
        best_params = best_model.model.get_params()
        return best_model, best_params

    print("No pretrained model found. Starting hyperparameter tuning...")

    # Initialize progress bar
    param_combinations = [
        {'C': c, 'kernel': k, 'gamma': g, 'class_weight': cw}
        for c in param_grid['C']
        for k in param_grid['kernel']
        for g in param_grid['gamma']
        for cw in param_grid['class_weight']
    ]
    total_combinations = len(param_combinations)
    progress_bar = tqdm(total=total_combinations, desc="GridSearch Progress")

    best_score = 0
    best_params = None
    best_model = None

    for params in param_combinations:
        print(f"Testing parameters: {params}")
        current_model = BreastMNISTSVM(C=params['C'], kernel=params['kernel'])
        current_model.model.gamma = params['gamma']  # Explicitly set gamma
        current_model.model.class_weight = params['class_weight']  # Set class weight

        # Train the model
        current_model.fit(X_train, y_train)

        # Evaluate the model
        val_score = current_model.score(X_val, y_val)
        print(f"Validation score: {val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_params = params
            best_model = current_model

        progress_bar.update(1)
    progress_bar.close()

    # Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path} with validation accuracy: {best_score:.4f}")

    return best_model, best_params

def test_svm(model, X_test, y_test, class_names=["Benign", "Malignant"]):
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