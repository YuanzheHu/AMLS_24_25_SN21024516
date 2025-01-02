# Setup default hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 128
LEARNING_RATE = 0.001
N_COMPONENTS = 20  # PCA components for SVM

import os
import torch
from A.task_a_utils import load_breastmnist, load_breastmnist_flat, save_training_log_plaintext
from A.task_a_model import BreastMNISTCNN
from A.task_a_train import (
    train_model, load_model, plot_history, test_model, plot_confusion_matrix,
    train_svm_with_grid_search, test_traditional_model, preprocess_data
)
from A.task_a_predict import predict_and_visualize
from sklearn.metrics import classification_report
from itertools import product


def setup_directories():
    """Ensure necessary directories exist."""
    for directory in ["data", "A/models", "A/figure", "A/log"]:
        os.makedirs(directory, exist_ok=True)


def handle_cnn():
    """Workflow for CNN model with hyperparameter tuning and model loading."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load BreastMNIST dataset
        train_loader, val_loader, test_loader = load_breastmnist(batch_size=BATCH_SIZE, download=True, data_dir="data")
        print("Dataset loaded successfully!")

        # Define model path
        model_path = "A/models/best_cnn_model.pth"
        hyperparam_path = "A/models/best_cnn_hyperparams.pth"

        best_val_acc = 0.0
        best_model_state = None
        best_params = None  # Initialize to ensure it is always defined
        best_history = None

        # Check if pretrained model exists
        if os.path.exists(model_path) and os.path.exists(hyperparam_path):
            print(f"Pretrained model found. Loading model from {model_path}...")

            # Load hyperparameters and model
            saved_hyperparams = torch.load(hyperparam_path)
            hidden_units = saved_hyperparams["hidden_units"]
            print(f"Loaded hyperparameters: {saved_hyperparams}")

            model = BreastMNISTCNN(hidden_units=hidden_units).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded successfully!")
        else:
            print("No pretrained model found. Starting hyperparameter tuning...")

            # Define hyperparameter space
            hyperparam_space = product([64, 128, 256], [0.001, 0.0005, 0.0001], [10, 20])

            # Perform hyperparameter tuning
            for hidden_units, lr, num_epochs in hyperparam_space:
                print(f"Testing combination: hidden_units={hidden_units}, lr={lr}, num_epochs={num_epochs}")

                # Initialize and train the model
                model = BreastMNISTCNN(hidden_units=hidden_units).to(device)
                model, history = train_model(model, train_loader, val_loader, device, epochs=num_epochs, lr=lr)

                val_acc = max(history["val_acc"])

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()
                    best_params = {"hidden_units": hidden_units, "learning_rate": lr, "num_epochs": num_epochs}
                    best_history = history

            # Save the best model and hyperparameters
            os.makedirs("A/models", exist_ok=True)
            torch.save(best_model_state, model_path)
            torch.save(best_params, hyperparam_path)
            print(f"Best model saved to {model_path} with val_acc: {best_val_acc:.4f}")
            print(f"Best hyperparameters: {best_params}")

            # Load the best model for evaluation
            model = BreastMNISTCNN(hidden_units=best_params["hidden_units"]).to(device)
            model.load_state_dict(best_model_state)

        # Evaluate the best model on the test set
        print("Evaluating the model on the test set...")
        y_true, y_pred = test_model(model, test_loader, device)

        # Save confusion matrix and classification report
        plot_confusion_matrix(y_true, y_pred, save_dir="figure", file_name="confusion_matrix_cnn.png", class_names=["Benign", "Malignant"])

        report_text = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])
        save_training_log_plaintext(
            {
                "task": "A",
                "model": "CNN",
                "hidden_units": best_params["hidden_units"] if best_params else 128,
                "learning_rate": best_params["learning_rate"] if best_params else LEARNING_RATE,
                "num_epochs": best_params["num_epochs"] if best_params else NUM_EPOCHS,
                "batch_size": BATCH_SIZE
            },
            report_text,
            filename="log.txt"
        )
        print("Training log saved to A/log/log.txt")

        # Plot and save the training history
        if best_history:
            plot_history(best_history, save_dir="figure", file_name="training_history_cnn.png")
        
        # Predict and visualize some samples
        print("Predicting and visualizing test samples...")
        predict_and_visualize(model, test_loader, device, class_names=["Benign", "Malignant"], num_samples=9, save_dir="figure", save_file="cnn_predictions.png")
        print("Prediction visualization saved!")

    except Exception as e:
        print(f"Error in CNN workflow: {e}")


def handle_svm():
    """Workflow for SVM model."""
    try:
        print("Using SVM with preprocessing and hyperparameter tuning...")

        # Load and preprocess the dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_breastmnist_flat(batch_size=BATCH_SIZE)
        X_train, X_val, X_test, _, _ = preprocess_data(X_train, X_val, X_test, n_components=N_COMPONENTS)

        # Train or load SVM
        model_path = "A/models/best_svm_model.pkl"
        model, best_params = train_svm_with_grid_search(X_train, y_train, X_val, y_val, model_path=model_path)
        y_true, y_pred = test_traditional_model(model, X_test, y_test)

        # Save results
        plot_confusion_matrix(y_true, y_pred, save_dir="figure", file_name="confusion_matrix_svm.png", class_names=["Benign", "Malignant"])
        report_text = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])
        print(report_text)
        save_training_log_plaintext(
            {"task": "A", "model": "SVM", "kernel": best_params.get('kernel', 'N/A'), "C": best_params.get('C', 'N/A'),
             "gamma": best_params.get('gamma', 'N/A'), "pca_components": N_COMPONENTS},
            report_text,
            filename="log.txt"
        )
        print("Training log saved to A/log/log.txt")

    except Exception as e:
        print(f"Error in SVM workflow: {e}")


def main():
    """Main entry point for Task A."""
    setup_directories()
    print("Choose model: 1. CNN  2. SVM")
    choice = input("Enter your choice: ").strip()
    if choice == "1":
        handle_cnn()
    elif choice == "2":
        handle_svm()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
