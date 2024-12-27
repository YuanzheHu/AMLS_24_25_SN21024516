# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 128
LEARNING_RATE = 0.001

import os
import torch
from A.task_a_utils import load_breastmnist, load_breastmnist_flat, load_breastmnist_augmented,save_training_log_plaintext
from A.task_a_model import BreastMNISTCNN
from A.task_a_train import (
    train_model, load_model, plot_history, test_model, plot_confusion_matrix, train_svm, test_traditional_model
)
from A.task_a_predict import predict_and_visualize
from sklearn.metrics import classification_report


def setup_directories():
    """Ensure necessary directories exist."""
    os.makedirs("data", exist_ok=True)       # Dataset directory
    os.makedirs("A/models", exist_ok=True)   # Model directory
    os.makedirs("A/figure", exist_ok=True)   # Figure directory
    os.makedirs("A/log", exist_ok=True)      # Log directory

def handle_cnn():
    """Workflow for CNN model."""
    try:
        # Detect device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Ask user if data augmentation should be used
        use_augmentation = input("Use data augmentation? (y/n): ").strip().lower() == "y"
        # Load dataset with or without augmentation
        print("Loading BreastMNIST dataset...")
        if use_augmentation:
            from A.task_a_utils import load_breastmnist_augmented
            train_loader, val_loader, test_loader = load_breastmnist_augmented(batch_size=BATCH_SIZE, download=True, data_dir="data")
        else:
            train_loader, val_loader, test_loader = load_breastmnist(batch_size=BATCH_SIZE, download=True, data_dir="data")
        print("Dataset loaded successfully!")

        # Initialize the model
        model = BreastMNISTCNN(hidden_units=HIDDEN_UNITS).to(device)

        # Train the model
        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=NUM_EPOCHS, lr=LEARNING_RATE
        )
        print("Training completed!")

        # Plot training history
        plot_history(history, save_dir="figure", file_name="training_history.png")
        print("Training history saved!")

        # Test the model
        print("Testing the model...")
        y_true, y_pred = test_model(model, test_loader, device)
        print("Test results collected!")

        # Plot confusion matrix
        cm_file = "figure/confusion_matrix_cnn.png"
        plot_confusion_matrix(
            y_true, y_pred,
            save_dir="figure",
            file_name="confusion_matrix_cnn.png",
            class_names=["Benign", "Malignant"]
        )
        print("Confusion matrix saved!")

        # Generate classification report
        report_text = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])

        # Save training log
        hyperparameters = {
            "task": "A",
            "model": "CNN",
            "Data Augmentation": use_augmentation,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden_units": HIDDEN_UNITS,
        }
        save_training_log_plaintext(hyperparameters, report_text, augmentation=use_augmentation, filename="A_log.txt")
        print("Training log saved to A/log/A_log.txt")
        
        # Predict and visualize results
        print("Predicting and visualizing test samples...")
        predict_and_visualize(model, test_loader, device, save_dir="figure", save_file="predictions_cnn.png")
        print("Prediction visualization saved!")

    except Exception as e:
        print(f"Error in CNN workflow: {e}")

def handle_svm():
    """Workflow for SVM model."""
    try:
        print("Using SVM...")

        # Load and flatten the dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_breastmnist_flat(batch_size=BATCH_SIZE)

        # Train SVM
        model = train_svm(X_train, y_train, X_val, y_val, C=1.0, kernel='linear')

        # Test SVM
        y_true, y_pred = test_traditional_model(model, X_test, y_test)

        # Plot and save confusion matrix
        cm_file = "figure/confusion_matrix_svm.png"
        plot_confusion_matrix(
            y_true, y_pred,
            save_dir="figure",
            file_name="confusion_matrix_svm.png",
            class_names=["Benign", "Malignant"]
        )
        print("SVM Confusion matrix saved!")

        # Generate classification report as plain text
        report_text = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])

        # Save training log
        hyperparameters = {
            "task": "A",
            "model": "SVM",
            "kernel": "linear",
            "regularization_param": 1.0,
            "batch_size": BATCH_SIZE,
            "confusion_matrix": cm_file
        }
        save_training_log_plaintext(
            hyperparameters,
            report_text,
            save_dir="A/log",
            filename="A_log.txt"
        )

        print("SVM training log saved!")

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