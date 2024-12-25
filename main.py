# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 128
LEARNING_RATE = 0.001

import os
import torch
from A.task_a_utils import load_breastmnist, load_breastmnist_flat, save_training_log
from A.task_a_model import BreastMNISTCNN
from A.task_a_train import (
    train_model, load_model, plot_history, test_model, plot_confusion_matrix, train_svm, test_traditional_model
)
from A.task_a_predict import predict_and_visualize


def setup_directories():
    """Ensure necessary directories exist."""
    os.makedirs("A/models", exist_ok=True)
    os.makedirs("A/figure", exist_ok=True)
    os.makedirs("A/log", exist_ok=True) 

def handle_cnn():
    """Workflow for CNN model."""
    try:
        # Detect device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load dataset
        print("Loading BreastMNIST dataset...")
        train_loader, val_loader, test_loader = load_breastmnist(batch_size=BATCH_SIZE)
        print("Dataset loaded successfully!")

        # Initialize the model
        model = BreastMNISTCNN(hidden_units=HIDDEN_UNITS).to(device)

        # Check if a pre-trained model exists
        model_path = "A/models/best_model.pth"
        if os.path.exists(model_path):
            print("Loading pre-trained model...")
            model = load_model(model, model_path, device)
        else:
            # Train the model
            print("No pre-trained model found. Starting training...")
            model, history = train_model(
                model, train_loader, val_loader, device,
                epochs=NUM_EPOCHS, lr=LEARNING_RATE
            )
            print("Training completed!")

            # Plot training history
            plot_history(history, save_dir="A/figure", file_name="training_history.png")
            print("Training history saved!")

        # Test the model
        print("Testing the model...")
        y_true, y_pred = test_model(model, test_loader, device)
        print("Test results collected!")

        # Plot confusion matrix
        cm_file = "A/figure/confusion_matrix_cnn.png"
        plot_confusion_matrix(
            y_true, y_pred,
            save_dir="A/figure",
            file_name="confusion_matrix_cnn.png",
            class_names=["Benign", "Malignant"]
        )
        print("Confusion matrix saved!")

        # Generate classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True)

        # Save training log
        hyperparameters = {
            "task": "A",
            "model": "CNN",
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden_units": HIDDEN_UNITS,
            "confusion_matrix": cm_file,
            "training_history": "A/figure/training_history.png"
        }
        save_training_log(hyperparameters, report, save_dir="A/log")  # Save to A/log

        # Predict and visualize results
        print("Predicting and visualizing test samples...")
        predict_and_visualize(model, test_loader, device, save_dir="A/figure", save_file="predictions_cnn.png")
        print("Prediction visualization saved!")

    except Exception as e:
        print(f"Error in CNN workflow: {e}")


from A.task_a_utils import save_training_log

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
        cm_file = "A/figure/confusion_matrix_svm.png"
        plot_confusion_matrix(
            y_true, y_pred,
            save_dir="A/figure",
            file_name="confusion_matrix_svm.png",
            class_names=["Benign", "Malignant"]
        )
        print("SVM Confusion matrix saved!")

        # Generate classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True)

        # Save training log
        hyperparameters = {
            "task": "A",
            "model": "SVM",
            "kernel": "linear",
            "regularization_param": 1.0,
            "batch_size": BATCH_SIZE,
            "confusion_matrix": cm_file
        }
        save_training_log(hyperparameters, report, save_dir="A/log")  # Save to A/log

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