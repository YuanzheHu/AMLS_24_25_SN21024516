import os
import torch
from datetime import datetime
from itertools import product
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from A.task_a_utils import (
    load_breastmnist, load_breastmnist_flat, save_training_log_plaintext
)
from A.task_a_model import BreastMNISTCNN
from A.task_a_train import (
    train_model, plot_history, test_model, plot_confusion_matrix,
    train_svm_with_grid_search, test_traditional_model, preprocess_data
)
from A.task_a_predict import predict_and_visualize
from B.task_b_utils import load_bloodmnist, log_message, generate_classification_report, plot_and_save_confusion_matrix
from B.task_b_model import get_resnet_model
from B.task_b_train import train_model as train_resnet_model, test_model as test_resnet_model
from B.task_b_predict import predict_and_visualize_resnet
import matplotlib.pyplot as plt


# Default Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 128
LEARNING_RATE = 0.001
N_COMPONENTS = 20  # PCA components for SVM


def setup_directories():
    """Ensure necessary directories exist for Task A and Task B."""
    for directory in ["data", "A/models", "A/figure", "A/log", "B/models", "B/figure", "B/log"]:
        os.makedirs(directory, exist_ok=True)

def handle_task_a_cnn():
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


def handle_task_a_svm():
    """Workflow for Task A with SVM."""
    try:
        print("Using SVM with preprocessing and hyperparameter tuning...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_breastmnist_flat(batch_size=BATCH_SIZE)
        X_train, X_val, X_test, _, _ = preprocess_data(X_train, X_val, X_test, n_components=N_COMPONENTS)

        model_path = "A/models/best_svm_model.pkl"
        model, best_params = train_svm_with_grid_search(X_train, y_train, X_val, y_val, model_path=model_path)
        y_true, y_pred = test_traditional_model(model, X_test, y_test)

        plot_confusion_matrix(y_true, y_pred, save_dir="A/figure", file_name="confusion_matrix_svm.png", class_names=["Benign", "Malignant"])
        report_text = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])
        save_training_log_plaintext(
            {"task": "A", "model": "SVM", "kernel": best_params.get('kernel', 'N/A'), "C": best_params.get('C', 'N/A'),
             "gamma": best_params.get('gamma', 'N/A'), "pca_components": N_COMPONENTS},
            report_text,
            filename="A/log/log.txt"
        )
        print("Training log saved to A/log/log.txt")

    except Exception as e:
        print(f"Error in SVM workflow: {e}")

def handle_task_b_resnet():
    """
    Handle ResNet model for training, testing, and evaluating BloodMNIST.
    Includes data loading, model training, evaluation, and visualization.
    """
    try:
        # Initialize log file
        log_file = "B/log/training_log.txt"
        with open(log_file, "w") as f:
            f.write("Training Log\n")
            f.write("=" * 50 + "\n")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        log_message(f"Device: {device}", log_file)

        # Load BloodMNIST dataset
        print("Loading BloodMNIST dataset...")
        log_message("Loading BloodMNIST dataset...", log_file)
        train_loader, val_loader, test_loader = load_bloodmnist(batch_size=32, data_dir="data")
        print("Dataset loaded successfully!")
        log_message("Dataset loaded successfully!", log_file)

        # Initialize ResNet model
        model = get_resnet_model(num_classes=8).to(device)

        # Record training start time
        start_time = datetime.now()
        log_message(f"Training started at: {start_time}", log_file)

        # Check for pre-trained model
        model_path = "B/models/resnet_model.pth"
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}...")
            log_message(f"Loading pre-trained model from {model_path}", log_file)
            model.load_state_dict(torch.load(model_path))
        else:
            print("No pre-trained model found. Starting training...")
            log_message("No pre-trained model found. Starting training...", log_file)
            model, history = train_resnet_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=10,
                lr=0.001,
                save_dir="B/models"
            )
            print("Training completed!")
            log_message("Training completed!", log_file)

        # Record training end time
        end_time = datetime.now()
        log_message(f"Training ended at: {end_time}", log_file)
        log_message(f"Training duration: {end_time - start_time}", log_file)

        # Log hyperparameters
        log_message("Hyperparameters:", log_file)
        log_message("  - Epochs: 10", log_file)
        log_message("  - Learning Rate: 0.001", log_file)
        log_message("  - Batch Size: 32", log_file)

        # Test the model
        print("Testing the model...")
        log_message("Testing the model...", log_file)
        y_true, y_pred = test_resnet_model(model, test_loader, device)

        # Generate and save classification report
        target_names = [str(i) for i in range(8)]
        generate_classification_report(y_true, y_pred, target_names, log_file)

        # Plot and save confusion matrix
        cm_path = os.path.join("B/figure", "confusion_matrix_resnet.png")
        plot_and_save_confusion_matrix(y_true, y_pred, target_names, save_path=cm_path, log_file=log_file)

        # Predict and visualize test samples
        print("Predicting and visualizing test samples...")
        log_message("Predicting and visualizing test samples...", log_file)
        predict_and_visualize_resnet(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir="B/figure",
            save_file="predictions_resnet.png"
        )
        print("Prediction visualization saved!")
        log_message("Prediction visualization saved!", log_file)

    except Exception as e:
        print(f"Error in ResNet workflow: {e}")
        log_message(f"Error in ResNet workflow: {e}", log_file)

def main():
    """Main entry point for Task A and Task B."""
    setup_directories()
    print("Choose task: 1. Task A  2. Task B")
    task_choice = input("Enter your choice: ").strip()
    if task_choice == "1":
        print("Choose method for Task A: 1. CNN  2. SVM")
        method_choice = input("Enter your choice: ").strip()
        if method_choice == "1":
            handle_task_a_cnn()
        elif method_choice == "2":
            handle_task_a_svm()
        else:
            print("Invalid choice. Exiting.")
    elif task_choice == "2":
        handle_task_b_resnet()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
