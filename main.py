# Standard Library
import os
from datetime import datetime            

# Third-Party Libraries
import joblib
import torch
from sklearn.metrics import classification_report

# Project Modules
# Task A Utilities
from A.task_a_utils import (
    load_breastmnist, load_breastmnist_flat, preprocess_data, save_svm_log, save_cnn_log
)
from A.task_a_train import (
    plot_history, test_model, plot_confusion_matrix, train_cnn_with_hyperparameters,
    load_best_cnn_model, train_svm_with_grid_search, test_svm
)
from A.task_a_predict import predict_and_visualize, plot_roc_auc, plot_roc_auc_cnn,compare_model_performance_a

# Task B Utilities
from B.task_b_utils import (
    load_bloodmnist, log_message, generate_classification_report, plot_and_save_confusion_matrix,
    load_vit_bloodmnist
)
from B.task_b_model import get_resnet_model, get_vit_model
from B.task_b_train import train_resnet_model, test_resnet_model, train_vit_model, test_vit_model
from B.task_b_predict import predict_and_visualize_resnet, compare_model_performance

# Default Hyperparameters for Task A
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
    """
    Workflow for CNN model with hyperparameter tuning and model loading.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load BreastMNIST dataset
        train_loader, val_loader, test_loader = load_breastmnist(batch_size=BATCH_SIZE, download=True, data_dir="data")
        print("Dataset loaded successfully!")

        # Define model path
        model_path = "A/models/best_cnn_model.pth"
        log_path = "A/log/log.txt"

        # Check if pretrained model exists
        if os.path.exists(model_path):
            print(f"Pretrained model found. Loading model from {model_path}...")
            best_model = load_best_cnn_model(model_path, device)
            best_model.eval()
            print("Model loaded successfully!")
        else:
            print("No pretrained model found. Starting hyperparameter tuning...")

            # Define hyperparameter space
            hyperparam_space = {
                "hidden_units": [64, 128, 256],
                "learning_rate": [0.001, 0.0005, 0.0001],
                "num_epochs": [10, 20],
                "dropout": [0.3, 0.5]  # Regularization option
            }

            # Train and evaluate the model
            best_model, best_params, best_history, best_val_acc = train_cnn_with_hyperparameters(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                hyperparam_space=hyperparam_space,
                model_path=model_path
            )

            # Log results
            save_cnn_log(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model="CNN",
                best_val_acc=best_val_acc,
                best_params=best_params,
                classification_report="N/A"  # Will be updated after test evaluation
            )

        # Evaluate on test set
        print("Evaluating the model on the test set...")
        y_true, y_pred = test_model(best_model, test_loader, device)

        # Generate and print classification report
        print("\nClassification Report:")
        report = classification_report(
            y_true, y_pred, target_names=["Benign", "Malignant"],
        )
        print(report)

        plot_roc_auc_cnn(
            model=best_model,
            test_loader=test_loader,
            device=device,
            save_dir="A/figure",
            file_name="roc_auc_cnn.png",
            log_path="A/log/log.txt"
        )

        # Update log with classification report
        save_cnn_log(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model="CNN",
            best_val_acc=best_val_acc if 'best_val_acc' in locals() else None,
            best_params=best_params if 'best_params' in locals() else None,
            classification_report=report,
        )

        # Save confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, save_dir="figure", file_name="confusion_matrix_cnn.png", class_names=["Benign", "Malignant"]
        )

        # Save training history plot (if training was performed)
        if 'best_history' in locals():
            plot_history(best_history, save_dir="figure", file_name="training_history_cnn.png")
        

        # Predict and visualize some samples
        predict_and_visualize(best_model, test_loader, device, class_names=["Benign", "Malignant"], num_samples=9, save_dir="figure", save_file="cnn_predictions.png")
        print("Prediction visualization saved!")
        return classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True)

    except Exception as e:
        print(f"Error in CNN workflow: {e}")

def handle_task_a_svm():
    """
    Workflow for Task A using SVM with extended hyperparameter tuning and class imbalance handling.
    """
    try:
        print("Using SVM with extended hyperparameter tuning and class imbalance handling...")

        # Load and preprocess the BreastMNIST dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_breastmnist_flat(batch_size=BATCH_SIZE)
        X_train, X_val, X_test, scaler, pca = preprocess_data(X_train, X_val, X_test, n_components=N_COMPONENTS)

        # Define model path
        model_path = "A/models/best_svm_model.pkl"

        # Define an extended parameter grid with class_weight option
        param_grid = {
            'C': [0.1, 1, 10],  # A range that captures more regularization options
            'kernel': ['linear', 'rbf', 'poly'],  # Focus on commonly used kernels
            'gamma': ['scale', 'auto'],  # Add more granularity for gamma
            'degree': [2, 3],  # Include lower-degree polynomials for 'poly' kernel
            'class_weight': [None, 'balanced'], 
        }

        # Check if pretrained model exists
        if os.path.exists(model_path):
            print(f"Pretrained SVM model found. Loading model from {model_path}...")
            model, best_params = joblib.load(model_path), None
        else:
            print("No pretrained model found. Starting hyperparameter tuning...")

            # Train SVM with extended GridSearchCV
            model, best_params = train_svm_with_grid_search(
                X_train, y_train, X_val, y_val,
                model_path="A/models/best_svm_model.pkl",
                param_grid=param_grid
            )

            print(f"Best SVM model saved to {model_path} with params: {best_params}")

        # Evaluate the model on the test set
        print("Evaluating the SVM model on the test set...")
        y_true, y_pred = test_svm(model, X_test, y_test)

        # Save confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            save_dir="figure",
            file_name="confusion_matrix_svm.png",
            class_names=["Benign", "Malignant"]
        )

        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            target_names=["Benign", "Malignant"],
        )
        print("\nClassification Report:\n", report)

        # Plot and save ROC-AUC curve
        plot_roc_auc(
            model=model.model,  # Access internal SVC model
            X_test=X_test,
            y_test=y_test,
            save_dir="A/figure",
            file_name="roc_auc_svm.png",
            log_path="A/log/log.txt"
        )

        # Log results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        val_accuracy = model.score(X_val, y_val)
        save_svm_log(
            timestamp=timestamp,
            task="A",
            model="SVM",
            best_val_acc=val_accuracy,
            best_params=best_params,
            classification_report=report
        )
        print("Results logged successfully!")

        return classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True)
    
    except Exception as e:
        print(f"Error in SVM workflow: {e}")

def handle_comparison_a(svm_report, cnn_report):
    """
    Compare SVM and CNN performance using F1-Score.
    """
    try:
        # Extract class labels from the reports
        class_labels = [label for label in cnn_report.keys() if isinstance(cnn_report[label], dict)]

        # Extract F1-Scores for SVM and CNN
        svm_scores = [svm_report[label]["f1-score"] for label in class_labels]
        cnn_scores = [cnn_report[label]["f1-score"] for label in class_labels]

        # Call the comparison plot function
        compare_model_performance_a(
            svm_scores=svm_scores,
            cnn_scores=cnn_scores,
            class_labels=class_labels,
            metric="F1-Score",
            save_path="A/figure/comparison_f1_score.png"
        )
        print("Comparison chart saved successfully.")

    except Exception as e:
        print(f"Error in handle_comparison_a: {e}")

def handle_task_b_resnet():
    """
    Handle ResNet model for training, testing, and evaluating BloodMNIST.
    Includes data loading, model training, evaluation, and visualization.
    """
    try:
        # Initialize log file
        log_file = "B/log/log.txt"
        with open(log_file, "a") as f:
            f.write("Training Log - ResNet\n")
            f.write("=" * 50 + "\n")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        log_message(f"Device: {device}", log_file)

        # Load BloodMNIST dataset
        print("Loading BloodMNIST dataset...")
        train_loader, val_loader, test_loader = load_bloodmnist(batch_size=32, data_dir="data")
        print("Dataset loaded successfully!")

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
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print("No pre-trained model found. Starting training...")
            log_message("No pre-trained model found. Starting training...", log_file)
            model, history, lr_history = train_resnet_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=10,
                lr=0.001,
                save_dir="B/models"
            )
            print("Training completed!")

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
        
        return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    except Exception as e:
        print(f"Error in ResNet workflow: {e}")
        log_message(f"Error in ResNet workflow: {e}", log_file)

def handle_task_b_vit():
    """
    Handle Vision Transformer (ViT) model for training, testing, and evaluating BloodMNIST.
    """
    try:
        # Initialize log file
        log_file = "B/log/log.txt"
        with open(log_file, "a") as f:
            f.write("Training Log - Vision Transformer\n")
            f.write("=" * 50 + "\n")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        log_message(f"Device: {device}", log_file)

        # Load BloodMNIST dataset for ViT
        train_loader, val_loader, test_loader = load_vit_bloodmnist(batch_size=32, data_dir="data")

        # Initialize Vision Transformer model
        model = get_vit_model(num_classes=8).to(device)

        start_time = datetime.now()
        log_message(f"Training started at: {start_time}", log_file)

        # Check for pre-trained model
        model_path = "B/models/vit_model.pth"
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}...")
            log_message(f"Loading pre-trained model from {model_path}", log_file)
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print("No pre-trained model found. Starting training...")
            log_message("No pre-trained model found. Starting training...", log_file)
            # Train the ViT model
            model, history, lr_history = train_vit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs= 8,
                lr=0.001,
                save_dir="B/models"
            )
            print("Training completed!")

        end_time = datetime.now()
        log_message(f"Training ended at: {end_time}", log_file)
        log_message(f"Training duration: {end_time - start_time}", log_file)

        # Log hyperparameters
        log_message("Hyperparameters:", log_file)
        log_message("  - Epochs: 8", log_file)
        log_message("  - Learning Rate: 0.001", log_file)
        log_message("  - Batch Size: 32", log_file)

        # Test the ViT model
        print("Testing the ViT model...")
        y_true, y_pred = test_vit_model(model, test_loader, device)

        # Generate classification report and confusion matrix
        target_names = [str(i) for i in range(8)]
        generate_classification_report(y_true, y_pred, target_names, log_file)
        plot_and_save_confusion_matrix(y_true, y_pred, target_names, save_path="B/figure/confusion_matrix_vit.png", log_file=log_file)

        # Predict and visualize
        predict_and_visualize_resnet(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir="B/figure",
            save_file="predictions_vit.png"
        )
        print("Prediction visualization saved!")

        return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    except Exception as e:
        print(f"Error in ViT workflow: {e}")
        log_message(f"Error in ViT workflow: {e}", log_file)

def handle_comparison_b(resnet_report, vit_report):
    """
    Compare ResNet and ViT performance using F1-Score.
    """
    try:
        class_labels = [label for label in resnet_report.keys() if isinstance(resnet_report[label], dict)]

        resnet_scores = [resnet_report[label]["f1-score"] for label in class_labels]
        vit_scores = [vit_report[label]["f1-score"] for label in class_labels]

        compare_model_performance(
            resnet_scores=resnet_scores,
            vit_scores=vit_scores,
            class_labels=class_labels,
            metric="F1-Score",
            save_path="B/figure/comparison_f1_score.png"
        )
        print("Comparison chart saved successfully.")

    except Exception as e:
        print(f"Error in handle_comparison: {e}")
        
def main():
    """
    Main function to handle Task A (SVM vs CNN) and Task B (ResNet vs ViT).
    """
    setup_directories()

    print("Choose task:")
    print("1. Task A: BreastMNIST (SVM vs CNN)")
    print("2. Task B: BloodMNIST (ResNet vs ViT)")
    task_choice = input("Enter your choice (1 or 2): ").strip()

    if task_choice == "1":
        # Run Task A workflows
        svm_report = handle_task_a_svm()
        cnn_report = handle_task_a_cnn()
        handle_comparison_a(svm_report, cnn_report)

    elif task_choice == "2":
        # Run Task B workflows
        resnet_report = handle_task_b_resnet()
        vit_report = handle_task_b_vit()
        handle_comparison_b(resnet_report, vit_report)

    else:
        print("Invalid task choice. Please restart and choose Task A or Task B.")

if __name__ == "__main__":
    main()

