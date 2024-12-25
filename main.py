import os
import torch
from A.task_a_utils import load_breastmnist
from A.task_a_model import BreastMNISTCNN
from A.task_a_train import train_model, load_model, plot_history, test_model
from A.task_a_predict import predict_and_visualize
from A.task_a_train import plot_confusion_matrix  # Import the new function

def main():
    """
    Main function to handle the complete workflow for Task A:
    1. Load BreastMNIST dataset.
    2. Initialize or load the model.
    3. Train the model (if no pre-trained model exists).
    4. Test the model and save results.
    5. Predict random test samples and visualize results.
    """
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure necessary directories exist
    os.makedirs("A/models", exist_ok=True)
    os.makedirs("A/figure", exist_ok=True)

    # Load dataset
    print("Loading BreastMNIST dataset...")
    train_loader, val_loader, test_loader = load_breastmnist(batch_size=32)
    print("Dataset loaded successfully!")

    # Initialize the model
    model = BreastMNISTCNN().to(device)

    # Check if a pre-trained model exists
    model_path = "A/models/best_model.pth"
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model = load_model(model, model_path, device)
    else:
        # Train the model if no pre-trained model is available
        print("No pre-trained model found. Starting training...")
        model, history = train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001)
        print("Training completed!")

        # Plot training history
        plot_history(history, save_dir="figure", file_name="training_history.png")
        print("Training history saved!")

    # Test the model
    print("Testing the model...")
    y_true, y_pred = test_model(model, test_loader, device)  # Modify to return predictions
    print("Test results collected!")

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_dir="figure",
        file_name="confusion_matrix.png",
        class_names=["Benign", "Malignant"]
    )
    print("Confusion matrix saved!")

    # Predict and visualize results
    print("Predicting and visualizing test samples...")
    predict_and_visualize(model, test_loader, device, save_dir="figure", save_file="predictions.png")
    print("Prediction visualization saved!")

if __name__ == "__main__":
    main()