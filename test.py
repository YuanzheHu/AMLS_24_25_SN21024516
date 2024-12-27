import os
import torch
from B.task_b_utils import load_bloodmnist
from B.task_b_model import get_resnet_model
from B.task_b_train import train_model, test_model
from B.task_b_predict import predict_and_visualize_resnet
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    """
    Main script for training and testing ResNet on BloodMNIST.
    Includes data loading, model training, evaluation, and visualization.
    """
    # Setup directories
    os.makedirs("B/data", exist_ok=True)
    os.makedirs("B/models", exist_ok=True)
    os.makedirs("B/figure", exist_ok=True)
    os.makedirs("B/log", exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BloodMNIST dataset
    print("Loading BloodMNIST dataset...")
    train_loader, val_loader, test_loader = load_bloodmnist(batch_size=32, data_dir="B/data")
    print("Dataset loaded successfully!")

    # Initialize ResNet model
    model = get_resnet_model(num_classes=8).to(device)

    # Check if a pre-trained model exists
    model_path = "B/models/resnet_model.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        model.load_state_dict(torch.load(model_path))
    else:
        # Train the model if no pre-trained model exists
        print("No pre-trained model found. Starting training...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=10,
            lr=0.001,
            save_dir="B/models"
        )
        print("Training completed!")

    # Test the model
    print("Testing the model...")
    y_true, y_pred = test_model(model, test_loader, device)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(8)])
    print("\nClassification Report:")
    print(report)

    # Save classification report
    report_path = os.path.join("B/log", "classification_report_resnet.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(8)])
    disp.plot(cmap=plt.cm.Blues)

    # Save confusion matrix
    cm_path = os.path.join("B/figure", "confusion_matrix_resnet.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Predict and visualize test samples
    print("Predicting and visualizing test samples...")
    predict_and_visualize_resnet(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir="B/figure",
        save_file="predictions_resnet.png"
    )
    print("Prediction visualization saved!")

if __name__ == "__main__":
    main()