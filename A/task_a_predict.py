import random
import torch
import matplotlib.pyplot as plt
import os

from task_a_utils import load_breastmnist
from task_a_model import BreastMNISTCNN
from task_a_train import load_model

def predict_and_visualize(model, test_loader, device, class_names=["Benign", "Malignant"], num_samples=9, save_dir="figure", save_file="predictions.png"):
    """
    Predict and visualize random samples from the test set, and save the result.

    Args:
        model (torch.nn.Module): Trained CNN model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the model on.
        class_names (list): List of class names.
        num_samples (int): Number of samples to visualize (default: 9).
        save_dir (str): Directory to save the figure.
        save_file (str): File name for the saved figure.
    """
    model.eval()  # Set the model to evaluation mode
    test_samples, test_labels, pred_labels = [], [], []

    # Randomly sample test data
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(inputs)):
                if len(test_samples) < num_samples:
                    test_samples.append(inputs[i].cpu().squeeze().numpy())
                    test_labels.append(labels[i])
                    pred_labels.append(preds[i])
                else:
                    break
            if len(test_samples) >= num_samples:
                break

    # Plot the predictions with color-coded titles
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 3, 3  # We have 9 test samples

    for i in range(len(test_samples)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(test_samples[i], cmap='gray')
        pred_label = pred_labels[i].item()
        true_label = test_labels[i].item()
        title_color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}", color=title_color)
        ax.axis('off')

    plt.tight_layout()

    # Save the figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_file)
    plt.savefig(save_path)
    print(f"Prediction visualization saved to {save_path}")

if __name__ == "__main__":
    # Load data
    _, _, test_loader = load_breastmnist(batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BreastMNISTCNN()
    model_path = "models/best_model.pth"
    model = load_model(model, model_path, device)

    # Predict and visualize
    predict_and_visualize(model, test_loader, device, save_dir="figure", save_file="predictions.png")

    # Exit program
    import sys
    sys.exit(0)