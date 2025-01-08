import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

    # Ensure save directory exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the figure
    figure_path = os.path.join(save_path, save_file)
    plt.savefig(figure_path)
    print(f"Prediction visualization saved to {figure_path}")

def plot_roc_auc(model, X_test, y_test, save_dir="A/figure", file_name="roc_auc_svm.png", log_path="A/log/log.txt"):
    """
    Plot and save the ROC-AUC curve for the SVM model.

    Args:
        model: Trained SVM model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): True labels.
        save_dir (str): Directory to save the plot.
        file_name (str): File name for the saved plot.
        log_path (str): Path to save the ROC-AUC score in the log.
    """
    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()
    print(f"ROC-AUC curve saved to {os.path.join(save_dir, file_name)}")

    # Log ROC-AUC score
    with open(log_path, "a") as log_file:
        log_file.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
    print(f"ROC-AUC Score: {roc_auc:.4f} logged to {log_path}")

def compare_model_performance_a(svm_scores, cnn_scores, class_labels, metric, save_path):
    """
    Generate a bar chart comparing the performance of SVM and CNN.

    Args:
        svm_scores (list): F1-Scores for SVM across different classes.
        cnn_scores (list): F1-Scores for CNN across different classes.
        class_labels (list): Class labels.
        metric (str): Metric being compared (e.g., F1-Score).
        save_path (str): Path to save the comparison chart.
    """
    # Bar positions
    x = np.arange(len(class_labels))
    width = 0.35

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, svm_scores, width, label="SVM", alpha=0.8)
    plt.bar(x + width / 2, cnn_scores, width, label="CNN", alpha=0.8)

    # Chart details
    plt.xlabel("Class Labels")
    plt.ylabel(metric)
    plt.title(f"Comparison of SVM and CNN Performance ({metric})")
    plt.xticks(x, class_labels)
    plt.ylim(0, 1)  # F1-Score range is [0, 1]
    plt.legend()

    # Save the chart
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison chart saved to {save_path}")
