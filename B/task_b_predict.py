import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def predict_and_visualize_resnet(
    model, test_loader, device, save_dir="B/figure", save_file="predictions_resnet.png"
):
    """
    Predict and visualize results for test samples.

    Args:
        model (nn.Module): Trained ResNet model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to run predictions on.
        save_dir (str): Directory to save the visualization.
        save_file (str): File name for the saved visualization.
    """
    model.eval()
    test_samples, test_labels, pred_labels = [], [], []

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Custom class names (example: abbreviations or simplified names)
    class_names = {
        0: "RBC",
        1: "WBC",
        2: "Platelets",
        3: "Lymphocytes",
        4: "Monocytes",
        5: "Eosinophils",
        6: "Basophils",
        7: "Others"
    }

    # Collect predictions
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            test_samples.extend(inputs.cpu())
            test_labels.extend(labels.cpu())
            pred_labels.extend(predicted.cpu())

            # Only visualize the first batch for simplicity
            if len(test_samples) >= 9:
                break

    # Visualize predictions
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # 3x3 grid for 9 samples
    for idx, ax in enumerate(axes.flat):
        if idx < len(test_samples):
            # Convert tensor to numpy and adjust format
            sample = test_samples[idx].permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
            sample = np.clip(sample, 0, 1)  # Clip to valid range
            true_label = test_labels[idx].item()
            pred_label = pred_labels[idx].item()

            # Display sample
            ax.imshow(sample)
            title_color = "green" if true_label == pred_label else "red"
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                color=title_color
            )
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_file))
    print(f"Prediction visualization saved to {os.path.join(save_dir, save_file)}")


def compare_model_performance(
    resnet_scores, vit_scores, class_labels, metric="F1-Score", save_path="B/figure/comparison.png"
):
    """
    Compare performance between ResNet and ViT models using a bar chart.

    Args:
        resnet_scores (list): Scores (e.g., F1-Score) for ResNet on each class.
        vit_scores (list): Scores (e.g., F1-Score) for ViT on each class.
        class_labels (list): Labels for the classes.
        metric (str): Metric name (e.g., "F1-Score").
        save_path (str): Path to save the comparison chart.
    """
    assert len(resnet_scores) == len(vit_scores), "Score lists must have the same length."
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(len(resnet_scores))]

    # X-axis positions for the classes
    x = np.arange(len(class_labels))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width / 2, resnet_scores, width, label="ResNet", alpha=0.8, color="blue")
    bar2 = ax.bar(x + width / 2, vit_scores, width, label="ViT", alpha=0.8, color="orange")

    # Add labels, title, and legend
    ax.set_xlabel("Class Labels")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison Between ResNet and ViT")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")  # Rotate labels for better visibility
    ax.legend()

    # Add values on top of bars
    for bar in bar1:
        height = bar.get_height()
        if height > 0:  # Avoid adding annotations for zero values
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text above bars
                        textcoords="offset points",
                        ha='center', va='bottom')

    for bar in bar2:
        height = bar.get_height()
        if height > 0:  # Avoid adding annotations for zero values
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text above bars
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison chart saved to {save_path}")
