import os
import matplotlib.pyplot as plt
import torch

def predict_and_visualize_resnet(model, test_loader, device, save_dir="B/figure", save_file="predictions_resnet.png"):
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
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    class_names = [str(i) for i in range(8)]
    for idx, ax in enumerate(axes.flat):
        if idx < len(test_samples):
            sample = test_samples[idx].permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
            true_label = test_labels[idx].item()
            pred_label = pred_labels[idx].item()

            ax.imshow(sample)
            title_color = "green" if true_label == pred_label else "red"
            ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=title_color)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_file))
    print(f"Prediction visualization saved to {os.path.join(save_dir, save_file)}")