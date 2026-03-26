import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate region classifier")
    parser.add_argument("--val-dir", type=str, default="final_split_data/val")
    parser.add_argument("--model-path", type=str, default="region_classifier_new.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="eval_outputs")
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        text_values = [[f"{cm[i, j]:.2f}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    else:
        text_values = [[str(cm[i, j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, text_values[i][j], ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_multi_roc(
    results_dict,
    figsize=(6, 5),
    colors=None,
    zoom_factor=5,
    zoom_location=2,
    zoom_bbox=(0.5, 0.9),
    zoom_xlim=(0, 0.1),
    zoom_ylim=(0.9, 1.0),
    xlabel_size=18,
    ylabel_size=18,
    tick_size=14,
    legend_size=13,
    inset_tick_size=11,
    save_path=None
):
    if colors is None:
        colors = ['#8b91c1', '#d9a9cd', '#424874', '#e6d3ae', '#7aa6c2', '#c27a7a']

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    for idx, (model_name, results) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        fpr = results["fpr"]
        tpr = results["tpr"]
        roc_auc = results["auc"]

        ax.plot(
            fpr,
            tpr,
            color=color,
            label=f"{model_name} (AUC = {roc_auc:.4f})",
            lw=2,
            alpha=0.8
        )

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("1 - Specificity", fontsize=xlabel_size)
    ax.set_ylabel("Sensitivity", fontsize=ylabel_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.legend(loc="lower right", fontsize=legend_size)

    axins = zoomed_inset_axes(
        ax,
        zoom_factor,
        loc=zoom_location,
        bbox_to_anchor=zoom_bbox,
        bbox_transform=ax.transAxes
    )

    for idx, (model_name, results) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        axins.plot(results["fpr"], results["tpr"], color=color, lw=2)

    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    axins.set_xticks(np.linspace(zoom_xlim[0], zoom_xlim[1], 3))
    axins.set_yticks(np.linspace(zoom_ylim[0], zoom_ylim[1], 3))
    axins.tick_params(axis="both", labelsize=inset_tick_size)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    val_data = torchvision.datasets.ImageFolder(
        root=args.val_dir,
        transform=transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    class_list = val_data.classes
    num_classes = len(class_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    val_loss = 0.0
    val_correct = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            val_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            val_correct += torch.sum(preds == labels).item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss /= len(val_data)
    val_acc = val_correct / len(val_data)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    plot_confusion_matrix(
        cm=cm,
        class_names=class_list,
        title="Confusion Matrix",
        save_path=save_dir / "confusion_matrix.png",
        normalize=False
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_list,
        title="Normalized Confusion Matrix",
        save_path=save_dir / "normalized_confusion_matrix.png",
        normalize=True
    )

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    results_dict = {}
    for i, class_name in enumerate(class_list):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        results_dict[class_name.capitalize()] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc
        }

        print(f"{class_name} AUC: {roc_auc:.4f}")

    plot_multi_roc(
        results_dict=results_dict,
        save_path=save_dir / "roc_curve.png"
    )


if __name__ == "__main__":
    main()