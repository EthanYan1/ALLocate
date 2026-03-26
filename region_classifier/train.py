import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, num_classes: int):
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
    parser = argparse.ArgumentParser(description="Train region classifier")
    parser.add_argument("--train-dir", type=str, default="final_split_data/train")
    parser.add_argument("--val-dir", type=str, default="final_split_data/val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--save-path", type=str, default="region_classifier_new.pth")
    parser.add_argument("--plot-path", type=str, default="loss_curve.png")
    return parser.parse_args()


def build_dataloaders(train_dir, val_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )

    val_data = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_data, val_data, train_loader, val_loader


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, val_data, train_loader, val_loader = build_dataloaders(
        args.train_dir,
        args.val_dir,
        args.batch_size,
        args.num_workers
    )

    class_list = train_data.classes
    num_classes = len(class_list)

    print("Train classes:", train_data.classes)
    print("Val classes:", val_data.classes)

    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
            leave=False
        )

        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
            leave=False
        )

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_acc += torch.sum(preds == labels).item()

                for label, pred in zip(labels, preds):
                    label_idx = label.item()
                    pred_idx = pred.item()

                    class_total[label_idx] += 1
                    if label_idx == pred_idx:
                        class_correct[label_idx] += 1
                        tp[label_idx] += 1
                    else:
                        fp[pred_idx] += 1
                        fn[label_idx] += 1

                batch_acc = (preds == labels).float().mean().item()
                val_pbar.set_postfix(loss=loss.item(), acc=batch_acc)

        train_loss /= len(train_data)
        val_loss /= len(val_data)
        val_acc /= len(val_data)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        for cls_idx in range(num_classes):
            acc = class_correct[cls_idx] / class_total[cls_idx] if class_total[cls_idx] > 0 else 0.0
            precision = tp[cls_idx] / (tp[cls_idx] + fp[cls_idx]) if (tp[cls_idx] + fp[cls_idx]) > 0 else 0.0
            recall = tp[cls_idx] / (tp[cls_idx] + fn[cls_idx]) if (tp[cls_idx] + fn[cls_idx]) > 0 else 0.0

            print(f"Class {class_list[cls_idx]} Accuracy: {acc:.4f}")
            print(f"Class {class_list[cls_idx]} Precision: {precision:.4f}")
            print(f"Class {class_list[cls_idx]} Recall: {recall:.4f}")

    torch.save(model, args.save_path)
    print(f"Saved full model to: {args.save_path}")

    plt.figure(figsize=(6, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()