import argparse
from pathlib import Path
from collections import Counter

import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

REGION_CLASS_NAMES = ["adequate", "blood", "clot"]
ADEQUATE_CLASS_NAME = "adequate"

BLAST_IDX = 1
TYPICAL_IDX = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run region classifier, then YOLO cell counting on adequate regions."
    )
    parser.add_argument(
        "--regions-folder",
        type=str,
        required=True,
        help="Folder containing region images"
    )
    parser.add_argument(
        "--region-model",
        type=str,
        default="region_classifier_new.pth",
        help="Path to whole-model region classifier .pth"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="fold1_last.pt",
        help="Path to YOLO .pt model"
    )
    parser.add_argument(
        "--region-conf",
        type=float,
        default=None,
        help="Optional minimum softmax confidence for accepting adequate region. If omitted, all adequate predictions are accepted."
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images inside subfolders"
    )
    parser.add_argument(
        "--count-all-boxes",
        action="store_true",
        help="Count all YOLO detections instead of only the highest-confidence detection per region"
    )
    return parser.parse_args()


def load_region_classifier(model_path, device):
    # whole-model loading, matching your earlier workflow
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model


def get_image_paths(folder, recursive=False):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Regions folder not found: {folder}")

    if recursive:
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    return sorted(paths)


def predict_region_class(model, image_path, device, transform):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_conf = probs[0, pred_idx].item()

    pred_name = REGION_CLASS_NAMES[pred_idx]
    return pred_idx, pred_name, pred_conf


def run_yolo_on_region(yolo_model, image_path, conf_threshold, count_all_boxes=True):
    counts = Counter()

    results = yolo_model.predict(
        source=str(image_path),
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return counts

    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    if count_all_boxes:
        for cls_id in classes:
            counts[cls_id] += 1
    else:
        best_idx = confidences.argmax()
        best_cls = classes[best_idx]
        counts[best_cls] += 1

    return counts


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    region_model = load_region_classifier(args.region_model, device)
    yolo_model = load_yolo_model(args.yolo_model)

    yolo_class_names = yolo_model.names
    if isinstance(yolo_class_names, dict):
        ordered_yolo_names = [yolo_class_names[i] for i in sorted(yolo_class_names.keys())]
    else:
        ordered_yolo_names = list(yolo_class_names)

    image_paths = get_image_paths(args.regions_folder, recursive=args.recursive)

    if not image_paths:
        print("No images found.")
        return

    total_regions = 0
    adequate_regions = 0
    skipped_regions = 0

    region_class_counter = Counter()
    total_cell_counts = Counter()

    for img_path in image_paths:
        total_regions += 1

        pred_idx, pred_name, pred_conf = predict_region_class(
            region_model,
            img_path,
            device,
            transform
        )
        region_class_counter[pred_name] += 1

        is_adequate = (pred_name == ADEQUATE_CLASS_NAME)

        if args.region_conf is not None:
            is_adequate = is_adequate and (pred_conf >= args.region_conf)

        if not is_adequate:
            skipped_regions += 1
            continue

        adequate_regions += 1

        region_counts = run_yolo_on_region(
            yolo_model=yolo_model,
            image_path=img_path,
            conf_threshold=args.yolo_conf,
            count_all_boxes=args.count_all_boxes
        )

        total_cell_counts.update(region_counts)

    print("\nRegion Classifier Summary:")
    print(f"Total regions processed: {total_regions}")
    for cls_name in REGION_CLASS_NAMES:
        print(f"{cls_name}: {region_class_counter.get(cls_name, 0)}")
    print(f"Adequate regions sent to YOLO: {adequate_regions}")
    print(f"Skipped non-adequate regions: {skipped_regions}")

    print("\nCell Detection Summary on Adequate Regions")
    for cls_id, cls_name in enumerate(ordered_yolo_names):
        print(f"{cls_name}: {total_cell_counts.get(cls_id, 0)}")

    blast_count = total_cell_counts.get(BLAST_IDX, 0)
    typical_count = total_cell_counts.get(TYPICAL_IDX, 0)
    denom = blast_count + typical_count

    if denom == 0:
        print("\nBlast percentage: undefined (no normal or blast cells detected)")
    else:
        blast_percentage = blast_count / denom
        print(f"\nBlast percentage: {blast_percentage:.4f}")


if __name__ == "__main__":
    main()