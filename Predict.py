"""
Predict.py
==========
Run RT-DETR inference on a single image or an entire folder.
Saves annotated output images and a JSON file of all predictions.

Usage:
    conda activate inf117_rtdetr

    # single image:
    CUDA_VISIBLE_DEVICES=1 python Predict.py --input path/to/image.jpeg

    # folder of images:
    CUDA_VISIBLE_DEVICES=1 python Predict.py --input dataset/images/test/

    # adjust confidence threshold:
    CUDA_VISIBLE_DEVICES=1 python Predict.py --input path/to/image.jpeg --threshold 0.5

    # use a specific checkpoint:
    CUDA_VISIBLE_DEVICES=1 python Predict.py --input path/to/image.jpeg --weights output/checkpoints/model_epoch_0072.pth
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import cv2
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
PRED_DIR   = OUTPUT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Apical Lesion",
    "Main Root",
    "Main Canal",
    "Mesial Root",
    "Mesial Canal",
    "Distal Root",
    "Distal Canal",
    "Palatal Canal",
    "Palatal Root",
    "Root Canal Filling",
    "decay",
]
NUM_CLASSES      = len(CLASS_NAMES)
PRETRAINED_MODEL = "PekingU/rtdetr_r50vd"
IMG_SIZE         = 640
IMG_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

COLORS = [
    (255,  69,   0), (  0, 128, 255), (  0, 255, 128), (255, 215,   0),
    (128,   0, 255), (255,  20, 147), (  0, 255, 255), (255, 140,   0),
    ( 50, 205,  50), (220,  20,  60), (  0, 191, 255),
]


# ── LOGGING ───────────────────────────────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# ── MODEL LOADER ──────────────────────────────────────────────────────────────
def load_model(weights_path: Path, device):
    model = RTDetrForObjectDetection.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_image(model, processor, img_path: Path, device, threshold: float):
    """
    Returns list of dicts:
        {"label": str, "score": float, "box": [x1, y1, x2, y2]}
    """
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return None
    image_rgb      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = image_bgr.shape[:2]

    encoding = processor(
        images=image_rgb, return_tensors="pt",
        do_resize=True, size={"height": IMG_SIZE, "width": IMG_SIZE},
    )
    pixel_values = encoding["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([[H_orig, W_orig]]),
    )[0]

    detections = []
    for box, score, label in zip(
        results["boxes"].tolist(),
        results["scores"].tolist(),
        results["labels"].tolist(),
    ):
        x1, y1, x2, y2 = box
        detections.append({
            "label": CLASS_NAMES[label],
            "score": round(score, 4),
            "box":   [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    return detections


# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def draw_and_save(img_path: Path, detections: list, out_path: Path):
    image = cv2.imread(str(img_path))
    if image is None:
        return
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls_id = CLASS_NAMES.index(det["label"])
        color  = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det['label']} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), image)


# ── COLLECT IMAGES ─────────────────────────────────────────────────────────
def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMG_EXTENSIONS else []
    return sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True, help="Image file or folder")
    parser.add_argument("--weights",   default=str(OUTPUT_DIR / "model_best.pth"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--no-vis",    action="store_true", help="Skip saving annotated images")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("INF-117 Dental Detection — RT-DETR Predict")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("No GPU found.")
        sys.exit(1)
    device = torch.device("cuda:0")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        logger.error("Train the model first: python Train.py")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    images = collect_images(input_path)
    if not images:
        logger.error(f"No images found at: {input_path}")
        sys.exit(1)

    logger.info(f"Found {len(images)} image(s)")
    logger.info(f"Weights:   {weights_path}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"GPU:       {torch.cuda.get_device_name(0)}")
    logger.info("-" * 60)

    processor = RTDetrImageProcessor.from_pretrained(PRETRAINED_MODEL)
    model     = load_model(weights_path, device)

    all_results = []

    for img_path in images:
        detections = predict_image(model, processor, img_path, device, args.threshold)
        if detections is None:
            logger.warning(f"Could not read: {img_path}")
            continue

        n = len(detections)
        logger.info(f"  {img_path.name}  →  {n} detection(s)")
        for det in detections:
            logger.info(f"    [{det['label']}] score={det['score']}  box={det['box']}")

        # Save annotated image
        if not args.no_vis:
            out_path = PRED_DIR / img_path.name
            draw_and_save(img_path, detections, out_path)

        all_results.append({
            "file": str(img_path),
            "detections": detections,
        })

    # Save JSON results
    json_path = PRED_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("")
    logger.info(f"Annotated images  → {PRED_DIR}")
    logger.info(f"JSON results      → {json_path}")
    logger.info("Prediction complete.")


if __name__ == "__main__":
    main()
