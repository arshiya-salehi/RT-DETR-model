"""
Validate.py
===========
Evaluate a trained RT-DETR model on the validation set using COCO metrics.
Run this after every few epochs during training to track progress,
or on demand to inspect a specific checkpoint.

Outputs:
  - output/val_results/val_report.json    — full metrics report
  - output/val_results/visualizations/   — annotated prediction images
  - output/val_results/val.log           — console log

Usage:
    conda activate inf117_rtdetr
    CUDA_VISIBLE_DEVICES=1 python Validate.py

    # evaluate a specific checkpoint:
    CUDA_VISIBLE_DEVICES=1 python Validate.py --weights output/checkpoints/model_epoch_0030.pth

    # skip visualization images (faster):
    CUDA_VISIBLE_DEVICES=1 python Validate.py --no-vis

    # adjust confidence threshold:
    CUDA_VISIBLE_DEVICES=1 python Validate.py --threshold 0.5
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import ops
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
ANN_DIR     = DATASET_DIR / "annotations"
IMG_DIR     = DATASET_DIR / "images"
OUTPUT_DIR  = BASE_DIR / "output"

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
NUM_CLASSES    = len(CLASS_NAMES)
PRETRAINED_MODEL = "PekingU/rtdetr_r50vd"
IMG_SIZE       = 1024

# Distinct BGR colors per class (for cv2 visualization)
COLORS = [
    (255,  69,   0), (  0, 128, 255), (  0, 255, 128), (255, 215,   0),
    (128,   0, 255), (255,  20, 147), (  0, 255, 255), (255, 140,   0),
    ( 50, 205,  50), (220,  20,  60), (  0, 191, 255),
]


# ── LOGGING ───────────────────────────────────────────────────────────────────
def setup_logging(res_dir: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(res_dir / "eval.log"),
            logging.StreamHandler(sys.stdout),
        ],
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


# ── INFERENCE HELPERS ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, processor, img_path: Path, device, threshold: float):
    """
    Run RT-DETR inference on a single image.

    Returns:
        boxes_xyxy  — (N, 4) float32 in pixel coords [x1, y1, x2, y2]
        scores      — (N,) float32
        class_ids   — (N,) int
        orig_size   — (H, W) of the original image
    """
    image_bgr  = cv2.imread(str(img_path))
    if image_bgr is None:
        return None, None, None, None
    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = image_bgr.shape[:2]

    encoding = processor(
        images=image_rgb, return_tensors="pt",
        do_resize=True, size={"height": IMG_SIZE, "width": IMG_SIZE},
    )
    pixel_values = encoding["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)

    # Post-process: returns list of dicts with 'scores', 'labels', 'boxes' (cx,cy,w,h norm)
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([[H_orig, W_orig]]),
    )[0]

    boxes_xyxy = results["boxes"].cpu()       # (N, 4) pixel coords xyxy
    scores     = results["scores"].cpu()
    class_ids  = results["labels"].cpu()

    return boxes_xyxy, scores, class_ids, (H_orig, W_orig)


# ── COCO EVALUATION ───────────────────────────────────────────────────────────
def run_coco_eval(coco_gt, all_predictions, split: str, logger):
    """
    Run official COCO evaluation (AP@50:95, AP50, AP75, per-class AP).

    all_predictions: list of dicts matching COCO results format
    """
    if not all_predictions:
        logger.warning("No predictions to evaluate.")
        return {}

    coco_dt  = coco_gt.loadRes(all_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "mAP@50:95": round(float(stats[0]), 4),
        "mAP@50":    round(float(stats[1]), 4),
        "mAP@75":    round(float(stats[2]), 4),
        "AP_small":  round(float(stats[3]), 4),
        "AP_medium": round(float(stats[4]), 4),
        "AP_large":  round(float(stats[5]), 4),
    }

    # Per-class AP@50 ─────────────────────────────────────────────────────────
    per_class = {}
    cat_ids = coco_gt.getCatIds()
    for cat_id, cat_name in zip(cat_ids, CLASS_NAMES):
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap50 = float(coco_eval.stats[1]) if len(coco_eval.stats) > 1 else -1.0
        per_class[cat_name] = round(ap50, 4)
    metrics["per_class_AP50"] = per_class

    logger.info(f"\n── {split} Detection Metrics ──────────────────────────────")
    for k, v in metrics.items():
        if k != "per_class_AP50":
            logger.info(f"  {k:<20}: {v}")
    logger.info(f"\n── Per-class AP@50 ───────────────────────────────────────")
    for cls, ap in per_class.items():
        note = "  ← rare class (26 train instances)" if cls == "Palatal Canal" else ""
        logger.info(f"  {cls:<22}: {ap}{note}")

    return metrics


# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def save_visualization(img_path: Path, boxes_xyxy, scores, class_ids, out_path: Path):
    image = cv2.imread(str(img_path))
    if image is None:
        return
    for (x1, y1, x2, y2), score, cls_id in zip(
        boxes_xyxy.tolist(), scores.tolist(), class_ids.tolist()
    ):
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(image, label, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), image)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default=str(OUTPUT_DIR / "model_best.pth"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--no-vis",    action="store_true")
    parser.add_argument("--max-vis",   type=int, default=20)
    parser.add_argument("--split",     choices=["val", "test"], default="val")
    args = parser.parse_args()

    res_dir = OUTPUT_DIR / f"{args.split}_results"
    vis_dir = res_dir / "visualizations"
    res_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(res_dir)
    logger.info("=" * 60)
    logger.info(f"INF-117 Dental Detection — RT-DETR {args.split.capitalize()}")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("No GPU found.")
        sys.exit(1)
    device = torch.device("cuda:0")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        logger.error("Train the model first with: python Train.py")
        sys.exit(1)

    logger.info(f"Weights:   {weights_path}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"GPU:       {torch.cuda.get_device_name(0)}")
    logger.info("-" * 60)

    processor = RTDetrImageProcessor.from_pretrained(PRETRAINED_MODEL)
    model     = load_model(weights_path, device)

    coco_gt = COCO(str(ANN_DIR / f"{args.split}.json"))
    img_ids = sorted(coco_gt.imgs.keys())
    predictions = []
    vis_saved   = 0

    logger.info(f"Running inference on {len(img_ids)} {args.split} images...")

    for img_id in img_ids:
        img_info = coco_gt.imgs[img_id]
        img_path = IMG_DIR / args.split / img_info["file_name"]

        boxes_xyxy, scores, class_ids, orig_size = run_inference(
            model, processor, img_path, device, args.threshold
        )
        if boxes_xyxy is None:
            logger.warning(f"Could not read: {img_path}")
            continue

        # Accumulate COCO-format predictions
        for (x1, y1, x2, y2), score, cls_id in zip(
            boxes_xyxy.tolist(), scores.tolist(), class_ids.tolist()
        ):
            predictions.append({
                "image_id":    img_id,
                "category_id": int(cls_id),
                "bbox":        [x1, y1, x2 - x1, y2 - y1],   # COCO: [x,y,w,h]
                "score":       float(score),
            })

        # Save visualization
        if not args.no_vis and vis_saved < args.max_vis:
            out_path = vis_dir / img_info["file_name"]
            save_visualization(img_path, boxes_xyxy, scores, class_ids, out_path)
            vis_saved += 1

    logger.info(f"Total predictions: {len(predictions)}")

    # COCO evaluation
    metrics = run_coco_eval(coco_gt, predictions, args.split.capitalize(), logger)

    # Save report
    report = {
        "timestamp":  datetime.now().isoformat(),
        "weights":    str(weights_path),
        "split":      args.split,
        "threshold":  args.threshold,
        "n_images":   len(img_ids),
        "n_preds":    len(predictions),
        "metrics":    metrics,
    }
    report_path = res_dir / f"{args.split}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nReport saved      → {report_path}")
    logger.info(f"Visualizations    → {vis_dir}  ({vis_saved} images)")
    logger.info(f"{args.split.capitalize()} complete.")


if __name__ == "__main__":
    main()
