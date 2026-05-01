"""
Test.py
=======
Final evaluation of a trained RT-DETR model on the held-out test set.

⚠️  Run this script ONLY ONCE, after training is complete and a final
     model checkpoint has been selected via Validate.py.
     Running it multiple times can cause inadvertent overfitting to test data.

Outputs:
  - output/test_results/test_report.json    — full metrics report
  - output/test_results/visualizations/    — annotated prediction images
  - output/test_results/test.log           — console log

Usage:
    conda activate inf117_rtdetr
    CUDA_VISIBLE_DEVICES=1 python Test.py

    # test a specific checkpoint:
    CUDA_VISIBLE_DEVICES=1 python Test.py --weights output/checkpoints/model_epoch_0072.pth

    # skip visualizations (faster):
    CUDA_VISIBLE_DEVICES=1 python Test.py --no-vis

    # adjust confidence threshold:
    CUDA_VISIBLE_DEVICES=1 python Test.py --threshold 0.5
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
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
ANN_DIR     = DATASET_DIR / "annotations"
IMG_DIR     = DATASET_DIR / "images"
OUTPUT_DIR  = BASE_DIR / "output"
TEST_DIR    = OUTPUT_DIR / "test_results"
VIS_DIR     = TEST_DIR / "visualizations"

TEST_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

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
        handlers=[
            logging.FileHandler(TEST_DIR / "test.log"),
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


# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, processor, img_path: Path, device, threshold: float):
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return None, None, None, None
    image_rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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

    boxes_xyxy = results["boxes"].cpu()
    scores     = results["scores"].cpu()
    class_ids  = results["labels"].cpu()
    return boxes_xyxy, scores, class_ids, (H_orig, W_orig)


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


# ── COCO EVALUATION ───────────────────────────────────────────────────────────
def run_coco_eval(coco_gt, predictions, logger):
    if not predictions:
        logger.warning("No predictions — evaluation skipped.")
        return {}

    coco_dt   = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    metrics = {
        "mAP@50:95": round(float(stats[0]), 4),
        "mAP@50":    round(float(stats[1]), 4),
        "mAP@75":    round(float(stats[2]), 4),
        "AP_small":  round(float(stats[3]), 4),
        "AP_medium": round(float(stats[4]), 4),
        "AP_large":  round(float(stats[5]), 4),
    }

    logger.info("\n── Bounding Box Metrics ──────────────────────────────")
    for k, v in metrics.items():
        logger.info(f"  {k:<20}: {v}")

    # Per-class AP@50
    per_class = {}
    cat_ids = coco_gt.getCatIds()
    for cat_id, cat_name in zip(cat_ids, CLASS_NAMES):
        evaluator.params.catIds = [cat_id]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        ap50 = round(float(evaluator.stats[1]), 4) if len(evaluator.stats) > 1 else -1.0
        per_class[cat_name] = ap50
    metrics["per_class_AP50"] = per_class

    logger.info("\n── Per-Class AP@50 ───────────────────────────────────")
    for cls, ap in per_class.items():
        note = "  ← rare class" if cls == "Palatal Canal" else ""
        logger.info(f"  {cls:<22}: {ap}{note}")

    return metrics


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default=str(OUTPUT_DIR / "model_best.pth"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--no-vis",    action="store_true")
    parser.add_argument("--max-vis",   type=int, default=20)
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("INF-117 Dental Detection — RT-DETR Final Test Evaluation")
    logger.info("=" * 60)
    logger.warning("⚠  Only run this script ONCE. Running it multiple times on")
    logger.warning("   the test set can inflate reported performance.")
    logger.info("")

    if not torch.cuda.is_available():
        logger.error("No GPU found.")
        sys.exit(1)
    device = torch.device("cuda:0")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        logger.error("Train the model first: python Train.py")
        sys.exit(1)

    logger.info(f"Weights:   {weights_path}")
    logger.info(f"GPU:       {torch.cuda.get_device_name(0)}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("-" * 60)

    processor = RTDetrImageProcessor.from_pretrained(PRETRAINED_MODEL)
    model     = load_model(weights_path, device)

    coco_gt  = COCO(str(ANN_DIR / "test.json"))
    img_ids  = sorted(coco_gt.imgs.keys())
    predictions = []
    vis_saved   = 0

    logger.info(f"Running inference on {len(img_ids)} test images...")

    for img_id in img_ids:
        img_info = coco_gt.imgs[img_id]
        img_path = IMG_DIR / "test" / img_info["file_name"]

        boxes_xyxy, scores, class_ids, orig_size = run_inference(
            model, processor, img_path, device, args.threshold
        )
        if boxes_xyxy is None:
            logger.warning(f"Could not read: {img_path}")
            continue

        for (x1, y1, x2, y2), score, cls_id in zip(
            boxes_xyxy.tolist(), scores.tolist(), class_ids.tolist()
        ):
            predictions.append({
                "image_id":    img_id,
                "category_id": int(cls_id),
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       float(score),
            })

        if not args.no_vis and vis_saved < args.max_vis:
            out_path = VIS_DIR / img_info["file_name"]
            save_visualization(img_path, boxes_xyxy, scores, class_ids, out_path)
            vis_saved += 1

    logger.info(f"Total predictions: {len(predictions)}")

    metrics = run_coco_eval(coco_gt, predictions, logger)

    report = {
        "timestamp":  datetime.now().isoformat(),
        "weights":    str(weights_path),
        "split":      "test",
        "n_images":   len(img_ids),
        "n_preds":    len(predictions),
        "threshold":  args.threshold,
        "class_names": CLASS_NAMES,
        "metrics":    metrics,
    }
    report_path = TEST_DIR / "test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nReport saved         → {report_path}")
    logger.info(f"Visualizations saved → {VIS_DIR}  ({vis_saved} images)")
    logger.info("Test evaluation complete.")


if __name__ == "__main__":
    main()
