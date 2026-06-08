"""
Test_TwoStage.py
================
Unified Two-Stage Hierarchical Inference and Evaluation Pipeline.

Process:
  1. Load Stage 1 (Roots/Pathologies) Joint model (output/model_best.pth)
  2. Load Stage 2 (Canal Focus) Crop model (output/model_stage2_best.pth)
  3. For each test radiograph:
     - Run Stage 1 to detect roots, fillings, lesions, and decay.
     - For each predicted root (Main/Mesial/Distal/Palatal):
       - Crop root with padding and apply CLAHE local contrast enhancement.
       - Run Stage 2 on the crop to predict canals.
       - Map canal coordinates back to original radiograph coordinates.
  4. Aggregate all predictions and run unified COCO evaluations (mAP@50).
  5. Save visualizations of the unified detections.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Lock to GPU 1 (default if not set)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
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
TEST_DIR    = OUTPUT_DIR / "test_twostage_results"
VIS_DIR     = TEST_DIR / "visualizations"

TEST_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ── CLASSES ───────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apical Lesion",       # 0
    "Main Root",           # 1
    "Main Canal",          # 2
    "Mesial Root",         # 3
    "Mesial Canal",        # 4
    "Distal Root",         # 5
    "Distal Canal",        # 6
    "Palatal Canal",       # 7
    "Palatal Root",        # 8
    "Root Canal Filling",  # 9
    "decay",               # 10
]
NUM_CLASSES = len(CLASS_NAMES)

ROOT_CLASSES = {1, 3, 5, 8}       # Main Root, Mesial Root, Distal Root, Palatal Root
CANAL_CLASSES = {2, 4, 6, 7}      # Main Canal, Mesial Canal, Distal Canal, Palatal Canal
PATHOLOGY_CLASSES = {0, 9, 10}    # Apical Lesion, Root Canal Filling, decay

PRETRAINED_MODEL = "PekingU/rtdetr_r50vd"
IMG_SIZE         = 1024
PADDING          = 20

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
            logging.FileHandler(TEST_DIR / "test_twostage.log"),
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

ROOT_TO_CANAL_MAP = {
    1: 2,  # Main Root -> Main Canal
    3: 4,  # Mesial Root -> Mesial Canal
    5: 6,  # Distal Root -> Distal Canal
    8: 7   # Palatal Root -> Palatal Canal
}

# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_stage1_inference(model, processor, img_path: Path, device, threshold: float):
    """Runs Stage 1 on the full image."""
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return None, None, None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = image_bgr.shape[:2]

    encoding = processor(
        images=image_rgb, return_tensors="pt",
        do_resize=True, size={"height": IMG_SIZE, "width": IMG_SIZE},
    )
    pixel_values = encoding["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=torch.tensor([[H_orig, W_orig]])
    )[0]

    return results["boxes"].cpu(), results["scores"].cpu(), results["labels"].cpu(), (H_orig, W_orig)

@torch.no_grad()
def run_stage2_inference(model, processor, crop_img_bgr, device, threshold: float):
    """Runs Stage 2 on a single root crop."""
    if crop_img_bgr is None or crop_img_bgr.size == 0:
        return None, None, None
    
    # Radiographs are grayscale; apply local CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    H_crop, W_crop = crop_img_bgr.shape[:2]

    encoding = processor(
        images=enhanced_rgb, return_tensors="pt",
        do_resize=True, size={"height": IMG_SIZE, "width": IMG_SIZE},
    )
    pixel_values = encoding["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=torch.tensor([[H_crop, W_crop]])
    )[0]

    return results["boxes"].cpu(), results["scores"].cpu(), results["labels"].cpu()

# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def save_visualization(img_path: Path, predictions, out_path: Path):
    """Draws bounding boxes for unified two-stage predictions on the original image."""
    image = cv2.imread(str(img_path))
    if image is None:
        return
        
    for pred in predictions:
        cls_id = pred["category_id"]
        score = pred["score"]
        x, y, w, h = pred["bbox"]
        
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (int(x), int(y) - th - 4), (int(x) + tw, int(y)), color, -1)
        cv2.putText(image, label, (int(x), int(y) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
    cv2.imwrite(str(out_path), image)

# ── COCO EVALUATION ───────────────────────────────────────────────────────────
def run_coco_eval(coco_gt, predictions, logger):
    if not predictions:
        logger.warning("No predictions compiled — evaluation skipped.")
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

    logger.info("\n── Bounding Box Metrics (Two-Stage) ────────────────────────")
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

    logger.info("\n── Per-Class AP@50 (Two-Stage) ─────────────────────────────")
    for cls, ap in per_class.items():
        note = "  ← rare class" if cls == "Palatal Canal" else ""
        logger.info(f"  {cls:<22}: {ap}{note}")

    return metrics

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-weights", default=str(OUTPUT_DIR / "model_best.pth"))
    parser.add_argument("--stage2-weights", default=str(OUTPUT_DIR / "model_stage2_best.pth"))
    parser.add_argument("--threshold1", type=float, default=0.05, help="Stage 1 confidence threshold")
    parser.add_argument("--threshold2", type=float, default=0.05, help="Stage 2 confidence threshold")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--max-vis", type=int, default=20)
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("INF-117 Dental Detection — Two-Stage Hierarchical Evaluation")
    logger.info("=" * 70)

    if not torch.cuda.is_available():
        logger.error("No GPU found.")
        sys.exit(1)
    device = torch.device("cuda:0")

    stage1_path = Path(args.stage1_weights)
    stage2_path = Path(args.stage2_weights)

    if not stage1_path.exists():
        logger.error(f"Stage 1 weights not found: {stage1_path}")
        sys.exit(1)
    if not stage2_path.exists():
        logger.error(f"Stage 2 weights not found: {stage2_path}")
        sys.exit(1)

    logger.info(f"Stage 1 Weights: {stage1_path} (Thresh: {args.threshold1})")
    logger.info(f"Stage 2 Weights: {stage2_path} (Thresh: {args.threshold2})")
    logger.info("-" * 70)

    processor = RTDetrImageProcessor.from_pretrained(PRETRAINED_MODEL)
    model_s1 = load_model(stage1_path, device)
    model_s2 = load_model(stage2_path, device)

    coco_gt = COCO(str(ANN_DIR / "test.json"))
    img_ids = sorted(coco_gt.imgs.keys())
    
    final_predictions = []
    vis_saved = 0

    logger.info(f"Running two-stage pipeline on {len(img_ids)} test images...")

    for step, img_id in enumerate(img_ids):
        img_info = coco_gt.imgs[img_id]
        img_path = IMG_DIR / "test" / img_info["file_name"]

        # Read full image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        h_orig, w_orig = img_bgr.shape[:2]

        # 1. Run Stage 1 (Roots, Fillings, Lesions, Decay)
        s1_boxes, s1_scores, s1_labels, _ = run_stage1_inference(
            model_s1, processor, img_path, device, args.threshold1
        )
        if s1_boxes is None:
            continue

        image_preds = []

        # 2. Iterate through predictions
        for box, score, label_id in zip(s1_boxes.tolist(), s1_scores.tolist(), s1_labels.tolist()):
            x1, y1, x2, y2 = box
            
            # Map predictions we want to KEEP from Stage 1 directly (non-canal predictions)
            if label_id in ROOT_CLASSES or label_id in PATHOLOGY_CLASSES:
                # Convert back to COCO format [x, y, w, h]
                image_preds.append({
                    "image_id": img_id,
                    "category_id": label_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score
                })

            # If it's a predicted root, crop it and run Stage 2 (Canals focus)
            if label_id in ROOT_CLASSES:
                rx1 = max(0, int(x1 - PADDING))
                ry1 = max(0, int(y1 - PADDING))
                rx2 = min(w_orig, int(x2 + PADDING))
                ry2 = min(h_orig, int(y2 + PADDING))

                crop_w = rx2 - rx1
                crop_h = ry2 - ry1

                if crop_w <= 0 or crop_h <= 0:
                    continue

                crop_img = img_bgr[ry1:ry2, rx1:rx2]

                # Run Stage 2 inference on this cropped root region
                s2_boxes, s2_scores, s2_labels = run_stage2_inference(
                    model_s2, processor, crop_img, device, args.threshold2
                )

                if s2_boxes is not None and len(s2_boxes) > 0:
                    for s2_box, s2_score, s2_label_id in zip(
                        s2_boxes.tolist(), s2_scores.tolist(), s2_labels.tolist()
                    ):
                        # Keep only canal labels from Stage 2
                        if s2_label_id in CANAL_CLASSES:
                            s2_x1, s2_y1, s2_x2, s2_y2 = s2_box

                            # Translate back from crop space to original image space
                            orig_cx1 = rx1 + s2_x1
                            orig_cy1 = ry1 + s2_y1
                            orig_cw = s2_x2 - s2_x1
                            orig_ch = s2_y2 - s2_y1

                            # Anatomical mapping override: map canal to parent root's canal category
                            mapped_canal_id = ROOT_TO_CANAL_MAP.get(label_id, s2_label_id)

                            # Append translated canal prediction
                            image_preds.append({
                                "image_id": img_id,
                                "category_id": mapped_canal_id,
                                "bbox": [orig_cx1, orig_cy1, orig_cw, orig_ch],
                                "score": s2_score
                            })

        final_predictions.extend(image_preds)

        # Save Visualizations
        if not args.no_vis and vis_saved < args.max_vis:
            out_path = VIS_DIR / img_info["file_name"]
            save_visualization(img_path, image_preds, out_path)
            vis_saved += 1

        if (step + 1) % 10 == 0 or (step + 1) == len(img_ids):
            logger.info(f"  Processed {step + 1}/{len(img_ids)} images...")

    logger.info(f"\nTotal Two-Stage predictions: {len(final_predictions)}")

    # 5. Run COCO evaluations
    metrics = run_coco_eval(coco_gt, final_predictions, logger)

    # 6. Save final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "stage1_weights": str(stage1_path),
        "stage2_weights": str(stage2_path),
        "threshold1": args.threshold1,
        "threshold2": args.threshold2,
        "n_images": len(img_ids),
        "n_preds": len(final_predictions),
        "metrics": metrics
    }
    
    report_path = TEST_DIR / "twostage_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nUnified Two-Stage Report saved → {report_path}")
    logger.info(f"Visualizations saved          → {VIS_DIR}  ({vis_saved} images)")
    logger.info("Hierarchical evaluation complete.")

if __name__ == "__main__":
    main()
