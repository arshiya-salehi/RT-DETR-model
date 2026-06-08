"""
Train_Stage2.py
===============
Specialized RT-DETR training script for Stage 2 (Canal Focus) of the Hierarchical pipeline.
Trains exclusively on root crops to detect canals (Main Canal, Mesial Canal, Distal Canal, Palatal Canal).

Key Enhancements for Canal Detection:
  - Input: Crop root images from Prepare_Stage2_Data.py
  - Preprocessing: Local CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Augmentations: Random Crop/Scale-up (RandomResizedCrop) & Random Vertical Flips
  - Model weights saved to: output/stage2_checkpoints/ and output/model_stage2_best.pth
"""

import os
import sys
import json
import logging
import math
import argparse
from datetime import datetime
from pathlib import Path

# GPU Lock to GPU 1 (default if not set)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import cv2
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
)
from pycocotools.coco import COCO

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
ANN_DIR     = DATASET_DIR / "annotations"
IMG_DIR     = DATASET_DIR / "images_stage2"
OUTPUT_DIR  = BASE_DIR / "output"
LOG_DIR     = BASE_DIR / "logs"
CKPT_DIR    = OUTPUT_DIR / "stage2_checkpoints"

for d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
NUM_CLASSES = len(CLASS_NAMES)  # Keep 11 classes to map head correctly without surgical modifications

FOCUS_CLASSES = {2, 4, 6, 7}  # Train strictly on canal instances

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
BATCH_SIZE    = 16
NUM_EPOCHS    = 100       # Crops converge faster than complex radiographs
BASE_LR       = 2e-4
WEIGHT_DECAY  = 1e-4
MAX_GRAD_NORM = 0.1
IMG_SIZE      = 1024      # High resolution to capture fine canal boundaries
NUM_WORKERS   = 16
SAVE_EVERY    = 10

PRETRAINED_MODEL = "PekingU/rtdetr_r50vd"

# ── LOGGING ───────────────────────────────────────────────────────────────────
def setup_logging():
    log_path = LOG_DIR / "train_stage2.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)

# ── DATASET ───────────────────────────────────────────────────────────────────
class DentalStage2Dataset(torch.utils.data.Dataset):
    """
    Reads Stage 2 (Root-cropped) images and translated canal annotations.
    Applies CLAHE local contrast enhancement and customized Stage 2 augmentations.
    """
    def __init__(self, ann_path: Path, img_dir: Path, processor, augment: bool = False):
        self.coco      = COCO(str(ann_path))
        self.img_dir   = img_dir
        self.processor = processor
        self.augment   = augment
        self.img_ids   = sorted(self.coco.imgs.keys())
        self.tv_tensors = tv_tensors
        
        # Setup CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Augmentation pipeline
        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),   # Added vertical flip
            # Random Crop / Scale-up: zooms in/out slightly and rescales to uniform 1024x1024
            T.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), antialias=True),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.GaussianBlur(kernel_size=(5, 7), sigma=(0.1, 3.0)),
            T.SanitizeBoundingBoxes(),
            T.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
        ]) if augment else None

    def _apply_clahe(self, image_tensor):
        """Applies CLAHE enhancement on the loaded PyTorch uint8 image tensor."""
        # Convert tensor to numpy grayscale (radiographs are essentially grayscale)
        img_np = image_tensor[0].numpy()
        enhanced_np = self.clahe.apply(img_np)
        
        # Convert back to tensor and replicate to 3 identical color channels
        enhanced_tensor = torch.from_numpy(enhanced_np).unsqueeze(0).repeat(3, 1, 1)
        return enhanced_tensor

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.img_dir / img_info["file_name"]

        # 1. Read original crop
        image = torchvision.io.read_image(str(img_path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # 2. Apply local CLAHE contrast enhancement
        image = self._apply_clahe(image)

        W_orig = img_info["width"]
        H_orig = img_info["height"]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes_xyxy = []
        labels = []
        for ann in anns:
            if ann["category_id"] not in FOCUS_CLASSES:
                continue
            x, y, bw, bh = ann["bbox"]
            boxes_xyxy.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            
        if len(boxes_xyxy) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        # 3. Apply augmentations
        if self.aug is not None:
            image_tv = self.tv_tensors.Image(image)
            boxes_tv = self.tv_tensors.BoundingBoxes(
                boxes_tensor, format="XYXY", canvas_size=(H_orig, W_orig)
            )
            out = self.aug({"image": image_tv, "boxes": boxes_tv, "labels": labels_tensor})
            image = out["image"]
            boxes_tv = out["boxes"]
            labels_tensor = out["labels"]
            boxes_tensor = boxes_tv.as_subclass(torch.Tensor)

        _, H_new, W_new = image.shape
        
        # 4. Normalize to CXCYWH format for RT-DETR
        final_boxes = []
        for box in boxes_tensor:
            x1, y1, x2, y2 = box.tolist()
            bw, bh = (x2 - x1), (y2 - y1)
            cx = (x1 + bw / 2) / W_new
            cy = (y1 + bh / 2) / H_new
            nw = bw / W_new
            nh = bh / H_new
            
            cx, cy, nw, nh = (
                max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)),
                max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))
            )
            final_boxes.append([cx, cy, nw, nh])

        # 5. Image Processor resizing + float normalization
        encoding = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": IMG_SIZE, "width": IMG_SIZE},
        )
        pixel_values = encoding["pixel_values"].squeeze(0)
        
        target = {
            "class_labels": labels_tensor,
            "boxes":        torch.tensor(final_boxes, dtype=torch.float32)
                            if final_boxes else torch.zeros((0, 4), dtype=torch.float32),
        }
        return pixel_values, target, img_id

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets      = [item[1] for item in batch]
    img_ids      = [item[2] for item in batch]
    return pixel_values, targets, img_ids

# ── TRAINING FUNCTIONS ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, logger):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (pixel_values, targets, _) in enumerate(loader):
        pixel_values = pixel_values.to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            lr_now = optimizer.param_groups[0]['lr']
            logger.info(
                f"  Epoch {epoch:3d} | Batch {batch_idx+1:3d}/{n_batches} "
                f"| Loss {loss.item():.4f} | LR {lr_now:.2e}"
            )

    return total_loss / n_batches

@torch.no_grad()
def validate(model, loader, device, logger):
    model.eval()
    total_loss = 0.0

    for pixel_values, targets, _ in loader:
        pixel_values = pixel_values.to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    logger.info(f"  Validation loss: {avg_loss:.4f}")
    return avg_loss

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("INF-117 Dental Detection — RT-DETR STAGE 2 Training (Canal-Focus)")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("No GPU found.")
        sys.exit(1)
    device   = torch.device("cuda:0")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    processor = RTDetrImageProcessor.from_pretrained(PRETRAINED_MODEL)

    # Datasets using the crop datasets
    train_ds = DentalStage2Dataset(
        ANN_DIR / "train_stage2.json", IMG_DIR / "train", processor, augment=True
    )
    val_ds = DentalStage2Dataset(
        ANN_DIR / "val_stage2.json", IMG_DIR / "val", processor, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )

    logger.info(f"Train crops: {len(train_ds)}")
    logger.info(f"Val crops:   {len(val_ds)}")
    logger.info(f"Epochs:      {args.epochs}  |  Batch size: {args.batch_size}")
    logger.info("-" * 60)

    model = RTDetrForObjectDetection.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Checkpoint for resume not found: {resume_path}")
            sys.exit(1)
        logger.info(f"Resuming model weights from checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state)
        
        # Parse epoch from checkpoint name (e.g. model_stage2_epoch_0070.pth -> 70)
        try:
            filename = resume_path.stem
            parts = filename.split("_")
            start_epoch = int(parts[-1]) + 1
            logger.info(f"Detected checkpoint epoch: {start_epoch - 1}. Resuming training from Epoch {start_epoch}...")
        except Exception as e:
            logger.warning(f"Could not parse epoch from filename: {e}. Starting from Epoch 1.")
            start_epoch = 1

    model.to(device)

    # Optimization Setup
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n],
         "lr": args.lr * 0.1},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": args.lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, logger
        )
        val_loss = validate(model, val_loader, device, logger)
        
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        # Save best Stage 2 weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = OUTPUT_DIR / "model_stage2_best.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ✓ New best Stage 2 model saved → {best_path}")

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"model_stage2_epoch_{epoch:04d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  Checkpoint saved → {ckpt_path}")

    # Save final model
    final_path = OUTPUT_DIR / "model_stage2_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\nFinal Stage 2 model saved → {final_path}")

    # Save history
    history_path = OUTPUT_DIR / "stage2_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Stage 2 training history saved → {history_path}")
    logger.info("Stage 2 training complete.")

if __name__ == "__main__":
    main()
