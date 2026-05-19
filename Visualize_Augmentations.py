import os
import cv2
import json
import random
import argparse
import torch
import numpy as np
from pathlib import Path
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

parser = argparse.ArgumentParser()
parser.add_argument("--focus", choices=["all", "roots", "canals", "pathologies"], default="all",
                    help="Filter the visualization to show only specific anatomical groups.")
args = parser.parse_args()

# Setup directories
BASE_DIR = Path(__file__).parent
ANN_PATH = BASE_DIR / "dataset" / "annotations" / "train.json"
IMG_DIR = BASE_DIR / "dataset" / "images" / "train"
OUT_DIR = BASE_DIR / "augmented_samples"
OUT_DIR.mkdir(exist_ok=True)

with open(ANN_PATH, "r") as f:
    coco = json.load(f)

img_dicts = coco["images"]
random.shuffle(img_dicts)
sample_imgs = img_dicts[:20]

# Class Names
CLASS_NAMES = [
    "Apical Lesion", "Main Root", "Main Canal", "Mesial Root", "Mesial Canal",
    "Distal Root", "Distal Canal", "Palatal Canal", "Palatal Root", "Root Canal Filling", "decay"
]

FOCUS_GROUPS = {
    "all": set(range(11)),
    "roots": {1, 3, 5, 8},
    "canals": {2, 4, 6, 7},
    "pathologies": {0, 10}
}
focus_classes = FOCUS_GROUPS[args.focus]

# Color Palette
COLORS = [(255, 69, 0), (0, 128, 255), (0, 255, 128), (255, 215, 0),
          (128, 0, 255), (255, 20, 147), (0, 255, 255), (255, 140, 0),
          (50, 205, 50), (220, 20, 60), (0, 191, 255)]

# Augmentations (Same as Train.py)
aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    T.SanitizeBoundingBoxes(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])

print(f"Generating 20 augmented samples (Focus: {args.focus.upper()}) in {OUT_DIR}/ ...")

count = 0
for img_info in sample_imgs:
    if count >= 20: break
    
    img_id = img_info["id"]
    file_name = img_info["file_name"]
    img_path = IMG_DIR / file_name
    
    # Read Image
    image = torchvision.io.read_image(str(img_path))
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    
    H_orig, W_orig = image.shape[1], image.shape[2]
    
    # Get annotations
    anns = [ann for ann in coco["annotations"] if ann["image_id"] == img_id]
    boxes_xyxy = []
    labels = []
    for ann in anns:
        if ann["category_id"] not in focus_classes:
            continue
        x, y, bw, bh = ann["bbox"]
        boxes_xyxy.append([x, y, x+bw, y+bh])
        labels.append(ann["category_id"])
        
    if len(boxes_xyxy) == 0:
        continue
        
    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Apply TV Tensors and Augmentations
    image_tv = tv_tensors.Image(image)
    boxes_tv = tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(H_orig, W_orig))
    
    # Run augmentation
    out = aug({"image": image_tv, "boxes": boxes_tv, "labels": labels_tensor})
    aug_image = out["image"]
    aug_boxes = out["boxes"]
    aug_labels = out["labels"]
    
    # Convert back to numpy for CV2 drawing
    img_np = aug_image.permute(1, 2, 0).numpy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_np = np.ascontiguousarray(img_np)
    
    # Draw boxes
    for box, label in zip(aug_boxes, aug_labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_id = int(label.item())
        color = COLORS[cls_id % len(COLORS)]
        
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        
        text = CLASS_NAMES[cls_id]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_np, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img_np, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    out_path = OUT_DIR / f"aug_{file_name}"
    cv2.imwrite(str(out_path), img_np)
    count += 1

print("Done! Check the 'augmented_samples' folder.")
