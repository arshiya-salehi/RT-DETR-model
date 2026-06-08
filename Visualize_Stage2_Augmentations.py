import os
import cv2
import json
import random
import torch
import numpy as np
from pathlib import Path
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

# Setup directories
BASE_DIR = Path(__file__).parent
ANN_PATH = BASE_DIR / "dataset" / "annotations" / "train_stage2.json"
IMG_DIR = BASE_DIR / "dataset" / "images_stage2" / "train"
OUT_DIR = BASE_DIR / "augmented_samples_stage2"
OUT_DIR.mkdir(exist_ok=True)

if not ANN_PATH.exists():
    print(f"⚠️ Stage 2 annotations not found at: {ANN_PATH}")
    print("Please ensure Prepare_Stage2_Data.py has been run successfully.")
    exit(1)

with open(ANN_PATH, "r") as f:
    coco = json.load(f)

img_dicts = coco["images"]
random.shuffle(img_dicts)
sample_imgs = img_dicts[:20]

CLASS_NAMES = [
    "Apical Lesion", "Main Root", "Main Canal", "Mesial Root", "Mesial Canal",
    "Distal Root", "Distal Canal", "Palatal Canal", "Palatal Root", "Root Canal Filling", "decay"
]

FOCUS_CLASSES = {2, 4, 6, 7}  # Canal classes

COLORS = [
    (255, 69, 0), (0, 128, 255), (0, 255, 128), (255, 215, 0),
    (128, 0, 255), (255, 20, 147), (0, 255, 255), (255, 140, 0),
    (50, 205, 50), (220, 20, 60), (0, 191, 255)
]

IMG_SIZE = 1024

# Setup CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Stage 2 Augmentations (matching Train_Stage2.py)
aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), antialias=True),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    T.GaussianBlur(kernel_size=(5, 7), sigma=(0.1, 3.0)),
    T.SanitizeBoundingBoxes(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
])

def apply_clahe(image_tensor):
    img_np = image_tensor[0].numpy()
    enhanced_np = clahe.apply(img_np)
    enhanced_tensor = torch.from_numpy(enhanced_np).unsqueeze(0).repeat(3, 1, 1)
    return enhanced_tensor

print(f"Generating 20 augmented Stage 2 (Canal Focus Crop) samples in {OUT_DIR}/ ...")

count = 0
for img_info in sample_imgs:
    if count >= 20:
        break
    
    img_id = img_info["id"]
    file_name = img_info["file_name"]
    img_path = IMG_DIR / file_name
    
    if not img_path.exists():
        continue
        
    # Read Image
    image = torchvision.io.read_image(str(img_path))
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
        
    # Apply CLAHE
    image = apply_clahe(image)
    
    H_orig, W_orig = image.shape[1], image.shape[2]
    
    # Get annotations
    anns = [ann for ann in coco["annotations"] if ann["image_id"] == img_id]
    boxes_xyxy = []
    labels = []
    for ann in anns:
        if ann["category_id"] not in FOCUS_CLASSES:
            continue
        x, y, bw, bh = ann["bbox"]
        boxes_xyxy.append([x, y, x + bw, y + bh])
        labels.append(ann["category_id"])
        
    if len(boxes_xyxy) == 0:
        continue
        
    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Apply TV Tensors and Augmentations
    image_tv = tv_tensors.Image(image)
    boxes_tv = tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(H_orig, W_orig))
    
    try:
        out = aug({"image": image_tv, "boxes": boxes_tv, "labels": labels_tensor})
        aug_img = out["image"]
        aug_boxes = out["boxes"]
        aug_labels = out["labels"]
    except Exception as e:
        print(f"⚠️ Skipping an image due to transform error: {e}")
        continue
        
    # Convert back to numpy for CV2 drawing
    img_np = aug_img.permute(1, 2, 0).numpy().astype(np.uint8).copy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    boxes_list = aug_boxes.as_subclass(torch.Tensor).tolist()
    labels_list = aug_labels.tolist()
    
    for box, cls_id in zip(boxes_list, labels_list):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 3)
        
        label = CLASS_NAMES[cls_id]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_np, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img_np, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    out_path = OUT_DIR / f"aug_stage2_{count:02d}_{file_name}"
    cv2.imwrite(str(out_path), img_np)
    count += 1

print(f"✅ Generated {count} visualization samples successfully.")
