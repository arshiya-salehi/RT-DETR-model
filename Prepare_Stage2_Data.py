import os
import json
import cv2
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
ANN_DIR = DATASET_DIR / "annotations"
IMG_DIR = DATASET_DIR / "images"
OUT_IMG_DIR = DATASET_DIR / "images_stage2"

ROOT_CLASSES = {1, 3, 5, 8}       # Main Root, Mesial Root, Distal Root, Palatal Root
CANAL_CLASSES = {2, 4, 6, 7}      # Main Canal, Mesial Canal, Distal Canal, Palatal Canal
PADDING = 20                      # Add padding around the root crop to prevent boundary truncation

def get_overlap_fraction(root_bbox, canal_bbox):
    """
    Computes the fraction of the canal bounding box that is inside the root bounding box.
    COCO format: [x, y, w, h]
    """
    rx1, ry1, rw, rh = root_bbox
    rx2, ry2 = rx1 + rw, ry1 + rh
    
    cx1, cy1, cw, ch = canal_bbox
    cx2, cy2 = cx1 + cw, cy1 + ch
    
    # Calculate intersection box
    ix1 = max(rx1, cx1)
    iy1 = max(ry1, cy1)
    ix2 = min(rx2, cx2)
    iy2 = min(ry2, cy2)
    
    if ix2 > ix1 and iy2 > iy1:
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        canal_area = cw * ch
        return intersection_area / canal_area if canal_area > 0 else 0.0
    return 0.0

def process_split(split_name):
    print(f"\nProcessing '{split_name}' split...")
    
    ann_path = ANN_DIR / f"{split_name}.json"
    if not ann_path.exists():
        print(f"⚠️ Annotation file {ann_path} does not exist. Skipping.")
        return
        
    with open(ann_path, "r") as f:
        coco_data = json.load(f)
        
    images_dict = {img["id"]: img for img in coco_data["images"]}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)
        
    # Prepare output stage 2 COCO structure
    stage2_coco = {
        "images": [],
        "annotations": [],
        "categories": coco_data["categories"]  # Keep same 11 categories for consistency
    }
    
    split_out_dir = OUT_IMG_DIR / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)
    
    crop_image_id_counter = 1
    crop_ann_id_counter = 1
    
    for img_id, img_info in images_dict.items():
        original_img_path = IMG_DIR / split_name / img_info["file_name"]
        
        # Read image
        img = cv2.imread(str(original_img_path))
        if img is None:
            print(f"⚠️ Could not load image: {original_img_path}")
            continue
            
        h_orig, w_orig = img.shape[:2]
        img_anns = annotations_by_image.get(img_id, [])
        
        # Filter root annotations and canal annotations
        root_anns = [ann for ann in img_anns if ann["category_id"] in ROOT_CLASSES]
        canal_anns = [ann for ann in img_anns if ann["category_id"] in CANAL_CLASSES]
        
        # Crop each root instance
        for i, root_ann in enumerate(root_anns):
            rx, ry, rw, rh = root_ann["bbox"]
            
            # Add padding and clamp to boundaries
            x1 = max(0, int(rx - PADDING))
            y1 = max(0, int(ry - PADDING))
            x2 = min(w_orig, int(rx + rw + PADDING))
            y2 = min(h_orig, int(ry + rh + PADDING))
            
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            # Skip invalid crops
            if crop_w <= 0 or crop_h <= 0:
                continue
                
            crop_img = img[y1:y2, x1:x2]
            
            # Save cropped image
            base_name = Path(img_info["file_name"]).stem
            crop_filename = f"{base_name}_root_{i}.jpeg"
            crop_filepath = split_out_dir / crop_filename
            cv2.imwrite(str(crop_filepath), crop_img)
            
            # Register in stage 2 images
            stage2_coco["images"].append({
                "id": crop_image_id_counter,
                "file_name": crop_filename,
                "width": crop_w,
                "height": crop_h
            })
            
            # Search for overlapping canal instances to translate
            for canal_ann in canal_anns:
                overlap = get_overlap_fraction(root_ann["bbox"], canal_ann["bbox"])
                if overlap >= 0.5:
                    # Canal belongs to this root! Translate coordinates
                    cx, cy, cw, ch = canal_ann["bbox"]
                    
                    # Compute translated coordinates relative to crop origin (x1, y1)
                    new_cx = max(0.0, cx - x1)
                    new_cy = max(0.0, cy - y1)
                    
                    # Clamp bounding box sizes to stay within the cropped boundary
                    new_cw = min(float(crop_w) - new_cx, cw)
                    new_ch = min(float(crop_h) - new_cy, ch)
                    
                    if new_cw > 0 and new_ch > 0:
                        stage2_coco["annotations"].append({
                            "id": crop_ann_id_counter,
                            "image_id": crop_image_id_counter,
                            "category_id": canal_ann["category_id"],
                            "bbox": [new_cx, new_cy, new_cw, new_ch],
                            "area": float(new_cw * new_ch),
                            "iscrowd": 0
                        })
                        crop_ann_id_counter += 1
                        
            crop_image_id_counter += 1
            
    # Save the new stage 2 annotation file
    out_ann_path = ANN_DIR / f"{split_name}_stage2.json"
    with open(out_ann_path, "w") as f:
        json.dump(stage2_coco, f, indent=2)
        
    print(f"✅ Generated {len(stage2_coco['images'])} root crops and {len(stage2_coco['annotations'])} canal annotations.")
    print(f"✅ Saved Stage 2 annotations to: {out_ann_path}")

def main():
    print("=" * 60)
    print("Dental Object Detection — Crop Dataset Generator (Stage 2)")
    print("=" * 60)
    
    # Process train, validation, and test splits
    for split in ["train", "val", "test"]:
        process_split(split)
        
    print("\n🎉 Stage 2 Dataset Preparation Complete!")

if __name__ == "__main__":
    main()
