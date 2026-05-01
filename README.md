# INF-117 Dental Detection — RT-DETR

This folder contains the **RT-DETR** implementation for detecting dental
endodontic structures (roots, canals, lesions, etc.) in periapical X-rays.
It is a parallel implementation alongside the Mask R-CNN project and uses
the **same dataset** and **same COCO-format annotations**.

---

## Key Differences: RT-DETR vs Mask R-CNN

| Feature | Mask R-CNN (Detectron2) | RT-DETR (Transformers) |
|---|---|---|
| Task | Instance segmentation + detection | Detection only (bounding boxes) |
| Architecture | Two-stage CNN | End-to-end Transformer |
| Framework | Detectron2 | Hugging Face Transformers |
| Pretrained on | COCO | COCO |
| Input format | COCO JSON (segmentation) | COCO JSON (bbox only) |
| Inference speed | ~5 FPS | ~30 FPS |
| Accuracy | High (with masks) | Comparable on detection |

> **Note:** RT-DETR produces bounding boxes only, not segmentation masks.
> If you need pixel-level segmentation, use the Mask R-CNN scripts.

---

## Project Structure

```
rt_detr_dental/
├── Train.py              # Training script
├── Validate.py           # Validation metrics (run anytime)
├── Test.py               # Final test evaluation (run ONCE)
├── Predict.py            # Inference on new images
├── Check_env.py          # Dependency and GPU verification
├── requirements.txt      # Python dependencies
├── dataset/              # Symlink or copy from Mask R-CNN project
│   ├── annotations/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── images/
│       ├── train/
│       ├── val/
│       └── test/
├── output/
│   ├── model_best.pth         # Best checkpoint (lowest val loss)
│   ├── model_final.pth        # Last epoch checkpoint
│   ├── checkpoints/           # Periodic saves every 5 epochs
│   ├── val_results/           # Validation reports and visualizations
│   ├── test_results/          # Test reports and visualizations
│   ├── predictions/           # Single-image inference output
│   └── training_history.json  # Loss curve data
└── logs/
    └── train.log
```

---

## Setup Guide

### 1. Create Environment

```bash
conda create -n inf117_rtdetr python=3.11 -y
conda activate inf117_rtdetr
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Link the Dataset

The RT-DETR scripts use the **same dataset folder** as Mask R-CNN.
Copy or symlink it:

```bash
# Option A — symlink (saves disk space)
ln -s ../INF-117-project-main/dataset ./dataset

# Option B — copy
cp -r ../INF-117-project-main/dataset ./dataset
```

If you have not yet run `convert_to_coco.py` from the Mask R-CNN project,
do that first to create the COCO JSON files and the train/val/test splits.

### 4. Verify Setup

```bash
CUDA_VISIBLE_DEVICES=1 python Check_env.py
```

---

## Training Workflow

### Step 1 — Train

```bash
# Foreground (you can see live output):
CUDA_VISIBLE_DEVICES=1 python Train.py

# Background job (recommended on shared server):
CUDA_VISIBLE_DEVICES=1 nohup python Train.py > logs/train.log 2>&1 &
```

Training runs for **72 epochs** (~35–60 min on an A100).
A checkpoint is saved every 5 epochs to `output/checkpoints/`.
`model_best.pth` is updated whenever validation loss improves.

### Step 2 — Monitor

```bash
tail -f logs/train.log
```

### Step 3 — Validate (anytime)

```bash
CUDA_VISIBLE_DEVICES=1 python Validate.py
# Or on a specific checkpoint:
CUDA_VISIBLE_DEVICES=1 python Validate.py --weights output/checkpoints/model_epoch_0050.pth
```

### Step 4 — Test (ONCE, after training is done)

```bash
CUDA_VISIBLE_DEVICES=1 python Test.py
```

### Step 5 — Predict on New Images

```bash
# Single image:
CUDA_VISIBLE_DEVICES=1 python Predict.py --input path/to/image.jpeg

# Folder of images:
CUDA_VISIBLE_DEVICES=1 python Predict.py --input dataset/images/test/

# Adjust threshold:
CUDA_VISIBLE_DEVICES=1 python Predict.py --input path/to/image.jpeg --threshold 0.5
```

### Step 6 — Archive Results (GitHub Tracking)

When a training run finishes and you want to save the results to GitHub without committing massive `.pth` model weights, run:

```bash
python Save_Experiment.py "optional_note"
```

This will safely copy your logs, validation metrics, and exact hyperparameters (extracted automatically from `Train.py`) into a timestamped folder inside `experiments/`. You can then `git commit` this folder to track your progress safely!

---

## Output Metrics

All evaluation scripts report **COCO detection metrics** (bbox only):

- **mAP@50:95** — primary metric (strict)
- **mAP@50** — looser IoU threshold (easier to achieve)
- **mAP@75** — stricter IoU threshold
- **AP_small / AP_medium / AP_large** — by object size
- **Per-class AP@50** — one number per dental class

---

## Safety & Resource Notes

- **GPU Lock**: All scripts are hardcoded with `CUDA_VISIBLE_DEVICES=1`.
- **Background Jobs**: Always use `nohup` for training.
- **Test Set**: Only run `Test.py` once. Use `Validate.py` during development.
- **Disk**: `output/` can grow to ~2–3 GB. Monitor with `du -sh output/`.
