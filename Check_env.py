"""
Check_env.py
============
Verify that all dependencies for RT-DETR training are correctly installed
and that the GPU is accessible.

Usage:
    conda activate inf117_rtdetr
    CUDA_VISIBLE_DEVICES=1 python Check_env.py
"""

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("=" * 60)
print("INF-117 RT-DETR Environment Check")
print("=" * 60)

# ── Python ────────────────────────────────────────────────────────────────────
print(f"\n[1] Python: {sys.version}")

# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
try:
    import torch
    print(f"[2] PyTorch: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA version:   {torch.version.cuda}")
        print(f"    GPU count:      {torch.cuda.device_count()}")
        print(f"    GPU name:       {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU memory:     {mem:.1f} GB")
        # Quick tensor test
        x = torch.randn(100, 100).cuda()
        y = x @ x.T
        print(f"    GPU tensor test: PASSED (100×100 matmul)")
    else:
        print("    ⚠  No GPU detected — training will fail.")
except ImportError as e:
    print(f"[2] ✗ PyTorch not installed: {e}")

# ── Transformers (RT-DETR) ────────────────────────────────────────────────────
try:
    import transformers
    print(f"\n[3] Transformers: {transformers.__version__}")
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    print("    RTDetrForObjectDetection: OK")
    print("    RTDetrImageProcessor:     OK")
except ImportError as e:
    print(f"\n[3] ✗ Transformers not installed or too old: {e}")
    print("    Install: pip install transformers>=4.40.0")

# ── pycocotools ───────────────────────────────────────────────────────────────
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    print(f"\n[4] pycocotools: OK")
except ImportError as e:
    print(f"\n[4] ✗ pycocotools not installed: {e}")
    print("    Install: pip install pycocotools")

# ── torchvision ───────────────────────────────────────────────────────────────
try:
    import torchvision
    print(f"\n[5] torchvision: {torchvision.__version__}")
except ImportError as e:
    print(f"\n[5] ✗ torchvision not installed: {e}")

# ── OpenCV ────────────────────────────────────────────────────────────────────
try:
    import cv2
    print(f"\n[6] OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"\n[6] ✗ OpenCV not installed: {e}")
    print("    Install: pip install opencv-python")

# ── Dataset files ─────────────────────────────────────────────────────────────
from pathlib import Path
BASE_DIR = Path(__file__).parent
print(f"\n[7] Dataset structure check:")
expected = [
    BASE_DIR / "dataset" / "annotations" / "train.json",
    BASE_DIR / "dataset" / "annotations" / "val.json",
    BASE_DIR / "dataset" / "annotations" / "test.json",
    BASE_DIR / "dataset" / "images" / "train",
    BASE_DIR / "dataset" / "images" / "val",
    BASE_DIR / "dataset" / "images" / "test",
]
all_ok = True
for path in expected:
    exists = path.exists()
    status = "✓" if exists else "✗ MISSING"
    print(f"    {status}  {path.relative_to(BASE_DIR)}")
    if not exists:
        all_ok = False
if not all_ok:
    print("    Run python convert_to_coco.py first to build the dataset folder.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Environment check complete.")
print("If all items above show ✓ or OK, you are ready to train.")
print("  python Train.py")
print("=" * 60)
