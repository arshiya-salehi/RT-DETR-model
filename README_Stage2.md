# INF-117 Dental Detection — Stage 2 (Canal Focus) Training Instructions

This document outlines the background queueing commands, active monitoring processes, and next steps for the 120-epoch Stage 2 training run on GPU 1. Use this file as a reference if you lose access to the chat log.

---

## 1. Current State & Objective
* **Goal:** Fine-tune the Stage 2 (Canal Focus) RT-DETR model for **120 epochs** to improve **Distal Canal** detection without overfitting.
* **Why 120 Epochs?** 
  * At **Epoch 70**, the model scored **22.60% AP50 on Distal Canal**.
  * By **Epoch 500**, due to severe overfitting, the score dropped to **8.13% AP50**.
  * Therefore, we are capping the training at **120 epochs** with validation-based early saving to capture the optimal generalization peak.
* **Hardware Constraint:** Must run strictly on **GPU 1** (GPU 0 is reserved for other students). 
* **Resource Sharing:** Your teammate's process (PID `1370829`, running `train_unet`) is currently occupying 18.4 GB of VRAM on GPU 1. We must wait for it to finish so we can train at the full batch size of **16**.

---

## 2. Training Launch Commands

### Option A: The "Queue and Run" Background Command (Highly Recommended)
This command polls your teammate's process every 10 seconds. The instant their training completes, it automatically launches your 120-epoch training at the full batch size of `16`.
> [!TIP]
> You can safely close your SSH connection immediately after running this command.

```bash
nohup bash -c 'while ps -p 1370829 > /dev/null; do sleep 10; done; CUDA_VISIBLE_DEVICES=1 conda run -n inf117_rtdetr python Train_Stage2.py --epochs 120 --batch-size 16 > logs/train_stage2_short.log 2>&1' &
```

### Option B: The Manual Command
If you prefer to wait manually and start the script yourself after their training finishes:
```bash
CUDA_VISIBLE_DEVICES=1 nohup conda run -n inf117_rtdetr python Train_Stage2.py --epochs 120 --batch-size 16 > logs/train_stage2_short.log 2>&1 &
```

### Option C: The Safe Fallback (If you want to train *while* they are still running)
If you cannot wait and must train immediately alongside their active job, reduce the batch size to `8` so it fits in the remaining memory:
```bash
CUDA_VISIBLE_DEVICES=1 nohup conda run -n inf117_rtdetr python Train_Stage2.py --epochs 120 --batch-size 8 > logs/train_stage2_short.log 2>&1 &
```

---

## 3. Monitoring & Troubleshooting

* **Check if the queue/monitor is still waiting:**
  ```bash
  ps aux | grep 1370829
  ```
  *(If you see a background bash script sleeping, the queue is active).*

* **Watch live training log and metrics:**
  ```bash
  tail -f logs/train_stage2_short.log
  ```

* **Verify active GPU memory and utilization:**
  ```bash
  nvidia-smi
  ```

---

## 4. Post-Training Evaluation

Once the 120-epoch run is complete, follow these steps to evaluate and visualize your results:

1. **Evaluate the Stage 2 Best Weights:**
   Run the Two-Stage evaluation pipeline using the newly trained weights:
   ```bash
   CUDA_VISIBLE_DEVICES=1 conda run -n inf117_rtdetr python Test_TwoStage.py --weights output/model_stage2_best.pth
   ```

2. **Inspect the Results:**
   * **Metrics report:** Saved to `output/test_twostage_results/metrics.json`
   * **Annotated visual predictions:** Saved in `output/test_twostage_results/visualizations/`

3. **Compare Results:**
   Compare the new AP50 score for Distal and Main canals against previous runs:
   * **Epoch 70 (Baseline):** Distal Canal = **22.60%** | Main Canal = **49.20%**
   * **Epoch 500 (Overfit):** Distal Canal = **8.13%** | Main Canal = **38.84%**
   * **Epoch 120 (Target):** *Aims to exceed 22.60% without dropping Main Canal performance.*
