<div align="center">

# 🏜️ OFF-ROAD AUTONOMOUS VISION
### Robust Semantic Segmentation for Desert Terrains

<br>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-FF6B2B?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Hackathon_Submission-22C55E?style=for-the-badge)

<br>

<img src="assets/banner.png" alt="Project Banner" width="100%">

<br>

**Hybrid loss optimization for class-imbalanced terrain segmentation**
**in autonomous off-road navigation systems.**

<br>

[📊 View Report](#-report) · [🚀 Quick Start](#-quick-start) · [🎮 Launch UI](#-web-interface) · [📈 Results](#-results) · [🧠 Models](#-models-trained)

<br>

---
</div>

<br>

## 🎯 Project Overview

<table>
<tr>
<td width="60%">

We built an end-to-end **semantic segmentation system** that identifies **10 terrain classes** in desert off-road environments for autonomous vehicle navigation.

The core challenge? **Extreme class imbalance.** Sky dominates ~40% of pixels while safety-critical obstacles like Logs represent just ~0.5%. Standard models go blind to what matters most.

**Our solution:** A hybrid **Cross-Entropy + Dice Loss** function that forces the model to detect small, dangerous obstacles with the same priority as large background regions.

We trained **4 models** in a systematic ablation study to isolate and quantify the impact of each optimization strategy.

</td>
<td width="40%">

```text
📊 KEY RESULTS
─────────────────────
Baseline mIoU → XX.XX%
Final mIoU → XX.XX%
Logs IoU Gain → XX.XX%
Models Trained → 4
Classes → 10
Resolution → 512×512
```

</td>
</tr>
</table>

<br>

## ⚡ Tech Stack

<div align="center">

| Category | Technologies |
|:--------:|:------------|
| 🧠 **Core ML** | `PyTorch 2.x` · `segmentation-models-pytorch` · `CUDA 12.1` |
| 📸 **Vision** | `OpenCV` · `Albumentations` · `Pillow` |
| 📊 **Analysis** | `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` |
| 🌐 **Web UI** | `Gradio` · `FastAPI` (optional) |
| 🛠️ **DevOps** | `Miniconda` · `Git` · `GitHub` |
| 🏗️ **Architecture** | `U-Net` · `ResNet34 (ImageNet)` |
| 📉 **Loss** | `CrossEntropy + DiceLoss (Hybrid)` |

</div>

<br>

## 📁 Project Structure

```text
off-road-vision/
│
├── 🧠 model/
│   ├── train.py          ← Training script (all 4 models)
│   ├── test.py           ← Evaluation + per-class IoU
│   ├── model.py          ← U-Net architecture definition
│   ├── dataset.py        ← Custom dataset + map_mask()
│   ├── losses.py         ← CE Loss, Dice Loss, Hybrid Loss
│   └── config.py         ← All hyperparameters
│
├── 🎮 ui/
│   ├── app.py            ← Gradio web interface
│   ├── inference.py      ← Single-image prediction pipeline
│   └── utils.py          ← Visualization helpers
│
├── 📊 outputs/
│   ├── predictions/      ← Model prediction visualizations
│   ├── graphs/           ← Loss curves, IoU charts
│   └── failure_cases/    ← Documented failure examples
│
├── 💾 weights/
│   ├── model_a_baseline.pth ← CE only, no augmentation
│   ├── model_b_ce_aug.pth   ← CE + augmentation
│   ├── model_c_hybrid.pth   ← CE+Dice, no augmentation
│   └── model_d_final.pth    ← CE+Dice + augmentation ★
│
├── 📄 report/
│   ├── report.pdf        ← Final hackathon report
│   └── assets/           ← Report images and figures
│
├── requirements.txt
├── environment.yml
└── README.md             ← You are here
```

<br>

## 🚀 Quick Start

### Prerequisites
```bash
# Make sure you have:
# ✓ Python 3.10+
# ✓ NVIDIA GPU with CUDA 12.1+ (recommended)
# ✓ ~4GB free GPU memory
# ✓ ~2GB disk space for dataset
```

**Option 1: Conda (Recommended)**
```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)[your-username]/off-road-vision.git
cd off-road-vision

# 2. Create conda environment
conda create -n offroad python=3.10 -y
conda activate offroad

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio \
    --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Option 2: pip only**
```bash
git clone [https://github.com/](https://github.com/)[your-username]/off-road-vision.git
cd off-road-vision
pip install -r requirements.txt
```

<br>

## 🧠 Models Trained

We trained **4 models** in a systematic **2×2 ablation study** to isolate contributions:

<div align="center">

```text
                    ┌─────────────────────────────────────────┐
                    │         AUGMENTATION                    │
                    │     None          H+V Flip              │
                    ├──────────────┬──────────────────────────┤
          CE Only   │   MODEL A    │      MODEL B             │
LOSS                │   Baseline   │   + Augmentation         │
FUNCTION            │   🔴         │   🟠                     │
                    ├──────────────┼──────────────────────────┤
          CE+Dice   │   MODEL C    │      MODEL D  ★          │
          (Hybrid)  │   + Dice     │   + Both (FINAL)         │
                    │   🔵         │   🟢                     │
                    └──────────────┴──────────────────────────┘
```

</div>

<br>

| Model | Loss Function | Augmentation | mIoU | Logs IoU | Status |
|:---|:---|:---|:---|:---|:---|
| 🔴 **A** | CE Only | None | XX.XX% | XX.XX% | Baseline |
| 🟠 **B** | CE Only | H+V Flip | XX.XX% | XX.XX% | + Aug |
| 🔵 **C** | CE + Dice | None | XX.XX% | XX.XX% | + Loss |
| 🟢 **D** | CE + Dice | H+V Flip | XX.XX% | XX.XX% | Final ★ |

<br>

## 🏋️ Training

**Train All 4 Models**
```bash
# Train Model A — Baseline (CE only, no augmentation)
python model/train.py --model a --loss ce --augment none

# Train Model B — CE + Augmentation
python model/train.py --model b --loss ce --augment flip

# Train Model C — Hybrid Loss, no augmentation
python model/train.py --model c --loss hybrid --augment none

# Train Model D — Hybrid Loss + Augmentation (Final)
python model/train.py --model d --loss hybrid --augment flip
```

**Train with Custom Config**
```bash
python model/train.py \
    --model d \
    --loss hybrid \
    --augment flip \
    --epochs 15 \
    --batch-size 6 \
    --lr 0.0001 \
    --input-size 512 \
    --seed 42
```

**Training Configuration**
```yaml
# config.py — All hyperparameters
ARCHITECTURE:     U-Net + ResNet34
ENCODER_WEIGHTS:  imagenet
OPTIMIZER:        Adam (β1=0.9, β2=0.999, ε=1e-8)
LEARNING_RATE:    1e-4
LR_SCHEDULER:     CosineAnnealingLR (T_max=15)
LOSS_FUNCTION:    CrossEntropy + DiceLoss (Hybrid)
EPOCHS:           15
BATCH_SIZE:       6
INPUT_SIZE:       512 × 512
NUM_CLASSES:      10
RANDOM_SEED:      42
CHECKPOINT:       Min Validation Loss
```

<br>

## 📊 Evaluation

**Evaluate Any Model**
```bash
# Evaluate Model D (final submission)
python model/test.py --weights weights/model_d_final.pth

# Evaluate all 4 models and compare
python model/test.py --compare-all

# Generate per-class IoU breakdown
python model/test.py --weights weights/model_d_final.pth --detailed
```

**Expected Output**
```text
╔══════════════════════════════════════════════════════╗
║           MODEL D — EVALUATION RESULTS               ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Class            IoU        Status                  ║
║  ─────────────────────────────────────               ║
║  Sky              XX.XX%     ✅ Strong               ║
║  Trees            XX.XX%     ✅ Strong               ║
║  Lush Bushes      XX.XX%     ✅ Improved             ║
║  Landscape        XX.XX%     ✅ Improved             ║
║  Rocks            XX.XX%     ✅ Improved             ║
║  Logs ⭐          XX.XX%     🏆 Critical Win         ║
║  Dry Bushes       XX.XX%     ✅ Improved             ║
║  Gravel Path      XX.XX%     ✅ Improved             ║
║  Sand             XX.XX%     ✅ Improved             ║
║  Dry Grass        XX.XX%     ✅ Improved             ║
║  ─────────────────────────────────────               ║
║  Mean IoU         XX.XX%     🏆 Final Score          ║
║  Pixel Accuracy   XX.XX%                             ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

<br>

## 🎮 Web Interface

We built a **Gradio-powered web UI** for real-time terrain segmentation inference.

**Launch the UI**
```bash
# Start the web interface
python ui/app.py

# Or specify a custom port
python ui/app.py --port 7860 --share
```

**What You Can Do**
```text
┌─────────────────────────────────────────────────────────┐
│  🎮 OFF-ROAD VISION — WEB INTERFACE                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📤 UPLOAD           Drop any desert terrain image      │
│                      (supports JPG, PNG, BMP)           │
│                                                         │
│  🧠 SELECT MODEL     Choose from all 4 trained models   │
│                      Model A / B / C / D                │
│                                                         │
│  🎯 SEGMENT          One-click semantic segmentation    │
│                      Real-time inference on GPU         │
│                                                         │
│  📊 VIEW RESULTS     Side-by-side comparison:           │
│                      Original → Overlay → Class Mask    │
│                                                         │
│  📋 CLASS LEGEND     Color-coded terrain class labels   │
│                      with confidence percentages        │
│                                                         │
│  💾 DOWNLOAD         Save prediction mask as PNG        │
│                                                         │
│  🔄 COMPARE          Run same image through all 4       │
│                      models side-by-side                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**UI Preview**
```text
┌─────────────────────────────────────────────────────────────────┐
│  🏜️ Off-Road Autonomous Vision                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │                 │  │                 │  │                │   │
│  │   📤 Upload     │  │  🎯 Segmented   │  │ 🗺️ Class Mask  │   │
│  │   Image Here    │  │  Overlay        │  │                │   │
│  │                 │  │                 │  │                │   │
│  │  [DROP IMAGE]   │  │  [PREDICTION]   │  │  [COLOR MAP]   │   │
│  │                 │  │                 │  │                │   │
│  └─────────────────┘  └─────────────────┘  └────────────────┘   │
│                                                                 │
│  Model: [Model D ★ ▼]    Resolution: 512×512                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  CLASS LEGEND                                           │    │
│  │  🟦 Sky  🟩 Trees  🟫 Rocks  🟧 Logs  🟨 Sand  ...      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  [ 🎯 Segment ]  [ 🔄 Compare All Models ]  [ 💾 Download ]     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Gradio App Code (`ui/app.py`)**
```python
"""
Off-Road Autonomous Vision — Web Interface
Gradio-powered real-time terrain segmentation
"""
import gradio as gr
import torch
import numpy as np
import cv2
from inference import SegmentationInference

# ─── Initialize Model ────────────────────────────────
engine = SegmentationInference(
    weights_path="weights/model_d_final.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_size=512,
    num_classes=10
)

# ─── Class Color Map ─────────────────────────────────
CLASS_NAMES = [
    "Sky", "Trees", "Lush Bushes", "Landscape", "Rocks",
    "Logs", "Dry Bushes", "Gravel Path", "Sand", "Dry Grass"
]

CLASS_COLORS = [
    [135, 206, 235],  # Sky - light blue
    [34, 139, 34],    # Trees - forest green
    [0, 128, 0],      # Lush Bushes - green
    [210, 180, 140],  # Landscape - tan
    [128, 128, 128],  # Rocks - gray
    [139, 69, 19],    # Logs - brown
    [189, 183, 107],  # Dry Bushes - khaki
    [169, 169, 169],  # Gravel Path - dark gray
    [244, 164, 96],   # Sand - sandy brown
    [154, 205, 50],   # Dry Grass - yellow green
]

# ─── Prediction Function ─────────────────────────────
def predict(image, model_choice):
    """Run segmentation on uploaded image."""
    
    # Select model weights
    weight_map = {
        "Model A — Baseline (CE Only)": "weights/model_a_baseline.pth",
        "Model B — CE + Augmentation": "weights/model_b_ce_aug.pth",
        "Model C — Hybrid Loss": "weights/model_c_hybrid.pth",
        "Model D — Final (Hybrid + Aug) ★": "weights/model_d_final.pth",
    }
    
    engine.load_weights(weight_map[model_choice])
    
    # Run inference
    mask = engine.predict(image)
    
    # Create colored overlay
    overlay = engine.create_overlay(image, mask, alpha=0.5)
    
    # Create class mask visualization
    color_mask = engine.create_color_mask(mask, CLASS_COLORS)
    
    # Generate class distribution text
    stats = engine.get_class_stats(mask, CLASS_NAMES)
    
    return overlay, color_mask, stats

# ─── Compare All Models ──────────────────────────────
def compare_all(image):
    """Run same image through all 4 models."""
    results = []
    for weight_file in [
        "weights/model_a_baseline.pth",
        "weights/model_b_ce_aug.pth",
        "weights/model_c_hybrid.pth",
        "weights/model_d_final.pth"
    ]:
        engine.load_weights(weight_file)
        mask = engine.predict(image)
        overlay = engine.create_overlay(image, mask, alpha=0.5)
        results.append(overlay)
        
    return results[0], results[1], results[2], results[3]

# ─── Build Gradio Interface ──────────────────────────
with gr.Blocks(
    title="Off-Road Autonomous Vision",
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #0A1628, #162033);
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .header h1 {
        color: #FF6B2B;
        font-size: 2em;
    }
    .header p {
        color: #94A3B8;
    }
    """
) as demo:
    
    # ── Header ──
    gr.HTML("""
    <div class="header">
        <h1>🏜️ Off-Road Autonomous Vision</h1>
        <p>Real-time semantic segmentation for desert terrain navigation</p>
        <p style="color: #FF6B2B; font-size: 0.9em;">
            U-Net + ResNet34  ·  Hybrid CE + Dice Loss  ·  10 Terrain Classes  ·  512×512
        </p>
    </div>
    """)
    
    # ── Single Model Tab ──
    with gr.Tab("🎯 Single Model Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📤 Upload Terrain Image",
                    type="numpy",
                    height=400
                )
                model_dropdown = gr.Dropdown(
                    choices=[
                        "Model A — Baseline (CE Only)",
                        "Model B — CE + Augmentation",
                        "Model C — Hybrid Loss",
                        "Model D — Final (Hybrid + Aug) ★",
                    ],
                    value="Model D — Final (Hybrid + Aug) ★",
                    label="🧠 Select Model"
                )
                segment_btn = gr.Button(
                    "🎯 Segment Terrain",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                output_overlay = gr.Image(
                    label="🎯 Segmented Overlay",
                    height=400
                )
                
            with gr.Column(scale=1):
                output_mask = gr.Image(
                    label="🗺️ Class Mask",
                    height=400
                )
                output_stats = gr.Textbox(
                    label="📊 Class Distribution",
                    lines=12
                )
                
        segment_btn.click(
            fn=predict,
            inputs=[input_image, model_dropdown],
            outputs=[output_overlay, output_mask, output_stats]
        )

    # ── Compare All Models Tab ──
    with gr.Tab("🔄 Compare All Models"):
        with gr.Row():
            compare_input = gr.Image(
                label="📤 Upload Image",
                type="numpy",
                height=300
            )
            compare_btn = gr.Button(
                "🔄 Compare All 4 Models",
                variant="primary",
                size="lg"
            )
            
        with gr.Row():
            out_a = gr.Image(label="🔴 Model A — Baseline")
            out_b = gr.Image(label="🟠 Model B — + Aug")
            out_c = gr.Image(label="🔵 Model C — + Dice")
            out_d = gr.Image(label="🟢 Model D — Final ★")
            
        compare_btn.click(
            fn=compare_all,
            inputs=[compare_input],
            outputs=[out_a, out_b, out_c, out_d]
        )

    # ── Class Legend Tab ──
    with gr.Tab("📋 Class Legend"):
        gr.HTML("""
        <div style="padding: 20px; background: #0A1628; border-radius: 12px;">
            <h3 style="color: #FF6B2B;">10 Terrain Classes</h3>
            <table style="width: 100%; color: white; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #1E293B;">
                    <th style="padding: 8px;">ID</th>
                    <th>Color</th>
                    <th>Class Name</th>
                    <th>Category</th>
                </tr>
                <tr><td>0</td><td>🟦</td><td>Sky</td><td>Background</td></tr>
                <tr><td>1</td><td>🟩</td><td>Trees</td><td>Vegetation</td></tr>
                <tr><td>2</td><td>🌿</td><td>Lush Bushes</td><td>Vegetation</td></tr>
                <tr><td>3</td><td>🟫</td><td>Landscape</td><td>Background</td></tr>
                <tr><td>4</td><td>⬜</td><td>Rocks</td><td>Obstacle</td></tr>
                <tr><td>5</td><td>🟫</td><td>Logs</td><td>Obstacle ⚠️</td></tr>
                <tr><td>6</td><td>🟨</td><td>Dry Bushes</td><td>Vegetation</td></tr>
                <tr><td>7</td><td>⬛</td><td>Gravel Path</td><td>Navigable</td></tr>
                <tr><td>8</td><td>🟧</td><td>Sand</td><td>Navigable</td></tr>
                <tr><td>9</td><td>🟡</td><td>Dry Grass</td><td>Vegetation</td></tr>
            </table>
        </div>
        """)

# ─── Launch ───────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
```

<br>

## 📈 Results

**Performance Progression**
```text
mIoU Performance Across 4 Models
═══════════════════════════════════════════════════════
Model A ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░  XX.XX%
Model B ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  XX.XX%
Model C ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░  XX.XX%
Model D ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░  XX.XX%  ★
═══════════════════════════════════════════════════════
                                     Target: Maximum mIoU
```

**Ablation Analysis**
```text
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  WHAT CONTRIBUTED MORE?                                 │
│                                                         │
│  Augmentation alone (A→B):      +XX.XX% mIoU            │
│  Hybrid Loss alone  (A→C):      +XX.XX% mIoU            │
│  Combined effect    (A→D):      +XX.XX% mIoU    ★       │
│                                                         │
│  → [Loss function / Augmentation] was the primary       │
│    performance driver for minority class recovery       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Critical Safety Metric**
```text
Logs IoU (Most Dangerous Obstacle Class)
═════════════════════════════════════════
Model A  ▓░░░░░░░░░░░░░░░░░░░░░░░░  XX.XX%   Invisible
Model B  ▓▓▓░░░░░░░░░░░░░░░░░░░░░░  XX.XX%   Partial
Model C  ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░  XX.XX%   Detected
Model D  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░  XX.XX%   Reliable ★
```

<br>

## 🔁 Reproducing Results

**Full Reproduction Pipeline**
```bash
# Step 1: Environment
conda create -n offroad python=3.10 -y
conda activate offroad
pip install -r requirements.txt

# Step 2: Dataset
# Place dataset in project root:
# Offroad_Segmentation_Training_Dataset/
# ├── images/
# └── masks/

# Step 3: Train all 4 models
python model/train.py --model a --loss ce --augment none
python model/train.py --model b --loss ce --augment flip
python model/train.py --model c --loss hybrid --augment none
python model/train.py --model d --loss hybrid --augment flip

# Step 4: Evaluate
python model/test.py --compare-all

# Step 5: Generate report visuals
python generate_report_visuals.py

# Step 6: Launch UI
python ui/app.py
```

**Verify Your Results Match Ours**
```bash
python model/test.py --weights weights/model_d_final.pth --detailed
# Expected output should show:
# Mean IoU:     ~XX.XX%  (± 0.5%)
# Pixel Acc:    ~XX.XX%  (± 0.3%)
# Logs IoU:     ~XX.XX%  (± 1.0%)
```

<br>

## ⚙️ Requirements

**`requirements.txt`**
```text
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.1
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
gradio>=4.0.0
tqdm>=4.65.0
```

**`environment.yml`**
```yaml
name: offroad
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - torch>=2.0.0
    - torchvision>=0.15.0
    - segmentation-models-pytorch>=0.3.3
    - albumentations>=1.3.1
    - opencv-python>=4.8.0
    - numpy>=1.24.0
    - matplotlib>=3.7.0
    - seaborn>=0.12.0
    - scikit-learn>=1.3.0
    - gradio>=4.0.0
    - tqdm>=4.65.0
```

<br>

## 🗺️ Class Definitions

| ID | Class | Pixel Freq | Category | Safety Level |
|:---|:---|:---|:---|:---|
| 0 | Sky | ~40.2% | Background | 🟢 None |
| 1 | Trees | ~15.3% | Vegetation | 🟡 Low |
| 2 | Lush Bushes | ~X.X% | Vegetation | 🟡 Low |
| 3 | Landscape | ~22.1% | Background | 🟢 None |
| 4 | Rocks | ~1.2% | Obstacle | 🔴 High |
| 5 | Logs | ~0.5% | Obstacle | 🔴 Critical |
| 6 | Dry Bushes | ~X.X% | Vegetation | 🟡 Low |
| 7 | Gravel Path | ~X.X% | Navigable | 🟢 None |
| 8 | Sand | ~X.X% | Navigable | 🟢 None |
| 9 | Dry Grass | ~X.X% | Vegetation | 🟡 Low |

<br>

## ⚠️ Known Limitations

| Failure Mode | Description | Severity | Proposed Fix |
|:---|:---|:---|:---|
| Shadow → Rock | Shadows misclassified as rocks | Medium | LiDAR depth fusion |
| Grass ↔ Bush | Boundary bleeding at transitions | Low | Boundary loss terms |
| Low light | Reduced accuracy in dark images | Medium | PhotoAugmentation |

<br>

## 📄 Report

📎 **View Full Hackathon Report (PDF)**

The report covers:
* 4-model ablation study with complete comparison tables
* Per-class IoU analysis across all models
* Loss curve analysis and convergence behavior
* Failure mode documentation with visual evidence
* Future work recommendations

<br>

## 🙏 Acknowledgments

* **segmentation-models-pytorch** — Pre-built architectures
* **Albumentations** — Fast image augmentation
* **Gradio** — Web interface framework
* **PyTorch** — Deep learning framework

<br>

<div align="center">
Built with 🧠 and ☕ for [HACKATHON NAME] 2025<br>
[TEAM NAME]
</div>
