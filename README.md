# 🏜️ Offroad Environment Segmentation AI

> **Semantic segmentation model for autonomous offroad navigation in desert terrain.**
>
> Built for the **Startathon Desert Hackathon** — classifies every pixel of a terrain image into one of 10 environmental categories to enable safe autonomous offroading.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Sample Predictions

> Each row shows: **Input Image** | **Ground Truth** | **AI Prediction**

| | | |
|:-:|:-:|:-:|
| ![Result 0](final_submission_results/result_0.png) | ![Result 1](final_submission_results/result_1.png) | ![Result 2](final_submission_results/result_2.png) |
| ![Result 3](final_submission_results/result_3.png) | ![Result 4](final_submission_results/result_4.png) | |

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| **Architecture** | U-Net |
| **Encoder** | ResNet-34 (ImageNet pretrained) |
| **Framework** | [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) |
| **Input Resolution** | 512 × 512 |
| **Output Classes** | 10 |
| **Loss Function** | CrossEntropy + Dice (hybrid) |
| **Optimizer** | Adam (LR: 1e-5 for fine-tuning) |
| **LR Scheduler** | Cosine Annealing |
| **Augmentation** | Horizontal flip, Vertical flip |

---

## 🏷️ Terrain Classes

| Class ID | Raw Pixel Value | Class Name | Legend Color |
|:--------:|:---------------:|------------|:------------:|
| 0 | 100 | Trees | 🟩 `#228B22` |
| 1 | 200 | Lush Bushes | 🟢 `#9ACD32` |
| 2 | 300 | Dry Grass | 🟨 `#DAA520` |
| 3 | 500 | Dry Bushes | 🟫 `#8B4513` |
| 4 | 550 | Ground Clutter | ⬜ `#808080` |
| 5 | 600 | Flowers | 🩷 `#FF69B4` |
| 6 | 700 | Logs | 🟤 `#A0522D` |
| 7 | 800 | Rocks | ⬛ `#696969` |
| 8 | 7100 | Landscape | 🟧 `#F4A460` |
| 9 | 10000 | Sky | 🔵 `#87CEEB` |

---

## 📊 Performance

### Overall Metrics

| Metric | Score |
|--------|------:|
| **Pixel Accuracy** | 87.78% |
| **Mean IoU** | 65.38% |

### Per-Class IoU (Intersection over Union)

| Class | IoU | Rating |
|-------|----:|:------:|
| Sky | 98.73% | 🟢 Excellent |
| Trees | 87.63% | 🟢 Excellent |
| Dry Grass | 70.37% | 🟡 Good |
| Lush Bushes | 70.14% | 🟡 Good |
| Landscape | 69.78% | 🟡 Good |
| Flowers | 64.22% | 🟡 Good |
| Logs | 56.21% | 🟠 Fair |
| Dry Bushes | 48.93% | 🟠 Fair |
| Rocks | 47.84% | 🟠 Fair |
| Ground Clutter | 39.98% | 🔴 Needs Work |

> **Note:** Small / rare objects (Logs, Rocks, Ground Clutter) are harder to detect. The hybrid CrossEntropy + Dice loss was specifically added to improve these classes.

### Confusion Matrix

![Confusion Matrix](final_submission_results/confusion_matrix.png)

---

## 🏋️ Training Evolution

The model was iteratively improved across **4 training versions**:

| Version | File | Resolution | Batch | Loss | Augmentation | Key Improvement |
|:-------:|------|:----------:|:-----:|------|:------------:|-----------------|
| V0 | `run_training.py` | 256 | 8 | CE | ❌ | Baseline |
| V1 | `local_train.py` | 256 | 6 | CE | ❌ | Local GPU tuning |
| V2 | `local_train_v2.py` | 256 | 6 | CE | ❌ | **Fixed mask ID mapping** (100→0, 200→1, …) |
| V3 | `local_train_v3.py` | 256 | 6 | CE + Dice | ✅ Flip H/V | Augmentation, hybrid loss, cosine LR |
| V4 | `local_train_final.py` | 512 | 2 | CE + Dice | ✅ Flip H/V | High-res fine-tuning (LR=1e-5) |

### What Changed at Each Step

- **V0 → V1**: Adjusted batch size to fit RTX 4050's 6 GB VRAM
- **V1 → V2**: 🐛 **Critical bug fix** — masks were being read as grayscale (`cv2.imread(path, 0)`), truncating raw IDs (100, 200, …, 10000). Changed to `cv2.imread(path, -1)` and added `ID_MAPPING` to remap to 0–9
- **V2 → V3**: Added horizontal/vertical flip augmentation, switched to hybrid CrossEntropy + Dice loss (massive IoU improvement for small classes like Logs), added cosine annealing LR scheduler
- **V3 → V4**: Bumped resolution to 512×512, lowered batch to 2, fine-tuned with LR=1e-5 from V3 weights

---

## 📂 Project Structure

```
desert_hackathon/
├── app.py                      # Streamlit web app for live inference
├── best_model.pth              # Trained model weights (~93 MB)
├── requirements.txt            # Python dependencies
├── generate_readme_assets.py   # (Optional) Generate extra charts on a GPU machine
│
├── run_training.py             # V0 — Baseline training
├── local_train.py              # V1 — Local GPU training
├── local_train_v2.py           # V2 — Fixed mask ID mapping
├── local_train_v3.py           # V3 — Augmentation + hybrid loss + scheduler
├── local_train_final.py        # V4 — 512×512 high-res fine-tuning
│
├── check_model.py              # Quick single-image visual check
├── accurate_check.py           # Corrected mask reading validation
├── check_iou.py                # Full validation set IoU computation
├── check_split.py              # Train/val split ratio verification
├── final_test.py               # Final eval: IoU + confusion matrix + visuals
│
├── final_submission_results/   # Pre-generated evaluation outputs
│   ├── confusion_matrix.png
│   └── result_0..4.png
│
└── Offroad_Segmentation_Training_Dataset/  # Dataset (gitignored)
    ├── train/
    │   ├── Color_Images/
    │   └── Segmentation/
    └── val/
        ├── Color_Images/
        └── Segmentation/
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on RTX 4050 — 6 GB VRAM)

### Installation

```bash
git clone https://github.com/<YOUR_USERNAME>/desert_hackathon.git
cd desert_hackathon
pip install -r requirements.txt
```

### Dataset Setup

1. Download the **Offroad Segmentation Training Dataset** (provided by hackathon organizers).
2. Place it in the project root:
   ```
   desert_hackathon/
   └── Offroad_Segmentation_Training_Dataset/
       ├── train/
       │   ├── Color_Images/
       │   └── Segmentation/
       └── val/
           ├── Color_Images/
           └── Segmentation/
   ```

---

## 🏋️ Reproducing Training

```bash
# Step 1 — Baseline with corrected mask mapping
python local_train_v2.py

# Step 2 — Improve with augmentation + hybrid loss (resumes from V2)
python local_train_v3.py

# Step 3 — Fine-tune at 512×512 (resumes from V3)
python local_train_final.py
```

---

## 🧪 Evaluation

```bash
# Quick visual check on a random validation image
python check_model.py

# Accurate visual check with correct mask reading
python accurate_check.py

# Per-class IoU on the full validation set
python check_iou.py

# Full evaluation: IoU + confusion matrix + 5 visual results
python final_test.py

# Verify train/val split ratio
python check_split.py

# (Optional, requires GPU) Generate bar charts & sample prediction grids
python generate_readme_assets.py
```

---

## 🌐 Web App

Launch the interactive Streamlit demo:

```bash
streamlit run app.py
```

**Features:**
- 📤 Upload any terrain image for real-time segmentation
- 🖼️ Side-by-side original vs. AI perception view
- 📈 Live confidence score (softmax-based)
- 📊 Pre-computed baseline metrics dashboard
- 🔍 Expandable detailed per-class IoU breakdown
- 🗺️ Color-coded terrain legend

---

## 🔑 Key Technical Decisions

### 1. Raw Mask Reading
Segmentation masks encode class IDs as raw pixel values (100, 200, …, 10000). Reading as `cv2.imread(path, 0)` (grayscale) truncates values above 255, causing incorrect labels. Using `cv2.imread(path, -1)` reads unchanged values and preserves the original IDs.

### 2. Hybrid CE + Dice Loss
CrossEntropy alone struggles with underrepresented classes (Logs, Rocks, Ground Clutter). Dice loss focuses on per-class overlap and significantly boosted IoU for these minority classes.

### 3. Progressive Training
Instead of training at 512×512 from scratch (GPU memory-prohibitive at batch sizes needed), we first converge at 256×256 and then fine-tune at 512×512 with a very low learning rate. Faster convergence, lower memory usage.

### 4. Default Class Fallback
Unknown / unmapped pixel values in masks are assigned to class 8 (Landscape) as a safe default to prevent training crashes from out-of-range class indices.

---

## 👥 Team

| Name | Role |
|------|------|
| `Dhruv Bajpai` | `Team Lead` |
| `Samarth Shukla` | `Backend` |
| `Kshitij Trivedi` | `Frontend` |

---

## 📄 License

This project was developed for the **Startathon Desert Hackathon**. Please check with the organizers for dataset licensing and usage terms.

---

## 🙏 Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Iakubovskii
- [Streamlit](https://streamlit.io/) for the interactive demo framework
- Hackathon organizers for the Offroad Segmentation dataset
