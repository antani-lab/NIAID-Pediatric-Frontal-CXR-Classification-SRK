# 🩺📊 Train–Infer–Visualize (Pediatric CXR Classification with Explanation)

**One notebook, end‑to‑end** for research on chest X‑ray (CXR) classification: **train** with k‑fold cross‑validation, **infer** on validation/test, **explain** with a family of CAM from Pytorch Grad-CAM, and **compare** model performance against three radiologists.  
> ⚠️ **Clinical use disclaimer:** Research use only. Not for diagnostic or therapeutic decisions.

---

## 🗂️ Table of Contents
- [🧭 Overview](#-overview)
- [⚙️ Install & Environment](#️-install--environment)
- [🚀 Quick Start](#-quick-start)
- [🗃️ Data & Layout](#️-data--layout)
- [🧪 Configuration](#-configuration)
- [▶️ Usage](#️-usage)
- [📁 Outputs & Artifacts](#-outputs--artifacts)
- [🎯 Reproducibility](#-reproducibility)
- [🛠️ Troubleshooting](#️-troubleshooting)

---

## 🧭 Overview
- 🧱 **Frameworks:** PyTorch, PyTorch Lightning, `timm`, Albumentations, scikit‑learn, matplotlib.
- 🔍 **Explainability:** `jacobgil/pytorch-grad-cam` family (GradCAM, GradCAM++, XGradCAM, EigenCAM, HiResCAM, …).
- 🔂 **Cross‑Validation:** k‑fold with EarlyStopping + ModelCheckpoint (best by `val_loss`).
- 📈 **Metrics:** Balanced Accuracy, Sensitivity/Specificity, Precision, F1, MCC, Cohen’s κ, Youden’s J, ROC‑AUC.
- 🩻 **Reader Study:** Compare the best fold (`selected based on the highest validation Youden Index at the fixed 0.5 boundary`) to the radiologists.


---

## ⚙️ Install & Environment
```bash
# 1) Create and activate environment (example with conda)
conda create -n pediatric-cxr python=3.10 -y
conda activate pediatric-cxr

# 2) Install requirements
pip install -r requirements.txt
```

---

## 🚀 Quick Start
1. 📓 **Open the notebook:** `train_infer_visualize.ipynb` and run cells in order.  
2. ⚙️ **Edit hyperparameters:** Update the `h` dictionary (backbone name, `img_size`, `batch_size`, `n_splits`, `output_dir`, `test_dir`, etc.).  
3. 🧮 **Build baseline characteristics:** Generates `baseline_characteristics.csv` for Train vs Test cohorts based on the `image_data_github.csv` file.  
4. 🏋️ **Train with k‑fold CV:** Best checkpoint per fold saved by lowest validation loss; overwrites the previous checkpoint, so we finally have only the best checkpoint saved for a given fold.  
5. ✅ **Validation/Test inference:** Saves metrics and ROC curves (per fold + combined).  
6. 🔥 **Explanation:** Exports CAM heatmaps/contours/bounding boxes at original resolution.  
7. 👩‍⚕️👨‍⚕️ **Radiologist comparison:** Computes metrics and extracts misclassified sets per reader.

---

## 🗃️ Data & Layout
```
project/
├─ train_infer_visualize.ipynb
├─ data/
│  ├─ train/
│  │  ├─ normal/
│  │  └─ tb/
│  └─ test/
│     ├─ normal/
│     └─ tb/
└─ image_data_github.csv   # metadata (filenames, labels, demographics, reader cols)
```
- 🔎 Only **frontal** images are used for characteristics/splits (filenames contain `-AP` or `-PA`).  
- 📄 CSV fields typically include: `original`, `Type` (`Cases`/`Controls`), `MC_status`, demographics, and reader predictions: `alopezca_prediction`, `lamorenog_prediction`, `rdmontoyac_prediction`.

> 🧭 **Class convention:** `class_names = ["normal", "tb"]`; positive class = `tb` (CSV `Type == "Cases"`). Ensure folders and CSV agree.

---

## 🧪 Configuration
Edit the `h` dictionary near the top of the notebook:
- 🧠 **Model:** `model` (timm backbone), `dropout`, `num_classes=2`
- 🏋️ **Training:** `img_size`, `batch_size`, `num_workers`, `lr`, `max_epochs`, `patience`, `n_splits`
- ⚖️ **Imbalance:** `balance=True` → `WeightedRandomSampler`
- 💾 **IO:** `test_dir`, `output_dir`
- 🔬 **CAM:** method (e.g., `GradCAM`, `EigenCAM`), heatmap alpha, contour thickness, save masks/bboxes

---

## ▶️ Usage
- **Notebook flow:**
  - 🧮 Baseline table → 🏋️ CV training → ✅ Validation/Test inference → 🔥 Explanation → 👩‍⚕️👨‍⚕️ Reader comparison.
- **Script mode (optional):**
  ```bash
  jupyter nbconvert --to script train_infer_visualize.ipynb
  python train_infer_visualize.py
  ```

---

## 📁 Outputs & Artifacts
```
{output_dir}/
├─ fold_*/{model}/
│  ├─ *.ckpt
│  ├─ validation_inference/ (confusion_matrix.png, metrics.csv, roc.png)
│  └─ test_inference/       (confusion_matrix.png, metrics.csv, roc.png)
├─ {model}/
│  ├─ all_folds_validation_metrics.csv
│  ├─ combined_validation_roc.png
│  ├─ combined_test_roc.png
│  ├─ cam/…                 # heatmaps, masks, contours, bboxes
│  └─ comparison_to_radiologists/
└─ baseline_characteristics.csv
```

---

## 🎯 Reproducibility
- 🎲 Global seeds for Python/NumPy/PyTorch (CPU & CUDA).
- ⚙️ Deterministic cuDNN where appropriate; TF32 enabled for matmuls.
- 🧮 Mixed precision (bf16) during training/inference where supported.
- 🧴 Consistent Albumentations transforms (CLAHE for train; ImageNet mean/std).

---

## 🛠️ Troubleshooting
- ❓ **No checkpoints found:** Verify fold setup and that training completed; filenames follow the template in the notebook.  
- 🔁 **CSV–folder mismatch:** Ensure `original` names align with files in `data/*/*` and `Cases` map to `tb`.  
- 🔬 **CAM target not found:** Use a supported backbone; the helper chooses a valid Conv2d or inserts an adapter where needed.  
- 🧠 **Out‑of‑memory:** Reduce `batch_size` / `img_size` or disable some CAM exports.

---
