# In[ ]:


#check GPU
get_ipython().system('nvidia-smi')


# In[ ]:


#get current working directory
import os
os.getcwd()


# In[ ]:


#print python version and location
import sys
print(sys.executable)
print(sys.version)


# In[ ]:


# =========================
# Standard library imports
# =========================
import contextlib
import glob
import io
import logging
import os
import random
import re
import shutil
import time
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# =========================
# Third-party imports
# =========================
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchmetrics
from torchmetrics.classification import Accuracy, F1Score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from IPython.display import display
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image, ImageDraw


# In[ ]:


# Set Seeds for Full Reproducibility

def set_seed(seed):
    """
    Set the random seed
    """
    # 1. Python random
    random.seed(seed)
    # 2. NumPy
    np.random.seed(seed)
    # 3. PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. Enable TF32 for float32 matmuls on A100 (max throughput, minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)


# In[ ]:


# Model Configuration dictionary; 5 fold cross validation, so we will save five models, each model with a single checkpoint with the lowest validation loss

h = {
    "model": "dpn68",
    "img_size": 224,
    "batch_size": 64,
    "num_workers": 2,
    "dropout": 0.3,
    "num_classes": 2,
    "pin_memory": True,
    "lr": 5e-5,
    "max_epochs": 64,
    "patience": 10,
    "balance": True,
    "n_splits": 5,
    "test_dir": "path/to/test",
    "output_dir": "path/to/output",
    "cmap": "Blues",
    "alpha": 0.8,
    "text_color": "black",
    "text_font_size": 12,
    "xt_label_rotation": 45,
    "xt_label_size": 12,
    "yt_label_size": 12,
    "xlabel_font_size": 12,
    "ylabel_font_size": 12,
    "title_font_size": 14,
    "train_color": "blue",
    "val_color": "orange",
    "legend_loc": "best",
    "font_size": 12
    }


# In[ ]:


# Dataset + DataModule

class CXRDataset(Dataset):
    """
    - Train/Val: returns (image, label)
    - Test: returns (image, label, filename) when return_paths=True
    """
    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        return_paths: bool = False
    ) -> None:
        assert len(file_paths) == len(labels), "file_paths and labels must align"
        self.file_paths  = file_paths
        self.labels = labels
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int],
                                             Tuple[torch.Tensor, int, str]]:
        # reads grayscale and expand to 3-channel RGB for pretrained backbones
        img = cv2.imread(str(self.file_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {self.file_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        y = int(self.labels[idx])
        if self.return_paths:
            return img, y, self.file_paths[idx].name
        return img, y

class DataModule(pl.LightningDataModule):
    def __init__(self, h: Dict, fold_idx: Optional[int] = None) -> None:
        super().__init__()
        self.h = h
        self.fold_idx = fold_idx
        # Class definitions (index 1 == 'tb' == positive)
        self.class_names  = ["normal", "tb"]
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        print(f"[DataModule] Class to index mapping: {self.class_to_idx}")
        # Transforms
        size = int(h["img_size"])
        # Train (with CLAHE)
        self.train_transform = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Resize(size, size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        # Val/Test (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.test_transform = self.val_transform

        # DataLoader args 
        num_workers = int(h["num_workers"])
        self.loader_args = dict(
            batch_size=int(h["batch_size"]),
            num_workers=num_workers,
            pin_memory=bool(h["pin_memory"]),
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False
        )
        # torch DataLoader disallows None prefetch_factor; remove key if None
        if self.loader_args["prefetch_factor"] is None:
            self.loader_args.pop("prefetch_factor")

        # Expose test filenames for downstream alignment 
        self.test_filenames: List[str] = []

    def _make_split(self, base: Path, split: str) -> Tuple[List[Path], List[int]]:
        """
        Reads images from either:
          - base/<split>/<class>/   when split is non-empty, or
          - base/<class>/           when split == "".
        Ensures a deterministic, filename-sorted order across classes.
        """
        fps, labs = [], []
        for cls in self.class_names:
            folder = (base / split / cls) if split else (base / cls)
            if not folder.exists():
                continue
            imgs: List[Path] = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                imgs.extend(folder.glob(ext))
            # sort deterministically by filename
            imgs = sorted(imgs, key=lambda p: p.name.lower())
            fps.extend(imgs)
            labs.extend([self.class_to_idx[cls]] * len(imgs))
        # final global sort by filename to make cross-class order stable as well
        if fps:
            order = np.argsort([p.name.lower() for p in fps])
            fps = [fps[i] for i in order]
            labs = [labs[i] for i in order]
        return fps, labs

    def setup(self, stage: Optional[str] = None) -> None:
        # Train/Val for cross-validation
        if self.fold_idx is not None:
            base = Path(self.h["output_dir"]) / f"fold_{self.fold_idx}"
            tr_fps, tr_labs = self._make_split(base, "train")
            vl_fps, vl_labs = self._make_split(base, "val")
            if self.h.get("balance", False) and len(tr_labs) > 0:
                counts = np.bincount(tr_labs, minlength=len(self.class_names))
                weights = (1.0 / np.maximum(counts, 1))[np.array(tr_labs)]
                weights = torch.as_tensor(weights, dtype=torch.double)
                self.train_sampler = WeightedRandomSampler(
                    weights, num_samples=len(weights), replacement=True
                )
                print("[DataModule] Using WeightedRandomSampler to correct class-imbalance")
            else:
                self.train_sampler = None
                print("[DataModule] Using standard shuffling (no imbalance correction)")

            self.train_dataset = CXRDataset(tr_fps, tr_labs, transform=self.train_transform, return_paths=False)
            self.val_dataset = CXRDataset(vl_fps, vl_labs, transform=self.val_transform,   return_paths=False)
        else:
            self.train_dataset = None
            self.val_dataset = None
            self.train_sampler = None
            print("[DataModule] Skipping train/val setup (fold_idx is None)")

        # build TEST
        td = Path(self.h["test_dir"])
        ts_fps, ts_labs = self._make_split(td, "")
        if not ts_fps:
            ts_fps, ts_labs = self._make_split(td, "test")
        self.test_dataset = CXRDataset(ts_fps, ts_labs, transform=self.test_transform, return_paths=True)
        self.test_filenames = [p.name for p in ts_fps]
        print(f"[DataModule] Test set: {len(ts_fps)} samples")

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        if self.train_sampler is not None:
            return DataLoader(
                self.train_dataset,
                sampler=self.train_sampler,
                shuffle=False,
                **self.loader_args
            )
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.loader_args
        )
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.loader_args
        )
    def test_dataloader(self):
        # returns (image, label, filename) batches
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.loader_args
        )


# In[ ]:


# Model architecture
# helpers to find the last Conv2d that writes the final stage feature map 
class _ConvSpy:
    def __init__(self):
        self.records: List[Tuple[str, nn.Module, Tuple[int,int,int,int]]] = []  # (name, mod, out.shape)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def hook(self, name: str, mod: nn.Conv2d):
        def _h(m, x, y):
            if isinstance(y, torch.Tensor) and y.ndim == 4:
                self.records.append((name, mod, tuple(y.shape)))
        return _h

    def attach(self, module: nn.Module):
        for n, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                self.hooks.append(m.register_forward_hook(self.hook(n, m)))

    def detach(self):
        for h in self.hooks:
            try: h.remove()
            except: pass
        self.hooks.clear()

def _discover_laststage_conv_k(backbone: nn.Module, img_size: int) -> Tuple[Optional[nn.Conv2d], Optional[Tuple[int,int]]]:
    """
    Returns (last_conv_in_final_stage, kernel_size) if any Conv2d writes the final stage map; else (None, None).
    We run on CPU with a 1x3xH×W dummy input to avoid GPU binding during Lightning __init__.
    """
    spy = _ConvSpy()
    spy.attach(backbone)
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
        feats = backbone(x)  # timm features_only → list[tensor]
        if isinstance(feats, (list, tuple)):
            last = feats[-1]
        else:
            last = feats
        last_hw = tuple(last.shape[-2:])
    spy.detach()

    # keep convs that produced tensors with same H,W as final map
    same_hw: List[Tuple[str, nn.Module, Tuple[int,int,int,int]]] = []
    for (n, m, shp) in spy.records:
        if tuple(shp[-2:]) == last_hw:
            same_hw.append((n, m, shp))
    if not same_hw:
        return None, None
    # execution order ~ records order → take the last one
    _, last_conv, _ = same_hw[-1]
    k = last_conv.kernel_size if isinstance(last_conv.kernel_size, tuple) else (last_conv.kernel_size, last_conv.kernel_size)
    return last_conv, (int(k[0]), int(k[1]))

def _resolve_timm_name(user_name: str, table: dict) -> str:
    """
    Normalize a user-friendly alias into an exact timm model id using `table`.
    If not found, return the original `user_name` (assume it's already a valid timm id).
    """
    key = user_name.strip().lower()
    # common normalizations
    norm = (
        key.replace("-", "_")
           .replace(" ", "")
           .replace("patch16_224", "patch16_224")  # no-op but explicit
           .replace("rw224", "rw_224")
    )
    if norm in table:
        return table[norm]
    # also try raw key and original text, so exact timm ids pass through
    if key in table:
        return table[key]
    return user_name  # assume caller already supplied a valid timm id

class TBModel(pl.LightningModule):
    def __init__(self, h: Dict):
        super().__init__()
        self.h: Dict = h
        self.save_hyperparameters('h')
        name: str = h['model']
        img_size: int = int(h['img_size'])
        drop_rate: float = float(h['dropout'])
        num_classes: int = int(h['num_classes'])
        self.num_classes = num_classes
        # Backbones
        self.is_transformer: bool = False
        if name in ['vgg16']:
            weights = getattr(models, f"{name.upper()}_Weights").IMAGENET1K_V1
            vgg = getattr(models, name)(weights=weights)
            self.backbone = vgg.features
            feat_dim = 512
            self.post3x3 = nn.Identity()  # VGG already ends with conv blocks suited for CAM
        elif name in ['dpn68'
        ]:
            timm_map = {
                'dpn68': 'dpn68.mx_in1k',
            }
            base = timm.create_model(timm_map[name], pretrained=True, features_only=True)
            self.backbone = base
            feat_dim = base.feature_info[-1]['num_chs']

            # Discover the last conv that writes the final stage map; if not 3×3 → add one
            last_conv, k = _discover_laststage_conv_k(self.backbone, img_size)
            need_3x3 = (k is None) or (k != (3,3))
            if need_3x3:
                self.post3x3 = nn.Sequential(
                    nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feat_dim),
                    nn.SiLU(inplace=True)
                )
            else:
                self.post3x3 = nn.Identity()       
        # ViT / DeiT / BEiT (features_only)
        elif any(s in name.lower() for s in ["deit", "beit"]):
            vt_map = {
                # DeiT3 small
                "deit3_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
             
                # BEiT v2 base
                "beit3_base": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
            }
            resolved = _resolve_timm_name(name, vt_map)
            print(f"[Backbone] resolved '{name}' → timm id '{resolved}'")        
            base = timm.create_model(
                resolved,
                pretrained=True,
                features_only=True,   # timm returns a spatial [B,C,H,W] map for features_only
                img_size=img_size,
                drop_rate=0.0,
                drop_path_rate=0.0,
                pretrained_strict=False
            )
            self.backbone = base
            feat_dim = base.feature_info[-1]['num_chs']
            self.is_transformer = True
            # keep ViTs as-is for CAM in pipeline
            self.post3x3 = nn.Identity()

        else:
            raise ValueError(f"Unsupported model '{name}'")
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),              # → (B, C)
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes)
        )
        # Metrics 
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = self.post3x3(feats)
        logits = self.head(feats)
        return logits

    @staticmethod
    def _nll_from_onehot_logits(logits: torch.Tensor, labels_oh: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        return -(labels_oh * logp).sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        if labels.ndim == 1:
            labels_oh = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            labels_oh = labels.float()
        logits = self(imgs)
        loss = self._nll_from_onehot_logits(logits, labels_oh)
        preds = logits.argmax(dim=1)
        tgt = labels_oh.argmax(dim=1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc(preds, tgt), on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1(preds, tgt),  on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        if labels.ndim == 1:
            labels_oh = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            labels_oh = labels.float()
        logits = self(imgs)
        loss = self._nll_from_onehot_logits(logits, labels_oh)
        preds = logits.argmax(dim=1)
        tgt = labels_oh.argmax(dim=1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(preds, tgt), on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1(preds, tgt),  on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.h.get("lr", 5e-5),
                                weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.h.get("max_epochs", 64)
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

# History callback (epoch-level)
class History(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train_loss" in m: self.train_losses.append(m["train_loss"].item())
        if "train_acc" in m: self.train_accs.append(m["train_acc"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "val_loss" in m: self.val_losses.append(m["val_loss"].item())
        if "val_acc" in m: self.val_accs.append(m["val_acc"].item())


# In[ ]:


# fine-tuning end-to-end
# Checkpoint callback
def get_model_checkpoint_callback(h: Dict, fold: int) -> ModelCheckpoint:
    model_name = h["model"]
    ckpt_dir = os.path.join(h["output_dir"], f"fold_{fold}", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    filename_template = (
        f"{model_name}_fold{fold}_"
        "epoch{epoch:02d}_"
        "{val_loss:.4f}_{val_acc:.4f}"
    )
    return ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=filename_template,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last=False,
        save_weights_only=False,
        auto_insert_metric_name=False
    )
# helpers
# Plotting Helpers
def plot_training_curves(
    train_losses,
    val_losses,
    train_accs=None,
    val_accs=None,
    save_path=None,
    figsize=(8, 6),
    train_color="C0",
    val_color="C1",
    alpha=0.8,
    xlabel_font_size=14,
    ylabel_font_size=14,
    title_font_size=16,
    legend_loc="upper right",
    font_size=12
):
    """
    Plots training & validation loss vs. epoch.
    Each curve uses its own epoch range so mismatched lengths (due to early stopping) won't error out.
    """
    train_epochs = range(1, len(train_losses) + 1)
    val_epochs = range(1, len(val_losses) + 1)

    plt.figure(figsize=figsize)
    plt.plot(train_epochs, train_losses, label="Train Loss",
             color=train_color, alpha=alpha, linewidth=2)
    plt.plot(val_epochs, val_losses, label="Val Loss",
             color=val_color, alpha=alpha, linewidth=2)

    if (train_accs is not None) and (val_accs is not None):
        train_acc_epochs = range(1, len(train_accs) + 1)
        val_acc_epochs = range(1, len(val_accs) + 1)
        plt.plot(train_acc_epochs, train_accs, label="Train Acc",
                 linestyle="--", color=train_color, linewidth=2)
        plt.plot(val_acc_epochs, val_accs, label="Val Acc",
                 linestyle="--", color=val_color, linewidth=2)

    plt.xlabel("Epoch", fontsize=xlabel_font_size)
    plt.ylabel("Value", fontsize=ylabel_font_size)
    plt.title("Training Curves", fontsize=title_font_size)
    plt.legend(loc=legend_loc, fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
        plt.close()
    else:
        plt.show()

def plot_confusion(
    cm: np.ndarray,
    classes: List[str],
    save_path: Path,
    figsize: Tuple[int,int]=(6,6),
    cmap: str="Purples",
    alpha: float=0.8,
    xt_label_rotation: int=0,
    xt_label_size: int=12,
    yt_label_size: int=12,
    xlabel_font_size: int=14,
    ylabel_font_size: int=14,
    title_font_size: int=16,
    text_color: str="black",
    text_font_size: int=20
) -> None:
    """
    Plot confusion matrix cm=[[TP,FP],[FN,TN]] with
      rows = predicted class (0=tb,1=normal)
      cols = true      class (0=tb,1=normal)
    """
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap, alpha=alpha)
    fig.colorbar(cax)

    ax.set_xticks([0,1])
    ax.set_xticklabels(classes, rotation=xt_label_rotation, fontsize=xt_label_size)
    ax.set_yticks([0,1])
    ax.set_yticklabels(classes, fontsize=yt_label_size)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel("True label", fontsize=xlabel_font_size)
    ax.set_ylabel("Predicted label", fontsize=ylabel_font_size)
    #ax.set_title("Confusion Matrix", fontsize=title_font_size)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v),
                ha="center", va="center",
                color=text_color, fontsize=text_font_size)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
        plt.close()
    else:
        plt.show()

#logging stats
def _format_param_count(n: int) -> str:
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train

def _probe_backbone_last_writer(backbone: nn.Module, img_size: int) -> Tuple[str, Tuple[int,int,int,int], Tuple[int,int]]:
    """
    Run a CPU dummy forward to record all Conv2d writers; return the LAST one that writes the final feature map.
    Returns:
      (name, out_shape, kernel_tuple)
    Raises RuntimeError if none found.
    """
    records: List[Tuple[str, nn.Module, Tuple[int,int,int,int]]] = []
    hooks = []

    def _make_hook(name: str, m: nn.Conv2d):
        def _h(_m, _in, out):
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                records.append((name, m, tuple(out.shape)))
        return _h
    for n, m in backbone.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_make_hook(n, m)))
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32, device="cpu")
        feats = backbone(x.cpu())
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats
        last_hw = tuple(last.shape[-2:])  # (H, W)
    for h in hooks:
        try: h.remove()
        except: pass
    same_hw = [(n, m, shp) for (n, m, shp) in records if tuple(shp[-2:]) == last_hw]
    if not same_hw:
        raise RuntimeError("No Conv2d writer found for the final stage feature map.")
    name, mod, shp = same_hw[-1]
    k = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
    k = (int(k[0]), int(k[1]))
    return name, shp, k

def _summarize_inserted_3x3(model: TBModel) -> Tuple[bool, int, Tuple[int,int,int,int]]:
    """
    Returns:
      (inserted, param_count, weight_shape) where weight_shape is (C_out, C_in, 3, 3) if inserted else (0,0,0,0).
    """
    post = getattr(model, "post3x3", None)
    if isinstance(post, nn.Sequential) and len(post) > 0 and isinstance(post[0], nn.Conv2d):
        conv = post[0]
        bn = post[1] if len(post) > 1 and isinstance(post[1], nn.BatchNorm2d) else None
        cout, cin, kh, kw = tuple(conv.weight.shape)
        # conv params
        p_conv = conv.weight.numel() + (conv.bias.numel() if conv.bias is not None else 0)
        # bn params (gamma, beta); running stats are buffers, not parameters
        p_bn = (bn.weight.numel() + bn.bias.numel()) if bn is not None else 0
        return True, int(p_conv + p_bn), (cout, cin, kh, kw)
    return False, 0, (0,0,0,0)

# Plot per-fold training curves into each {output_dir}/fold_n/{model}/
def plot_all_curves(h: Dict, histories: List[History]) -> None:
    for fold_idx, hist in enumerate(histories):
        model_dir = os.path.join(h["output_dir"], f"fold_{fold_idx}", h["model"])
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, "training_curve.png")

        plot_training_curves(
            train_losses=hist.train_losses,
            val_losses=hist.val_losses,
            train_accs=hist.train_accs,
            val_accs=hist.val_accs,
            save_path=save_path,
            figsize=(8, 6),
            train_color=f"C{fold_idx*2}",
            val_color=f"C{fold_idx*2+1}",
            alpha=0.9,
            xlabel_font_size=14,
            ylabel_font_size=14,
            title_font_size=16,
            legend_loc="upper right",
            font_size=12
        )
        print(f"[INFO] Saved fold-{fold_idx} curves → {save_path}")

# Combined curves across folds, saved under {output_dir}/{model}/
def plot_combined_curves(h: Dict, histories: List[History]) -> None:
    model_root = os.path.join(h["output_dir"], h["model"])
    os.makedirs(model_root, exist_ok=True)
    combined_path = os.path.join(model_root, f"combined_curves_{h['model']}.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for fold_idx, hist in enumerate(histories):
        te = range(1, len(hist.train_losses) + 1)
        ax.plot(te, hist.train_losses,
                label=f"fold-{fold_idx} train",
                color=colors[fold_idx % len(colors)],
                linestyle='-')
        ve = range(1, len(hist.val_losses) + 1)
        ax.plot(ve, hist.val_losses,
                label=f"fold-{fold_idx} val",
                color=colors[fold_idx % len(colors)],
                linestyle='--')

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(f"Combined Training & Validation Curves ({h['model']})", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(combined_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved combined curves → {combined_path}")

# Training loop 
def train_cv(h: Dict):
    """
    Cross-validated training on physical fold_i/train & fold_i/val folders.
    Prints model architecture, parameter counts, and CAM-relevant sanity checks
    BEFORE training each fold.
    """
    os.makedirs(h["output_dir"], exist_ok=True)
    histories = []

    for fold in range(h["n_splits"]):
        print(f"\n==== Starting training for Fold {fold} ====")

        # Data module
        dm = DataModule(h, fold_idx=fold)
        dm.setup()

        # Sanity check: count images actually seen from disk
        print(f" [Fold {fold}] Checking file counts in physical folders:")
        base_fold_dir = os.path.join(h["output_dir"], f"fold_{fold}")
        for root, dirs, files in os.walk(base_fold_dir):
            img_count = len([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))])
            if img_count > 0:
                print(f"{root:<80} → {img_count} images")

        # Build model
        model = TBModel(h)

        # print architecture & trainable params
        print("\n-----------------------------------")
        print(f"[Fold {fold}] Model architecture:")
        print(model)  # full nn.Module tree
        total_p, train_p = _count_params(model)
        print(f"[Fold {fold}] Parameters: total={_format_param_count(total_p)}  "
              f"trainable={_format_param_count(train_p)}")

        # CAM-relevant sanity checks; Was a 3×3 inserted? If yes, how many params?
        inserted, pcount, wshape = _summarize_inserted_3x3(model)
        if inserted:
            cout, cin, kh, kw = wshape
            print(f"[Fold {fold}] Inserted final 3×3 conv (Conv-BN-SiLU): "
                  f"weight_shape={wshape}  params={_format_param_count(pcount)}")
        else:
            print(f"[Fold {fold}] No extra 3×3 inserted (backbone already ends with a suitable 3×3).")

        # What does the backbone itself end with (writer of the final feature map)?
        try:
            name, shp, k = _probe_backbone_last_writer(model.backbone, int(h["img_size"]))
            print(f"[Fold {fold}] Backbone last-stage writer: {name}  out={shp}  kernel={k}")
        except Exception as e:
            print(f"[Fold {fold}] WARNING: could not probe backbone last writer ({type(e).__name__}: {e})")

        print("-----------------------------------\n")

        # Callbacks
        hist_cb = History()
        es_cb = EarlyStopping(monitor="val_loss", patience=h["patience"], mode="min", verbose=True)
        ckpt_cb = get_model_checkpoint_callback(h, fold)

        # Trainer
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            max_epochs=h["max_epochs"],
            default_root_dir=h["output_dir"],
            callbacks=[es_cb, ckpt_cb, hist_cb],
            log_every_n_steps=30,
            enable_progress_bar=True
        )
        # Train
        trainer.fit(model, dm)
        # Save history handle
        histories.append(hist_cb)
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    return histories

# Entrypoint
if __name__ == "__main__":
    histories = train_cv(h)
    plot_all_curves(h, histories)
    plot_combined_curves(h, histories)


# In[ ]:


# validation performance

def inference_validation_with_curves(h):
    """
    For each fold:
      1) Find best‐val_loss checkpoint under output_dir/fold_i/{model_name}.
      2) Build a DataLoader from fold_i/val/*.
      3) Run inference in bf16, then cast logits back to float32 before softmax.
      4) Compute confusion‐matrix and scalar metrics.
      5) Save per‐fold confusion‐matrix and metrics CSV under
         output_dir/fold_i/{model_name}/validation_inference.
      6) At the end, compute cross‐fold summary stats and plot combined ROC + mean ROC
         (with inset) into that same validation_inference folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_fold_records = []
    roc_data = []

    fold_colors = ["red", "blue", "green", "orange", "brown"]
    fold_curve_alpha = 0.9
    fold_curve_lw = 2
    mean_line_color = "black"
    mean_line_lw = 2
    axis_fontsize = h["xlabel_font_size"]  # use same font size for axes

    for fold in range(h["n_splits"]):
        # Base folder for this fold’s model outputs:
        fold_base = os.path.join(h["output_dir"], f"fold_{fold}", h["model"])
        if not os.path.isdir(fold_base):
            print(f"⚠️ Skipping fold {fold}: folder not found → {fold_base}")
            continue

        # Find best‐val_loss checkpoint inside fold_base:
        pattern = os.path.join(fold_base, f"{h['model']}_fold{fold}_epoch*_*.ckpt")
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            print(f"⚠️ Skipping fold {fold}: no checkpoints in {fold_base}")
            continue

        best_loss = float("inf")
        best_ckpt = None
        for path in ckpts:
            name = os.path.basename(path)
            m = re.match(rf".*_fold{fold}_epoch\d+_(\d+\.\d+)_\d+\.\d+\.ckpt$", name)
            if not m:
                continue
            loss_val = float(m.group(1))
            if loss_val < best_loss:
                best_loss, best_ckpt = loss_val, path

        if best_ckpt is None:
            print(f"⚠️ No valid checkpoint for fold {fold}")
            continue

        print(f"\n🎯 Fold {fold} validation inference")
        print(f" → Best checkpoint: {os.path.basename(best_ckpt)} (val_loss={best_loss:.4f})")

        # Build validation‐only DataLoader from fold_i/val
        val_normal_dir = os.path.join(h["output_dir"], f"fold_{fold}", "val", "normal")
        val_abnormal_dir = os.path.join(h["output_dir"], f"fold_{fold}", "val", "tb")
        val_normal_paths = sorted(glob.glob(os.path.join(val_normal_dir, "*")))
        val_abnormal_paths = sorted(glob.glob(os.path.join(val_abnormal_dir, "*")))

        if (not val_normal_paths) and (not val_abnormal_paths):
            print(f"⚠️ Fold {fold} has no validation images.")
            continue

        val_paths = np.array(val_normal_paths + val_abnormal_paths)
        val_labels = np.array([0]*len(val_normal_paths) + [1]*len(val_abnormal_paths), dtype=int)

        # Same transform as in training
        val_transform = A.Compose([
            A.Resize(h["img_size"], h["img_size"], interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        val_dataset = CXRDataset(val_paths, val_labels, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=h["batch_size"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
            drop_last=False
        )

        # Load model from best checkpoint
        model = TBModel.load_from_checkpoint(best_ckpt, h=h)
        model.to(device)
        model.eval()

        # Run inference, collecting hard preds & probabilities
        y_true, y_pred, y_score = [], [], []

        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(imgs)
                logits_fp32 = logits.float()            # cast back to float32
                probs = torch.softmax(logits_fp32, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                y_score.extend(probs)
                y_pred.extend(preds)
                y_true.extend(labels.numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        # Compute confusion
        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))
        cm_clinician = np.array([[TP, FP],
                                 [FN, TN]], dtype=int)

        # Compute scalar metrics
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        precision = precision_score(y_true, y_pred, pos_label=1)
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        fscore = f1_score(y_true, y_pred, average="macro")
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        youden_j = sensitivity + specificity - 1

        print(f"▶️ Val BalAcc={bal_acc:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}")

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc_val = auc(fpr, tpr)
        roc_data.append({
            "fold": fold,
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc_val
        })

        # Prepare the validation_inference folder under fold_i/{model_name}
        valinf_root = os.path.join(fold_base, "validation_inference")
        os.makedirs(valinf_root, exist_ok=True)

        # Save confusion matrix plot into valinf_root
        cm_path = os.path.join(valinf_root, f"confusion_fold{fold}.png")
        plot_confusion(
            cm_clinician,
            classes=["tb", "normal"],
            save_path=cm_path,
            figsize=(6, 6),
            cmap="Blues",
            alpha=0.8,
            xt_label_rotation=0,
            xt_label_size=h["xt_label_size"],
            yt_label_size=h["yt_label_size"],
            xlabel_font_size=h["xlabel_font_size"],
            ylabel_font_size=h["ylabel_font_size"],
            title_font_size=h["title_font_size"],
            text_color=h["text_color"],
            text_font_size=h["text_font_size"]
        )
        print(f" [INFO] Saved confusion matrix → {cm_path}")

        # Save per‐fold metrics CSV into valinf_root
        record = {
            "fold": fold,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "balanced_accuracy": bal_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "NPV": npv,
            "F_score": fscore,
            "MCC": mcc,
            "Cohen_Kappa": kappa,
            "Youden_J": youden_j,
            "roc_auc": roc_auc_val
        }
        all_fold_records.append(record)
        metrics_df = pd.DataFrame([record])
        metrics_path = os.path.join(valinf_root, f"metrics_fold{fold}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f" [INFO] Saved per‐fold validation metrics → {metrics_path}")

        del model
        torch.cuda.empty_cache()

    # Save combined metrics CSV under the model folder’s validation_inference
    if not all_fold_records:
        print("⚠️ No validation metrics to save (no folds processed).")
        return

    # Choose a representative valinf_root for "all folds" output
    combined_root = os.path.join(h["output_dir"], h["model"])
    os.makedirs(combined_root, exist_ok=True)
    all_out_path = os.path.join(combined_root, "all_folds_validation_metrics.csv")
    pd.DataFrame(all_fold_records).to_csv(all_out_path, index=False)
    print(f"\n✅ Saved all‐fold validation metrics → {all_out_path}")

    # Print cross‐fold summary statistics
    stats = pd.DataFrame(all_fold_records).drop(columns=["fold", "TP", "FP", "FN", "TN"]).agg(["mean", "std"])
    display_map = {
        "balanced_accuracy": "balanced_accuracy",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
        "precision": "precision",
        "NPV": "NPV",
        "F_score": "F_score",
        "MCC": "MCC",
        "Cohen_Kappa": "Cohen_Kappa",
        "Youden_J": "Youden_J",
        "roc_auc": "ROC_AUC"
    }
    print("\nCross-Fold Summary Statistics 📊")
    for col in stats.columns:
        disp_name = display_map.get(col, col)
        mu = stats.loc["mean", col]
        sd = stats.loc["std",  col]
        print(f" • {disp_name}: {mu:.4f} ± {sd:.4f}")

    # Plot combined ROC curves + mean ROC with inset zoom; save under combined_root
    mean_fpr = np.linspace(0.0, 1.0, 200)
    interp_tprs = []

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for idx, rd in enumerate(roc_data):
        fpr_vals = rd["fpr"]
        tpr_vals = rd["tpr"]
        roc_auc_val = rd["auc"]

        ax_roc.plot(
            fpr_vals,
            tpr_vals,
            color=fold_colors[idx % len(fold_colors)],
            alpha=fold_curve_alpha,
            lw=fold_curve_lw,
            label=f"fold-{rd['fold']}: AUC={roc_auc_val:.4f}"
        )

        interp_tpr = np.interp(mean_fpr, fpr_vals, tpr_vals)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    mean_tpr = np.mean(np.vstack(interp_tprs), axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    ax_roc.plot(
        mean_fpr,
        mean_tpr,
        color=mean_line_color,
        lw=mean_line_lw,
        linestyle="--",
        label=f"Mean ROC: AUC={mean_auc:.4f}"
    )

    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.set_xlabel("False Positive Rate", fontsize=axis_fontsize)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=axis_fontsize)
    ax_roc.legend(loc="best", fontsize=h["font_size"])
    ax_roc.grid(False)

    # Inset zoom: [0.0–0.4 FPR, 0.9–1.0 TPR]
    x1, x2 = 0.0, 0.4
    y1, y2 = 0.9, 1.0

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=1.1,
        edgecolor="black",
        facecolor="none",
        linestyle="--"
    )
    ax_roc.add_patch(rect)

    ax_inset = inset_axes(
        ax_roc,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.5, 0.4, 0.3, 0.3),
        bbox_transform=ax_roc.transAxes,
        loc="lower left"
    )
    for idx, rd in enumerate(roc_data):
        ax_inset.plot(
            rd["fpr"],
            rd["tpr"],
            color=fold_colors[idx % len(fold_colors)],
            alpha=fold_curve_alpha,
            lw=fold_curve_lw
        )
    ax_inset.plot(
        mean_fpr,
        mean_tpr,
        color=mean_line_color,
        lw=mean_line_lw,
        linestyle="--"
    )
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(y1, y2)
    ax_inset.set_xlabel("FPR", fontsize=10)
    ax_inset.set_ylabel("TPR", fontsize=10)
    ax_inset.tick_params(axis="both", labelsize=8)

    # Connect rectangle corners to inset
    #Bottom-left
    con_bl = ConnectionPatch(
        xyA=(x1, y1), coordsA=ax_roc.transData,
        xyB=(x1, y1), coordsB=ax_inset.transData,
        color="black", linewidth=1.0, linestyle="--"
    )
    ax_roc.add_artist(con_bl)

    # Bottom-right
    con_br = ConnectionPatch(
        xyA=(x2, y1), coordsA=ax_roc.transData,
        xyB=(x2, y1), coordsB=ax_inset.transData,
        color="black", linewidth=1.0, linestyle="--"
    )
    ax_roc.add_artist(con_br)

    plt.tight_layout()
    roc_fig_path = os.path.join(combined_root, "roc_validation_summary.png")
    fig_roc.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Saved Validation ROC curves plot → {roc_fig_path}")

    print("\n🎉 Validation inference + ROC plotting complete!")

# Run validation inference
inference_validation_with_curves(h)


# In[ ]:


# Test inference 

OUTPUT_DIR = Path(h["output_dir"])
MODEL_NAME = h["model"]
TEST_DIR   = Path(h["test_dir"])
CLASS_LABELS = ["normal", "tb"] # class_to_idx = {"normal":0, "tb":1}

# Model summary helper
def print_and_save_model_summary(model: torch.nn.Module,
                                 device: torch.device,
                                 img_size: int,
                                 out_txt: Path) -> None:
    """
    Try torchsummary; if unavailable, fall back to manual parameter counts.
    Also prints to stdout and writes to out_txt.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    header = []
    header.append("="*80)
    header.append(f"Model: {MODEL_NAME}")
    header.append(f"Device: {device.type}")
    header.append(f"Input size (C,H,W): (3, {img_size}, {img_size})")
    header.append(f"Total params:      {total_params:,}")
    header.append(f"Trainable params:  {trainable_params:,}")
    header.append(f"Frozen params:     {frozen_params:,}")
    header.append("="*80)
    header_text = "\n".join(header)
    print(header_text)

    # torchsummary for layer-wise table
    table_text = ""
    try:
        from torchsummary import summary as ts_summary
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ts_summary(model, input_size=(3, img_size, img_size),
                       device=("cuda" if device.type=="cuda" else "cpu"))
        table_text = buf.getvalue()
        print(table_text)
    except Exception as e:
        fallback = (
            "\n[torchsummary not available or failed]\n"
            "Falling back to parameter counts only.\n"
            f"Error: {e}\n"
        )
        print(fallback)
        table_text = fallback

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(header_text + "\n" + table_text)

# Inference + Evaluation
def inference_test_with_curves(h: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = OUTPUT_DIR / MODEL_NAME / "test_inference"
    save_root.mkdir(parents=True, exist_ok=True)

    # Build test DataLoader
    dm_test = DataModule(h, fold_idx=None)
    dm_test.setup()
    test_loader = dm_test.test_dataloader()

    records, fpr_list, tpr_list, auc_list = [], [], [], []
    fold_colors = ['red', 'blue', 'green', 'orange', 'brown']
    axis_fontsize = 12
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

    for fold in range(h["n_splits"]):
        ckpt_dir = OUTPUT_DIR / f"fold_{fold}" / MODEL_NAME
        if not ckpt_dir.is_dir():
            print(f"⚠️ Skipping fold {fold}: {ckpt_dir} not found")
            continue

        ckpts = sorted(glob.glob(str(ckpt_dir / f"{MODEL_NAME}_fold{fold}_epoch*_*.ckpt")))
        if not ckpts:
            print(f"⚠️ Skipping fold {fold}: no checkpoints")
            continue

        # Pick ckpt
        best_loss, best_ckpt = float("inf"), None
        for p in ckpts:
            m = re.match(rf".*_fold{fold}_epoch\d+_(\d+\.\d+)_\d+\.\d+\.ckpt$", os.path.basename(p))
            if not m: 
                continue
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss, best_ckpt = loss, p
        if best_ckpt is None:
            print(f"⚠️ No valid ckpt for fold {fold}")
            continue
        print(f"\n🎯 Fold {fold}, ckpt={Path(best_ckpt).name}, val_loss={best_loss:.4f}")

        # Load & eval model
        model = TBModel.load_from_checkpoint(best_ckpt, h=h)
        model.to(device).eval()

        # Optional: save a model summary once per fold
        summary_txt = save_root / f"fold_{fold}" / "model_summary.txt"
        print_and_save_model_summary(model, device, int(h["img_size"]), summary_txt)

        # Inference
        y_true, y_pred, y_prob = [], [], []
        with torch.inference_mode():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        imgs, labels, _filenames = batch
                    elif len(batch) == 2:
                        imgs, labels = batch
                    else:
                        raise ValueError(f"Unexpected batch structure with length {len(batch)}")
                else:
                    raise ValueError("Unexpected batch type from DataLoader")

                imgs = imgs.to(device)
                with amp_ctx:
                    logits = model(imgs)
                logits = logits.float()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                y_true.extend(np.asarray(labels))
                y_pred.extend(preds)
                y_prob.extend(probs)

        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        y_prob = np.array(y_prob, dtype=float)

        # confusion matrix
        TP = int(((y_true==1)&(y_pred==1)).sum())
        FP = int(((y_true==0)&(y_pred==1)).sum())
        FN = int(((y_true==1)&(y_pred==0)).sum())
        TN = int(((y_true==0)&(y_pred==0)).sum())
        cm = np.array([[TP, FP],[FN, TN]], dtype=int)

        # Save confusion matrix
        fold_dir = save_root / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        plot_confusion(cm, CLASS_LABELS, fold_dir / f"confusion_fold{fold}.png", cmap="Blues")

        # Metrics
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, pos_label=1)
        spec = TN/(TN+FP) if TN+FP else 0.0
        prec = precision_score(y_true, y_pred, pos_label=1)
        npv = TN/(TN+FN) if TN+FN else 0.0
        f1s = f1_score(y_true, y_pred, pos_label=1)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        youden = sens + spec - 1
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        records.append({
            "fold": fold, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "balanced_accuracy": bal_acc, "sensitivity": sens,
            "specificity": spec, "precision": prec, "NPV": npv,
            "F1_score": f1s, "MCC": mcc, "Cohen_Kappa": kappa,
            "Youden_J": youden, "ROC_AUC": roc_auc
        })
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)
        del model
        torch.cuda.empty_cache()
    
    # Save metrics CSV
    df = pd.DataFrame(records)
    df.to_csv(save_root / "test_inference_metrics.csv", index=False)
    print(f"\n✅ Saved metrics to {save_root/'test_inference_metrics.csv'}")
    
    # Print cross-fold mean ± std for metrics
    if not df.empty:
        stats = df.drop(columns=["fold", "TP", "FP", "FN", "TN"]).agg(["mean", "std"])
        print("\n📊 Cross-Fold Summary Statistics  📊")
        for metric in stats.columns:
            mu, sd = stats.loc["mean", metric], stats.loc["std", metric]
            print(f" • {metric}: {mu:.4f} ± {sd:.4f}")

    # Plot ROC curves (per-fold + mean) with inset zoom
    if fpr_list:
        common_fpr = np.linspace(0, 1, 200)
        tprs_interp = []
        for fpr_vals, tpr_vals in zip(fpr_list, tpr_list):
            tprs_interp.append(np.interp(common_fpr, fpr_vals, tpr_vals))
        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_auc = np.mean(auc_list)

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for i, (fpr_vals, tpr_vals, auc_val) in enumerate(zip(fpr_list, tpr_list, auc_list)):
            ax_roc.plot(
                fpr_vals,
                tpr_vals,
                color=fold_colors[i % len(fold_colors)],
                alpha=0.9,
                lw=2,
                label=f"fold-{i}: AUC={auc_val:.4f}"
            )
        ax_roc.plot(
            common_fpr,
            mean_tpr,
            color='black',
            lw=2,
            linestyle='--',
            label=f"Mean ROC (AUC={mean_auc:.4f})"
        )
        ax_roc.set_xlabel("False Positive Rate", fontsize=axis_fontsize)
        ax_roc.set_ylabel("True Positive Rate", fontsize=axis_fontsize)
        ax_roc.legend(loc="best", fontsize=10)
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.02)
        ax_roc.grid(False)

        # Inset zoom: [0.0–0.4 FPR, 0.9–1.0 TPR]
        x1, x2 = 0.0, 0.4
        y1, y2 = 0.9, 1.0
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1.1, edgecolor='black',
                                 facecolor='none', linestyle='--')
        ax_roc.add_patch(rect)
        ax_inset = inset_axes(
            ax_roc, width="100%", height="100%",
            bbox_to_anchor=(0.5, 0.4, 0.3, 0.3),
            bbox_transform=ax_roc.transAxes, loc='lower left'
        )
        for i, (fpr_vals, tpr_vals) in enumerate(zip(fpr_list, tpr_list)):
            ax_inset.plot(
                fpr_vals,
                tpr_vals,
                color=fold_colors[i % len(fold_colors)],
                alpha=0.9,
                lw=2
            )
        ax_inset.plot(common_fpr, mean_tpr, color='black', lw=2, linestyle='--')
        ax_inset.set_xlim(x1, x2)
        ax_inset.set_ylim(y1, y2)
        ax_inset.set_xlabel("FPR", fontsize=10)
        ax_inset.set_ylabel("TPR", fontsize=10)
        ax_inset.tick_params(axis="both", labelsize=8)

        con_bl = ConnectionPatch(
            xyA=(x1, y1), coordsA=ax_roc.transData,
            xyB=(x1, y1), coordsB=ax_inset.transData,
            color='black', linewidth=1.0, linestyle='--'
        )
        ax_roc.add_artist(con_bl)
        con_br = ConnectionPatch(
            xyA=(x2, y1), coordsA=ax_roc.transData,
            xyB=(x2, y1), coordsB=ax_inset.transData,
            color='black', linewidth=1.0, linestyle='--'
        )
        ax_roc.add_artist(con_br)

        plt.tight_layout()
        roc_fig_path = os.path.join(save_root, "test_roc_curves.png")
        fig_roc.savefig(roc_fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved ROC curves plot → {roc_fig_path}")
        plt.show()

    print("\n🎉 Inference and evaluation complete!")

# perform inference
inference_test_with_curves(h)


# ## Class activation visualization using Pytorch Grad-CAM package (jacobgil)
# 
# 1. Works for CNNs, hybrids, and ViTs.
# 
# 2. Auto-selects the best target layer in this priority order: The newly added post3x3 conv if it exists and is 3×3
# 
# 3. Handles class-dependent (GradCAM, GradCAM++, XGradCAM, HiResCAM…) and class-agnostic (EigenCAM) methods.
# 
# 4. Saves raw masks, heatmaps, contours, and bounding boxes at original resolution.
# 
# 5. Prints which target layer was used to extract activations.

# In[ ]:


# Helper functions

#Preprocessing
def build_preprocess(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
def preprocess_image(orig_rgb: np.ndarray,
                     preprocess: transforms.Compose,
                     device: torch.device) -> torch.Tensor:
    """
    orig_rgb expected in [0,1], H×W×3 float32
    """
    img_u8 = (np.clip(orig_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return preprocess(img_u8).unsqueeze(0).to(device)

# Utility: find fully-qualified module name for logging
def _find_module_name(root: nn.Module, target: nn.Module) -> Optional[str]:
    for name, m in root.named_modules():
        if m is target:
            return name or "<root>"
    return None
def _kernel_tuple(m: nn.Conv2d) -> Tuple[int, int]:
    if isinstance(m.kernel_size, tuple):
        return (m.kernel_size[0], m.kernel_size[1])
    return (int(m.kernel_size), int(m.kernel_size))

# Target layer resolution
def _last_conv_kgt1(module: nn.Module) -> Optional[nn.Conv2d]:
    """
    Return the last Conv2d with kernel > 1; if none, return last Conv2d.
    """
    last_any = None
    last_kgt1 = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_any = m
            k = _kernel_tuple(m)
            if max(k) > 1:
                last_kgt1 = m
    return last_kgt1 if last_kgt1 is not None else last_any

def _vit_patch_embed_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    """
    For timm ViTs, the patch embed is usually a Conv2d at module.patch_embed.proj.
    We'll try to find the last Conv2d whose qualified name contains 'patch_embed'.
    """
    candidates = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d) and ("patch_embed" in name or "patch" in name):
            candidates.append((name, m))
    if candidates:
        return candidates[-1][1]
    return None

def select_target_layer_for_cam(model: nn.Module) -> nn.Conv2d:
    """
    Heuristic order:
      1) model.post3x3 if it exists and is Conv2d with kernel (3,3)
      2) Last Conv2d with kernel > 1 in model.backbone
      3) ViT patch-embed Conv2d if present (features_only=True path)
      4) Last Conv2d anywhere in the model
    """
    # Case 1: prefer the trainable 3×3 adapter if present
    post3x3 = getattr(model, "post3x3", None)
    if isinstance(post3x3, nn.Sequential):
        for m in post3x3.modules():
            if isinstance(m, nn.Conv2d):
                if _kernel_tuple(m) == (3, 3):
                    return m
    elif isinstance(post3x3, nn.Conv2d):
        if _kernel_tuple(post3x3) == (3, 3):
            return post3x3

    # Case 2: scan backbone for last Conv2d with k>1
    if hasattr(model, "backbone"):
        target = _last_conv_kgt1(model.backbone)
        if isinstance(target, nn.Conv2d):
            return target

        # Case 3: ViT patch embed conv
        target = _vit_patch_embed_conv(model.backbone)
        if isinstance(target, nn.Conv2d):
            return target

    # Case 4: fallback: last Conv2d anywhere
    target = _last_conv_kgt1(model)
    if isinstance(target, nn.Conv2d):
        return target
    raise RuntimeError("No suitable Conv2d layer found for Grad-CAM.")

# Model loader with auto strict fallback
def load_model_auto_strict(ckpt_path: str,
                           h: Dict,
                           device: torch.device):
    """
    Try strict=True first; if it fails, fall back to strict=False.
    Returns model.eval() on the requested device.
    """
    try:
        model = TBModel.load_from_checkpoint(
            ckpt_path, h=h, strict=True, map_location=device
        )
        print("[LOAD] strict=True worked.")
    except Exception as e:
        print(f"[LOAD] strict=True failed: {e}\n[LOAD] retrying with strict=False.")
        model = TBModel.load_from_checkpoint(
            ckpt_path, h=h, strict=False, map_location=device
        )
        print("[LOAD] strict=False used.")
    return model.to(device).eval()

# Main CAM runner
def run_cam_per_fold(
    h: Dict[str, Union[str, int, float, bool]],
    method: str = "gradcam",
    abnormal_class_index: int = 1,
    threshold: Optional[float] = 0.4,
    heatmap_alpha: float = 0.5,
    save_contours: bool = True,
    contour_color: Tuple[int,int,int] = (0, 0, 255),
    contour_thickness: int = 3,
    contour_line_type: int = cv2.LINE_AA,
    only_abnormal_folder: bool = True
) -> None:
    # config checks
    for key in ("test_dir", "output_dir", "model", "n_splits", "img_size"):
        if key not in h:
            raise KeyError(f"Missing h['{key}']")
    img_size = int(h["img_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = str(h["model"])
    preprocess = build_preprocess(img_size)
    # choose images from the abnormal class test folder
    test_dir = Path(h["test_dir"])
    image_paths: List[Path] = []
    if only_abnormal_folder:
        abn_dir = test_dir / "tb"
        if not abn_dir.exists():
            abn_dir = test_dir / "test" / "tb"
        if abn_dir.exists():
            for p in sorted(abn_dir.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    image_paths.append(p)    else:
        image_paths = [
            p for p in test_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {test_dir} (abnormal_only={only_abnormal_folder})."
        )
    print(f"[CAM] Loaded {len(image_paths)} images from {test_dir}")

    # CAM method map
    CAMS = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "hirescam": HiResCAM,
    }
    mname = method.lower()
    if mname not in CAMS:
        raise ValueError(f"Unknown CAM method '{method}'")
    CAMClass = CAMS[mname]

    # iterate folds
    for fold in range(int(h["n_splits"])):
        ckpt_dir = Path(h["output_dir"]) / f"fold_{fold}" / model_id
        if not ckpt_dir.exists():
            print(f"[CAM] No checkpoint dir for fold {fold}: {ckpt_dir}")
            continue

        ckpts = sorted(ckpt_dir.glob(f"{model_id}_fold{fold}_epoch*_*.ckpt"))
        if not ckpts:
            print(f"[CAM] No checkpoints in {ckpt_dir}")
            continue

        # choose the checkpoint with the lowest val_loss from filename
        def _val_loss_from_name(p: Path) -> float:
            m = re.search(rf"_fold{fold}_epoch\d+_([\d\.]+)_", p.name)
            return float(m.group(1)) if m else float("inf")

        best_ckpt = min(ckpts, key=_val_loss_from_name)
        print(f"[CAM] Fold {fold}: using {best_ckpt.name}")

        # load the finetuned model
        pm = load_model_auto_strict(str(best_ckpt), h=h, device=device)

        # resolve target layer and log its qualified name and kernel
        try:
            target_layer = select_target_layer_for_cam(pm)
        except Exception as e:
            print(f"[CAM] Failed to resolve target layer: {e}")
            del pm
            torch.cuda.empty_cache()
            continue

        tl_name = _find_module_name(pm, target_layer) or "<unnamed>"
        k_hw = _kernel_tuple(target_layer)
        print(f"[CAM] Target layer: {tl_name}  kernel={k_hw}")

        # init CAM with the full LightningModule; pm.forward returns logits
        cam = CAMClass(model=pm, target_layers=[target_layer], reshape_transform=None)
        cam.batch_size = 1

        # output folders
        base = ckpt_dir / f"cam_viz" / method / f"fold_{fold}"
        (base / "masks").mkdir(parents=True, exist_ok=True)
        (base / "heatmaps").mkdir(parents=True, exist_ok=True)
        if save_contours:
            (base / "contours").mkdir(parents=True, exist_ok=True)
            (base / "bboxes").mkdir(parents=True, exist_ok=True)

        # process each image
        for img_fp in image_paths:
            bgr = cv2.imread(str(img_fp))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            H0, W0 = bgr.shape[:2]

            inp = preprocess_image(rgb, preprocess, device)

            # class-agnostic vs class-specific behavior
            if mname in {"eigencam"}:
                mask = cam(input_tensor=inp)[0]  # EigenCAM ignores targets
            else:
                mask = cam(
                    input_tensor=inp,
                    targets=[ClassifierOutputTarget(int(abnormal_class_index))],
                    aug_smooth=True,
                    eigen_smooth=True
                )[0]

            # normalize ∈ [0,1]
            mmin, mmax = float(np.min(mask)), float(np.max(mask))
            if mmax > mmin:
                mask = (mask - mmin) / (mmax - mmin)
            else:
                mask = np.zeros_like(mask, dtype=np.float32)

            # save raw mask @ original resolution
            mask_u8_small = np.uint8(np.clip(mask, 0, 1) * 255)
            mask_u8 = cv2.resize(mask_u8_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                str(base / "masks" / f"{img_fp.stem}__{model_id}__fold{fold}__{method}__mask.png"),
                mask_u8
            )

            # heatmap overlay @ original resolution
            heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_HOT)
            overlay = cv2.addWeighted(heat, float(heatmap_alpha), bgr, 1.0 - float(heatmap_alpha), 0.0)
            cv2.imwrite(
                str(base / "heatmaps" / f"{img_fp.stem}__{model_id}__fold{fold}__{method}__heat.png"),
                overlay
            )

            if save_contours:
                thr = int(255 * float(threshold)) if threshold is not None else 0
                _, binary = cv2.threshold(mask_u8, thr, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # draw contours
                cont_img = bgr.copy()
                cv2.drawContours(cont_img, contours, -1, contour_color, contour_thickness,
                                 lineType=contour_line_type)
                cv2.imwrite(
                    str(base / "contours" / f"{img_fp.stem}__{model_id}__fold{fold}__{method}__contours.png"),
                    cont_img
                )

                # draw bounding boxes
                box_img = bgr.copy()
                for cnt in contours:
                    x, y, w, h_ = cv2.boundingRect(cnt)
                    cv2.rectangle(box_img, (x, y), (x + w, y + h_),
                                  contour_color, contour_thickness,
                                  lineType=contour_line_type)
                cv2.imwrite(
                    str(base / "bboxes" / f"{img_fp.stem}__{model_id}__fold{fold}__{method}__bboxes.png"),
                    box_img
                )

        # cleanup
        if hasattr(cam, "activations_and_grads"):
            try:
                cam.activations_and_grads.release()
            except Exception:
                pass
        del cam, pm
        torch.cuda.empty_cache()
        print(f"[CAM] Fold {fold} done → {base}")

    print("[CAM] All folds complete.")

if __name__ == "__main__":
    method = "eigencam" # gradcam, gradcam++, scorecam, ablationcam, xgradcam, layercam, fullgrad, eigencam, eigengradcam, hirescam
    abnormal_class = 1 # TB = class index 1
    threshold = 0.4
    heatmap_alpha = 0.5
    save_contours = True
    contour_color = (0, 0, 255) # red in BGR
    contour_thickness = 3
    contour_line_type = cv2.LINE_AA
    run_cam_per_fold(
        h=h,
        method=method,
        abnormal_class_index=abnormal_class,
        threshold=threshold,
        heatmap_alpha=heatmap_alpha,
        save_contours=save_contours,
        contour_color=contour_color,
        contour_thickness=contour_thickness,
        contour_line_type=contour_line_type,
        only_abnormal_folder=True
    )


# ## The code below performs the following:
# 
# 1. Computes and compares the best model performance to the individual radiologits. First, identify image filenames with AP, PA, and exclude LAT.
# 2. Pair against radiologists using the CSV "image_data_github.csv"
# 3. Run subgroup analyses under separate directories: Confirmed/ (Confirmed Cases + Controls), Unconfirmed/ (Unconfirmed Cases + Controls), and All/ (All Cases + Controls)
# 4. For each subgroup, compute + save: confusion matrices (Model, Rad-1, Rad-2, Rad-3), metrics CSV + bar plot + heatmap
# 5. ROC curve for model (swept threshold) with radiologist operating points annotated by Youden’s J
# 6. Implement the paired statistics: McNemar overall (exact binomial for small discordants; continuity-corrected chi-square otherwise)
# 7. Sensitivity-focused McNemar restricted to TB-positive cases
# 8. Paired case-resampling bootstrap for ΔYouden’s J at τ=0.5 with percentile CI

# In[ ]:


from __future__ import annotations

import re
import json
import contextlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    roc_auc_score,
)

# ----------------------------
# User paths (edit if needed)
# ----------------------------
CKPT_FOLD2 = Path(
    "path/to/model/checkpoint.ckpt"
)

RAD_CSV = Path("path/to/metadata.csv"
)

OUT_ROOT = Path(
    "path/to/output"
)

# -------------------------------------------------
# Plot + bootstrap controls
# -------------------------------------------------
@dataclass(frozen=True)
class PlotCfg:
    # Confusions (unchanged behavior from your current working version)
    cm_figsize: Tuple[int, int] = (6, 6)
    cm_alpha: float = 0.95
    cm_xt_rot: int = 0
    cm_xt_size: int = 14
    cm_yt_size: int = 14
    cm_xlabel_size: int = 14
    cm_ylabel_size: int = 14
    cm_cbar_tick_size: int = 14
    cm_text_size: int = 20
    cm_text_color: str = "black"

    # ROC
    roc_axis_fontsize: int = 14
    roc_legend_fontsize: int = 20

    # Fig-1 (discordance)
    dis_figsize: Tuple[int, int] = (5, 5)
    dis_axis_size: int = 12
    dis_text_size: int = 18

    # Fig-2 (UpSet-like)
    upset_figsize: Tuple[int, int] = (10, 6)
    upset_bar_label_size: int = 11
    upset_set_label_size: int = 12
    upset_dot_size: int = 70

PLOT = PlotCfg()

@dataclass(frozen=True)
class BootstrapCfg:
    # supervisor asked 2k–10k; keep configurable
    B_metrics_ci: int = 5000
    B_deltaJ: int = 5000
    seed: int = 1337

BOOT = BootstrapCfg()

# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _is_frontal_ap_pa(fname: str) -> bool:
    """
    Keep filenames that contain AP/PA (case-insensitive) and exclude LAT.
    """
    s = str(fname)
    if re.search(r"lat", s, flags=re.IGNORECASE):
        return False
    return re.search(r"(?:^|[^A-Za-z])(AP|PA)(?:[^A-Za-z]|$)", s, flags=re.IGNORECASE) is not None

def _to_bin_pred(x: Any) -> float:
    """
    Radiologist predictions are 'Cases'/'Controls'. Map Cases->1, Controls->0.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if s == "cases":
        return 1.0
    if s == "controls":
        return 0.0
    return np.nan

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0

# -------------------------------------------------------------------
# Correct TN/FP/FN/TP extraction (positive class = 1 = TB)
# -------------------------------------------------------------------
def _counts_from_y(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Returns (TN, FP, FN, TP) using standard definitions.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    return TN, FP, FN, TP

def _confusion_predrows_truecols(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    REQUIRED layout (Predicted rows, True columns):
        [[TN, FN],
         [FP, TP]]
    """
    TN, FP, FN, TP = _counts_from_y(y_true, y_pred)
    return np.array([[TN, FN],
                     [FP, TP]], dtype=int)

def _metrics_from_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    TN, FP, FN, TP = _counts_from_y(y_true, y_pred)

    sens = _safe_div(TP, TP + FN)
    spec = _safe_div(TN, TN + FP)
    bal_acc = 0.5 * (sens + spec)
    prec = _safe_div(TP, TP + FP)
    npv = _safe_div(TN, TN + FN)
    f1s = _safe_div(2.0 * TP, 2.0 * TP + FP + FN)

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    youden = sens + spec - 1.0

    out: Dict[str, float] = dict(
        TP=float(TP), FP=float(FP), FN=float(FN), TN=float(TN),
        balanced_accuracy=float(bal_acc),
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(prec),
        NPV=float(npv),
        F1_score=float(f1s),
        MCC=float(mcc),
        Cohen_Kappa=float(kappa),
        Youden_J=float(youden),
    )

    if y_prob is not None:
        try:
            out["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["ROC_AUC"] = float("nan")
    else:
        out["ROC_AUC"] = float("nan")

    return out

def _youden_j_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    TN, FP, FN, TP = _counts_from_y(y_true, y_pred)
    sens = _safe_div(TP, TP + FN)
    spec = _safe_div(TN, TN + FP)
    return float(sens + spec - 1.0)

def plot_confusion_pred_true(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    *,
    figsize: Tuple[int, int] = (6, 6),
    cmap: str = "Purples",
    alpha: float = 0.95,
    xt_label_rotation: int = 0,
    xt_label_size: int = 14,
    yt_label_size: int = 14,
    xlabel_font_size: int = 14,
    ylabel_font_size: int = 14,
    cbar_tick_size: int = 14,
    text_color: str = "black",
    text_font_size: int = 20,
) -> None:
    """
    cm is expected in layout:
        [[TN, FN],
         [FP, TP]]
    """
    cm = np.asarray(cm, dtype=int)
    assert cm.shape == (2, 2), f"Expected 2x2 cm, got {cm.shape}"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=plt.get_cmap(cmap), alpha=alpha)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=cbar_tick_size)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, rotation=xt_label_rotation, fontsize=xt_label_size)
    ax.set_yticklabels(class_names, fontsize=yt_label_size)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    ax.set_xlabel("True Label", fontsize=xlabel_font_size)
    ax.set_ylabel("Predicted Label", fontsize=ylabel_font_size)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                fontsize=text_font_size, color=text_color)

    plt.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def infer_fold2_exact(
    h: Dict,
    ckpt_path: Path,
) -> pd.DataFrame:
    """    
      - amp_ctx = autocast(bfloat16) on CUDA
      - logits.float()
      - probs = softmax(logits)[:,1]
      - preds = argmax(logits)
      - y_true from DataLoader labels
    """
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

    dm_test = DataModule(h, fold_idx=None)
    dm_test.setup()
    test_loader = dm_test.test_dataloader()

    model = TBModel.load_from_checkpoint(str(ckpt_path), h=h)
    model.to(device).eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    fnames: List[str] = []

    with torch.inference_mode():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    imgs, labels, _filenames = batch
                elif len(batch) == 2:
                    imgs, labels = batch
                    _filenames = None
                else:
                    raise ValueError(f"Unexpected batch structure with length {len(batch)}")
            else:
                raise ValueError("Unexpected batch type from DataLoader")

            imgs = imgs.to(device)

            with amp_ctx:
                logits = model(imgs)

            logits = logits.float()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            y_true.extend(np.asarray(labels))
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

            if _filenames is not None:
                fnames.extend([str(x) for x in _filenames])

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)

    if fnames and (len(fnames) != len(y_true_arr)):
        raise RuntimeError(f"Filename count mismatch: {len(fnames)} vs labels {len(y_true_arr)}")

    return pd.DataFrame({
        "original": fnames if fnames else [f"idx_{i:05d}" for i in range(len(y_true_arr))],
        "y_true": y_true_arr,
        "model_pred": y_pred_arr,
        "model_prob": y_prob_arr,
    })

# ----------------------------
# ROC plot
# ----------------------------
def plot_roc_model_plus_rads(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    rad_points: Dict[str, Tuple[float, float, float]],  # name -> (FPR, TPR, J)
    model_J: float,
    model_auc: float,
    model_point: Tuple[float, float],
    save_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    pt_color = {
        "Model": "green",
        "Rad-1": "red",
        "Rad-2": "blue",
        "Rad-3": "saddlebrown",
    }

    fig, ax = plt.subplots(figsize=(7, 7))

    # Model ROC curve
    ax.plot(fpr, tpr, color="red", lw=2)

    # Scatter operating points
    ax.scatter([model_point[0]], [model_point[1]], s=90, marker="o",
               color=pt_color["Model"], edgecolors="black")

    marker_map = {"Rad-1": "^", "Rad-2": "s", "Rad-3": "D"}
    for name, (xfpr, ytpr, _) in rad_points.items():
        ax.scatter([xfpr], [ytpr], s=90, marker=marker_map.get(name, "^"),
                   color=pt_color.get(name, "black"), edgecolors="black")

    ax.set_xlabel("False Positive Rate", fontsize=PLOT.roc_axis_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=PLOT.roc_axis_fontsize)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(False)

    # Legend only
    handles: List[Line2D] = []
    handles.append(Line2D(
        [0], [0], color="red", lw=2,
        marker="o", markerfacecolor=pt_color["Model"], markeredgecolor="black",
        label=f"Model (AUC={model_auc:.4f}, J={model_J:.4f})"
    ))
    for rname in ["Rad-1", "Rad-2", "Rad-3"]:
        if rname in rad_points:
            jv = float(rad_points[rname][2])
            handles.append(Line2D(
                [0], [0],
                color="none", lw=0,
                marker=marker_map.get(rname, "^"),
                markerfacecolor=pt_color[rname],
                markeredgecolor="black",
                label=f"{rname} (J={jv:.4f})"
            ))
    ax.legend(handles=handles, loc="best", fontsize=PLOT.roc_legend_fontsize)

    plt.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# McNemar exact + Holm + paired bootstrap ΔJ
# ----------------------------
def _mcnemar_exact_pvalue(b: int, c: int) -> float:
    """
    Exact McNemar test via binomial test on discordant pairs.
    Prefers SciPy; falls back to continuity-corrected chi-square if SciPy absent.
    """
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        # Use the smaller discordant count, two-sided exact
        return float(binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        try:
            from scipy.stats import chi2
            chi2_stat = (abs(b - c) - 1.0) ** 2 / float(n)
            return float(1.0 - chi2.cdf(chi2_stat, df=1))
        except Exception:
            return float("nan")

def _holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    prev = 0.0
    for i, (k, p) in enumerate(items, start=1):
        a = min(1.0, (m - i + 1) * p)
        a = max(a, prev)
        adj[k] = a
        prev = a
    return adj

def paired_bootstrap_delta_J(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_reader: np.ndarray,
    B: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    assert n == len(y_pred_model) == len(y_pred_reader)

    delta0 = _youden_j_from_preds(y_true, y_pred_model) - _youden_j_from_preds(y_true, y_pred_reader)

    deltas = np.empty(B, dtype=float)
    idx = np.arange(n)
    for b in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        deltas[b] = _youden_j_from_preds(y_true[samp], y_pred_model[samp]) - _youden_j_from_preds(y_true[samp], y_pred_reader[samp])

    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975]).tolist()
    return float(delta0), float(ci_low), float(ci_high)

# ----------------------------
# Bootstrap CI for metrics (Model + each reader)
# ----------------------------
def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    B: int,
    seed: int,
) -> pd.DataFrame:
    """
    Case bootstrap (resample indices with replacement).
    Returns tidy table:
      metric, point, ci_low, ci_high
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)

    # Point estimate
    point = _metrics_from_outputs(y_true, y_pred, y_prob)

    keys = list(point.keys())
    boot = {k: [] for k in keys}

    idx = np.arange(len(y_true))
    for _ in range(B):
        samp = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[samp]
        yp = y_pred[samp]
        yp_prob = (y_prob[samp] if y_prob is not None else None)
        m = _metrics_from_outputs(yt, yp, yp_prob)
        for k in keys:
            boot[k].append(m[k])

    rows = []
    for k in keys:
        arr = np.asarray(boot[k], dtype=float)
        # ignore NaNs if any (e.g., rare AUC failures)
        lo, hi = np.nanquantile(arr, [0.025, 0.975]).tolist()
        rows.append({"metric": k, "point": float(point[k]), "ci_low": float(lo), "ci_high": float(hi)})

    return pd.DataFrame(rows)

# ----------------------------
# Figure — TB+ discordance squares (Model vs each radiologist)
# ----------------------------
DISCORDANCE_CMAPS = {
    "All": "Greens",
    "Confirmed": "Blues",
    "Unconfirmed": "YlOrBr",  
}

def plot_tbpos_discordance_square(
    m_hit: np.ndarray,
    r_hit: np.ndarray,
    save_path: Path,
    *,
    reader_name: str,
    cohort: str,
    cmap: str = "Greys",
) -> Tuple[int, int]:
    """
    TB-positive only.
    m_hit/r_hit are boolean arrays indicating "called TB" among TB positives.

    Matrix (rows=Model, cols=Reader):
        [[both miss, reader only],
         [model only, both hit]]

    Returns (b, c) where:
      b = model only (Model=1, Reader=0)  -> model rescued
      c = reader only (Model=0, Reader=1) -> reader rescued
    """
    m_hit = np.asarray(m_hit, dtype=bool)
    r_hit = np.asarray(r_hit, dtype=bool)
    assert len(m_hit) == len(r_hit)

    both_miss = int((~m_hit & ~r_hit).sum())
    reader_only = int((~m_hit &  r_hit).sum())  # c
    model_only  = int(( m_hit & ~r_hit).sum())  # b
    both_hit    = int(( m_hit &  r_hit).sum())

    mat = np.array([[both_miss, reader_only],
                    [model_only,  both_hit]], dtype=int)

    fig, ax = plt.subplots(figsize=PLOT.dis_figsize)
    im = ax.imshow(mat, cmap=plt.get_cmap(cmap), alpha=0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=PLOT.dis_axis_size)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "TB"], fontsize=PLOT.dis_axis_size)
    ax.set_yticklabels(["Normal", "TB"], fontsize=PLOT.dis_axis_size)   
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xlabel(f"{reader_name}", fontsize=PLOT.dis_axis_size)
    ax.set_ylabel("Model", fontsize=PLOT.dis_axis_size)    
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=PLOT.dis_text_size)   
    plt.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return model_only, reader_only

# ----------------------------
# Figure — UpSet-like intersection plot
# ----------------------------
def _pattern_counts(bits: np.ndarray) -> pd.DataFrame:
    """
    bits: shape (N, K) boolean/int (0/1)
    Returns df with columns: pattern(str), count(int), and each set bit column.
    Pattern order is K bits in the provided column order.
    """
    bits = np.asarray(bits, dtype=int)
    K = bits.shape[1]
    patterns = ["".join(map(str, row.tolist())) for row in bits]
    vc = pd.Series(patterns).value_counts()

    out = vc.reset_index()
    out.columns = ["pattern", "count"]
    # expand bits
    for k in range(K):
        out[f"b{k}"] = out["pattern"].str[k].astype(int)
    return out.sort_values("count", ascending=False).reset_index(drop=True)

def plot_upset_like(
    bits: np.ndarray,
    set_names: List[str],
    set_colors: List[str],
    save_path: Path,
    *,
    caption: str,
) -> pd.DataFrame:
    """
    Pure matplotlib UpSet-like:
      - top bars: pattern counts
      - bottom: dot matrix membership with connecting lines
    Returns the pattern count table used to plot (also saved by caller if desired).
    """
    assert bits.shape[1] == len(set_names) == len(set_colors)

    tbl = _pattern_counts(bits)
    # Keep only patterns that actually occur
    patterns = tbl["pattern"].tolist()
    counts = tbl["count"].to_numpy(dtype=int)

    K = len(set_names)
    M = len(patterns)
    x = np.arange(M)

    fig = plt.figure(figsize=PLOT.upset_figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.6], hspace=0.05)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[1, 0], sharex=ax_bar)

    ax_bar.bar(x, counts)
    ax_bar.set_ylabel("Count", fontsize=PLOT.upset_bar_label_size)
    ax_bar.tick_params(axis="x", labelbottom=False)
    ax_bar.tick_params(axis="y", labelsize=PLOT.upset_bar_label_size)

    # Dot matrix
    ax_mat.set_yticks(np.arange(K))
    ax_mat.set_yticklabels(list(reversed(set_names)), fontsize=PLOT.upset_set_label_size)
    ax_mat.set_xlabel(caption, fontsize=11)
    ax_mat.tick_params(axis="x", labelsize=10)

    # Build membership matrix in table order, and plot from top to bottom
    # We stored b0..b{K-1} aligned with set_names order.
    mem = np.stack([tbl[f"b{k}"].to_numpy(dtype=int) for k in range(K)], axis=1)  # (M, K)
    mem = mem[:, ::-1]  # reverse to match reversed y labels

    for j in range(M):
        ys_on = np.where(mem[j] == 1)[0]
        # light background dots for off
        for yi in range(K):
            ax_mat.scatter(j, yi, s=PLOT.upset_dot_size * 0.35, color="lightgray", edgecolors="none")
        # colored dots for on
        for yi in ys_on:
            # map yi back to correct color (because we reversed)
            color = set_colors[::-1][yi]
            ax_mat.scatter(j, yi, s=PLOT.upset_dot_size, color=color, edgecolors="black", linewidths=0.6)
        if len(ys_on) >= 2:
            ax_mat.plot([j, j], [ys_on.min(), ys_on.max()], color="black", lw=1.0)

    ax_mat.set_xlim(-0.6, M - 0.4)
    ax_mat.set_ylim(-0.6, K - 0.4)

    plt.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return tbl

# ----------------------------
# Reader agreement stats (pairwise Cohen κ + Fleiss κ + pattern counts)
# ----------------------------
def fleiss_kappa_binary(ratings_3xN: np.ndarray) -> float:
    """
    Fleiss' kappa for 3 raters, binary categories {0,1}.
    ratings_3xN: shape (N, 3) in {0,1}
    """
    R = np.asarray(ratings_3xN, dtype=int)
    assert R.ndim == 2 and R.shape[1] == 3
    N = R.shape[0]
    n = 3

    # n_ik counts for each item i: how many raters voted for class 0 and class 1
    n0 = (R == 0).sum(axis=1)
    n1 = (R == 1).sum(axis=1)

    # P_i
    P_i = (n0 * (n0 - 1) + n1 * (n1 - 1)) / float(n * (n - 1))
    Pbar = float(P_i.mean())

    # p_k
    p0 = float(n0.sum()) / float(N * n)
    p1 = float(n1.sum()) / float(N * n)
    Pe = p0 * p0 + p1 * p1

    if (1.0 - Pe) <= 0:
        return float("nan")
    return float((Pbar - Pe) / (1.0 - Pe))

def reader_agreement_summary(rad1: np.ndarray, rad2: np.ndarray, rad3: np.ndarray) -> Dict[str, Any]:
    """
    Uses only rows where all 3 readers are present.
    Returns kappa pairwise + fleiss + pattern counts.
    """
    r1 = np.asarray(rad1, dtype=float)
    r2 = np.asarray(rad2, dtype=float)
    r3 = np.asarray(rad3, dtype=float)
    keep = (~np.isnan(r1)) & (~np.isnan(r2)) & (~np.isnan(r3))

    r1i = r1[keep].astype(int)
    r2i = r2[keep].astype(int)
    r3i = r3[keep].astype(int)

    out: Dict[str, Any] = {}
    out["N_all3"] = int(len(r1i))
    if len(r1i) == 0:
        out.update({
            "kappa_R1_R2": np.nan, "kappa_R1_R3": np.nan, "kappa_R2_R3": np.nan,
            "fleiss_kappa": np.nan,
            "count_3of3_TB": 0, "count_3of3_Normal": 0, "count_2of3_TB": 0, "count_1of3_TB": 0,
        })
        return out

    out["kappa_R1_R2"] = float(cohen_kappa_score(r1i, r2i))
    out["kappa_R1_R3"] = float(cohen_kappa_score(r1i, r3i))
    out["kappa_R2_R3"] = float(cohen_kappa_score(r2i, r3i))
    out["fleiss_kappa"] = float(fleiss_kappa_binary(np.stack([r1i, r2i, r3i], axis=1)))

    s = r1i + r2i + r3i
    out["count_3of3_TB"] = int((s == 3).sum())
    out["count_3of3_Normal"] = int((s == 0).sum())
    out["count_2of3_TB"] = int((s == 2).sum())
    out["count_1of3_TB"] = int((s == 1).sum())
    return out

# ----------------------------
# Error analysis (TB misses, rescues, FP on controls)
# ----------------------------
def error_analysis(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    rad1: np.ndarray,
    rad2: np.ndarray,
    rad3: np.ndarray,
) -> pd.DataFrame:
    """
    Uses only rows where all 3 readers are present for multi-reader intersection stats.
    Also reports per-method FN/FP counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    m = np.asarray(model_pred, dtype=int)

    r1f = np.asarray(rad1, dtype=float)
    r2f = np.asarray(rad2, dtype=float)
    r3f = np.asarray(rad3, dtype=float)

    # Per-method FN/FP (allow NaNs separately)
    rows = []

    def _fn_fp_for(pred_float: np.ndarray, name: str) -> None:
        keep = ~np.isnan(pred_float)
        yt = y_true[keep]
        yp = pred_float[keep].astype(int)

        fn_tb = int(((yt == 1) & (yp == 0)).sum())
        fp_ctl = int(((yt == 0) & (yp == 1)).sum())
        rows.append({"method": name, "FN_TB": fn_tb, "FP_controls": fp_ctl})

    _fn_fp_for(m.astype(float), "Model")
    _fn_fp_for(r1f, "Rad-1")
    _fn_fp_for(r2f, "Rad-2")
    _fn_fp_for(r3f, "Rad-3")

    # Multi-reader intersection / rescues (require all 3 readers present)
    keep3 = (~np.isnan(r1f)) & (~np.isnan(r2f)) & (~np.isnan(r3f))
    yt3 = y_true[keep3]
    m3 = m[keep3]
    r1 = r1f[keep3].astype(int)
    r2 = r2f[keep3].astype(int)
    r3 = r3f[keep3].astype(int)

    tbpos = (yt3 == 1)

    # TB misses
    model_miss_tb = int(((tbpos) & (m3 == 0)).sum())
    r1_miss_tb = int(((tbpos) & (r1 == 0)).sum())
    r2_miss_tb = int(((tbpos) & (r2 == 0)).sum())
    r3_miss_tb = int(((tbpos) & (r3 == 0)).sum())
    all3_miss_tb = int(((tbpos) & (r1 == 0) & (r2 == 0) & (r3 == 0)).sum())

    # Rescue counts
    # model correct among each reader's TB misses:
    model_rescue_r1 = int(((tbpos) & (r1 == 0) & (m3 == 1)).sum())
    model_rescue_r2 = int(((tbpos) & (r2 == 0) & (m3 == 1)).sum())
    model_rescue_r3 = int(((tbpos) & (r3 == 0) & (m3 == 1)).sum())

    # any reader correct among model's TB misses:
    any_reader_rescue_model = int(((tbpos) & (m3 == 0) & ((r1 == 1) | (r2 == 1) | (r3 == 1))).sum())

    # FP on controls (all 3 present)
    ctl = (yt3 == 0)
    fp_model = int(((ctl) & (m3 == 1)).sum())
    fp_r1 = int(((ctl) & (r1 == 1)).sum())
    fp_r2 = int(((ctl) & (r2 == 1)).sum())
    fp_r3 = int(((ctl) & (r3 == 1)).sum())

    summary = pd.DataFrame([{
        "N_all3_present": int(len(yt3)),
        "N_TBpos_all3_present": int(tbpos.sum()),
        "TB_missed_by_model": model_miss_tb,
        "TB_missed_by_rad1": r1_miss_tb,
        "TB_missed_by_rad2": r2_miss_tb,
        "TB_missed_by_rad3": r3_miss_tb,
        "TB_missed_by_all3_readers": all3_miss_tb,
        "model_rescues_rad1_FNs": model_rescue_r1,
        "model_rescues_rad2_FNs": model_rescue_r2,
        "model_rescues_rad3_FNs": model_rescue_r3,
        "any_reader_rescues_model_FNs": any_reader_rescue_model,
        "FP_controls_model": fp_model,
        "FP_controls_rad1": fp_r1,
        "FP_controls_rad2": fp_r2,
        "FP_controls_rad3": fp_r3,
    }])

    return pd.concat([pd.DataFrame(rows), summary], axis=0, ignore_index=True)

def plot_fp_controls_bar(err_df: pd.DataFrame, save_path: Path, caption: str) -> None:
    """
    Simple bar plot of FP on controls for Model/R1/R2/R3 (from per-method rows).
    """
    per = err_df.loc[err_df["method"].notna()].copy()
    per = per[per["method"].isin(["Model", "Rad-1", "Rad-2", "Rad-3"])]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(per["method"], per["FP_controls"].astype(int).to_numpy())
    ax.set_ylabel("False Positives on Controls")
    ax.set_xlabel(caption)
    plt.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Cohorts
# ----------------------------
def build_cohort(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    cohort = cohort.strip().lower()
    if cohort == "all":
        return df.copy()
    if cohort in ("confirmed", "unconfirmed"):
        status = "Confirmed" if cohort == "confirmed" else "Unconfirmed"
        mask = (df["Type"].str.lower() == "controls") | (
            (df["Type"].str.lower() == "cases") & (df["MC_status"].fillna("") == status)
        )
        return df.loc[mask].copy()
    raise ValueError(f"Unknown cohort: {cohort}")

# ----------------------------
# Main driver
# ----------------------------
def run_fold2_vs_radiologists_exact(h: Dict) -> None:
    _ensure_dir(OUT_ROOT)

    # 1) Inference ONCE (exact match)
    df_pred = infer_fold2_exact(h=h, ckpt_path=CKPT_FOLD2)

    # Sanity for full test loader
    y_true_all = df_pred["y_true"].to_numpy(dtype=int)
    y_pred_all = df_pred["model_pred"].to_numpy(dtype=int)
    y_prob_all = df_pred["model_prob"].to_numpy(dtype=float)

    m_all = _metrics_from_outputs(y_true_all, y_pred_all, y_prob_all)
    print("\n[Sanity] Fold-2 EXACT inference metrics (full test_loader):")
    print(json.dumps(m_all, indent=2))

    # 2) Load radiologist table
    rad = pd.read_csv(RAD_CSV)
    need_cols = [
        "original", "Type", "MC_status",
        "Rad-1_prediction", "Rad-2_prediction", "Rad-3_prediction",
    ]
    for c in need_cols:
        if c not in rad.columns:
            raise KeyError(f"Missing column in RAD CSV: {c}")

    rad = rad[need_cols].copy()
    rad["original"] = rad["original"].astype(str)
    rad = rad.loc[rad["original"].map(_is_frontal_ap_pa)].copy()

    # Merge
    df = df_pred.merge(rad, on="original", how="left", validate="one_to_one")
    if df["Type"].isna().any():
        missing = df.loc[df["Type"].isna(), "original"].tolist()[:25]
        raise RuntimeError(
            "Some test images could not be matched to RAD_CSV by 'original'. "
            f"First missing examples: {missing}"
        )

    # Reader calls -> {0,1}
    df["rad1"] = df["Rad-1_prediction"].map(_to_bin_pred)
    df["rad2"] = df["Rad-2_prediction"].map(_to_bin_pred)
    df["rad3"] = df["Rad-3_prediction"].map(_to_bin_pred)

    CLASS_LABELS = ["Normal", "TB"]

    CMAPS = {
        "Model": "Greens",
        "Rad-1": "Reds",
        "Rad-2": "Blues",
        "Rad-3": "YlOrBr",
    }

    cohorts = ["All", "Confirmed", "Unconfirmed"]
    for cohort in cohorts:
        cdir = OUT_ROOT / cohort
        _ensure_dir(cdir / "confusion")
        _ensure_dir(cdir / "roc")
        _ensure_dir(cdir / "metrics")
        _ensure_dir(cdir / "metrics_ci")
        _ensure_dir(cdir / "stats")
        _ensure_dir(cdir / "fig_disagreement")
        _ensure_dir(cdir / "fig_agreement")
        _ensure_dir(cdir / "error_analysis")

        dfc = build_cohort(df, cohort)

        y_true = dfc["y_true"].to_numpy(dtype=int)
        y_prob = dfc["model_prob"].to_numpy(dtype=float)
        y_pred_model = dfc["model_pred"].to_numpy(dtype=int)

        methods: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = [
            ("Model", y_pred_model, y_prob),
            ("Rad-1", dfc["rad1"].to_numpy(dtype=float), None),
            ("Rad-2", dfc["rad2"].to_numpy(dtype=float), None),
            ("Rad-3", dfc["rad3"].to_numpy(dtype=float), None),
        ]

        # --- Confusions + metrics point estimates ---
        rows: List[Dict[str, Any]] = []
        confs: Dict[str, np.ndarray] = {}

        # also keep per-method arrays for CI/plots
        method_arrays: Dict[str, Dict[str, Any]] = {}

        for name, yp_raw, yprob_opt in methods:
            if yp_raw.dtype.kind == "f":
                keep = ~np.isnan(yp_raw)
                yt = y_true[keep]
                yp = yp_raw[keep].astype(int)
                yprob = None
            else:
                yt = y_true
                yp = yp_raw.astype(int)
                yprob = yprob_opt

            cm_loc = _confusion_predrows_truecols(yt, yp)
            confs[name] = cm_loc
            met = _metrics_from_outputs(yt, yp, yprob)
            met["method"] = name
            met["cohort"] = cohort
            rows.append(met)

            method_arrays[name] = {"y_true": yt, "y_pred": yp, "y_prob": yprob}

            plot_confusion_pred_true(
                cm_loc,
                CLASS_LABELS,
                save_path=cdir / "confusion" / f"confusion_{name.replace(' ', '_')}.png",
                figsize=PLOT.cm_figsize,
                cmap=CMAPS.get(name, "Purples"),
                alpha=PLOT.cm_alpha,
                xt_label_rotation=PLOT.cm_xt_rot,
                xt_label_size=PLOT.cm_xt_size,
                yt_label_size=PLOT.cm_yt_size,
                xlabel_font_size=PLOT.cm_xlabel_size,
                ylabel_font_size=PLOT.cm_ylabel_size,
                cbar_tick_size=PLOT.cm_cbar_tick_size,
                text_color=PLOT.cm_text_color,
                text_font_size=PLOT.cm_text_size,
            )

        metrics_df = pd.DataFrame(rows).sort_values(["cohort", "method"])
        metrics_df.to_csv(cdir / "metrics" / "metrics_summary.csv", index=False)

        # --- ROC: model curve + 3 radiologist points ---
        model_auc = float(roc_auc_score(y_true, y_prob))
        model_J = float(_youden_j_from_preds(y_true, y_pred_model))

        TNm, FPm, FNm, TPm = _counts_from_y(y_true, y_pred_model)
        model_fpr = _safe_div(FPm, FPm + TNm)
        model_tpr = _safe_div(TPm, TPm + FNm)

        rad_points: Dict[str, Tuple[float, float, float]] = {}
        for rname, col in [("Rad-1", "rad1"), ("Rad-2", "rad2"), ("Rad-3", "rad3")]:
            yp_r_raw = dfc[col].to_numpy(dtype=float)
            keep = ~np.isnan(yp_r_raw)
            yt = y_true[keep]
            yr = yp_r_raw[keep].astype(int)

            TN, FP, FN, TP = _counts_from_y(yt, yr)
            fpr_r = _safe_div(FP, FP + TN)
            tpr_r = _safe_div(TP, TP + FN)
            J = float(tpr_r + (1.0 - fpr_r) - 1.0)
            rad_points[rname] = (float(fpr_r), float(tpr_r), float(J))

        plot_roc_model_plus_rads(
            y_true=y_true,
            y_prob=y_prob,
            rad_points=rad_points,
            model_J=model_J,
            model_auc=model_auc,
            model_point=(float(model_fpr), float(model_tpr)),
            save_path=cdir / "roc" / "roc_with_readers.png",
        )

        # ============================================================
        # Bootstrap 95% CI for each metric (Model + each reader)
        # ============================================================
        ci_rows = []
        for name in ["Model", "Rad-1", "Rad-2", "Rad-3"]:
            arr = method_arrays[name]
            ci_tbl = bootstrap_metrics_ci(
                y_true=arr["y_true"],
                y_pred=arr["y_pred"],
                y_prob=arr["y_prob"],
                B=BOOT.B_metrics_ci,
                seed=BOOT.seed,
            )
            ci_tbl.insert(0, "method", name)
            ci_tbl.insert(0, "cohort", cohort)
            ci_rows.append(ci_tbl)

        metrics_ci_df = pd.concat(ci_rows, axis=0, ignore_index=True)
        metrics_ci_df.to_csv(cdir / "metrics_ci" / "metrics_ci.csv", index=False)

        # ============================================================
        # Paired stats (McNemar exact + Holm + bootstrap ΔJ)
        # ============================================================
        stats_rows: List[Dict[str, Any]] = []
        p_overall: Dict[str, float] = {}
        p_sens: Dict[str, float] = {}

        ym = y_pred_model.astype(int)

        for name, col in [("Rad-1", "rad1"), ("Rad-2", "rad2"), ("Rad-3", "rad3")]:
            yp_r_raw = dfc[col].to_numpy(dtype=float)
            keep = ~np.isnan(yp_r_raw)

            yt = y_true[keep]
            ym_k = ym[keep]
            yr = yp_r_raw[keep].astype(int)

            # Overall McNemar on correctness
            m_correct = (ym_k == yt).astype(int)
            r_correct = (yr == yt).astype(int)
            b = int(((m_correct == 1) & (r_correct == 0)).sum())
            c = int(((m_correct == 0) & (r_correct == 1)).sum())
            p_mc = _mcnemar_exact_pvalue(b, c)
            p_overall[name] = p_mc

            # TB-positive McNemar (sensitivity focus): hit == predicted TB among TB positives
            pos = (yt == 1)
            if int(pos.sum()) > 0:
                ym_pos = ym_k[pos]
                yr_pos = yr[pos]
                m_hit = (ym_pos == 1).astype(int)
                r_hit = (yr_pos == 1).astype(int)
                b2 = int(((m_hit == 1) & (r_hit == 0)).sum())
                c2 = int(((m_hit == 0) & (r_hit == 1)).sum())
                p_mc_sens = _mcnemar_exact_pvalue(b2, c2)
            else:
                b2 = c2 = 0
                p_mc_sens = float("nan")
            p_sens[name] = p_mc_sens

            deltaJ, ci_lo, ci_hi = paired_bootstrap_delta_J(
                y_true=yt,
                y_pred_model=ym_k,
                y_pred_reader=yr,
                B=BOOT.B_deltaJ,
                seed=BOOT.seed,
            )

            stats_rows.append({
                "cohort": cohort,
                "reader": name,
                "b_overall": b, "c_overall": c,
                "mcnemar_p_overall": p_mc,
                "b_sens_tbpos": b2, "c_sens_tbpos": c2,
                "mcnemar_p_sens_tbpos": p_mc_sens,
                "delta_YoudenJ": deltaJ,
                "deltaJ_CI95_low": ci_lo,
                "deltaJ_CI95_high": ci_hi,
            })

        adj_overall = _holm_bonferroni(p_overall)
        adj_sens = _holm_bonferroni(p_sens)
        for r in stats_rows:
            nm = r["reader"]
            r["mcnemar_p_overall_holm"] = adj_overall.get(nm, float("nan"))
            r["mcnemar_p_sens_tbpos_holm"] = adj_sens.get(nm, float("nan"))

        paired_df = pd.DataFrame(stats_rows).sort_values(["cohort", "reader"])
        paired_df.to_csv(cdir / "stats" / "paired_stats.csv", index=False)

        # ============================================================
        # Figure: TB+ discordance squares (Model vs each reader)
        # ============================================================
        # Use TB-positive subset (per reader availability)
       
        for rname, col in [("Rad-1", "rad1"), ("Rad-2", "rad2"), ("Rad-3", "rad3")]:
            yp_r_raw = dfc[col].to_numpy(dtype=float)
            keep = ~np.isnan(yp_r_raw)
            yt = y_true[keep]
            yr = yp_r_raw[keep].astype(int)
            ym_k = ym[keep]

            tbpos = (yt == 1)
            m_hit = (ym_k[tbpos] == 1)
            r_hit = (yr[tbpos] == 1)
            plot_tbpos_discordance_square(
                m_hit=m_hit,
                r_hit=r_hit,
                save_path=cdir / "fig_disagreement" / f"tbpos_discordance_{rname}.png",
                reader_name=rname,
                cohort=cohort,
                cmap=DISCORDANCE_CMAPS.get(cohort, "Greys"),
            )

        # ============================================================
        # Figure: UpSet-like intersection plot
        # ============================================================
        # Require all 3 readers present to make intersections meaningful
        r1f = dfc["rad1"].to_numpy(dtype=float)
        r2f = dfc["rad2"].to_numpy(dtype=float)
        r3f = dfc["rad3"].to_numpy(dtype=float)
        keep3 = (~np.isnan(r1f)) & (~np.isnan(r2f)) & (~np.isnan(r3f))

        yt3 = y_true[keep3]
        m3 = ym[keep3]
        r1 = r1f[keep3].astype(int)
        r2 = r2f[keep3].astype(int)
        r3 = r3f[keep3].astype(int)

        set_names = ["Model", "Rad-1", "Rad-2", "Rad-3"]
        set_colors = ["green", "red", "blue", "saddlebrown"]

        # All cases intersection
        bits_all = np.stack([m3, r1, r2, r3], axis=1)
        tbl_all = plot_upset_like(
            bits=bits_all,
            set_names=set_names,
            set_colors=set_colors,
            save_path=cdir / "fig_agreement" / "upset_allcases.png",
            caption=f"{cohort} — intersections (all cases; all 3 readers present)",
        )
        tbl_all.to_csv(cdir / "fig_agreement" / "upset_allcases_table.csv", index=False)

        # TB-positive only intersection
        tbpos3 = (yt3 == 1)
        bits_tb = np.stack([m3[tbpos3], r1[tbpos3], r2[tbpos3], r3[tbpos3]], axis=1)
        tbl_tb = plot_upset_like(
            bits=bits_tb,
            set_names=set_names,
            set_colors=set_colors,
            save_path=cdir / "fig_agreement" / "upset_TBpos_only.png",
            caption=f"{cohort} — intersections (TB-positive only; all 3 readers present)",
        )
        tbl_tb.to_csv(cdir / "fig_agreement" / "upset_TBpos_only_table.csv", index=False)

        # ============================================================
        # Reader agreement stats (pairwise κ + Fleiss κ + pattern counts)
        # ============================================================
        agree = reader_agreement_summary(r1f, r2f, r3f)
        agree_df = pd.DataFrame([{"cohort": cohort, **agree}])
        agree_df.to_csv(cdir / "fig_agreement" / "reader_agreement_stats.csv", index=False)

        # ============================================================
        # Error analysis (TB misses, rescues, FP on controls)
        # ============================================================
        err_df = error_analysis(
            y_true=y_true,
            model_pred=ym,
            rad1=r1f,
            rad2=r2f,
            rad3=r3f,
        )
        err_df.to_csv(cdir / "error_analysis" / "error_analysis.csv", index=False)

        plot_fp_controls_bar(
            err_df=err_df,
            save_path=cdir / "error_analysis" / "fp_controls_bar.png",
            caption=f"{cohort}",
        )

    print(f"\n✅ Done. Outputs saved under: {OUT_ROOT}")

# ----------------------------
# RUN 
# ----------------------------
run_fold2_vs_radiologists_exact(h)


# ## Saving mis-classified samples from the best model across the folds and for the radiologists
# 
# 1. We Pick only the single “best” fold (highest Youden_J) based on validation performance.
# 2. Extract misclassified samples for that best model
# 3. Build test loader and run inference over test_loader.
# 4. Whenever true_label == 0 and pred_label == 1: record + copy into output_dir/{model}/misclassified/normal_as_abnormal/
# 5. Whenever true_label == 1 and pred_label == 0: record + copy into output_dir/{model}/misclassified/abnormal_as_normal/
# 6. At the end, we have two lists (normals_as_abnorm, abnormals_as_norm) and the copied images.
# 7. We then extract misclassified samples for each radiologist
# 8.  For each radiologist, partition the test set rows into: normal_as_abnormal: where y_true==0 & preds==1 abnormal_as_normal: where y_true==1 & preds==0
# 9.  Copy those image files into output_dir/{model}/radiologists/{p}/misclassified/normal_as_abnormal/ …/abnormal_as_normal/ and record the file names in two lists per reader.
# 10.  Build one “comparison” CSV

# In[ ]:


# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RAD_CSV = Path("metadata.csv")
TI_METRICS_CSV = Path(h["output_dir"]) / h["model"] / "all_fold_validation_metrics.csv"
COMP_DIR = Path(h["output_dir"]) / h["model"] / "comparison_to_radiologists"
COMP_DIR.mkdir(exist_ok=True, parents=True)

PARTICIPANTS = ["Rad-1", "Rad-2", "Rad-3"]
CMAP_MAP = {
    "Rad-1":   "Reds",
    "Rad-2":  "Blues",
    "Rad-3": "Greens",
    "avg_reader": "Purples",
    "best_model": "Oranges",
}

def _normalize_filenames(fnames: Union[str, Sequence[str]]) -> List[str]:
    """
    Collate can produce a list of strings; with batch_size=1 you may get ['file.png'].
    This helper returns a flat list of str.
    """
    if isinstance(fnames, str):
        return [fnames]
    return list(fnames)

def save_misclassified_comparison(h: Dict) -> None:
    """
    1) Finds the fold with highest Youden_J with the validation data in TI_METRICS_CSV
    2) Loads that fold’s best checkpoint
    3) Runs inference on the test set (uses DataModule’s test_dataloader which returns (img, label, filename))
       and copies misclassified images into:
         COMPARISON_DIR/misclassified/model/{normal_as_abnormal,abnormal_as_normal}
    4) Uses RAD_CSV to identify each radiologist’s misclassified images,
       copies them into:
         COMPARISON_DIR/misclassified/radiologists/{reader}/{normal_as_abnormal,abnormal_as_normal}
    5) Builds a 2‐row‐header CSV of semicolon‐separated filenames per column
    """
    #  Paths & dirs
    MIS_DIR = COMP_DIR / "misclassified"
    MODEL_MIS_DIR = MIS_DIR / "model"
    RAD_MIS_BASEDIR = MIS_DIR / "radiologists"

    # -best fold by Youden_J
    ti_df = pd.read_csv(TI_METRICS_CSV)
    best_fold = int(ti_df.loc[ti_df["Youden_J"].idxmax(), "fold"])
    logger.info("Best fold by Youden J: %d", best_fold)

    # locate checkpoint
    ckpt_dir = Path(h["output_dir"]) / f"fold_{best_fold}" / h["model"]
    ckpts = sorted(ckpt_dir.glob(f"{h['model']}_fold{best_fold}_epoch*_*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    def _val_loss_from_name(p: Path) -> float:
        m = re.search(rf"_fold{best_fold}_epoch\d+_([\d\.]+)_", p.name)
        return float(m.group(1)) if m else float("inf")
    best_ckpt = min(ckpts, key=_val_loss_from_name)
    logger.info("Using checkpoint %s", best_ckpt.name)

    # build test loader (use DataModule; it returns (img, label, filename))
    dm_test = DataModule(h, fold_idx=None)
    dm_test.setup()
    test_dataset = dm_test.test_dataset
    # file_paths in dataset are Paths; map basename -> full path
    file_paths: List[Path] = [Path(p) for p in test_dataset.file_paths]
    file_map = {p.name: p for p in file_paths}
    test_loader: DataLoader = dm_test.test_dataloader()  # may have batch_size > 1

    # make model misclassification dirs
    norm2abn_dir = MODEL_MIS_DIR / "normal_as_abnormal"
    abn2norm_dir = MODEL_MIS_DIR / "abnormal_as_normal"
    for d in (norm2abn_dir, abn2norm_dir):
        d.mkdir(parents=True, exist_ok=True)
    normals_as_abnorm_model: List[str] = []
    abnormals_as_norm_model: List[str] = []

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TBModel.load_from_checkpoint(best_ckpt, h=h, strict=False, map_location=device)
    model.to(device).eval()
    # inference -> copy misclassified
    with torch.no_grad():
        for batch in test_loader:
            # Expect (images, labels, filenames)
            if len(batch) == 3:
                images, labels, fnames = batch
                names: List[str] = _normalize_filenames(fnames)
            elif len(batch) == 2:
                images, labels = batch
                raise ValueError(
                    "Test DataLoader returned (image, label) only. "
                    "Ensure CXRDataset(..., return_paths=True) for reliable filename mapping."
                )
            else:
                raise ValueError(f"Unexpected test batch structure: len={len(batch)}")

            images = images.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(images).float()
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # Iterate items in batch
            for i in range(images.size(0)):
                true_label = int(labels[i].item())
                pred_label = int(preds[i].item())
                fname = names[i]  # already a basename

                # ensure we have full path for the filename
                if fname not in file_map:
                    # Try to recover by matching on basename only
                    candidate = Path(fname).name
                    if candidate in file_map:
                        fname = candidate
                    else:
                        logger.warning("Filename %s not found in file_map; skipping copy.", fname)
                        continue
                if true_label == 0 and pred_label == 1:
                    # normal -> abnormal
                    try:
                        shutil.copy2(file_map[fname], norm2abn_dir / fname)
                    except Exception as e:
                        logger.warning("Copy failed for %s → %s: %s", file_map[fname], norm2abn_dir / fname, e)
                    normals_as_abnorm_model.append(fname)
                elif true_label == 1 and pred_label == 0:
                    # abnormal -> normal
                    try:
                        shutil.copy2(file_map[fname], abn2norm_dir / fname)
                    except Exception as e:
                        logger.warning("Copy failed for %s → %s: %s", file_map[fname], abn2norm_dir / fname, e)
                    abnormals_as_norm_model.append(fname)

    del model
    torch.cuda.empty_cache()
    logger.info("Model misclassified: %d normal→abnormal, %d abnormal→normal",
                len(normals_as_abnorm_model),
                len(abnormals_as_norm_model))

    # radiologist misclassifications
    df = pd.read_csv(RAD_CSV)
    valid_files = set(file_map.keys())
    df_test = df[df["original"].isin(valid_files)].copy()
    # "Type" == "Cases" → abnormal (1), "Controls" → normal (0)
    df_test["y_true"] = (df_test["Type"] == "Cases").astype(int)
    normals_as_abnorm: Dict[str, List[str]] = {}
    abnormals_as_norm: Dict[str, List[str]] = {}
    for p in PARTICIPANTS:
        # normal (0) predicted as abnormal
        normals = df_test.loc[
            (df_test["y_true"] == 0) &
            (df_test[f"{p}_prediction"] == "Cases"),
            "original"
        ].tolist()
        # abnormal (1) predicted as normal
        abnormals = df_test.loc[
            (df_test["y_true"] == 1) &
            (df_test[f"{p}_prediction"] == "Controls"),
            "original"
        ].tolist()
        normals_as_abnorm[p] = normals
        abnormals_as_norm[p] = abnormals

        # make dirs and copy images
        base = RAD_MIS_BASEDIR / p
        d1, d2 = base / "normal_as_abnormal", base / "abnormal_as_normal"
        d1.mkdir(parents=True, exist_ok=True)
        d2.mkdir(parents=True, exist_ok=True)
        for fn in normals:
            try:
                shutil.copy2(file_map[fn], d1 / fn)
            except Exception as e:
                logger.warning("Copy failed for %s → %s: %s", file_map.get(fn, fn), d1 / fn, e)
        for fn in abnormals:
            try:
                shutil.copy2(file_map[fn], d2 / fn)
            except Exception as e:
                logger.warning("Copy failed for %s → %s: %s", file_map.get(fn, fn), d2 / fn, e)

        logger.info(
            "%s misclassified: %d normal→abnormal, %d abnormal→normal",
             p, len(normals), len(abnormals)
        )

    # build comparison CSV with two‐row header
    data: Dict[tuple, str] = {
        ("Model", "Normal misclassified as abnormal"): ";".join(normals_as_abnorm_model),
        ("Model", "Abnormal misclassified as normal"): ";".join(abnormals_as_norm_model),
    }
    for p in PARTICIPANTS:
        data[(p.capitalize(), "Normal misclassified as abnormal")] = ";".join(normals_as_abnorm[p])
        data[(p.capitalize(), "Abnormal misclassified as normal")] = ";".join(abnormals_as_norm[p])

    cols = pd.MultiIndex.from_tuples(list(data.keys()))
    df_comp = pd.DataFrame([list(data.values())], columns=cols)

    out_csv = COMP_DIR / "misclassified_comparison.csv"
    df_comp.to_csv(out_csv, index=False)
    logger.info("Comparison CSV saved to %s", out_csv)

if __name__ == "__main__":
    save_misclassified_comparison(h)


# ## END OF CODE
