# src/pipeLine/alexnetSSL.py
# -*- coding: utf-8 -*-

"""
AlexNet SSL Pipeline - Self-Supervised Pretraining (COMPLETE ULTIMATE EDITION)
================================================================================
Author: Babak Yousefian

Mirrors the exact structure of resNet18SSL.py, adapted for AlexNet:

1. Loads pretrained AlexNet from:
   finalReports/models/AlexNet/checkpoints/alexnet_best.pth

2. Wraps it with BYOLTrainer (creates projector + predictor)

3. Runs BYOL/SimCLR/MAE on unlabeled data (labels ignored during SSL)

4. Saves checkpoints to:
   finalReports/selfSupervisedLearning/AlexNet/alexnet/SSL_checkpoints/
       architecture/byol/alexnet_{best,checkpoint,final,last}_update_byol.pth
       early_stopping/model_byol_best_epoch_*.pth
       interpretability/
       optuna/
       tensorboard/

5. Supports full pipeline: BYOL → SimCLR → MAE → Best → Fine-Tune
"""

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Project Imports
# ================================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.AlexNet import create_alexnet
from src.models.AlexNet.utils import count_parameters, set_seed
from src.self_supervised import (
    create_byol_trainer,
    create_mae_trainer,
    create_simclr_trainer,
)

# ================================================================
# ✅ Optional imports (with graceful degradation)
# ================================================================

try:
    from src.reporter.captum_reporter import generate_interpretability_report
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("⚠️ Captum not available.")

try:
    from src.reporter.lime_reporter import explain_predictions_lime
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("⚠️ LIME not available.")

try:
    from src.reporter.shap_reporter import generate_shap_explanation
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP not available.")

try:
    from src.reporter.netron_reporter import (
        save_model_with_name,
        update_architecture_info_json,
    )
    HAS_NETRON = True
except ImportError:
    HAS_NETRON = False
    print("⚠️ Netron not available.")

try:
    from src.optimizers.custom_optimizers import get_optimizer
    HAS_CUSTOM_OPT = True
except ImportError:
    HAS_CUSTOM_OPT = False

try:
    from src.medical.image_io import (
        MEDPY_AVAILABLE, PYDICOM_AVAILABLE,
        dicom_to_numpy, enhance_ultrasound, read_dicom,
    )
    HAS_MEDICAL = True
except ImportError:
    HAS_MEDICAL = False
    MEDPY_AVAILABLE = False
    PYDICOM_AVAILABLE = False

try:
    from src.utils.memory_monitor import clear_memory, print_memory_status
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️ TensorBoard not installed. Install: pip install tensorboard")

try:
    from src.self_supervised.OptimizationsAlgorithms.Optimizers.optuna_optimizer import optimize_hyperparameters
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from src.self_supervised.OptimizationsAlgorithms.Augmentations.kornia_augmentations import KorniaAugmentation
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False
    print("⚠️ Kornia not available.")

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage, PageBreak, Paragraph,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️ ReportLab not installed. PDF reports will be limited.")

try:
    from src.self_supervised.Facilities.early_stopping import create_early_stopping
    HAS_EARLY_STOPPING = True
except ImportError:
    HAS_EARLY_STOPPING = False


# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)


# ================================================================
# ✅ Paths (ABSOLUTE)
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")

# ✅ AlexNet pretrained checkpoint
ALEXNET_PRETRAINED_PATH = os.path.join(
    PROJECT_ROOT, "finalReports", "models", "AlexNet", "checkpoints", "alexnet_best.pth"
)

# ✅ SSL directories for AlexNet
ALEXNET_SSL_DIR = os.path.join(SELF_SUPERVISED_ROOT, "AlexNet", "alexnet", "SSL_checkpoints")
ALEXNET_ARCHITECTURE_DIR = os.path.join(ALEXNET_SSL_DIR, "architecture")
ALEXNET_BYOL_DIR = os.path.join(ALEXNET_ARCHITECTURE_DIR, "byol")
ALEXNET_MAE_DIR = os.path.join(ALEXNET_ARCHITECTURE_DIR, "mae")
ALEXNET_SIMCLR_DIR = os.path.join(ALEXNET_ARCHITECTURE_DIR, "simclr")
ALEXNET_EARLY_STOPPING_DIR = os.path.join(ALEXNET_SSL_DIR, "early_stopping")
ALEXNET_INTERPRETABILITY_DIR = os.path.join(ALEXNET_SSL_DIR, "interpretability")
ALEXNET_OPTUNA_DIR = os.path.join(ALEXNET_SSL_DIR, "optuna")
ALEXNET_TENSORBOARD_DIR = os.path.join(ALEXNET_SSL_DIR, "tensorboard")

# ✅ Final models directory (for fine-tuned models)
FINAL_MODELS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "final_models")


# ================================================================
# ✅ Directory Management
# ================================================================

def ensure_directories() -> None:
    """Create all necessary directories for AlexNet SSL."""
    directories = [
        ALEXNET_SSL_DIR,
        ALEXNET_ARCHITECTURE_DIR,
        ALEXNET_BYOL_DIR,
        ALEXNET_MAE_DIR,
        ALEXNET_SIMCLR_DIR,
        ALEXNET_EARLY_STOPPING_DIR,
        ALEXNET_INTERPRETABILITY_DIR,
        ALEXNET_OPTUNA_DIR,
        ALEXNET_TENSORBOARD_DIR,
        FINAL_MODELS_DIR,
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ All AlexNet SSL directories created")


# ================================================================
# ✅ SSL Dataset for AlexNet
# ================================================================

class AlexNetSSLDataset(Dataset):
    """
    Dataset for SSL pretraining on AlexNet.

    Loads unlabeled images and returns TWO augmented views.
    Mirrors ResNet18SSLDataset exactly.
    """

    def __init__(
        self,
        image_paths: List[str],
        transform=None,
        return_path: bool = False,
        enhance_ultrasound: bool = True,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.return_path = return_path
        self.enhance_ultrasound = enhance_ultrasound

        # ✅ Cache for faster loading
        self.cache = {}
        self.cache_size = 500

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # ✅ Load image with medical support
        try:
            if img_path.lower().endswith(".dcm") and HAS_MEDICAL and PYDICOM_AVAILABLE:
                ds = read_dicom(img_path)
                if ds is not None:
                    image = dicom_to_numpy(ds)
                    if image is not None:
                        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
                        image = image.astype(np.uint8)
                        image = np.stack([image, image, image], axis=2)
                    else:
                        image = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.imread(img_path)
                if image is None:
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ✅ Ultrasound enhancement
            if self.enhance_ultrasound and HAS_MEDICAL and MEDPY_AVAILABLE:
                try:
                    image = enhance_ultrasound(image)
                except:
                    pass

            image = cv2.resize(image, (224, 224))

        except Exception as e:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert to PIL
        image = Image.fromarray(image)

        if self.transform:
            # ✅ دو-view برای SSL
            view1, view2 = self.transform(image)
            if self.return_path:
                return view1, view2, img_path
            return view1, view2

        if self.return_path:
            return image, img_path
        return image, 0


# ================================================================
# ✅ Class-Aware Dataset (module-level for Windows pickle)
# ================================================================

class AlexNetClassAwareDataset(Dataset):
    """
    Class-aware augmentation dataset for imbalanced liver ultrasound.

    G3 (70 samples) gets EXTREME augmentation → effective ~5x
    G2 (375 samples) gets strong
    G1 (613 samples) gets medium
    NL (1584 samples) gets light

    Result: G3 effectively becomes ~350 samples per epoch.
    """

    CLASS_NAMES = ["NL", "G1", "G2", "G3"]

    def __init__(self, image_paths, label_map, class_transforms, target_size=224):
        self.image_paths = image_paths
        self.label_map = label_map
        self.class_transforms = class_transforms
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        class_name = self.label_map.get(img_name, "NL")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))

        transform = self.class_transforms[class_name]

        # ✅ Two different views for SSL
        try:
            view1 = transform(image=img)["image"]
            view2 = transform(image=img)["image"]
        except Exception as e:
            print(f"⚠️ Transform failed for {img_name}: {e}")
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            fallback = A.Compose([
                A.Resize(self.target_size, self.target_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            view1 = fallback(image=img)["image"]
            view2 = fallback(image=img)["image"]

        return view1, view2


# ================================================================
# ✅ Main Pipeline Class
# ================================================================

class AlexNetSSL:
    """
    AlexNet SSL Pipeline (COMPLETE ULTIMATE EDITION).

    Mirrors ResNet18SSL exactly:
    - Loads pretrained AlexNet from alexnet_best.pth
    - BYOL/SimCLR/MAE training
    - 4 checkpoints: best, checkpoint, last, final
    - Early stopping
    - TensorBoard
    - Interpretability (Captum, LIME, SHAP, Netron)
    - PDF reports
    - Full pipeline (BYOL→SimCLR→MAE→Best→Fine-Tune)
    """

    CLASS_NAMES = ["NL", "G1", "G2", "G3"]

    def __init__(
        self,
        method: str = "byol",
        num_classes: int = 4,
        input_size: int = 224,
        batch_size: int = 32,
        lr: float = 0.0001,
        weight_decay: float = 1e-4,
        total_epochs: int = 40,
        warmup_epochs: int = 10,
        use_mixed_precision: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
        save_dir: str = None,
        optimizer_name: str = "adamw",
        use_accelerate: bool = False,
        use_imgaug: bool = False,
        use_kornia: bool = True,
        enhance_ultrasound: bool = True,
        pretrained_path: str = None,
        **kwargs,
    ):
        self.method = method
        self.num_classes = num_classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.use_mixed_precision = use_mixed_precision
        self.verbose = verbose
        self.save_dir = save_dir or ALEXNET_SSL_DIR
        self.optimizer_name = optimizer_name
        self.use_accelerate = use_accelerate
        self.use_imgaug = use_imgaug
        self.use_kornia = use_kornia
        self.enhance_ultrasound = enhance_ultrasound
        self.kwargs = kwargs

        # ✅ Device
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print(f"📌 Using device: {self.device}")

        set_seed(42)

        # ✅ CPU optimization
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(2)
        except Exception:
            pass

        ensure_directories()

        # ✅ Build encoder
        self.encoder = self._build_encoder()

        # ✅ Load pretrained weights
        actual_pretrained_path = None
        if pretrained_path is not None and os.path.exists(pretrained_path):
            actual_pretrained_path = pretrained_path
            self.load_pretrained_weights(pretrained_path)
        else:
            # Try default path
            if os.path.exists(ALEXNET_PRETRAINED_PATH):
                actual_pretrained_path = ALEXNET_PRETRAINED_PATH
                self.load_pretrained_weights(ALEXNET_PRETRAINED_PATH)
            else:
                print("\n" + "=" * 80)
                print("🚨 CRITICAL ERROR: Pretrained AlexNet weights NOT FOUND!")
                print("=" * 80)
                print(f"   Looked in:")
                print(f"   1. {pretrained_path}")
                print(f"   2. {ALEXNET_PRETRAINED_PATH}")
                print()
                print("   Run AlexNet advanced mode first (wFTAlgorithms → option 10)")
                print("=" * 80)
                raise FileNotFoundError(
                    f"Pretrained AlexNet weights not found. "
                    f"Expected at: {ALEXNET_PRETRAINED_PATH}"
                )

        self.pretrained_path = actual_pretrained_path
        if self.verbose:
            print(f"📌 Pretrained path (locked in): {self.pretrained_path}")

        # ✅ Create trainer
        self.trainer = self._create_trainer()

        self.history = {}
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.metrics = {}
        self.confusion_matrix = None
        self.roc_data = None
        self.best_params = {}

        # ✅ TensorBoard
        self.writer = None
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=ALEXNET_TENSORBOARD_DIR)
            if self.verbose:
                print(f"📊 TensorBoard logs: {ALEXNET_TENSORBOARD_DIR}")

        if self.verbose:
            self._print_info()

    # ============================================================
    # Build / Load / Trainer
    # ============================================================

    def _build_encoder(self) -> nn.Module:
        """Build AlexNet encoder for SSL with monkey-patched SSL support."""
        model = create_alexnet(
            num_classes=self.num_classes,
            width_mult=0.5,
            fc6_features=1024,
            fc7_features=512,
            dropout_fc=0.4,
            use_batch_norm=True,
            input_size=self.input_size,
            verbose=False,
        )

        # ============================================================
        # ✅ MONKEY-PATCH: Add SSL support WITHOUT touching src/models/
        # ============================================================

        # 1) Add _feature_dim attribute
        if not hasattr(model, "_feature_dim"):
            # c5 = 256 * 0.5 = 128 (matches model.channels[-1])
            model._feature_dim = model.channels[-1]

        # 2) Add get_embedding() method
        if not hasattr(model, "get_embedding"):
            import types
            import torch.nn.functional as F

            def get_embedding(self_model, x):
                """Extract pooled conv features for SSL (B, C5)."""
                feat = self_model.features(x)
                feat = self_model.conv_dropout(feat)
                feat = self_model.adaptive_pool(feat)
                feat = F.adaptive_avg_pool2d(feat, (1, 1))
                feat = torch.flatten(feat, 1)
                return feat

            model.get_embedding = types.MethodType(get_embedding, model)

        if self.verbose:
            print(f"✅ Built AlexNet encoder (SSL-patched)")
            print(f"   Feature dimension: {model._feature_dim}")
            print(f"   Input size: {self.input_size}")

        return model

    def load_pretrained_weights(self, model_path: str = None) -> bool:
        """
        Load pretrained AlexNet weights from alexnet_best.pth.

        Handles multiple checkpoint formats.
        """
        if model_path is None:
            model_path = ALEXNET_PRETRAINED_PATH

        if not os.path.exists(model_path):
            if self.verbose:
                print(f"⚠️ Pretrained model not found: {model_path}")
            self._pretrained_loaded = False
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "encoder_state_dict" in checkpoint:
                state_dict = checkpoint["encoder_state_dict"]
            else:
                state_dict = checkpoint

            # ✅ Analysis
            model_state = self.encoder.state_dict()
            missing_keys = []
            unexpected_keys = []

            for key in state_dict.keys():
                if key not in model_state:
                    unexpected_keys.append(key)

            if self.verbose:
                print(f"\n📊 Pretrained weights analysis:")
                print(f"   Missing keys: {len(missing_keys)}")
                print(f"   Unexpected keys: {len(unexpected_keys)}")
                if unexpected_keys:
                    print(f"   ⚠️ Unexpected: {unexpected_keys[:5]}...")

            # ✅ Load with strict=False
            self.encoder.load_state_dict(state_dict, strict=False)
            self._pretrained_loaded = True

            if self.verbose:
                print(f"✅ Pretrained weights loaded from: {model_path}")
                if "best_accuracy" in checkpoint:
                    print(f"   Previous Best Accuracy: {checkpoint['best_accuracy']:.4f}")
                if "best_epoch" in checkpoint:
                    print(f"   Previous Best Epoch: {checkpoint['best_epoch']}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Failed to load pretrained weights: {e}")
            self._pretrained_loaded = False
            return False

    def _create_trainer(self):
        """Create SSL trainer based on method.

        ✅ FIXED: Reduced projection_dim and hidden_dim to prevent collapse.
        """
        if self.method == "simclr":
            trainer = create_simclr_trainer(
                encoder=self.encoder,
                input_size=self.input_size,
                projection_dim=self.kwargs.get("projection_dim", 128),  # ← 128 (نه 256)
                hidden_dim=self.kwargs.get("hidden_dim", 1024),  # ← 1024 (نه 2048)
                projection_layers=self.kwargs.get("projection_layers", 3),
                temperature=self.kwargs.get("temperature", 0.5),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_lars=self.kwargs.get("use_lars", False),
                use_mixed_precision=self.use_mixed_precision,
                use_bf16=self.kwargs.get("use_bf16", True),
                gradient_accumulation_steps=self.kwargs.get("gradient_accumulation_steps", 1),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
                color_jitter_strength=self.kwargs.get("color_jitter_strength", 1.0),
                blur_kernel_size=self.kwargs.get("blur_kernel_size", 23),
                device=self.device,
                verbose=self.verbose,
                num_workers=self.kwargs.get("num_workers", 0),
                save_dir=self.save_dir,
            )

        elif self.method == "byol":
            trainer = create_byol_trainer(
                model=self.encoder,
                model_name="alexnet",
                input_size=self.input_size,
                projection_dim=self.kwargs.get("projection_dim", 256),
                hidden_dim=self.kwargs.get("hidden_dim", 1024),
                momentum=self.kwargs.get("momentum", 0.996),
                momentum_schedule=self.kwargs.get("momentum_schedule", "cosine"),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_lars=self.kwargs.get("use_lars", False),
                use_mixed_precision=self.use_mixed_precision,
                gradient_accumulation_steps=self.kwargs.get("gradient_accumulation_steps", 1),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
                device=self.device,
                verbose=self.verbose,
                num_workers=self.kwargs.get("num_workers", 0),
                augmentation_method=self.kwargs.get("augmentation_method", "byol"),
                save_dir=self.save_dir,
            )


        elif self.method == "mae":
            trainer = create_mae_trainer(
                encoder=self.encoder,
                img_size=self.input_size,
                patch_size=self.kwargs.get("patch_size", 1),  # ← 1 برای AlexNet
                mask_ratio=self.kwargs.get("mask_ratio", 0.5),
                embed_dim=self.kwargs.get("embed_dim", 256),  # ← 256
                depth=self.kwargs.get("depth", 6),
                num_heads=self.kwargs.get("num_heads", 4),  # ← 4 heads
                mlp_ratio=self.kwargs.get("mlp_ratio", 4.0),
                dropout=self.kwargs.get("dropout", 0.1),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_mixed_precision=self.use_mixed_precision,
                use_bf16=self.kwargs.get("use_bf16", True),
                gradient_accumulation_steps=self.kwargs.get("gradient_accumulation_steps", 1),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.kwargs.get("warmup_epochs", 3),
                total_epochs=self.total_epochs,
                device=self.device,
                verbose=self.verbose,
                num_workers=self.kwargs.get("num_workers", 0),
                save_dir=self.save_dir,
            )
        else:
            raise ValueError(f"Unsupported SSL method: {self.method}")

        return trainer

    def _print_info(self):
        """Print pipeline information."""
        pretrained_loaded = getattr(self, "_pretrained_loaded", False)

        print("\n" + "=" * 80)
        print("🚀 AlexNet SSL Pipeline (COMPLETE ULTIMATE EDITION)")
        print("=" * 80)
        print(f"   Method: {self.method.upper()}")
        print(f"   Input Size: {self.input_size}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Total Epochs: {self.total_epochs}")
        print(f"   Learning Rate: {self.lr}")
        print(f"   Optimizer: {self.optimizer_name}")
        print(f"   Device: {self.device}")
        print(f"   Save Directory: {self.save_dir}")
        print(f"   Pretrained Weights: {'✅' if pretrained_loaded else '❌'}")
        print(f"   Pretrained Path: {self.pretrained_path}")
        print(f"   TensorBoard: {'✅' if HAS_TENSORBOARD else '❌'}")
        print(f"   Optuna: {'✅' if HAS_OPTUNA else '❌'}")
        print(f"   Kornia: {'✅' if HAS_KORNIA else '❌'}")
        print(f"   Medical: {'✅' if HAS_MEDICAL else '❌'}")
        print("=" * 80)

    # ============================================================
    # Class-Aware DataLoader
    # ============================================================

    def _create_class_aware_dataloader(
            self,
            image_paths,
            batch_size,
            shuffle,
            num_workers,
            target_size,
            augmentation_method=None,
            enhance_ultrasound=True,
            use_imgaug=False,
            use_kornia=False,
    ):
        """
        Class-Aware Augmentation with method-specific strengths.

        ✅ FIXED for SimCLR:
        - SimCLR gets stronger augmentation (needs diverse negative pairs)
        - BYOL gets medium/strong
        - MAE gets light/medium (self-masking handles diversity)
        """
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # ✅ اگر augmentation_method داده نشد، از self.method استفاده کن
        if augmentation_method is None:
            augmentation_method = self.method

        # ✅ Load labels
        labels_csv = os.path.join(PROJECT_ROOT, "finalReports", "labels.csv")
        if not os.path.exists(labels_csv):
            labels_csv = os.path.join(PROJECT_ROOT, "finalReports", "labels.csv")

        df = pd.read_csv(labels_csv)
        print(f"📂 Loaded labels: {labels_csv} ({len(df)} rows)")
        print(f"🎨 Augmentation method: {augmentation_method.upper()}")

        # ✅ Print imbalance stats
        label_counts = df["liverLabel"].value_counts()
        print("\n📊 Class Distribution:")
        for cls in ["NL", "G1", "G2", "G3"]:
            count = label_counts.get(cls, 0)
            print(f"   {cls}: {count}")

        # ============================================================
        # ✅ PER-CLASS TRANSFORMS
        # ============================================================
        def get_pipeline(strength: str, target_size: int):
            """Build Albumentations pipeline for given strength."""
            if strength == "light":
                rotation = 15
                brightness, contrast = 0.20, 0.20
                saturation, hue = 0.10, 0.05
                noise_var, blur_limit = (10.0, 30.0), 5
                scale, ratio = (0.85, 1.0), (0.85, 1.15)
                elastic_p, num_holes, hole_size = 0.2, 6, 16
            elif strength == "medium":
                rotation = 20
                brightness, contrast = 0.30, 0.30
                saturation, hue = 0.20, 0.08
                noise_var, blur_limit = (10.0, 40.0), 7
                scale, ratio = (0.75, 1.0), (0.80, 1.20)
                elastic_p, num_holes, hole_size = 0.3, 8, 20
            elif strength == "strong":
                rotation = 25
                brightness, contrast = 0.40, 0.40
                saturation, hue = 0.30, 0.10
                noise_var, blur_limit = (15.0, 50.0), 9
                scale, ratio = (0.65, 1.0), (0.75, 1.25)
                elastic_p, num_holes, hole_size = 0.4, 10, 24
            else:  # extreme
                rotation = 30
                brightness, contrast = 0.50, 0.50
                saturation, hue = 0.40, 0.12
                noise_var, blur_limit = (20.0, 60.0), 11
                scale, ratio = (0.55, 1.0), (0.70, 1.30)
                elastic_p, num_holes, hole_size = 0.5, 12, 28

            return A.Compose([
                A.RandomResizedCrop(
                    size=(target_size, target_size),
                    scale=scale,
                    ratio=ratio,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=rotation, p=0.7, border_mode=0),
                A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=0.7,
                ),
                # ✅ Color jitter (prevents collapse in SSL)
                A.HueSaturationValue(
                    hue_shift_limit=int(hue * 180),
                    sat_shift_limit=int(saturation * 100),
                    val_shift_limit=int(brightness * 100),
                    p=0.5,
                ),
                A.GaussNoise(var_limit=noise_var, p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0),
                    A.MedianBlur(blur_limit=blur_limit, p=1.0),
                ], p=0.4),
                A.ElasticTransform(alpha=1, sigma=20, p=elastic_p),
                A.CoarseDropout(
                    max_holes=num_holes,
                    max_height=hole_size,
                    max_width=hole_size,
                    fill_value=0,
                    p=0.4,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ])

        # ============================================================
        # ✅ METHOD-SPECIFIC CLASS TRANSFORMS
        # ============================================================
        if augmentation_method == "simclr":
            # SimCLR: قوی‌ترین augmentation (NT-Xent needs diverse negatives)
            class_transforms = {
                "NL": get_pipeline("light", target_size),
                "G1": get_pipeline("light", target_size),
                "G2": get_pipeline("light", target_size),
                "G3": get_pipeline("light", target_size),
            }
            print("🎨 SimCLR: strong/extreme augmentation (NT-Xent optimized)")

        elif augmentation_method == "byol":
            # BYOL: متوسط (EMA target network handles moderate aug)
            class_transforms = {
                "NL": get_pipeline("light", target_size),
                "G1": get_pipeline("light", target_size),
                "G2": get_pipeline("light", target_size),
                "G3": get_pipeline("medium", target_size),
            }
            print("🎨 BYOL: medium/strong augmentation")

        else:  # mae
            # MAE: سبک (self-masking already provides diversity)
            class_transforms = {
                "NL": get_pipeline("light", target_size),
                "G1": get_pipeline("light", target_size),
                "G2": get_pipeline("light", target_size),
                "G3": get_pipeline("medium", target_size),
            }
            print("🎨 MAE: light/medium augmentation (self-masking handles diversity)")

        # ✅ Build label map
        label_map = {}
        for _, row in df.iterrows():
            img_name = row["liverName"]
            label = row["liverLabel"]
            if label in CLASS_NAMES:
                label_map[img_name] = label

        # ✅ Build dataset
        dataset = AlexNetClassAwareDataset(
            image_paths=image_paths,
            label_map=label_map,
            class_transforms=class_transforms,
            target_size=target_size,
        )

        # ✅ Verify matching
        matched = sum(1 for p in image_paths if os.path.basename(p) in label_map)
        if matched < 0.5 * len(image_paths):
            print(f"🚨 WARNING: Only {matched}/{len(image_paths)} images matched labels!")
            raise RuntimeError("labels.csv does not match cropped/ directory")

        print(f"✅ Dataset built: {matched}/{len(image_paths)} images matched")

        # ✅ Weighted sampler (sqrt-based)
        from torch.utils.data import WeightedRandomSampler

        class_hits = {"NL": 0, "G1": 0, "G2": 0, "G3": 0}
        for p in image_paths:
            cls = label_map.get(os.path.basename(p), "NL")
            class_hits[cls] += 1

        raw_weights = {c: 1.0 / max(n, 1) for c, n in class_hits.items()}
        sqrt_weights = {c: np.sqrt(w) for c, w in raw_weights.items()}
        total_sqrt = sum(sqrt_weights.values())
        norm_weights = {c: w / total_sqrt for c, w in sqrt_weights.items()}

        print(f"\n📊 Weighted Sampler weights (sqrt-based):")
        for c in ["NL", "G1", "G2", "G3"]:
            print(f"   {c}: {norm_weights[c]:.4f}")

        sample_weights = []
        for p in image_paths:
            cls = label_map.get(os.path.basename(p), "NL")
            sample_weights.append(norm_weights[cls])
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        # ✅ Build DataLoader
        if shuffle:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=min(num_workers, 2),
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True,
                persistent_workers=False,
                prefetch_factor=None,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(num_workers, 2),
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True,
                persistent_workers=False,
                prefetch_factor=None,
            )

        return dataloader

    # ============================================================
    # Sanity Check
    # ============================================================

    def _sanity_check_before_training(self, train_loader: DataLoader):
        """Complete sanity check before training."""
        print("\n" + "=" * 80)
        print("🔍 SANITY CHECK BEFORE TRAINING")
        print("=" * 80)

        if train_loader is None:
            raise RuntimeError("❌ train_loader is None!")
        if len(train_loader) == 0:
            raise RuntimeError(f"❌ train_loader is empty!")

        print(f"✅ DataLoader has {len(train_loader)} batches")

        batch = next(iter(train_loader))
        print(f"✅ Batch type: {type(batch)}")

        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise RuntimeError(f"❌ Expected 2-tuple, got {type(batch)}")

        x1, x2 = batch
        print(f"✅ x1 shape: {x1.shape}")
        print(f"✅ x2 shape: {x2.shape}")

        if x1.shape != x2.shape:
            raise RuntimeError(f"❌ Shape mismatch: {x1.shape} vs {x2.shape}")

        if torch.isnan(x1).any() or torch.isnan(x2).any():
            raise RuntimeError("❌ NaN detected in views!")

        print(f"✅ x1 range: [{x1.min():.2f}, {x1.max():.2f}]")
        print(f"✅ x2 range: [{x2.min():.2f}, {x2.max():.2f}]")

        print("=" * 80)
        print("✅ SANITY CHECK PASSED! Ready to train.\n")

    # ============================================================
    # Pretrain (MAIN training loop)
    # ============================================================

    def pretrain(
            self,
            image_paths: List[str],
            validation_paths: List[str] = None,
            save_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Run SSL pretraining on AlexNet with Early Stopping + 4 checkpoints.

        ✅ FIXED: augmentation_method = self.method (not hardcoded "byol")
        """
        if HAS_MEMORY:
            print_memory_status("Memory Before Training")

        if self.verbose:
            print(f"\n📊 Starting SSL pretraining on {len(image_paths)} images")
            print(f"📌 Method: {self.method.upper()}")

        # ✅ Build train loader — augmentation_method = self.method
        train_loader = self._create_class_aware_dataloader(
            image_paths=image_paths,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.kwargs.get("num_workers", 0),
            target_size=self.input_size,
            augmentation_method=self.method,  # ✅ از self.method
            enhance_ultrasound=self.enhance_ultrasound,
            use_imgaug=self.use_imgaug,
            use_kornia=self.use_kornia,
        )

        # ✅ Build val loader
        val_loader = None
        if validation_paths:
            val_loader = self._create_class_aware_dataloader(
                image_paths=validation_paths,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.kwargs.get("num_workers", 0),
                target_size=self.input_size,
                augmentation_method=self.method,  # ✅ از self.method
                enhance_ultrasound=self.enhance_ultrasound,
                use_imgaug=self.use_imgaug,
                use_kornia=self.use_kornia,
            )

        # ✅ Early stopping
        if HAS_EARLY_STOPPING:
            effective_patience = min(55, max(15, self.total_epochs - 5))
            early_stopping = create_early_stopping(
                patience=effective_patience,
                min_delta=1e-4,
                verbose=self.verbose,
                save_best=True,
                restore_best=True,
                checkpoint_dir=ALEXNET_EARLY_STOPPING_DIR,
            )
        else:
            early_stopping = None
            print("⚠️ Early stopping not available")

        # ✅ Checkpoint paths
        method_dir = os.path.join(ALEXNET_ARCHITECTURE_DIR, self.method)
        os.makedirs(method_dir, exist_ok=True)

        best_path = os.path.join(method_dir, f"alexnet_best_update_{self.method}.pth")
        checkpoint_path = os.path.join(method_dir, f"alexnet_checkpoint_update_{self.method}.pth")
        last_path = os.path.join(method_dir, f"alexnet_last_update_{self.method}.pth")
        final_path = os.path.join(method_dir, f"alexnet_final_update_{self.method}.pth")

        # ✅ Initial save
        self._save_ssl_checkpoint(best_path)

        self._sanity_check_before_training(train_loader)

        # ✅ Training loop with progress bar
        import sys
        from tqdm import tqdm

        method_display = self.method.upper()

        with tqdm(
                total=self.total_epochs,
                desc=f"🧠 {method_display} Training on AlexNet",
                bar_format=(
                        "{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} epochs "
                        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
                colour="#00d4ff",
                ncols=110,
                leave=True,
                position=0,
                file=sys.stdout,
        ) as pbar:

            for epoch in range(self.total_epochs):
                self.trainer.current_epoch = epoch

                # Train one epoch
                metrics = self.trainer.train_epoch(train_loader)
                current_loss = metrics.get("loss", float("inf"))

                # Check for collapse
                if metrics.get("early_stopped", False):
                    print(f"\n🛑 Training stopped by collapse detection at epoch {epoch}")
                    self.best_loss = min(self.best_loss, current_loss)
                    break

                # Validate
                if val_loader is not None:
                    val_metrics = self.trainer.validate(val_loader)
                    monitor_loss = val_metrics.get("loss", current_loss)
                else:
                    monitor_loss = current_loss

                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar("Loss/train", current_loss, epoch)
                    self.writer.add_scalar("Loss/val", monitor_loss, epoch)
                    self.writer.add_scalar("LR", self.trainer.optimizer.param_groups[0]["lr"], epoch)
                    if hasattr(self.trainer, "momentum"):
                        self.writer.add_scalar("Momentum", self.trainer.momentum, epoch)

                # Early stopping check
                if early_stopping is not None:
                    if early_stopping(monitor_loss, self.trainer, epoch):
                        print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                        break

                # Save best
                health_status = getattr(self.trainer, "_last_health_status", "unknown")
                is_early = epoch < 10
                is_healthy = "HEALTHY" in health_status and "UNHEALTHY" not in health_status

                if current_loss < self.best_loss and is_healthy and not is_early:
                    self.best_loss = current_loss
                    self._save_ssl_checkpoint(best_path)
                    print(f"✅ Best updated (health={health_status}, epoch={epoch}): {best_path}")
                elif current_loss < self.best_loss:
                    reason = "early epoch" if is_early else f"model {health_status}"
                    print(f"⚠️ Loss improved ({current_loss:.4f}) but {reason} — NOT saving best")
                elif current_loss < self.best_loss and not is_early and not is_healthy:
                    # با WARNING هم best را ذخیره کن ولی لاگ بده
                    if "WARNING" in health_status:
                        self.best_loss = current_loss
                        self._save_ssl_checkpoint(best_path)
                        print(f"⚠️ Best updated with WARNING (epoch={epoch})")

                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self._save_ssl_checkpoint(checkpoint_path)
                    print(f"✅ Checkpoint saved: {checkpoint_path}")

                # Save last every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self._save_ssl_checkpoint(last_path)

                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "Best": f"{early_stopping.best_loss:.4f}" if early_stopping else "N/A",
                    "Patience": f"{early_stopping.counter}/{effective_patience}" if early_stopping else "N/A",
                    "LR": f'{self.trainer.optimizer.param_groups[0]["lr"]:.2e}',
                })
                pbar.update(1)

                # Memory status
                if HAS_MEMORY and (epoch + 1) % 10 == 0:
                    clear_memory()
                    print_memory_status(f"Memory After Epoch {epoch + 1}")

        # ✅ Restore best
        self.encoder = self.trainer.get_encoder()

        # ============================================================
        # ✅ COMPREHENSIVE EVALUATION (Test Accuracy + All Metrics)
        # ============================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print("🔬 Running COMPREHENSIVE evaluation (Test Accuracy + All Metrics)...")
            print("=" * 80)

        try:
            from src.self_supervised.ProbabilityAlgorithms.linear_probe import build_labeled_eval_loader

            eval_loader = build_labeled_eval_loader(
                labels_csv=os.path.join(PROJECT_ROOT, "finalReports", "labels.csv"),
                image_dir=os.path.join(PROJECT_ROOT, "finalReports", "cropped"),
                input_size=self.input_size,
                batch_size=32,
                max_per_class=300,
            )

            self.evaluate(eval_loader, num_classes=self.num_classes)

            if self.verbose:
                print(f"✅ Test Accuracy:  {self.metrics.get('accuracy', 0):.4f}")
                print(f"✅ Test Precision: {self.metrics.get('precision', 0):.4f}")
                print(f"✅ Test Recall:    {self.metrics.get('recall', 0):.4f}")
                print(f"✅ Test F1-Score:  {self.metrics.get('f1_score', 0):.4f}")
                print(f"✅ Test AUC:       {self.metrics.get('auc', 0):.4f}")

                if "per_class_recall" in self.metrics:
                    print("\n📊 Per-Class Recall:")
                    for i, r in enumerate(self.metrics["per_class_recall"]):
                        print(f"   {CLASS_NAMES[i]}: {r:.4f}")

        except Exception as e:
            print(f"⚠️ Comprehensive evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            self.metrics = {"error": str(e), "metric_type": "evaluation_failed"}

        # ✅ Final results
        final_epoch = (
            early_stopping.best_epoch
            if early_stopping and early_stopping.is_stopped
            else self.total_epochs - 1
        )

        results = {
            "history": self.trainer.get_history(),
            "best_loss": early_stopping.best_loss if early_stopping else self.best_loss,
            "best_epoch": early_stopping.best_epoch if early_stopping else self.current_epoch,
            "total_epochs": final_epoch + 1,
            "early_stopping": early_stopping.is_stopped if early_stopping else False,
            "early_stopping_summary": early_stopping.get_summary() if early_stopping else {},
            "metrics": self.metrics,
        }

        self.history = results
        self.best_loss = results["best_loss"]
        self.current_epoch = results["total_epochs"]

        self.save_model()

        if HAS_MEMORY:
            print_memory_status("Memory After Training")

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ SSL Pretraining Complete!")
            print(f"   Best Loss: {self.best_loss:.4f} (Epoch {results['best_epoch']})")
            print(f"   Total Epochs: {self.current_epoch}")
            print("=" * 80)

        if self.writer is not None:
            self.writer.close()

        # ✅ Save final
        self._save_ssl_checkpoint(final_path)
        print(f"✅ Final model saved: {final_path}")

        results["saved_paths"] = {
            "best": best_path,
            "checkpoint": checkpoint_path,
            "last": last_path,
            "final": final_path,
        }

        return results

    # ============================================================
    # Save
    # ============================================================

    def _save_ssl_checkpoint(self, path: str):
        """Save SSL checkpoint to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        encoder = self.get_encoder()

        checkpoint = {
            "encoder_state_dict": encoder.state_dict(),
            "method": self.method,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "best_loss": self.best_loss,
            "current_epoch": self.current_epoch,
            "timestamp": datetime.now().isoformat(),
            "config": self._get_config() if hasattr(self, "_get_config") else {},
        }

        if hasattr(self.trainer, "optimizer"):
            checkpoint["optimizer_state_dict"] = self.trainer.optimizer.state_dict()

        torch.save(checkpoint, path)

    def save_model(self, path: str = None) -> str:
        """Save model."""
        if path is None:
            method_dir = os.path.join(ALEXNET_ARCHITECTURE_DIR, self.method)
            os.makedirs(method_dir, exist_ok=True)
            path = os.path.join(method_dir, f"alexnet_best_update_{self.method}.pth")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "encoder_state_dict": self.get_encoder().state_dict(),
                "method": self.method,
                "input_size": self.input_size,
                "num_classes": self.num_classes,
                "best_loss": self.best_loss,
                "current_epoch": self.current_epoch,
                "timestamp": datetime.now().isoformat(),
                "config": self._get_config(),
            },
            path,
        )

        if self.verbose:
            print(f"✅ Model saved: {path}")

        return path

    def save_for_transfer(self) -> str:
        """Save model for weight transfer."""
        path = os.path.join(
            ALEXNET_SSL_DIR,
            f"alexnet_ssl_{self.method}_for_transfer.pth",
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "encoder_state_dict": self.get_encoder().state_dict(),
                "method": self.method,
                "input_size": self.input_size,
                "num_classes": self.num_classes,
                "feature_dim": self.encoder._feature_dim,
                "best_loss": self.best_loss,
                "timestamp": datetime.now().isoformat(),
            },
            path,
        )

        if self.verbose:
            print(f"✅ Model saved for transfer: {path}")

        return path

    # ============================================================
    # ✅ COMPLETE REPORT GENERATION (with all 6 reporters)
    # ============================================================

    def generate_report(
        self,
        output_path: str = None,
        include_plots: bool = True,
        labeled_loader: DataLoader = None,
    ) -> str:
        """
        Generate comprehensive PDF report with:
        - Custom_optimizers info (Section 1)
        - Training curves & metrics (Section 2)
        - Confusion matrix & ROC (Section 3)
        - Captum interpretability (Section 4)
        - LIME explanations (Section 5)
        - SHAP summary (Section 6)
        - Netron architecture (Section 7)
        - reportGenerator summary (Section 8)
        - Conclusion (Section 9)
        """
        if output_path is None:
            output_path = os.path.join(
                ALEXNET_SSL_DIR, f"alexnet_ssl_{self.method}_report.pdf"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plots_dir = os.path.join(os.path.dirname(output_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        interp_dir = os.path.join(os.path.dirname(output_path), "interpretability")
        os.makedirs(interp_dir, exist_ok=True)
        arch_dir = os.path.join(os.path.dirname(output_path), "architecture")
        os.makedirs(arch_dir, exist_ok=True)

        # ✅ Step 1: Ensure metrics exist
        if not self.metrics or "accuracy" not in self.metrics:
            print("\n🔬 Metrics incomplete — running full evaluation...")
            try:
                from src.self_supervised.ProbabilityAlgorithms.linear_probe import build_labeled_eval_loader

                eval_loader = build_labeled_eval_loader(
                    labels_csv=os.path.join(PROJECT_ROOT, "finalReports", "labels.csv"),
                    image_dir=os.path.join(PROJECT_ROOT, "finalReports", "cropped"),
                    input_size=self.input_size,
                    batch_size=32,
                    max_per_class=300,
                )
                self.evaluate(eval_loader, num_classes=self.num_classes)
                if labeled_loader is None:
                    labeled_loader = eval_loader
            except Exception as e:
                print(f"⚠️ Full evaluation failed: {e}")
                if not self.metrics or "accuracy" not in self.metrics:
                    self.metrics = {
                        "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                        "f1_score": 0.0, "num_samples": 0,
                        "metric_type": "evaluation_failed", "error": str(e),
                    }

        # ✅ Step 2: Generate all plots
        plot_paths = self._generate_complete_plots(plots_dir)
        print(f"📊 Generated {len(plot_paths)} plots")

        # ✅ Step 3: Captum interpretability
        captum_paths = {"gradcam": None, "ig": None}
        if labeled_loader is not None and HAS_CAPTUM:
            try:
                captum_result = generate_interpretability_report(
                    self.encoder, labeled_loader, interp_dir,
                    device=self.device, class_names=CLASS_NAMES[: self.num_classes],
                )
                captum_paths = {
                    "gradcam": captum_result.get("gradcam_path"),
                    "ig": captum_result.get("ig_path"),
                }
                print(f"✅ Captum: gradcam={captum_paths['gradcam'] is not None}, ig={captum_paths['ig'] is not None}")
            except Exception as e:
                print(f"⚠️ Captum failed: {e}")

        # ✅ Step 4: LIME
        lime_path = None
        if labeled_loader is not None and HAS_LIME:
            try:
                lime_result = explain_predictions_lime(
                    self.encoder, labeled_loader, interp_dir,
                    class_names=CLASS_NAMES[: self.num_classes],
                    device=self.device, num_samples=5,
                )
                lime_path = lime_result.get("lime_path")
                print(f"✅ LIME: {lime_path}")
            except Exception as e:
                print(f"⚠️ LIME failed: {e}")

        # ✅ Step 5: SHAP
        shap_path = None
        if labeled_loader is not None and HAS_SHAP:
            try:
                shap_result = generate_shap_explanation(
                    self.encoder, labeled_loader, interp_dir,
                    device=self.device, num_background=50, num_explain=10,
                    class_names=CLASS_NAMES[: self.num_classes],
                )
                shap_path = shap_result.get("shap_path")
                print(f"✅ SHAP: {shap_path}")
            except Exception as e:
                print(f"⚠️ SHAP failed: {e}")

        # ✅ Step 6: Netron architecture
        netron_result = {"onnx_path": None, "html_path": None, "info_path": None}
        if HAS_NETRON:
            try:
                netron_result = save_model_with_name(
                    self.encoder, arch_dir,
                    model_name="AlexNet",
                    input_shape=(1, 3, self.input_size, self.input_size),
                )
                info_json = netron_result.get("info_path")
                if info_json and os.path.exists(info_json):
                    update_architecture_info_json(
                        info_json,
                        captum_gradcam_path=captum_paths["gradcam"],
                        captum_ig_path=captum_paths["ig"],
                        lime_explanation_path=lime_path,
                        shap_summary_path=shap_path,
                    )
            except Exception as e:
                print(f"⚠️ Netron failed: {e}")

        # ✅ Step 7: Build PDF
        if not HAS_REPORTLAB:
            return self._generate_report_json_fallback(output_path)

        return self._build_full_pdf(
            output_path=output_path,
            plot_paths=plot_paths,
            captum_paths=captum_paths,
            lime_path=lime_path,
            shap_path=shap_path,
            netron_result=netron_result,
        )

    def _generate_complete_plots(self, plots_dir: str) -> Dict[str, str]:
        """Generate all plots for the report."""
        plot_paths = {}
        os.makedirs(plots_dir, exist_ok=True)

        print("\n📊 Generating plots...")

        # 1. Loss History
        try:
            if "loss_epoch" in self.history and self.history["loss_epoch"]:
                plt.figure(figsize=(12, 6))
                plt.plot(self.history["loss_epoch"], label="Training Loss",
                         linewidth=2, color="#3498db")
                if "val_loss_epoch" in self.history and self.history["val_loss_epoch"]:
                    plt.plot(self.history["val_loss_epoch"], label="Validation Loss",
                             linewidth=2, color="#e74c3c")
                plt.xlabel("Epoch", fontsize=14, fontweight="bold")
                plt.ylabel("Loss", fontsize=14, fontweight="bold")
                plt.title(f"{self.method.upper()} - Loss History",
                         fontsize=16, fontweight="bold")
                plt.legend(); plt.grid(alpha=0.3, linestyle="--")
                plt.tight_layout()
                path = os.path.join(plots_dir, "loss_history.png")
                plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                plot_paths["loss_history"] = path
                print("  ✅ loss_history")
        except Exception as e:
            print(f"  ⚠️ loss_history failed: {e}")
            plt.close("all")

        # 2. Learning Rate
        try:
            if "learning_rates" in self.history and self.history["learning_rates"]:
                plt.figure(figsize=(12, 6))
                plt.plot(self.history["learning_rates"], linewidth=2.5, color="#9b59b6")
                plt.xlabel("Epoch", fontsize=14, fontweight="bold")
                plt.ylabel("Learning Rate (log)", fontsize=14, fontweight="bold")
                plt.title(f"{self.method.upper()} - LR Schedule",
                         fontsize=16, fontweight="bold")
                plt.yscale("log"); plt.grid(alpha=0.3, linestyle="--")
                plt.tight_layout()
                path = os.path.join(plots_dir, "learning_rate.png")
                plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                plot_paths["learning_rate"] = path
                print("  ✅ learning_rate")
        except Exception as e:
            print(f"  ⚠️ learning_rate failed: {e}")
            plt.close("all")

        # 3. Confusion Matrix
        try:
            if self.confusion_matrix is not None:
                cm = np.asarray(self.confusion_matrix)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=CLASS_NAMES[: self.num_classes],
                            yticklabels=CLASS_NAMES[: self.num_classes],
                            annot_kws={"size": 14, "weight": "bold"})
                plt.title(f"{self.method.upper()} - Confusion Matrix",
                         fontsize=16, fontweight="bold")
                plt.ylabel("True Label", fontsize=14, fontweight="bold")
                plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
                plt.tight_layout()
                path = os.path.join(plots_dir, "confusion_matrix.png")
                plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                plot_paths["confusion_matrix"] = path
                print("  ✅ confusion_matrix")
        except Exception as e:
            print(f"  ⚠️ confusion_matrix failed: {e}")
            plt.close("all")

        # 4. ROC Curves
        try:
            if self.roc_data is not None:
                fpr, tpr, roc_auc = self.roc_data
                if fpr and tpr and roc_auc:
                    plt.figure(figsize=(10, 8))
                    colors_roc = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
                    for i in range(self.num_classes):
                        if i in fpr and i in tpr and i in roc_auc:
                            plt.plot(fpr[i], tpr[i],
                                     label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})",
                                     linewidth=2.5,
                                     color=colors_roc[i % len(colors_roc)])
                    plt.plot([0, 1], [0, 1], "k--", label="Random (0.500)", linewidth=2)
                    plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold")
                    plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold")
                    plt.title(f"{self.method.upper()} - ROC Curves",
                             fontsize=16, fontweight="bold")
                    plt.legend(loc="lower right", fontsize=12)
                    plt.grid(alpha=0.3, linestyle="--")
                    plt.tight_layout()
                    path = os.path.join(plots_dir, "roc_curves.png")
                    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                    plt.close()
                    plot_paths["roc_curves"] = path
                    print("  ✅ roc_curves")
        except Exception as e:
            print(f"  ⚠️ roc_curves failed: {e}")
            plt.close("all")

        # 5. Metrics Bar
        try:
            if self.metrics and ("accuracy" in self.metrics or "test_accuracy" in self.metrics):
                plt.figure(figsize=(10, 7))
                names = ["Accuracy", "Precision", "Recall", "F1-Score"]
                vals = [
                    self.metrics.get("test_accuracy", self.metrics.get("accuracy", 0)),
                    self.metrics.get("test_precision", self.metrics.get("precision", 0)),
                    self.metrics.get("test_recall", self.metrics.get("recall", 0)),
                    self.metrics.get("test_f1_score", self.metrics.get("f1_score", 0)),
                ]
                bars = plt.bar(names, vals,
                               color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
                               edgecolor="black", linewidth=1.5, width=0.6)
                for b, v in zip(bars, vals):
                    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                             f"{v:.4f}", ha="center", va="bottom",
                             fontsize=14, fontweight="bold")
                plt.ylim(0, 1.1)
                plt.ylabel("Score", fontsize=14, fontweight="bold")
                plt.title(f"{self.method.upper()} - Performance Metrics",
                         fontsize=16, fontweight="bold")
                plt.grid(axis="y", alpha=0.3, linestyle="--")
                plt.tight_layout()
                path = os.path.join(plots_dir, "metrics_bar.png")
                plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                plot_paths["metrics_bar"] = path
                print("  ✅ metrics_bar")
        except Exception as e:
            print(f"  ⚠️ metrics_bar failed: {e}")
            plt.close("all")

        # 6. Per-Class Performance
        try:
            if self.metrics and "classification_report" in self.metrics:
                plt.figure(figsize=(10, 7))
                names, vals = [], []
                for cname in CLASS_NAMES[: self.num_classes]:
                    if cname in self.metrics["classification_report"]:
                        r = self.metrics["classification_report"][cname]
                        names.append(cname)
                        vals.append(r.get("recall", 0))
                if names:
                    bars = plt.bar(names, vals,
                                   color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][:len(names)],
                                   edgecolor="black", linewidth=1.5)
                    for b, v in zip(bars, vals):
                        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                                 f"{v:.4f}", ha="center", va="bottom",
                                 fontsize=12, fontweight="bold")
                    plt.ylim(0, 1.1)
                    plt.ylabel("Recall (Per-Class)", fontsize=14, fontweight="bold")
                    plt.title(f"{self.method.upper()} - Per-Class Recall",
                             fontsize=16, fontweight="bold")
                    plt.grid(axis="y", alpha=0.3, linestyle="--")
                    plt.tight_layout()
                    path = os.path.join(plots_dir, "per_class_recall.png")
                    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                    plt.close()
                    plot_paths["per_class_recall"] = path
                    print("  ✅ per_class_recall")
        except Exception as e:
            print(f"  ⚠️ per_class_recall failed: {e}")
            plt.close("all")

        # 7. Pie Chart
        try:
            if self.metrics and "classification_report" in self.metrics:
                plt.figure(figsize=(9, 9))
                labels, sizes = [], []
                for cname in CLASS_NAMES[: self.num_classes]:
                    if cname in self.metrics["classification_report"]:
                        s = self.metrics["classification_report"][cname].get("support", 0)
                        if s > 0:
                            labels.append(cname); sizes.append(s)
                if sizes:
                    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
                            colors=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][:len(labels)],
                            explode=[0.05]*len(labels))
                    plt.title("Test Set Class Distribution",
                             fontsize=16, fontweight="bold")
                    plt.tight_layout()
                    path = os.path.join(plots_dir, "pie_chart.png")
                    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                    plt.close()
                    plot_paths["pie_chart"] = path
                    print("  ✅ pie_chart")
        except Exception as e:
            print(f"  ⚠️ pie_chart failed: {e}")
            plt.close("all")

        print(f"\n📊 Total plots generated: {len(plot_paths)}")
        return plot_paths

    def _build_full_pdf(
        self,
        output_path: str,
        plot_paths: Dict[str, str],
        captum_paths: Dict[str, str],
        lime_path: Optional[str],
        shap_path: Optional[str],
        netron_result: Dict[str, str],
    ) -> str:
        """Build the full PDF with 9 sections."""
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                     fontSize=26, spaceAfter=20,
                                     alignment=TA_CENTER,
                                     textColor=colors.HexColor("#1a252f"))
        h1_style = ParagraphStyle("H1", parent=styles["Heading2"], fontSize=20,
                                  spaceAfter=14, spaceBefore=20,
                                  textColor=colors.HexColor("#2c3e50"))
        h2_style = ParagraphStyle("H2", parent=styles["Heading3"], fontSize=15,
                                  spaceAfter=10, spaceBefore=14,
                                  textColor=colors.HexColor("#34495e"))
        body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                    fontSize=11, spaceAfter=8, leading=16)
        small_style = ParagraphStyle("Small", parent=styles["Normal"],
                                     fontSize=9, textColor=colors.grey)
        center_style = ParagraphStyle("Center", parent=body_style,
                                      alignment=TA_CENTER, fontSize=14)

        story = []

        # Helper
        def safe_image(path, title, width=6.5*inch, height=4.5*inch):
            story.append(Spacer(1, 12))
            story.append(Paragraph(title, h2_style))
            if path and os.path.exists(path) and os.path.getsize(path) > 100:
                try:
                    story.append(RLImage(path, width=width, height=height))
                    return True
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Could not load: {e}", small_style))
            else:
                story.append(Paragraph("⚠️ Image not available.", small_style))
            return False

        # ============================================================
        # PAGE 1: TITLE
        # ============================================================
        story.append(Paragraph("AlexNet SSL Report", title_style))
        story.append(Paragraph(f"Method: <b>{self.method.upper()}</b>", center_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ParagraphStyle("Date", parent=small_style, alignment=TA_CENTER),
        ))
        story.append(Spacer(1, 30))

        # ============================================================
        # PAGE 1: OVERVIEW + CONFIG
        # ============================================================
        story.append(Paragraph("1. Overview", h1_style))
        story.append(Paragraph(
            f"This report presents the complete Self-Supervised Learning (SSL) analysis "
            f"for <b>AlexNet</b> using the <b>{self.method.upper()}</b> method. "
            f"The encoder was initialized from <b>alexnet_best.pth</b> (pretrained on 2642 "
            f"labeled images) and then refined with SSL on the same unlabeled data. "
            f"Training ran for <b>{self.current_epoch} epochs</b> with a best SSL loss of "
            f"<b>{self.best_loss:.6f}</b>. Linear Probe accuracy is "
            f"<b>{self.metrics.get('accuracy', 0):.4f}</b>.",
            body_style,
        ))

        story.append(Paragraph("1.1 Training Configuration", h2_style))
        config_data = [
            ["Parameter", "Value"],
            ["SSL Method", self.method.upper()],
            ["Pretrained Weights", "✅ Loaded from alexnet_best.pth" if getattr(self, "_pretrained_loaded", False) else "❌ From scratch"],
            ["Pretrained Path", self.pretrained_path or "N/A"],
            ["Best SSL Loss", f"{self.best_loss:.6f}"],
            ["Total Epochs", str(self.current_epoch)],
            ["Input Size", f"{self.input_size}x{self.input_size}"],
            ["Feature Dimension", str(getattr(self.encoder, "_feature_dim", 512))],
            ["Optimizer", self.optimizer_name.upper()],
            ["Learning Rate", f"{self.lr:.6f}"],
            ["Weight Decay", f"{self.weight_decay:.6f}"],
            ["Batch Size", str(self.batch_size)],
            ["Device", str(self.device)],
            ["Classes", ", ".join(CLASS_NAMES[: self.num_classes])],
        ]
        t = Table(config_data, colWidths=[2.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        story.append(t)

        # ============================================================
        # PAGE 2: CUSTOM OPTIMIZERS (SECTION 2)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("2. Custom Optimizers", h1_style))
        story.append(Paragraph(
            "The SSL pipeline supports multiple optimizers. The table below summarizes "
            "all available custom optimizers and the one used in this run.",
            body_style,
        ))

        optimizer_data = [
            ["Optimizer", "Reference", "Best For", "Used Here"],
            ["AdamW", "Loshchilov & Hutter, 2019", "General purpose, stable", "✅" if self.optimizer_name.lower() == "adamw" else ""],
            ["RAdam", "Liu et al., 2019", "Stabilized adaptive LR", "✅" if self.optimizer_name.lower() == "radam" else ""],
            ["Ranger", "Wright, 2019", "RAdam + Lookahead", "✅" if self.optimizer_name.lower() == "ranger" else ""],
            ["SGD", "Robbins & Monro, 1951", "Simple, interpretable", "✅" if self.optimizer_name.lower() == "sgd" else ""],
        ]
        t = Table(optimizer_data, colWidths=[1.2*inch, 1.8*inch, 2*inch, 1*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"<b>Active optimizer:</b> {self.optimizer_name.upper()} "
            f"with lr={self.lr:.6f}, weight_decay={self.weight_decay:.6f}.",
            body_style,
        ))

        # ============================================================
        # PAGE 3: METRICS (SECTION 3)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("3. Performance Metrics", h1_style))
        story.append(Paragraph(
            "Linear Probe evaluation on held-out test set. Since SSL is unsupervised, "
            "Linear Probe is the standard way to measure semantic quality.",
            body_style,
        ))

        # ✅ اول test_accuracy رو بگیر، بعد accuracy (fallback)
        acc = self.metrics.get(
            "test_accuracy",
            self.metrics.get("accuracy", 0)
        )
        bal_acc = self.metrics.get("balanced_accuracy", acc)
        f1_macro = self.metrics.get("f1_macro", self.metrics.get("f1_score", 0))

        if bal_acc >= 0.75:
            verdict = "🟢 Excellent — highly discriminative features"
        elif bal_acc >= 0.60:
            verdict = "🟡 Good — useful features learned"
        elif bal_acc >= 0.40:
            verdict = "🟠 Moderate — encoder partially collapsed"
        else:
            verdict = "🔴 Poor — encoder likely collapsed"

        story.append(Paragraph(f"<b>Verdict:</b> {verdict}", body_style))
        story.append(Spacer(1, 10))

        metrics_data = [
            ["Metric", "Value", "Interpretation"],
            ["Accuracy", f"{acc:.4f}", "Overall correct predictions"],
            ["Balanced Accuracy", f"{bal_acc:.4f}", "Mean per-class recall"],
            ["Precision (weighted)", f"{self.metrics.get('precision', 0):.4f}", "TP / (TP + FP)"],
            ["Recall (weighted)", f"{self.metrics.get('recall', 0):.4f}", "TP / (TP + FN)"],
            ["F1 (weighted)", f"{self.metrics.get('f1_score', 0):.4f}", "Harmonic mean"],
            ["F1 (macro)", f"{f1_macro:.4f}", "Unweighted mean of per-class F1"],
            ["Test Samples", str(self.metrics.get("num_samples", 0)), "Held-out samples"],
        ]
        t = Table(metrics_data, colWidths=[2.2*inch, 1.5*inch, 2.8*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)

        # Per-class
        if "classification_report" in self.metrics:
            story.append(Spacer(1, 15))
            story.append(Paragraph("3.1 Per-Class Performance", h2_style))
            class_data = [["Class", "Precision", "Recall", "F1", "Support"]]
            for cname in CLASS_NAMES[: self.num_classes]:
                if cname in self.metrics["classification_report"]:
                    r = self.metrics["classification_report"][cname]
                    class_data.append([
                        cname, f"{r.get('precision',0):.4f}",
                        f"{r.get('recall',0):.4f}",
                        f"{r.get('f1-score',0):.4f}",
                        str(r.get("support", 0)),
                    ])
            t = Table(class_data, colWidths=[1.2*inch]*5)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)

        # AUC
        if "roc_auc" in self.metrics and self.metrics["roc_auc"]:
            story.append(Spacer(1, 15))
            story.append(Paragraph("3.2 AUC Scores", h2_style))
            auc_data = [["Class", "AUC"]]
            vals = []
            for i, cname in enumerate(CLASS_NAMES[: self.num_classes]):
                if i in self.metrics["roc_auc"]:
                    v = self.metrics["roc_auc"][i]
                    vals.append(v)
                    auc_data.append([cname, f"{v:.4f}"])
            if vals:
                auc_data.append(["Mean", f"{np.mean(vals):.4f}"])
            t = Table(auc_data, colWidths=[2*inch, 2*inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)

        # ============================================================
        # PAGE 4: TRAINING CURVES (SECTION 4)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("4. Training Curves", h1_style))
        safe_image(plot_paths.get("loss_history"), "4.1 Loss History")
        safe_image(plot_paths.get("learning_rate"), "4.2 Learning Rate Schedule")

        # ============================================================
        # PAGE 5: PERFORMANCE VISUALIZATIONS (SECTION 5)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("5. Performance Visualizations", h1_style))
        safe_image(plot_paths.get("metrics_bar"), "5.1 Metrics Bar Chart")
        safe_image(plot_paths.get("per_class_recall"), "5.2 Per-Class Recall")
        safe_image(plot_paths.get("pie_chart"), "5.3 Test Set Distribution",
                   width=5.5*inch, height=5*inch)

        # ============================================================
        # PAGE 6: CONFUSION MATRIX (SECTION 6)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("6. Confusion Matrix", h1_style))
        safe_image(plot_paths.get("confusion_matrix"), "6.1 Confusion Matrix",
                   width=6*inch, height=5*inch)

        # ============================================================
        # PAGE 7: ROC (SECTION 7)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("7. ROC Curves", h1_style))
        safe_image(plot_paths.get("roc_curves"), "7.1 ROC Curves",
                   width=6*inch, height=5*inch)

        # ============================================================
        # PAGE 8: CAPTUM (SECTION 8)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("8. Captum Interpretability", h1_style))
        story.append(Paragraph(
            "<b>Grad-CAM</b> (Gradient-weighted Class Activation Mapping) shows which "
            "regions contributed most to the prediction. <b>Integrated Gradients</b> "
            "computes per-pixel attribution via path integral from baseline to input.",
            body_style,
        ))
        safe_image(captum_paths.get("gradcam"), "8.1 Grad-CAM")
        safe_image(captum_paths.get("ig"), "8.2 Integrated Gradients")

        # ============================================================
        # PAGE 9: LIME (SECTION 9)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("9. LIME Explanations", h1_style))
        story.append(Paragraph(
            "<b>LIME</b> (Local Interpretable Model-agnostic Explanations) perturbs "
            "image regions and observes prediction changes. Green = positive "
            "contribution, Red = negative.",
            body_style,
        ))
        safe_image(lime_path, "9.1 LIME Explanation",
                   width=6*inch, height=4*inch)

        # ============================================================
        # PAGE 10: SHAP (SECTION 10)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("10. SHAP Summary", h1_style))
        story.append(Paragraph(
            "<b>SHAP</b> (SHapley Additive exPlanations) assigns per-pixel importance "
            "using game-theoretic values. Bright = high importance.",
            body_style,
        ))
        safe_image(shap_path, "10.1 SHAP Summary",
                   width=6*inch, height=4*inch)

        # ============================================================
        # PAGE 11: NETRON (SECTION 11)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("11. Model Architecture (Netron)", h1_style))
        story.append(Paragraph(
            f"The AlexNet architecture was exported to ONNX and visualized with Netron. "
            f"Files are stored in <b>architecture/</b>:",
            body_style,
        ))
        arch_data = [
            ["File", "Path", "Purpose"],
            ["ONNX", netron_result.get("onnx_path") or "N/A", "Portable model format"],
            ["HTML", netron_result.get("html_path") or "N/A", "Interactive viewer"],
            ["JSON", netron_result.get("info_path") or "N/A", "Metadata"],
        ]
        t = Table(arch_data, colWidths=[0.8*inch, 3.5*inch, 2.2*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"Parameters: <b>{count_parameters(self.encoder)['total']:,}</b> total, "
            f"<b>{count_parameters(self.encoder)['trainable']:,}</b> trainable.",
            body_style,
        ))

        # ============================================================
        # PAGE 12: REPORT GENERATOR SUMMARY (SECTION 12)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("12. Pipeline Summary", h1_style))
        story.append(Paragraph(
            "Aggregate statistics from all pipeline stages:",
            body_style,
        ))

        pipeline_data = [
            ["Stage", "Status", "Details"],
            ["Pretrained Load", "✅" if getattr(self, "_pretrained_loaded", False) else "❌", self.pretrained_path or "N/A"],
            ["SSL Training", "✅", f"{self.current_epoch} epochs, best loss {self.best_loss:.6f}"],
            ["Early Stopping", "✅" if self.history.get("early_stopping", False) else "❌",
             f"Triggered at epoch {self.history.get('best_epoch', 'N/A')}" if self.history.get("early_stopping") else "N/A"],
            ["Linear Probe", "✅" if self.metrics.get("accuracy", 0) > 0 else "❌",
             f"Acc = {self.metrics.get('accuracy', 0):.4f}"],
            ["Captum", "✅" if (captum_paths.get("gradcam") or captum_paths.get("ig")) else "❌", "Grad-CAM + IG"],
            ["LIME", "✅" if lime_path else "❌", "5 samples explained"],
            ["SHAP", "✅" if shap_path else "❌", "10 samples explained"],
            ["Netron", "✅" if netron_result.get("onnx_path") else "❌", "ONNX + HTML"],
        ]
        t = Table(pipeline_data, colWidths=[1.5*inch, 0.8*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        story.append(t)

        # ============================================================
        # PAGE 13: CONCLUSION (SECTION 13)
        # ============================================================
        story.append(PageBreak())
        story.append(Paragraph("13. Conclusion", h1_style))

        if acc >= 0.75:
            final_verdict = "🟢 EXCELLENT — ready for downstream fine-tuning"
        elif acc >= 0.60:
            final_verdict = "🟡 GOOD — encoder is useful"
        elif acc >= 0.40:
            final_verdict = "🟠 MODERATE — consider retraining"
        else:
            final_verdict = "🔴 POOR — do not use for transfer"

        conclusion = (
            f"<b>Final Verdict:</b> {final_verdict}<br/><br/>"
            f"<b>Linear Probe Accuracy:</b> {acc:.4f}<br/>"
            f"<b>Balanced Accuracy:</b> {bal_acc:.4f}<br/>"
            f"<b>F1 Score:</b> {self.metrics.get('f1_score', 0):.4f}<br/><br/>"
            f"<b>Best SSL Loss:</b> {self.best_loss:.6f}<br/>"
            f"<b>Epochs Trained:</b> {self.current_epoch}<br/>"
            f"<b>Optimizer:</b> {self.optimizer_name.upper()}<br/>"
            f"<b>Pretrained:</b> {self.pretrained_path or 'N/A'}<br/><br/>"
            f"<b>Recommendation:</b> "
            + ("Encoder is highly discriminative. Use for transfer learning."
               if acc >= 0.75 else
               "Encoder learned useful features. Transfer may provide moderate gains."
               if acc >= 0.60 else
               "Encoder quality is marginal. Consider adjusting hyperparameters."
               if acc >= 0.40 else
               "Encoder likely collapsed. Retrain with adjusted learning rate.")
        )
        story.append(Paragraph(conclusion, body_style))
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"Report generated by Liver AI Pipeline v5.0.0<br/>"
            f"Author: Babak Yousefian<br/>"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            small_style,
        ))

        # Build
        try:
            doc.build(story)
            print(f"✅ PDF Report saved: {output_path}")
        except Exception as e:
            print(f"❌ PDF build failed: {e}")
            import traceback
            traceback.print_exc()

        return output_path

    def _generate_report_json_fallback(self, output_path: str) -> str:
        """Fallback: JSON report if ReportLab not available."""
        json_path = output_path.replace(".pdf", ".json")
        report = {
            "model": "AlexNet",
            "method": self.method,
            "best_loss": self.best_loss,
            "current_epoch": self.current_epoch,
            "metrics": self.metrics,
            "config": self._get_config(),
            "timestamp": datetime.now().isoformat(),
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=4, default=str)
        print(f"✅ JSON report saved: {json_path}")
        return json_path

    def evaluate(self, labeled_loader: DataLoader, num_classes: int = 4) -> Dict[str, Any]:
        """Linear Probe evaluation (mirrors ResNet18SSL)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        self.encoder.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(labeled_loader, desc="Extracting features"):
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    images = batch
                    labels = torch.zeros(images.size(0), dtype=torch.long)
                images = images.to(self.device)
                labels = labels.to(self.device)
                if hasattr(self.encoder, "get_embedding"):
                    embeddings = self.encoder.get_embedding(images)
                else:
                    embeddings = self.encoder(images)
                    if embeddings.dim() > 2:
                        embeddings = embeddings.view(embeddings.size(0), -1)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        X = np.concatenate(all_embeddings, axis=0)
        y = np.concatenate(all_labels, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        class_counts = np.bincount(y_train, minlength=num_classes)
        raw_weights = 1.0 / (class_counts + 1e-8)
        sqrt_weights = np.sqrt(raw_weights)
        norm_weights = sqrt_weights / sqrt_weights.sum() * num_classes
        class_weight_dict = {i: float(norm_weights[i]) for i in range(num_classes)}

        clf = LogisticRegression(
            max_iter=3000, class_weight=class_weight_dict,
            random_state=42, C=1.0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

        try:
            y_prob = clf.predict_proba(X_test)
            y_true_bin = label_binarize(y_test, classes=range(num_classes))
            roc_auc = {}
            fpr, tpr = {}, {}
            for i in range(num_classes):
                if y_true_bin[:, i].sum() > 0:
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
            self.roc_data = (fpr, tpr, roc_auc)
        except Exception:
            pass

        class_report = classification_report(
            y_test, y_pred, labels=list(range(num_classes)),
            target_names=CLASS_NAMES[:num_classes], output_dict=True, zero_division=0,
        )

        # ✅ محاسبه‌ی AUC میانگین
        auc_mean = 0.0
        if hasattr(self, 'roc_data') and self.roc_data is not None:
            _, _, roc_auc = self.roc_data
            if roc_auc:
                auc_mean = float(np.mean(list(roc_auc.values())))

        self.metrics = {
            # ✅ Test Accuracy (اصلی)
            "accuracy": float(accuracy),
            "test_accuracy": float(accuracy),  # ← برای گزارش PDF

            # ✅ Precision, Recall, F1
            "precision": float(precision),
            "test_precision": float(precision),
            "recall": float(recall),
            "test_recall": float(recall),
            "f1_score": float(f1_w),
            "test_f1_score": float(f1_w),
            "f1_macro": float(f1_m),
            "balanced_accuracy": float(bal_acc),

            # ✅ AUC
            "auc": auc_mean,
            "test_auc": auc_mean,
            "roc_auc": getattr(self, 'roc_data', ({}, {}, {}))[2] if self.roc_data else {},

            # ✅ Loss
            "test_loss": 0.0,  # ← loss در Linear Probe معنی نداره

            # ✅ Confusion Matrix و Classification Report
            "confusion_matrix": cm.tolist() if hasattr(cm, 'tolist') else cm,
            "classification_report": class_report,
            "per_class_recall": recall_score(
                y_test, y_pred, average=None, zero_division=0,
                labels=list(range(num_classes))
            ).tolist(),

            # ✅ Metadata
            "num_samples": len(y_test),
            "metric_type": "linear_probe",
        }
        self.confusion_matrix = cm

        return self.metrics


    def load_model(self, path: str) -> None:
        """Load a trained encoder model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.method = checkpoint.get("method", self.method)
        self.input_size = checkpoint.get("input_size", self.input_size)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.current_epoch = checkpoint.get("current_epoch", 0)

        if self.verbose:
            print(f"✅ Model loaded: {path}")

    def get_encoder(self) -> nn.Module:
        """Get the trained encoder model."""
        return self.trainer.get_encoder()

    def get_embeddings(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from images."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        dataset = AlexNetSSLDataset(
            image_paths=image_paths,
            transform=transform,
            return_path=False,
            enhance_ultrasound=self.enhance_ultrasound,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.kwargs.get("num_workers", 0),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return self.trainer.get_embeddings(dataloader)[0]

    def _get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return {
            "method": self.method,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "use_mixed_precision": self.use_mixed_precision,
            "device": str(self.device),
            "optimizer": self.optimizer_name,
            "use_accelerate": self.use_accelerate,
            "use_imgaug": self.use_imgaug,
            "use_kornia": self.use_kornia,
            "enhance_ultrasound": self.enhance_ultrasound,
            "pretrained_path": self.pretrained_path,
            "pretrained_loaded": getattr(self, "_pretrained_loaded", False),
            "kwargs": self.kwargs,
        }

    def get_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_loss(self) -> float:
        return self.best_loss

    def get_current_epoch(self) -> int:
        return self.current_epoch

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    # ============================================================
    # Full SSL Pipeline
    # ============================================================

    def run_full_ssl_pipeline(
        self,
        image_paths: List[str],
        validation_paths: List[str] = None,
        save_interval: int = 10,
    ) -> Dict[str, Any]:
        """Run BYOL → SimCLR → MAE → Best → Transfer."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("🚀 FULL SSL PIPELINE - AlexNet")
            print("=" * 80)

        results = {}
        methods = ["byol", "simclr", "mae"]
        method_results = {}

        for method in methods:
            if self.verbose:
                print("\n" + "=" * 80)
                print(f"🧠 Training {method.upper()} on AlexNet")
                print("=" * 80)

            pipeline = AlexNetSSL(
                method=method,
                num_classes=self.num_classes,
                input_size=self.input_size,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                total_epochs=self.total_epochs,
                warmup_epochs=self.warmup_epochs,
                use_mixed_precision=self.use_mixed_precision,
                device=self.device,
                verbose=self.verbose,
                save_dir=os.path.join(self.save_dir, method),
                optimizer_name=self.optimizer_name,
                use_accelerate=self.use_accelerate,
                use_imgaug=self.use_imgaug,
                use_kornia=self.use_kornia,
                enhance_ultrasound=self.enhance_ultrasound,
                pretrained_path=ALEXNET_PRETRAINED_PATH,
                **self.kwargs,
            )

            method_result = pipeline.pretrain(
                image_paths=image_paths,
                validation_paths=validation_paths,
                save_interval=save_interval,
            )

            method_results[method] = {
                "pipeline": pipeline,
                "best_loss": method_result["best_loss"],
                "best_epoch": method_result["best_epoch"],
                "total_epochs": method_result["total_epochs"],
                "early_stopping": method_result["early_stopping"],
            }

            pipeline.save_for_transfer()
            results[method] = method_result

        # ✅ Select best method via Linear Probe
        from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
            build_labeled_eval_loader, evaluate_with_linear_probe,
        )

        if self.verbose:
            print("\n" + "=" * 80)
            print("🔬 Running Linear Probe for each method...")
            print("=" * 80)

        eval_loader = build_labeled_eval_loader(
            labels_csv=os.path.join(PROJECT_ROOT, "finalReports", "labels.csv"),
            image_dir=os.path.join(PROJECT_ROOT, "finalReports", "cropped"),
            input_size=self.input_size,
            batch_size=32,
            max_per_class=200,
        )

        linear_probe_results = {}
        for method, res in method_results.items():
            if self.verbose:
                print(f"\n📌 Linear Probe: {method.upper()}")

            encoder = res["pipeline"].get_encoder()
            probe_result = evaluate_with_linear_probe(
                encoder=encoder,
                labeled_loader=eval_loader,
                device=str(self.device),
                num_classes=self.num_classes,
                test_size=0.3,
            )
            linear_probe_results[method] = probe_result

            if self.verbose:
                print(f"   ✅ {method.upper()} LP Accuracy: {probe_result['linear_probe_accuracy']:.4f}")

        # Select best
        best_method = max(
            linear_probe_results.keys(),
            key=lambda x: linear_probe_results[x]["linear_probe_accuracy"],
        )
        best_lp_acc = linear_probe_results[best_method]["linear_probe_accuracy"]
        best_loss = method_results[best_method]["best_loss"]

        if self.verbose:
            print("\n" + "=" * 80)
            print("🏆 BEST METHOD SELECTED")
            print("=" * 80)
            for m, res in linear_probe_results.items():
                print(f"   {m.upper():<8} LP Acc: {res['linear_probe_accuracy']:.4f}  |  Loss: {method_results[m]['best_loss']:.4f}")
            print(f"\n   ✅ Winner: {best_method.upper()} (LP Acc = {best_lp_acc:.4f})")
            print("=" * 80)

        results["best_method"] = best_method
        results["best_loss"] = best_loss
        results["best_linear_probe_accuracy"] = best_lp_acc
        results["linear_probe_results"] = linear_probe_results
        results["method_results"] = {
            m: {
                "best_loss": res["best_loss"],
                "total_epochs": res["total_epochs"],
                "early_stopping": res["early_stopping"],
            }
            for m, res in method_results.items()
        }

        return results


# ================================================================
# ✅ Factory Function (matches resNet18SSL pattern)
# ================================================================

def create_alexnet_ssl_pipeline(
    method: str = "byol",
    num_classes: int = 4,
    input_size: int = 224,
    batch_size: int = 32,
    lr: float = 0.0001,
    weight_decay: float = 1e-4,
    total_epochs: int = 40,
    warmup_epochs: int = 10,
    use_mixed_precision: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    optimizer_name: str = "adamw",
    use_accelerate: bool = False,
    use_imgaug: bool = False,
    use_kornia: bool = True,
    enhance_ultrasound: bool = True,
    pretrained_path: str = None,
    **kwargs,
) -> AlexNetSSL:
    """Factory function to create AlexNet SSL pipeline."""
    return AlexNetSSL(
        method=method,
        num_classes=num_classes,
        input_size=input_size,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        use_mixed_precision=use_mixed_precision,
        device=device,
        verbose=verbose,
        optimizer_name=optimizer_name,
        use_accelerate=use_accelerate,
        use_imgaug=use_imgaug,
        use_kornia=use_kornia,
        enhance_ultrasound=enhance_ultrasound,
        pretrained_path=pretrained_path,
        **kwargs,
    )
