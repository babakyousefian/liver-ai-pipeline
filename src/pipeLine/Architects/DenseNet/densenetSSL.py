# src/pipeLine/densenetSSL.py
# -*- coding: utf-8 -*-

"""
DenseNet SSL Pipeline - Self-Supervised Pretraining (COMPLETE ULTIMATE EDITION)
================================================================================
Author: Babak Yousefian

Mirrors vgg16SSL.py / mobilenetv2SSL.py / alexnetSSL.py exactly, adapted for DenseNet.

1. Loads pretrained DenseNet from:
   finalReports/models/DenseNet/checkpoints/DenseNet_best.pth

2. Wraps it with BYOLTrainer (creates projector + predictor)

3. Runs BYOL/SimCLR/MAE on unlabeled data (labels ignored during SSL)

4. Saves checkpoints to:
   finalReports/selfSupervisedLearning/DenseNet/densenet/SSL_checkpoints/
       architecture/byol/densenet_{best,checkpoint,final,last}_update_byol.pth
       early_stopping/model_byol_best_epoch_*.pth
       interpretability/
       optuna/
       tensorboard/

5. Supports full pipeline: BYOL → SimCLR → MAE → Best

⚠️ DenseNet WARNING:
   - DenseNet121: ~8M params → batch=8, img=224
   - DenseNet169: ~14M params → batch=4, img=160
   - DenseNet201: ~20M params → batch=2, img=128
   - RTX 2050 (4GB) → use small batches

Author: Babak Yousefian
Version: 1.0.0 (LIGHT)
"""

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from src.self_supervised import (
    create_byol_trainer,
    create_mae_trainer,
    create_simclr_trainer,
    set_seed,
)

# ================================================================
# ✅ Optional imports (graceful degradation)
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
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️ TensorBoard not installed.")

try:
    from src.self_supervised.OptimizationsAlgorithms.Augmentations.kornia_augmentations import KorniaAugmentation
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

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

# DenseNet121: feature dim = 1024 (last dense block)
# DenseNet169: 1664
# DenseNet201: 1920
# پیش‌فرض = DenseNet121
FEATURE_DIM = 1024


# ================================================================
# ✅ Paths (ABSOLUTE)
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")

# ✅ Pretrained checkpoint — DenseNet path
DENSENET_PRETRAINED_PATH = os.path.join(
    PROJECT_ROOT, "finalReports", "models", "", "checkpoints", "DenseNet_best.pth"
)

# ✅ SSL directories
DENSENET_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "", "")
DENSENET_SSL_DIR = os.path.join(DENSENET_ROOT, "SSL_checkpoints")
DENSENET_ARCHITECTURE_DIR = os.path.join(DENSENET_SSL_DIR, "architecture")
DENSENET_BYOL_DIR = os.path.join(DENSENET_ARCHITECTURE_DIR, "byol")
DENSENET_MAE_DIR = os.path.join(DENSENET_ARCHITECTURE_DIR, "mae")
DENSENET_SIMCLR_DIR = os.path.join(DENSENET_ARCHITECTURE_DIR, "simclr")
DENSENET_EARLY_STOPPING_DIR = os.path.join(DENSENET_SSL_DIR, "early_stopping")
DENSENET_INTERPRETABILITY_DIR = os.path.join(DENSENET_SSL_DIR, "interpretability")
DENSENET_OPTUNA_DIR = os.path.join(DENSENET_SSL_DIR, "optuna")
DENSENET_TENSORBOARD_DIR = os.path.join(DENSENET_SSL_DIR, "tensorboard")

FINAL_MODELS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "final_models")


# ================================================================
# ✅ Directory Management
# ================================================================

def ensure_directories() -> None:
    """Create all necessary directories."""
    directories = [
        DENSENET_SSL_DIR,
        DENSENET_ARCHITECTURE_DIR,
        DENSENET_BYOL_DIR,
        DENSENET_MAE_DIR,
        DENSENET_SIMCLR_DIR,
        DENSENET_EARLY_STOPPING_DIR,
        DENSENET_INTERPRETABILITY_DIR,
        DENSENET_OPTUNA_DIR,
        DENSENET_TENSORBOARD_DIR,
        FINAL_MODELS_DIR,
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
    print(f"✅ All DenseNet SSL directories created")


# ================================================================
# ✅ SSL Dataset for DenseNet
# ================================================================

class DenseNetSSLDataset(Dataset):
    """
    Dataset for SSL pretraining on DenseNet.
    Loads unlabeled images and returns TWO augmented views.
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

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

            if self.enhance_ultrasound and HAS_MEDICAL and MEDPY_AVAILABLE:
                try:
                    image = enhance_ultrasound(image)
                except:
                    pass

            image = cv2.resize(image, (224, 224))

        except Exception:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = Image.fromarray(image)

        if self.transform:
            view1, view2 = self.transform(image)
            if self.return_path:
                return view1, view2, img_path
            return view1, view2

        if self.return_path:
            return image, img_path
        return image, 0


# ================================================================
# ✅ Class-Aware Dataset
# ================================================================

class DenseNetClassAwareDataset(Dataset):
    """Class-aware augmentation for imbalanced liver ultrasound."""

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

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))

        transform = self.class_transforms[class_name]

        try:
            view1 = transform(image=img)["image"]
            view2 = transform(image=img)["image"]
        except Exception:
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

class DenseNetSSL:
    """
    DenseNet SSL Pipeline.
    Mirrors VGG16SSL / MobileNetV2SSL / AlexNetSSL exactly.
    """

    CLASS_NAMES = ["NL", "G1", "G2", "G3"]

    def __init__(
        self,
        method: str = "byol",
        num_classes: int = 4,
        input_size: int = 224,
        batch_size: int = 8,       # ✅ DenseNet121 LIGHT
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
        gradient_accumulation_steps: int = 4,  # ✅ DenseNet LIGHT
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
        self.save_dir = save_dir or DENSENET_SSL_DIR
        self.optimizer_name = optimizer_name
        self.use_accelerate = use_accelerate
        self.use_imgaug = use_imgaug
        self.use_kornia = use_kornia
        self.enhance_ultrasound = enhance_ultrasound
        self.kwargs = kwargs
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print(f"📌 Using device: {self.device}")
            print(f"📌 LIGHT: batch={batch_size}, grad_accum={gradient_accumulation_steps}, img={input_size}")

        set_seed(42)

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

        candidates = [pretrained_path, DENSENET_PRETRAINED_PATH]
        for p in candidates:
            if p is not None and os.path.exists(p):
                actual_pretrained_path = p
                break

        if actual_pretrained_path is not None:
            self.load_pretrained_weights(actual_pretrained_path)
        else:
            print("\n" + "=" * 80)
            print("🚨 CRITICAL ERROR: Pretrained DenseNet weights NOT FOUND!")
            print("=" * 80)
            print("   Looked in:")
            for p in candidates:
                if p is not None:
                    print(f"   - {p}")
            print("=" * 80)
            raise FileNotFoundError("Pretrained DenseNet weights not found")

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
            self.writer = SummaryWriter(log_dir=DENSENET_TENSORBOARD_DIR)
            if self.verbose:
                print(f"📊 TensorBoard logs: {DENSENET_TENSORBOARD_DIR}")

        if self.verbose:
            self._print_info()

    # ============================================================
    # Build / Load / Trainer
    # ============================================================

    def _build_encoder(self) -> nn.Module:
        """Build DenseNet with growth_rate=8, layers=(6,12,24,16) to match checkpoint."""
        from src.models.DenseNet.DenseNet import DenseNet
        import types

        model = DenseNet(
            num_classes=self.num_classes,
            input_channels=3,
            input_size=self.input_size,
            layers=[6, 12, 24, 16],  # ✅ DenseNet121 layout (not [4,8,12,8])
            growth_rate=24,  # ✅ growth_rate=8 (not 16)
            bn_size=4,
            compression=0.5,
            init_channels=32,
            dropout=0.2,
            verbose=False,
        )

        def get_embedding(self_model, x):
            # ✅ ترتیب صحیح: block → transition → block → transition ...
            h = self_model.stem(x)
            num_blocks = len(self_model.dense_blocks)
            for i, block in enumerate(self_model.dense_blocks):
                h = block(h)
                if i < num_blocks - 1 and i < len(self_model.transitions):
                    h = self_model.transitions[i](h)
            h = self_model.final_bn(h)
            h = F.relu(h, inplace=False)
            h = F.adaptive_avg_pool2d(h, (1, 1))
            return torch.flatten(h, 1)

        if not hasattr(model, "get_embedding"):
            model.get_embedding = types.MethodType(get_embedding, model)
        model._feature_dim = 766

        if self.verbose:
            print(f"✅ Built DenseNet (growth_rate=24, layers=[6,12,24,16])")
            print(f"   Feature dimension: {model._feature_dim}")
            print(f"   Total params: {len(model.state_dict())}")

        return model

    def load_pretrained_weights(self, model_path: str = None) -> bool:
        if model_path is None:
            model_path = DENSENET_PRETRAINED_PATH

        if not os.path.exists(model_path):
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

            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model_state = self.encoder.state_dict()
            new_state = {}
            loaded = 0
            shape_mismatch = 0
            missing = 0

            for name, param in model_state.items():
                if name in state_dict and state_dict[name].shape == param.shape:
                    new_state[name] = state_dict[name]
                    loaded += 1
                elif name in state_dict:
                    new_state[name] = param
                    shape_mismatch += 1
                else:
                    new_state[name] = param
                    missing += 1

            self.encoder.load_state_dict(new_state, strict=False)
            self._pretrained_loaded = True

            print(f"\n✅ Pretrained weights loaded:")
            print(f"   ✅ Loaded: {loaded} params")
            print(f"   ⚠️ Shape mismatch: {shape_mismatch}")
            print(f"   ⚠️ Missing: {missing}")
            print(f"   Total: {len(model_state)}")



            if loaded < len(model_state) * 0.95:
                print(f"\n🚨 Still {len(model_state) - loaded} params not loaded!")
                print("   Missing keys (first 10):")
                shown = 0
                for name in model_state.keys():
                    if name not in new_state and shown < 10:
                        print(f"      {name}")
                        shown += 1

            return True

        except Exception as e:
            print(f"⚠️ Failed: {e}")
            import traceback
            traceback.print_exc()
            self._pretrained_loaded = False
            return False

    def _create_trainer(self):
        """Create SSL trainer based on method."""
        common_kwargs = {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

        if self.method == "simclr":
            trainer = create_simclr_trainer(
                encoder=self.encoder,
                input_size=self.input_size,
                projection_dim=self.kwargs.get("projection_dim", 128),
                hidden_dim=self.kwargs.get("hidden_dim", 1024),
                projection_layers=self.kwargs.get("projection_layers", 3),
                temperature=self.kwargs.get("temperature", 0.5),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_lars=self.kwargs.get("use_lars", False),
                use_mixed_precision=self.use_mixed_precision,
                use_bf16=self.kwargs.get("use_bf16", True),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
                color_jitter_strength=self.kwargs.get("color_jitter_strength", 1.0),
                blur_kernel_size=self.kwargs.get("blur_kernel_size", 23),
                device=self.device,
                verbose=self.verbose,
                num_workers=self.kwargs.get("num_workers", 0),
                save_dir=self.save_dir,
                **common_kwargs,
            )


        elif self.method == "byol":
            trainer = create_byol_trainer(
                model=self.encoder,
                model_name="densenet",
                input_size=self.input_size,
                projection_dim=self.kwargs.get("projection_dim", 256),
                hidden_dim=self.kwargs.get("hidden_dim", 4096),
                momentum=self.kwargs.get("momentum", 0.996),
                momentum_schedule=self.kwargs.get("momentum_schedule", "cosine"),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_lars=self.kwargs.get("use_lars", False),
                use_mixed_precision=self.use_mixed_precision,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
                device=self.device,
                verbose=self.verbose,
                num_workers=self.kwargs.get("num_workers", 0),
                augmentation_method=self.kwargs.get("augmentation_method", "byol"),
                save_dir=self.save_dir,
                use_kornia=self.use_kornia,
                use_bf16=True,
                augmentation_intensity=0.8,
                use_medical_safe=True,
                optimizer_name=self.optimizer_name,
            )

        elif self.method == "mae":
            trainer = create_mae_trainer(
                encoder=self.encoder,
                img_size=self.input_size,
                patch_size=self.kwargs.get("patch_size", 1),
                mask_ratio=self.kwargs.get("mask_ratio", 0.5),
                embed_dim=self.kwargs.get("embed_dim", 256),
                depth=self.kwargs.get("depth", 6),
                num_heads=self.kwargs.get("num_heads", 4),
                mlp_ratio=self.kwargs.get("mlp_ratio", 4.0),
                dropout=self.kwargs.get("dropout", 0.1),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_mixed_precision=self.use_mixed_precision,
                use_bf16=self.kwargs.get("use_bf16", True),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.kwargs.get("warmup_epochs", 3),
                total_epochs=self.total_epochs,
                device=self.device,
                verbose=self.verbose,
                num_workers=self.kwargs.get("num_workers", 0),
                save_dir=self.save_dir,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported SSL method: {self.method}")

        return trainer

    def _print_info(self):
        """Print pipeline information."""
        pretrained_loaded = getattr(self, "_pretrained_loaded", False)

        print("\n" + "=" * 80)
        print("🚀 DenseNet SSL Pipeline")
        print("=" * 80)
        print(f"   Method: {self.method.upper()}")
        print(f"   Input Size: {self.input_size}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"   Effective Batch: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"   Total Epochs: {self.total_epochs}")
        print(f"   Learning Rate: {self.lr}")
        print(f"   Optimizer: {self.optimizer_name}")
        print(f"   Device: {self.device}")
        print(f"   Save Directory: {self.save_dir}")
        print(f"   Pretrained Weights: {'✅' if pretrained_loaded else '❌'}")
        print(f"   Pretrained Path: {self.pretrained_path}")
        print(f"   TensorBoard: {'✅' if HAS_TENSORBOARD else '❌'}")
        print(f"   Kornia: {'✅' if HAS_KORNIA else '❌'}")
        print(f"   Medical: {'✅' if HAS_MEDICAL else '❌'}")
        print("=" * 80)

    # ============================================================
    # DataLoader
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
        """Class-aware DataLoader with method-specific augmentation."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if augmentation_method is None:
            augmentation_method = self.method

        labels_csv = os.path.join(PROJECT_ROOT, "finalReports", "labels.csv")
        df = pd.read_csv(labels_csv)
        print(f"📂 Loaded labels: {labels_csv} ({len(df)} rows)")
        print(f"🎨 Augmentation method: {augmentation_method.upper()}")

        label_counts = df["liverLabel"].value_counts()
        print("\n📊 Class Distribution:")
        for cls in ["NL", "G1", "G2", "G3"]:
            count = label_counts.get(cls, 0)
            print(f"   {cls}: {count}")

        def get_pipeline(strength: str, target_size: int):
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
                A.RandomResizedCrop(size=(target_size, target_size), scale=scale, ratio=ratio, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=rotation, p=0.7, border_mode=0),
                A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=0.7),
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
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        if augmentation_method == "simclr":
            class_transforms = {
                "NL": get_pipeline("light", target_size),
                "G1": get_pipeline("light", target_size),
                "G2": get_pipeline("light", target_size),
                "G3": get_pipeline("light", target_size),
            }
            print("🎨 SimCLR: strong/extreme augmentation")
        elif augmentation_method == "byol":
            class_transforms = {
                "NL": get_pipeline("light", target_size),
                "G1": get_pipeline("light", target_size),
                "G2": get_pipeline("light", target_size),
                "G3": get_pipeline("medium", target_size),
            }
            print("🎨 BYOL: medium/strong augmentation")
        else:
            class_transforms = {
                "NL": get_pipeline("light", target_size),
                "G1": get_pipeline("light", target_size),
                "G2": get_pipeline("light", target_size),
                "G3": get_pipeline("medium", target_size),
            }
            print("🎨 MAE: light/medium augmentation")

        label_map = {}
        for _, row in df.iterrows():
            img_name = row["liverName"]
            label = row["liverLabel"]
            if label in CLASS_NAMES:
                label_map[img_name] = label

        dataset = DenseNetClassAwareDataset(
            image_paths=image_paths,
            label_map=label_map,
            class_transforms=class_transforms,
            target_size=target_size,
        )

        matched = sum(1 for p in image_paths if os.path.basename(p) in label_map)
        if matched < 0.5 * len(image_paths):
            print(f"🚨 WARNING: Only {matched}/{len(image_paths)} images matched labels!")
            raise RuntimeError("labels.csv does not match cropped/ directory")

        print(f"✅ Dataset built: {matched}/{len(image_paths)} images matched")

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

        sample_weights = [norm_weights[label_map.get(os.path.basename(p), "NL")] for p in image_paths]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

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
        """Run SSL pretraining on DenseNet."""
        if self.verbose:
            print(f"\n📊 Starting SSL pretraining on {len(image_paths)} images")
            print(f"📌 Method: {self.method.upper()}")

        train_loader = self._create_class_aware_dataloader(
            image_paths=image_paths,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.kwargs.get("num_workers", 0),
            target_size=self.input_size,
            augmentation_method=self.method,
            enhance_ultrasound=self.enhance_ultrasound,
            use_imgaug=self.use_imgaug,
            use_kornia=self.use_kornia,
        )

        val_loader = None
        if validation_paths:
            val_loader = self._create_class_aware_dataloader(
                image_paths=validation_paths,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.kwargs.get("num_workers", 0),
                target_size=self.input_size,
                augmentation_method=self.method,
                enhance_ultrasound=self.enhance_ultrasound,
                use_imgaug=self.use_imgaug,
                use_kornia=self.use_kornia,
            )

        if HAS_EARLY_STOPPING:
            effective_patience = min(55, max(15, self.total_epochs - 5))
            early_stopping = create_early_stopping(
                patience=effective_patience,
                min_delta=1e-4,
                verbose=self.verbose,
                save_best=True,
                restore_best=True,
                checkpoint_dir=DENSENET_EARLY_STOPPING_DIR,
            )
        else:
            early_stopping = None
            print("⚠️ Early stopping not available")

        method_dir = os.path.join(DENSENET_ARCHITECTURE_DIR, self.method)
        os.makedirs(method_dir, exist_ok=True)

        best_path = os.path.join(method_dir, f"densenet_best_update_{self.method}.pth")
        checkpoint_path = os.path.join(method_dir, f"densenet_checkpoint_update_{self.method}.pth")
        last_path = os.path.join(method_dir, f"densenet_last_update_{self.method}.pth")
        final_path = os.path.join(method_dir, f"densenet_final_update_{self.method}.pth")

        self._save_ssl_checkpoint(best_path)
        self._sanity_check_before_training(train_loader)

        import sys as _sys
        from tqdm import tqdm

        method_display = self.method.upper()

        with tqdm(
                total=self.total_epochs,
                desc=f"🧠 {method_display} Training on DenseNet",
                bar_format=(
                        "{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} epochs "
                        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
                colour="#00d4ff",
                ncols=110,
                leave=True,
                position=0,
                file=_sys.stdout,
        ) as pbar:

            for epoch in range(self.total_epochs):
                self.trainer.current_epoch = epoch

                metrics = self.trainer.train_epoch(train_loader)
                current_loss = metrics.get("loss", float("inf"))

                if metrics.get("early_stopped", False):
                    print(f"\n🛑 Training stopped by collapse detection at epoch {epoch}")
                    self.best_loss = min(self.best_loss, current_loss)
                    break

                if val_loader is not None:
                    val_metrics = self.trainer.validate(val_loader)
                    monitor_loss = val_metrics.get("loss", current_loss)
                else:
                    monitor_loss = current_loss

                # ✅ TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar("Loss/train", current_loss, epoch)
                    self.writer.add_scalar("Loss/val", monitor_loss, epoch)
                    self.writer.add_scalar("LR", self.trainer.optimizer.param_groups[0]["lr"], epoch)
                    if hasattr(self.trainer, "momentum"):
                        self.writer.add_scalar("Momentum", self.trainer.momentum, epoch)

                # ✅ Early stopping
                if early_stopping is not None:
                    if early_stopping(monitor_loss, self.trainer, epoch):
                        print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                        break

                # ✅ Save best
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

                if (epoch + 1) % 5 == 0:
                    self._save_ssl_checkpoint(checkpoint_path)
                    print(f"✅ Checkpoint saved: {checkpoint_path}")

                if (epoch + 1) % 5 == 0:
                    self._save_ssl_checkpoint(last_path)

                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "Best": f"{early_stopping.best_loss:.4f}" if early_stopping else "N/A",
                    "Patience": f"{early_stopping.counter}/{effective_patience}" if early_stopping else "N/A",
                    "LR": f'{self.trainer.optimizer.param_groups[0]["lr"]:.2e}',
                })
                pbar.update(1)

        # ✅ Restore best
        self.encoder = self.trainer.get_encoder()

        # ✅ Evaluation
        if self.verbose:
            print("\n" + "=" * 80)
            print("🔬 Running COMPREHENSIVE evaluation...")
            print("=" * 80)

        try:
            from src.self_supervised.ProbabilityAlgorithms.linear_probe import build_labeled_eval_loader

            eval_loader = build_labeled_eval_loader(
                labels_csv=os.path.join(PROJECT_ROOT, "finalReports", "labels.csv"),
                image_dir=os.path.join(PROJECT_ROOT, "finalReports", "cropped"),
                input_size=self.input_size,
                batch_size=self.batch_size,
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

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ SSL Pretraining Complete!")
            print(f"   Best Loss: {self.best_loss:.4f} (Epoch {results['best_epoch']})")
            print(f"   Total Epochs: {self.current_epoch}")
            print("=" * 80)

        if self.writer is not None:
            self.writer.close()

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
    # Save / Load
    # ============================================================

    def _save_ssl_checkpoint(self, path: str):
        """Save SSL checkpoint."""
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
            method_dir = os.path.join(DENSENET_ARCHITECTURE_DIR, self.method)
            os.makedirs(method_dir, exist_ok=True)
            path = os.path.join(method_dir, f"densenet_best_update_{self.method}.pth")

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
        path = os.path.join(DENSENET_SSL_DIR, f"densenet_ssl_{self.method}_for_transfer.pth")
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
        """Get the trained encoder."""
        return self.trainer.get_encoder()

    # ============================================================
    # Evaluation
    # ============================================================

    def evaluate(self, labeled_loader: DataLoader, num_classes: int = 4) -> Dict[str, Any]:
        """Linear Probe evaluation."""
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

        auc_mean = 0.0
        if hasattr(self, 'roc_data') and self.roc_data is not None:
            _, _, roc_auc = self.roc_data
            if roc_auc:
                auc_mean = float(np.mean(list(roc_auc.values())))

        self.metrics = {
            "accuracy": float(accuracy),
            "test_accuracy": float(accuracy),
            "precision": float(precision),
            "test_precision": float(precision),
            "recall": float(recall),
            "test_recall": float(recall),
            "f1_score": float(f1_w),
            "test_f1_score": float(f1_w),
            "f1_macro": float(f1_m),
            "balanced_accuracy": float(bal_acc),
            "auc": auc_mean,
            "test_auc": auc_mean,
            "roc_auc": getattr(self, 'roc_data', ({}, {}, {}))[2] if self.roc_data else {},
            "test_loss": 0.0,
            "confusion_matrix": cm.tolist() if hasattr(cm, 'tolist') else cm,
            "classification_report": class_report,
            "per_class_recall": recall_score(
                y_test, y_pred, average=None, zero_division=0,
                labels=list(range(num_classes))
            ).tolist(),
            "num_samples": len(y_test),
            "metric_type": "linear_probe",
        }
        self.confusion_matrix = cm

        return self.metrics

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_loss(self) -> float:
        return self.best_loss

    def get_current_epoch(self) -> int:
        return self.current_epoch

    # ============================================================
    # Report Generation
    # ============================================================

    def generate_report(self, output_path: str = None, labeled_loader: DataLoader = None) -> str:
        """Generate 24-page PDF report using densenetReporter."""
        try:
            from src.pipeLine.Architects.DenseNet.densenetReporter import generate_densenet_report
            from src.pipeLine.Architects.DenseNet.densenetFineTune import get_ssl_methods_comparison

            ssl_comparison = get_ssl_methods_comparison()

            if output_path is None:
                output_path = os.path.join(DENSENET_SSL_DIR, f"densenet_ssl_{self.method}_report.pdf")

            data_split = {
                "train": self.metrics.get("num_train", 0),
                "val": self.metrics.get("num_val", 0),
                "test": self.metrics.get("num_samples", 0),
                "total": 0,
            }

            image_paths, labels = [], []
            if labeled_loader is not None:
                try:
                    ds = labeled_loader.dataset
                    if hasattr(ds, "paths"):
                        image_paths = ds.paths
                    elif hasattr(ds, "image_paths"):
                        image_paths = ds.image_paths
                    if hasattr(ds, "labels"):
                        labels = ds.labels
                except:
                    pass

            pdf_path = generate_densenet_report(
                final_model=self.encoder,
                metrics=self.metrics,
                history=self.history.get("history", {}) if isinstance(self.history, dict) else {},
                cv_summary={},
                ci95={},
                image_paths=image_paths,
                labels=labels,
                ssl_checkpoint_path=self.pretrained_path or "N/A",
                ssl_method=self.method,
                config={
                    "input_size": self.input_size,
                    "epochs_final": self.total_epochs,
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                },
                data_split=data_split,
                tta_info={"n_aug": 1},
                ssl_comparison=ssl_comparison,
                device=str(self.device),
                generate_interpretability_flag=True,
                verbose=self.verbose,
            )
            return pdf_path
        except Exception as e:
            print(f"⚠️ Could not use densenetReporter: {e}")
            import traceback
            traceback.print_exc()

            json_path = output_path.replace(".pdf", ".json") if output_path else "densenet_ssl_report.json"
            with open(json_path, "w") as f:
                json.dump({
                    "model": "DenseNet",
                    "method": self.method,
                    "metrics": self.metrics,
                    "history": self.history,
                }, f, indent=4, default=str)
            return json_path

    def _get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return {
            "method": self.method,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "use_mixed_precision": self.use_mixed_precision,
            "device": str(self.device),
            "optimizer": self.optimizer_name,
            "use_kornia": self.use_kornia,
            "enhance_ultrasound": self.enhance_ultrasound,
            "pretrained_path": self.pretrained_path,
            "pretrained_loaded": getattr(self, "_pretrained_loaded", False),
            "kwargs": self.kwargs,
        }


# ================================================================
# ✅ Factory Function
# ================================================================

def create_densenet_ssl_pipeline(
    method: str = "byol",
    num_classes: int = 4,
    input_size: int = 224,
    batch_size: int = 8,        # ✅ DenseNet121 LIGHT
    lr: float = 0.0001,
    weight_decay: float = 1e-4,
    total_epochs: int = 40,
    warmup_epochs: int = 10,
    use_mixed_precision: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    optimizer_name: str = "adamw",
    use_kornia: bool = True,
    enhance_ultrasound: bool = True,
    pretrained_path: str = None,
    gradient_accumulation_steps: int = 4,  # ✅ DenseNet LIGHT
    **kwargs,
) -> DenseNetSSL:
    """Factory function."""
    return DenseNetSSL(
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
        use_kornia=use_kornia,
        enhance_ultrasound=enhance_ultrasound,
        pretrained_path=pretrained_path,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **kwargs,
    )