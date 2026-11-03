# src/pipeLine/efficientNetB0SSL.py

"""
EfficientNet-B0 SSL Pipeline — Complete Ultimate 2026 Edition

Features (2026 ULTIMATE):
- ✅ CUDA-only enforcement
- ✅ Windows-compatible paths
- ✅ Class-Aware + BYOL/SimCLR + Kornia augmentation pipeline
- ✅ Linear Probe-based best method selection (NOT loss)
- ✅ Complete metrics (accuracy, F1, confusion, ROC, AUC)
- ✅ 8-page PDF report with all visualizations
- ✅ Auto-tuning based on hardware (VRAM, cores)
- ✅ Load pretrained EfficientNet-B0 from Without Fine-Tuning
- ✅ Full SSL pipeline: BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune → B1
- ✅ Comprehensive interpretability (Captum, LIME, SHAP)
- ✅ Netron architecture visualization

Author: Babak Yousefian
Version: 6.0.0 (2026 ULTIMATE EDITION)
"""

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)  # ✅ FIX: جلوگیری از oversubscription در workers

# ================================================================
# ✅ PATH SETUP
# ================================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================================================
# ✅ CUDA CHECK
# ================================================================

def _require_cuda(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n" + "=" * 70 +
            "\n❌ CUDA IS REQUIRED FOR EfficientNet-B0 SSL 2026 EDITION"
            "\n" + "=" * 70
        )
    if device != "cuda":
        print(f"⚠️  Forcing device='cuda' (was '{device}')")
    dev = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✅ CUDA Available: {gpu_name} ({vram_gb:.2f} GB)")
    return dev


# ================================================================
# ✅ PROJECT ROOT (Windows)
# ================================================================

PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")

# ================================================================
# ✅ EfficientNet-B0 SSL Directories
# ================================================================

EFFICIENTNET_B0_SSL_DIR = os.path.join(SELF_SUPERVISED_ROOT, "EfficientNet", "", "SSL_checkpoints")
TENSORBOARD_DIR = os.path.join(EFFICIENTNET_B0_SSL_DIR, "tensorboard")
EFFICIENTNET_WEIGHT_TRANSFORMER_DIR = os.path.join(SELF_SUPERVISED_ROOT, "EfficientNet", "weightTransformer")
FINAL_MODELS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "final_models")

COMPARISONS_REPORTER_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "comparisonsReporter")
COMPARISONS_BETWEEN_MODELS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "comparisonsReporterBetweenModels")
FINAL_REPORTERS_AND_CONCLUSIONS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "finalReportersAndConclusions")
INDIVIDUALS_REPORTER_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "individualsReporter")

# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)

# ================================================================
# ✅ Imports
# ================================================================

from src.models.efficientNet.B0 import create_efficientnet_b0
from src.self_supervised import (
    create_byol_trainer, create_mae_trainer, create_simclr_trainer,
    set_seed,
)

# ================================================================
# ✅ Optional imports
# ================================================================

try:
    from src.reporter.captum_reporter import generate_interpretability_report
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

try:
    from src.reporter.lime_reporter import explain_predictions_lime
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    from src.reporter.shap_reporter import generate_shap_explanation
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from src.reporter.netron_reporter import save_model_with_name, update_architecture_info_json
    HAS_NETRON = True
except ImportError:
    HAS_NETRON = False

try:
    from src.optimizers.custom_optimizers import get_optimizer
    HAS_CUSTOM_OPT = True
except ImportError:
    HAS_CUSTOM_OPT = False

try:
    from src.medical.image_io import (
        MEDPY_AVAILABLE, PYDICOM_AVAILABLE, dicom_to_numpy,
        enhance_ultrasound, read_dicom,
    )
    HAS_MEDICAL = True
except ImportError:
    HAS_MEDICAL = False

try:
    from src.utils.memory_monitor import clear_memory, print_memory_status
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

try:
    from src.self_supervised.OptimizationsAlgorithms.Augmentations.imgaug_pipeline import IMGAUG_AVAILABLE, get_imgaug_pipeline
    HAS_IMGAUG = IMGAUG_AVAILABLE
except ImportError:
    HAS_IMGAUG = False

try:
    from src.self_supervised.OptimizationsAlgorithms.Augmentations.kornia_augmentations import (
        KorniaAugmentation, DualViewKorniaAugmentation,
    )
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

try:
    from src.self_supervised.OptimizationsAlgorithms.Optimizers.accelerate_trainer import AccelerateTrainer
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from src.self_supervised.OptimizationsAlgorithms.Optimizers.optuna_optimizer import optimize_hyperparameters
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ================================================================
# ✅ Ensure directories
# ================================================================

def ensure_directories() -> None:
    dirs = [
        EFFICIENTNET_B0_SSL_DIR, TENSORBOARD_DIR, EFFICIENTNET_WEIGHT_TRANSFORMER_DIR,
        FINAL_MODELS_DIR, COMPARISONS_REPORTER_DIR, COMPARISONS_BETWEEN_MODELS_DIR,
        FINAL_REPORTERS_AND_CONCLUSIONS_DIR, INDIVIDUALS_REPORTER_DIR,
        os.path.join(EFFICIENTNET_B0_SSL_DIR, "optuna"),
        os.path.join(EFFICIENTNET_B0_SSL_DIR, "interpretability"),
        os.path.join(EFFICIENTNET_B0_SSL_DIR, "architecture"),
        # ✅ NEW: per-method architecture directories
        os.path.join(EFFICIENTNET_B0_SSL_DIR, "architecture", "byol"),
        os.path.join(EFFICIENTNET_B0_SSL_DIR, "architecture", "simclr"),
        os.path.join(EFFICIENTNET_B0_SSL_DIR, "architecture", "mae"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_method_checkpoint_dir(method: str, model_prefix: str = "b0_ssl") -> str:
    """
    ✅ NEW: Get the architecture/{method}/ directory for checkpoints.

    Args:
        method: 'byol', 'simclr', 'mae'
        model_prefix: 'b0_ssl' for EfficientNet-B0, 'resnet18_ssl' for ResNet18

    Returns:
        Absolute path to architecture/{method}/ directory
    """
    method_dir = os.path.join(EFFICIENTNET_B0_SSL_DIR, "architecture", method.lower())
    os.makedirs(method_dir, exist_ok=True)
    return method_dir


def get_checkpoint_paths(method: str, model_prefix: str = "b0_ssl") -> Dict[str, str]:
    """
    ✅ NEW: Get all 4 checkpoint paths for a given method.

    Returns:
        dict with keys: 'best', 'last', 'checkpoint', 'final'
    """
    method_dir = get_method_checkpoint_dir(method, model_prefix)
    method_lower = method.lower()

    return {
        "best": os.path.join(method_dir, f"{model_prefix}_{method_lower}_best.pth"),
        "last": os.path.join(method_dir, f"{model_prefix}_{method_lower}_last.pth"),
        "checkpoint": os.path.join(method_dir, f"{model_prefix}_{method_lower}_checkpoint.pth"),
        "final": os.path.join(method_dir, f"{model_prefix}_{method_lower}_final.pth"),
    }


# ================================================================
# ✅ SSL Dataset
# ================================================================

class EfficientNetB0SSLDataset(Dataset):
    """
    Dataset for SSL pretraining on EfficientNet-B0.

    ✅ SPEED FIX: Augmentation منتقل شد به BYOLTrainer (روی GPU).
    این dataset فقط I/O ساده انجام میده:
        image_path → cv2.imread → numpy → tensor
    """

    def __init__(self, image_paths, transform=None, return_path=False,
                 enhance_ultrasound=True):
        self.image_paths = image_paths
        self.transform = transform  # ✅ None — augmentation در trainer
        self.return_path = return_path
        self.enhance_ultrasound = enhance_ultrasound
        self.cache = {}
        self.cache_size = len(image_paths)  # ✅ cache کامل

        # اگه transform داده شد، dual-view رو چک کن
        self._is_dual_view = False
        if self.transform is not None:
            self._is_dual_view = self._check_transform()
            if not self._is_dual_view:
                print(f"⚠️  Warning: Transform does NOT return two views!")

    def _check_transform(self):
        try:
            dummy = torch.randn(3, 224, 224)
            result = self.transform(dummy)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                return True
        except Exception:
            pass
        return False

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        path = self.image_paths[idx]
        try:
            if HAS_MEDICAL and PYDICOM_AVAILABLE and path.lower().endswith(".dcm"):
                ds = read_dicom(path)
                if ds is not None:
                    img = dicom_to_numpy(ds)
                    if img is not None:
                        #if self.enhance_ultrasound and HAS_MEDICAL and MEDPY_AVAILABLE:
                            #img = enhance_ultrasound(img)
                        img = cv2.resize(img, (224, 224))
                        img = np.stack([img] * 3, axis=2) if img.ndim == 2 else img
                    else:
                        img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.imread(path)
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #if self.enhance_ultrasound and HAS_MEDICAL and MEDPY_AVAILABLE:
                        #try:
                            #img = enhance_ultrasound(img)
                        #except Exception:
                            #pass
                    img = cv2.resize(img, (224, 224))
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if len(self.cache) < self.cache_size:
            self.cache[idx] = img
        return img

    def __getitem__(self, idx):
        """
        ✅ SPEED FIX: خروجی مستقیم tensor (3, 224, 224) در [0, 255].
        Augmentation (Kornia) در trainer روی batch اعمال میشه.
        """
        img = self._load_image(idx)  # numpy (224, 224, 3) uint8

        # ✅ مستقیم tensor — بدون PIL
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, 224, 224)

        if self.transform is not None:
            # حالتی که transform داره (fallback)
            if self._is_dual_view:
                v1, v2 = self.transform(img_t.unsqueeze(0))
                v1 = v1.squeeze(0)
                v2 = v2.squeeze(0)
            else:
                v1 = self.transform(img_t)
                v2 = self.transform(img_t)
            if self.return_path:
                return v1, v2, self.image_paths[idx]
            return v1, v2

        # ✅ حالت اصلی: tensor خام برمی‌گردون
        if self.return_path:
            return img_t, self.image_paths[idx]
        return img_t, 0


# ================================================================
# ✅ DataLoader factory
# ================================================================

def create_ssl_dataloader(
        image_paths, batch_size=64, shuffle=True, num_workers=4,
        target_size=224, augmentation_method="byol",
        enhance_ultrasound=True, use_imgaug=False, use_kornia=True,
        drop_last=True,
):
    """
    ✅ SPEED FIX: DataLoader سبک — augmentation در BYOLTrainer (GPU).
    """
    # ✅ دیتاست بدون transform — augmentation در trainer
    dataset = EfficientNetB0SSLDataset(
        image_paths=image_paths,
        transform=None,
        return_path=False,
        enhance_ultrasound=False,  # ✅ SPEED FIX: غیرفعال — Kornia خودش انجام میده
    )

    # ✅ Windows-safe workers
    import platform
    is_windows = platform.system() == "Windows"
    # ✅ Windows: 4 workers optimal (نه 6)
    safe_workers = min(num_workers, 4) if is_windows else num_workers

    # ✅ persistent_workers فقط غیر Windows
    persistent = (safe_workers > 0) and not is_windows

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=safe_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent,
        prefetch_factor=6 if safe_workers > 0 else None,  # ← 6 به جای 4
    )


# ================================================================
# ✅ Main pipeline class
# ================================================================

class EfficientNetB0SSL:
    """Complete EfficientNet-B0 SSL pipeline."""

    def __init__(
        self,
        method="byol",
        num_classes=4,
        input_size=224,
        batch_size=64,
        lr=1e-4,
        weight_decay=1e-4,
        total_epochs=100,
        warmup_epochs=10,
        use_mixed_precision=True,
        device="cuda",
        verbose=True,
        save_dir=None,
        optimizer_name="adamw",
        use_accelerate=False,
        use_imgaug=False,
        use_kornia=False,
        enhance_ultrasound=True,
        pretrained_path=None,
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
        self.save_dir = save_dir or EFFICIENTNET_B0_SSL_DIR
        self.optimizer_name = optimizer_name
        self.use_accelerate = use_accelerate
        self.use_imgaug = use_imgaug
        self.use_kornia = use_kornia
        self.enhance_ultrasound = enhance_ultrasound
        self.pretrained_path = pretrained_path
        self.kwargs = kwargs
        # ✅ SPEED FIX: num_workers پیش‌فرض
        if "num_workers" not in self.kwargs:
            import os as _os
            cpu_count = _os.cpu_count() or 4
            self.kwargs["num_workers"] = min(6, max(4, cpu_count // 2))

        # ✅ CUDA-ONLY
        self.device = _require_cuda(device)

        set_seed(42)
        ensure_directories()

        # ✅ Auto-tune based on VRAM
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 4.0 and batch_size > 32:
            print(f"⚠️  Reducing batch_size from {batch_size} to 32 (low VRAM)")
            self.batch_size = 32

        # Build encoder
        self.encoder = self._build_encoder()

        # Load pretrained weights
        actual_path = None
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            actual_path = self.pretrained_path
            self.load_pretrained_weights(self.pretrained_path)
        else:
            default_path = os.path.join(
                PROJECT_ROOT, "finalReports", "models",
                "efficientNet", "", "checkpoints", "efficientnet_b0_best.pth",
            )
            if os.path.exists(default_path):
                actual_path = default_path
                self.load_pretrained_weights(default_path)
            else:
                raise FileNotFoundError(
                    f"❌ Pretrained EfficientNet-B0 not found at: {default_path}"
                )

        self.pretrained_path = actual_path

        # Trainer
        self.trainer = self._create_trainer()

        # State
        self.history = {}
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.metrics = {}
        self.confusion_matrix = None
        self.roc_data = None

        # TensorBoard
        self.writer = None
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

        if self.verbose:
            self._print_info()

    # ============================================================
    # ✅ BUILD ENCODER
    # ============================================================

    def _build_encoder(self):
        model = create_efficientnet_b0(num_classes=self.num_classes, pretrained=False, verbose=False)
        if self.verbose:
            print(f"✅ Built EfficientNet-B0 encoder")
            print(f"   Feature dimension: {model._feature_dim}")
        return model

    # ============================================================
    # ✅ LOAD PRETRAINED
    # ============================================================

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not os.path.exists(model_path):
            print(f"⚠️  Not found: {model_path}")
            self._pretrained_loaded = False
            return False

        try:
            ckpt = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "encoder_state_dict" in ckpt:
                state_dict = ckpt["encoder_state_dict"]
            else:
                state_dict = ckpt

            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
            self._pretrained_loaded = True

            if self.verbose:
                print(f"✅ Pretrained loaded: {model_path}")
                print(f"   Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                if "best_loss" in ckpt:
                    print(f"   Previous Best Loss: {ckpt['best_loss']:.4f}")
                if "epoch" in ckpt:
                    print(f"   Previous Epoch: {ckpt['epoch']}")

            return True
        except Exception as e:
            print(f"⚠️  Failed to load pretrained: {e}")
            self._pretrained_loaded = False
            return False

    # ============================================================
    # ✅ CREATE TRAINER
    # ============================================================

    def _create_trainer(self):
        if self.method == "simclr":
            return create_simclr_trainer(
                encoder=self.encoder,
                input_size=self.input_size,
                projection_dim=self.kwargs.get("projection_dim", 256),
                hidden_dim=self.kwargs.get("hidden_dim", 2048),
                temperature=self.kwargs.get("temperature", 0.5),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_lars=self.kwargs.get("use_lars", True),
                use_mixed_precision=self.use_mixed_precision,
                use_bf16=self.kwargs.get("use_bf16", True),
                gradient_accumulation_steps=self.kwargs.get("gradient_accumulation_steps", 1),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
                device=str(self.device),
                verbose=self.verbose,
                save_dir=self.save_dir,
            )

        if self.method == "byol":
            return create_byol_trainer(
                model=self.encoder,
                model_name="efficientnet_b0",
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
                gradient_accumulation_steps=self.kwargs.get("gradient_accumulation_steps", 1),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
                device=str(self.device),
                verbose=self.verbose,
                augmentation_method="byol",
                save_dir=self.save_dir,
                # ✅ SPEED FIX: این ۴ خط رو اضافه کن
                use_kornia=True,
                use_bf16=True,
                augmentation_intensity=0.8,
                use_medical_safe=True,
            )

        if self.method == "mae":
            return create_mae_trainer(
                encoder=self.encoder,
                img_size=self.input_size,
                patch_size=self.kwargs.get("patch_size", 7),
                mask_ratio=self.kwargs.get("mask_ratio", 0.75),
                embed_dim=self.kwargs.get("embed_dim", 256),
                depth=self.kwargs.get("depth", 4),
                num_heads=self.kwargs.get("num_heads", 8),
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                use_mixed_precision=self.use_mixed_precision,
                gradient_accumulation_steps=self.kwargs.get("gradient_accumulation_steps", 1),
                clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
                device=str(self.device),
                verbose=self.verbose,
                save_dir=self.save_dir,
            )

        raise ValueError(f"Unknown SSL method: {self.method}")

    # ============================================================
    # ✅ PRINT INFO
    # ============================================================

    def _print_info(self):
        print("\n" + "=" * 80)
        print("🚀 EfficientNet-B0 SSL Pipeline — 2026 ULTIMATE")
        print("=" * 80)
        print(f"   Method: {self.method.upper()}")
        print(f"   Input Size: {self.input_size}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Epochs: {self.total_epochs}")
        print(f"   LR: {self.lr}")
        print(f"   Optimizer: {self.optimizer_name}")
        print(f"   Device: {self.device}")
        print(f"   Save Dir: {self.save_dir}")
        print(f"   Pretrained: ✅ Loaded")
        print(f"   TensorBoard: {'✅' if HAS_TENSORBOARD else '❌'}")
        print(f"   Optuna: {'✅' if HAS_OPTUNA else '❌'}")
        print(f"   Kornia: {'✅' if HAS_KORNIA else '❌'}")
        print(f"   Captum: {'✅' if HAS_CAPTUM else '❌'}")
        print(f"   LIME: {'✅' if HAS_LIME else '❌'}")
        print(f"   SHAP: {'✅' if HAS_SHAP else '❌'}")
        print(f"   Netron: {'✅' if HAS_NETRON else '❌'}")
        print("=" * 80)

    # ============================================================
    # ✅ PRETRAIN
    # ============================================================

    def pretrain(self, image_paths, validation_paths=None, save_interval=10):
        if HAS_MEMORY:
            print_memory_status("Memory Before Training")

        if self.verbose:
            print(f"\n📊 Starting SSL pretraining on {len(image_paths)} images")

        # Dataloaders
        train_loader = create_ssl_dataloader(
            image_paths=image_paths,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.kwargs.get("num_workers", 6),  # ← از 4 به 6
            target_size=self.input_size,
            augmentation_method=self.kwargs.get("augmentation_method", "byol"),
            enhance_ultrasound=self.enhance_ultrasound,
            use_imgaug=self.use_imgaug,
            use_kornia=self.use_kornia,
            drop_last=True,
        )
        val_loader = None
        if validation_paths:
            val_loader = create_ssl_dataloader(
                image_paths=validation_paths, batch_size=self.batch_size, shuffle=False,
                num_workers=self.kwargs.get("num_workers", 4),
                target_size=self.input_size,
                augmentation_method=self.kwargs.get("augmentation_method", "byol"),
                enhance_ultrasound=self.enhance_ultrasound,
                use_imgaug=self.use_imgaug, use_kornia=self.use_kornia, drop_last=False,
            )

        # Early stopping
        from src.self_supervised.Facilities.early_stopping import create_early_stopping
        effective_patience = min(55, max(15, self.total_epochs - 5))

        early_stopping = create_early_stopping(
            patience=effective_patience, min_delta=1e-4, verbose=self.verbose,
            save_best=True, restore_best=True,
            checkpoint_dir=os.path.join(self.save_dir, "early_stopping"),
        )

        # Progress bar
        from tqdm import tqdm

        with tqdm(
            total=self.total_epochs,
            desc=f"🧠 {self.method.upper()} on EfficientNet-B0",
            bar_format="{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            colour="#00d4ff", ncols=110,
        ) as pbar:
            for epoch in range(self.total_epochs):
                # ✅ FIX: set current_epoch روی trainer تا نمایش درست بشه
                self.trainer.current_epoch = epoch

                metrics = self.trainer.train_epoch(train_loader)
                current_loss = metrics.get("loss", float("inf"))

                # Early stop from trainer
                if metrics.get("early_stopped", False):
                    print(f"\n🛑 Collapse detected at epoch {epoch}")
                    break

                monitor_loss = current_loss
                if val_loader is not None:
                    try:
                        val_m = self.trainer.validate(val_loader)
                        monitor_loss = val_m.get("loss", current_loss)
                    except Exception:
                        pass

                # TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar("Loss/train", current_loss, epoch)
                    if val_loader:
                        self.writer.add_scalar("Loss/val", monitor_loss, epoch)
                    try:
                        self.writer.add_scalar("LR", self.trainer.optimizer.param_groups[0]["lr"], epoch)
                    except Exception:
                        pass

                # Early stopping
                if early_stopping(monitor_loss, self.trainer, epoch):
                    print(f"\n🛑 Early stopping at epoch {epoch}")
                    break

                # ✅ NEW: Save 'last' every epoch
                self.save_model(save_type="last")

                # ✅ NEW: Save 'best' if improved
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.save_model(save_type="best")

                # ✅ NEW: Save 'checkpoint' every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.save_model(save_type="checkpoint")

                # ✅ NEW: Save epoch-specific (for debugging)
                if (epoch + 1) % save_interval == 0:
                    self.save_model(
                        path=os.path.join(
                            get_method_checkpoint_dir(self.method),
                            f"b0_ssl_{self.method}_epoch_{epoch+1}.pth",
                        ),
                        save_type="checkpoint",
                    )

                # Progress update
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "Best": f"{self.best_loss:.4f}",
                    "Pat": f"{early_stopping.counter}/{effective_patience}",
                })
                pbar.update(1)

                if HAS_MEMORY and (epoch + 1) % 10 == 0:
                    clear_memory()

        # Restore best
        self.encoder = self.trainer.get_encoder()

        # ✅ NEW: Save 'final' — encoder after restore best
        self.save_model(save_type="final")

        # ✅ NEW: Save 'checkpoint' at the very end (with all state)
        self.save_model(save_type="checkpoint")


        results = {
            "history": self.trainer.get_history(),
            "best_loss": self.best_loss,
            "best_epoch": early_stopping.best_epoch,
            "total_epochs": epoch + 1,
            "early_stopping": early_stopping.is_stopped,
            "early_stopping_summary": early_stopping.get_summary(),
        }

        self.history = results
        self.current_epoch = epoch + 1

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ SSL Pretraining Complete!")
            print(f"   Best Loss: {self.best_loss:.4f}")
            print(f"   Epochs: {self.current_epoch}")
            print("=" * 80)

        if self.writer:
            self.writer.close()

        return results

    # ============================================================
    # ✅ EVALUATE (Linear Probe)
    # ============================================================

    def evaluate(self, labeled_loader, num_classes=4):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        self.encoder.eval()
        all_emb, all_lbl = [], []

        with torch.no_grad():
            for images, labels in tqdm(labeled_loader, desc="Extracting"):
                images = images.to(self.device)
                if hasattr(self.encoder, "get_embedding"):
                    emb = self.encoder.get_embedding(images)
                else:
                    emb = self.encoder(images)
                    if emb.dim() > 2:
                        emb = emb.view(emb.size(0), -1)
                all_emb.append(emb.float().cpu().numpy())
                all_lbl.append(labels.numpy())

        X = np.concatenate(all_emb, axis=0)
        y = np.concatenate(all_lbl, axis=0)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        try:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        except TypeError:
            clf = LogisticRegression(max_iter=2000, multi_class="ovr", class_weight="balanced", random_state=42)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)

        accuracy = accuracy_score(y_te, y_pred)
        precision = precision_score(y_te, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_te, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_te, y_pred)

        y_bin = label_binarize(y_te, classes=range(num_classes))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        class_report = classification_report(
            y_te, y_pred, labels=range(num_classes),
            target_names=CLASS_NAMES[:num_classes], output_dict=True, zero_division=0,
        )

        self.metrics = {
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1_score": f1, "confusion_matrix": cm, "roc_auc": roc_auc,
            "classification_report": class_report, "num_samples": len(y_te),
        }
        self.confusion_matrix = cm
        self.roc_data = (fpr, tpr, roc_auc)

        if self.verbose:
            print(f"\n📊 Evaluation:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1: {f1:.4f}")

        return self.metrics

    # ============================================================
    # ✅ SAVE / LOAD
    # ============================================================

    def save_model(self, path=None, save_type: str = "best"):
        """
        ✅ NEW: Save model to architecture/{method}/ directory.

        Args:
            path: Custom path (optional)
            save_type: 'best', 'last', 'checkpoint', 'final'
        """
        # ✅ NEW: Default paths in architecture/{method}/
        if path is None:
            paths = get_checkpoint_paths(self.method, model_prefix="b0_ssl")
            path = paths.get(save_type, paths["best"])

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # ✅ NEW: Different content for different save types
        save_dict = {
            "encoder_state_dict": self.get_encoder().state_dict(),
            "method": self.method,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "best_loss": self.best_loss,
            "current_epoch": self.current_epoch,
            "timestamp": datetime.now().isoformat(),
            "config": self._get_config(),
            "save_type": save_type,
        }

        # ✅ NEW: For 'checkpoint', add optimizer + scheduler + history
        if save_type == "checkpoint":
            try:
                save_dict["optimizer_state_dict"] = self.trainer.optimizer.state_dict()
                save_dict["scheduler_state_dict"] = self.trainer.scheduler.state_dict()
                save_dict["history"] = self.trainer.get_history()
                save_dict["momentum"] = self.trainer.momentum
            except Exception as e:
                print(f"⚠️ Could not save optimizer/scheduler state: {e}")

        torch.save(save_dict, path)

        if self.verbose:
            print(f"✅ Model saved ({save_type}): {path}")
        return path

    def save_for_transfer(self):
        path = os.path.join(
            EFFICIENTNET_WEIGHT_TRANSFORMER_DIR,
            f"efficientnet_b0_ssl_{self.method}_for_transfer.pth",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "encoder_state_dict": self.get_encoder().state_dict(),
            "method": self.method,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "feature_dim": self.encoder._feature_dim,
            "best_loss": self.best_loss,
            "timestamp": datetime.now().isoformat(),
        }, path)
        return path

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.method = ckpt.get("method", self.method)
        self.input_size = ckpt.get("input_size", self.input_size)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.current_epoch = ckpt.get("current_epoch", 0)

    def get_encoder(self):
        return self.trainer.get_encoder()

    # ============================================================
    # ✅ REPORT GENERATION
    # ============================================================

    def generate_report(self, output_path=None, labeled_loader=None):
        if output_path is None:
            output_path = os.path.join(
                COMPARISONS_REPORTER_DIR, f"efficientnet_b0_ssl_{self.method}_report.pdf"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plots_dir = os.path.join(os.path.dirname(output_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        interpret_dir = os.path.join(os.path.dirname(output_path), "interpretability")
        os.makedirs(interpret_dir, exist_ok=True)
        arch_dir = os.path.join(os.path.dirname(output_path), "architecture")
        os.makedirs(arch_dir, exist_ok=True)

        # ✅ Auto-build metrics if empty
        if not self.metrics or "accuracy" not in self.metrics:
            print("\n🔬 Auto-running Linear Probe...")
            try:
                from src.self_supervised.ProbabilityAlgorithms.linear_probe import build_labeled_eval_loader
                eval_loader = build_labeled_eval_loader(
                    labels_csv="finalReports/labels_extended.csv" if os.path.exists("finalReports/labels_extended.csv") else "finalReports/labels.csv",
                    image_dir="finalReports/cropped_extended" if os.path.exists("finalReports/cropped_extended") else "finalReports/cropped",
                    input_size=self.input_size, batch_size=32, max_per_class=300,
                )
                self.evaluate(eval_loader, num_classes=self.num_classes)
                if labeled_loader is None:
                    labeled_loader = eval_loader
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")

        # Build labeled_loader for interpretability
        if labeled_loader is None:
            try:
                from src.self_supervised.ProbabilityAlgorithms.linear_probe import build_labeled_eval_loader
                labeled_loader = build_labeled_eval_loader(
                    labels_csv="finalReports/labels_extended.csv" if os.path.exists("finalReports/labels_extended.csv") else "finalReports/labels.csv",
                    image_dir="finalReports/cropped_extended" if os.path.exists("finalReports/cropped_extended") else "finalReports/cropped",
                    input_size=self.input_size, batch_size=32, max_per_class=30,
                )
            except Exception as e:
                print(f"⚠️ Could not build labeled_loader: {e}")

        # Plots
        plot_paths = self._generate_plots(plots_dir)
        print(f"📊 Generated {len(plot_paths)} plots")

        # Interpretability
        interpret_paths = {
            "captum_gradcam_path": None, "captum_ig_path": None,
            "lime_explanation_path": None, "shap_summary_path": None,
        }

        if labeled_loader is not None:
            if HAS_CAPTUM:
                try:
                    res = generate_interpretability_report(
                        self.encoder, labeled_loader, interpret_dir,
                        device=self.device, class_names=CLASS_NAMES[:self.num_classes],
                    )
                    interpret_paths["captum_gradcam_path"] = res.get("gradcam_path")
                    interpret_paths["captum_ig_path"] = res.get("ig_path")
                except Exception as e:
                    print(f"⚠️ Captum failed: {e}")

            if HAS_LIME:
                try:
                    res = explain_predictions_lime(
                        self.encoder, labeled_loader, interpret_dir,
                        class_names=CLASS_NAMES[:self.num_classes], device=self.device,
                    )
                    if isinstance(res, dict):
                        interpret_paths["lime_explanation_path"] = res.get("lime_path")
                except Exception as e:
                    print(f"⚠️ LIME failed: {e}")

            if HAS_SHAP:
                try:
                    res = generate_shap_explanation(
                        self.encoder, labeled_loader, interpret_dir,
                        device=self.device, class_names=CLASS_NAMES[:self.num_classes],
                    )
                    if isinstance(res, dict):
                        interpret_paths["shap_summary_path"] = res.get("shap_path")
                except Exception as e:
                    print(f"⚠️ SHAP failed: {e}")

        # Netron
        if HAS_NETRON:
            try:
                netron_res = save_model_with_name(
                    self.encoder, arch_dir, model_name="EfficientNet-B0",
                    input_shape=(1, 3, self.input_size, self.input_size),
                )
                info_json = netron_res.get("info_path")
                if info_json and os.path.exists(info_json):
                    update_architecture_info_json(info_json, **interpret_paths)
            except Exception as e:
                print(f"⚠️ Netron failed: {e}")

        # Build PDF
        if not HAS_REPORTLAB:
            return self._generate_report_fallback(output_path)

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=60)
        styles = getSampleStyleSheet()
        title = ParagraphStyle("T", parent=styles["Heading1"], fontSize=24, alignment=TA_CENTER, spaceAfter=20)
        h1 = ParagraphStyle("H1", parent=styles["Heading2"], fontSize=16, spaceAfter=12, textColor=colors.HexColor("#2c3e50"))
        h2 = ParagraphStyle("H2", parent=styles["Heading3"], fontSize=12, spaceAfter=8)
        body = styles["Normal"]

        story = []

        # PAGE 1: Title + Summary
        story.append(Paragraph("EfficientNet-B0 SSL Report", title))
        story.append(Paragraph(f"Method: <b>{self.method.upper()}</b>",
                               ParagraphStyle("S", parent=body, alignment=TA_CENTER, fontSize=14)))
        story.append(Spacer(1, 20))
        story.append(Paragraph("1. Overview", h1))
        story.append(Paragraph(
            f"This report presents the complete SSL analysis for <b>EfficientNet-B0</b> "
            f"using <b>{self.method.upper()}</b>. Trained for <b>{self.current_epoch}</b> epochs, "
            f"achieving best loss <b>{self.best_loss:.6f}</b>. Quality assessed via Linear Probe.",
            body,
        ))
        story.append(Spacer(1, 15))

        pretrained_display = "✅ Loaded" if getattr(self, "_pretrained_loaded", False) else "❌ From Scratch"
        summary_data = [
            ["Parameter", "Value"],
            ["SSL Method", self.method.upper()],
            ["Best SSL Loss", f"{self.best_loss:.6f}"],
            ["Total Epochs", str(self.current_epoch)],
            ["Input Size", f"{self.input_size}x{self.input_size}"],
            ["Batch Size", str(self.batch_size)],
            ["LR", f"{self.lr}"],
            ["Weight Decay", f"{self.weight_decay}"],
            ["Optimizer", self.optimizer_name],
            ["Device", str(self.device)],
            ["Pretrained Weights", pretrained_display],
        ]
        t = Table(summary_data, colWidths=[2.5 * inch, 3.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)

        # PAGE 2: Metrics
        story.append(PageBreak())
        story.append(Paragraph("2. Performance Metrics", h1))
        if self.metrics and "accuracy" in self.metrics:
            acc = self.metrics.get("accuracy", 0)
            verdict = "🟢 Excellent" if acc > 0.75 else ("🟡 Good" if acc > 0.60 else "🔴 Poor")
            story.append(Paragraph(f"<b>Verdict:</b> {verdict} (acc={acc:.4f})", body))
            story.append(Spacer(1, 10))
            metrics_data = [
                ["Metric", "Value"],
                ["Accuracy", f"{acc:.4f}"],
                ["Precision", f"{self.metrics.get('precision', 0):.4f}"],
                ["Recall", f"{self.metrics.get('recall', 0):.4f}"],
                ["F1-Score", f"{self.metrics.get('f1_score', 0):.4f}"],
                ["Samples", str(self.metrics.get("num_samples", 0))],
            ]
            t = Table(metrics_data, colWidths=[3 * inch, 3 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]))
            story.append(t)

            if "classification_report" in self.metrics:
                story.append(Spacer(1, 15))
                story.append(Paragraph("Per-Class Performance", h2))
                cd = [["Class", "Precision", "Recall", "F1", "Support"]]
                for cname in CLASS_NAMES[: self.num_classes]:
                    if cname in self.metrics["classification_report"]:
                        r = self.metrics["classification_report"][cname]
                        cd.append([cname, f"{r.get('precision', 0):.4f}",
                                   f"{r.get('recall', 0):.4f}", f"{r.get('f1-score', 0):.4f}",
                                   str(r.get("support", 0))])
                t = Table(cd, colWidths=[1.2 * inch] * 5)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]))
                story.append(t)

        # Plots pages
        for pk, ptitle in [
            ("loss_history", "Loss History"),
            ("learning_rate", "Learning Rate"),
            ("confusion_matrix", "Confusion Matrix"),
            ("roc_curves", "ROC Curves"),
            ("metrics_bar_chart", "Metrics Chart"),
            ("accuracy_comparison", "Per-Class Performance"),
            ("pie_chart", "Class Distribution"),
        ]:
            if pk in plot_paths and os.path.exists(plot_paths[pk]):
                story.append(PageBreak())
                story.append(Paragraph(ptitle, h1))
                story.append(RLImage(plot_paths[pk], width=6.5 * inch, height=4.5 * inch))

        # Interpretability
        story.append(PageBreak())
        story.append(Paragraph("Interpretability", h1))
        for key, title_ in [("captum_gradcam_path", "Grad-CAM"),
                            ("captum_ig_path", "Integrated Gradients"),
                            ("lime_explanation_path", "LIME"),
                            ("shap_summary_path", "SHAP")]:
            p = interpret_paths.get(key)
            if p and os.path.exists(p):
                story.append(Paragraph(title_, h2))
                story.append(RLImage(p, width=6 * inch, height=4 * inch))

        # Conclusion
        story.append(PageBreak())
        story.append(Paragraph("Conclusion", h1))
        story.append(Paragraph(
            f"<b>Method:</b> {self.method.upper()}<br/>"
            f"<b>Best Loss:</b> {self.best_loss:.6f}<br/>"
            f"<b>Epochs:</b> {self.current_epoch}<br/>"
            f"<b>Accuracy:</b> {self.metrics.get('accuracy', 0):.4f}<br/>",
            body,
        ))

        try:
            doc.build(story)
            print(f"✅ PDF Report: {output_path}")
        except Exception as e:
            print(f"❌ PDF build failed: {e}")

        self.save_training_summary()
        return output_path

    def _generate_plots(self, plots_dir):
        plot_paths = {}

        # Loss
        try:
            if "loss_epoch" in self.history and self.history["loss_epoch"]:
                plt.figure(figsize=(10, 6))
                plt.plot(self.history["loss_epoch"], label="Train Loss", linewidth=2)
                plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{self.method.upper()} Loss")
                plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
                p = os.path.join(plots_dir, "loss_history.png")
                plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
                plot_paths["loss_history"] = p
        except Exception: plt.close("all")

        # LR
        try:
            if "learning_rates" in self.history and self.history["learning_rates"]:
                plt.figure(figsize=(10, 6))
                plt.plot(self.history["learning_rates"], linewidth=2, color="purple")
                plt.xlabel("Epoch"); plt.ylabel("LR"); plt.yscale("log")
                plt.title("LR Schedule"); plt.grid(True, alpha=0.3); plt.tight_layout()
                p = os.path.join(plots_dir, "learning_rate.png")
                plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
                plot_paths["learning_rate"] = p
        except Exception: plt.close("all")

        # Confusion Matrix
        try:
            if self.confusion_matrix is not None:
                cm = np.asarray(self.confusion_matrix)
                plt.figure(figsize=(8, 7))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=CLASS_NAMES[: self.num_classes],
                            yticklabels=CLASS_NAMES[: self.num_classes])
                plt.title(f"{self.method.upper()} Confusion Matrix")
                plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
                p = os.path.join(plots_dir, "confusion_matrix.png")
                plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
                plot_paths["confusion_matrix"] = p
        except Exception: plt.close("all")

        # ROC
        try:
            if self.roc_data is not None:
                fpr, tpr, roc_auc = self.roc_data
                plt.figure(figsize=(9, 7))
                colors_roc = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
                for i in range(self.num_classes):
                    if i in fpr and i in tpr and i in roc_auc:
                        plt.plot(fpr[i], tpr[i], linewidth=2.5,
                                 label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:.3f})",
                                 color=colors_roc[i % len(colors_roc)])
                plt.plot([0, 1], [0, 1], "k--", label="Random")
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
                plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
                p = os.path.join(plots_dir, "roc_curves.png")
                plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
                plot_paths["roc_curves"] = p
        except Exception: plt.close("all")

        # Metrics Bar
        try:
            if self.metrics and "accuracy" in self.metrics:
                mnames = ["Accuracy", "Precision", "Recall", "F1"]
                vals = [self.metrics.get("accuracy", 0), self.metrics.get("precision", 0),
                        self.metrics.get("recall", 0), self.metrics.get("f1_score", 0)]
                plt.figure(figsize=(9, 6))
                bars = plt.bar(mnames, vals, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
                for bar, v in zip(bars, vals):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                             f"{v:.4f}", ha="center", fontsize=11)
                plt.ylim(0, 1.1); plt.title("Metrics"); plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                p = os.path.join(plots_dir, "metrics_bar_chart.png")
                plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
                plot_paths["metrics_bar_chart"] = p
        except Exception: plt.close("all")

        # Per-class
        try:
            if self.metrics and "classification_report" in self.metrics:
                names_, accs_ = [], []
                for cn in CLASS_NAMES[: self.num_classes]:
                    if cn in self.metrics["classification_report"]:
                        r = self.metrics["classification_report"][cn]
                        names_.append(cn); accs_.append(r.get("recall", 0))
                if names_:
                    plt.figure(figsize=(9, 6))
                    bars = plt.bar(names_, accs_, color="#3498db")
                    for b, v in zip(bars, accs_):
                        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                                 f"{v:.3f}", ha="center", fontsize=11)
                    plt.ylim(0, 1.1); plt.title("Per-Class Recall"); plt.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    p = os.path.join(plots_dir, "accuracy_comparison.png")
                    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
                    plot_paths["accuracy_comparison"] = p
        except Exception: plt.close("all")

        # Pie
        try:
            if self.metrics and "classification_report" in self.metrics:
                lbs, szs = [], []
                for cn in CLASS_NAMES[: self.num_classes]:
                    if cn in self.metrics["classification_report"]:
                        s = self.metrics["classification_report"][cn].get("support", 0)
                        if s > 0: lbs.append(cn); szs.append(s)
                if szs:
                    plt.figure(figsize=(8, 8))
                    plt.pie(szs, labels=lbs, autopct="%1.1f%%", startangle=90)
                    plt.title("Class Distribution"); plt.tight_layout()
                    p = os.path.join(plots_dir, "pie_chart.png")
                    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
                    plot_paths["pie_chart"] = p
        except Exception: plt.close("all")

        return plot_paths

    def _generate_report_fallback(self, output_path):
        json_path = output_path.replace(".pdf", ".json")
        with open(json_path, "w") as f:
            json.dump({
                "model": "EfficientNet-B0", "method": self.method,
                "best_loss": self.best_loss, "metrics": self.metrics,
                "config": self._get_config(),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4, default=str)
        return json_path

    def save_training_summary(self, output_path=None):
        if output_path is None:
            output_path = os.path.join(COMPARISONS_REPORTER_DIR,
                                       f"efficientnet_b0_ssl_{self.method}_summary.json")
        pretrained_loaded = getattr(self, "_pretrained_loaded", False)
        summary = {
            "model": "EfficientNet-B0", "method": self.method,
            "best_loss": self.best_loss, "total_epochs": self.current_epoch,
            "history": self.history, "metrics": self.metrics,
            "config": self._get_config(), "optimizer": self.optimizer_name,
            "device": str(self.device),
            "pretrained_weights": "Loaded" if pretrained_loaded else "From Scratch",
            "pretrained_path": self.pretrained_path,
            "pretrained_loaded": pretrained_loaded,
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4, default=str)
        return output_path

    def _get_config(self):
        return {
            "method": self.method, "num_classes": self.num_classes,
            "input_size": self.input_size, "batch_size": self.batch_size,
            "lr": self.lr, "weight_decay": self.weight_decay,
            "total_epochs": self.total_epochs, "warmup_epochs": self.warmup_epochs,
            "use_mixed_precision": self.use_mixed_precision,
            "device": str(self.device), "optimizer": self.optimizer_name,
            "pretrained_path": self.pretrained_path,
            "pretrained_loaded": getattr(self, "_pretrained_loaded", False),
            "kwargs": self.kwargs,
        }

    # ============================================================
    # ✅ FULL SSL PIPELINE
    # ============================================================

    def run_full_ssl_pipeline(self, image_paths, validation_paths=None, save_interval=10):
        print("\n" + "=" * 80)
        print("🚀 FULL SSL PIPELINE (EfficientNet-B0 → B1)")
        print("=" * 80)

        results = {}
        methods = ["byol", "simclr", "mae"]
        method_results = {}

        for method in methods:
            print(f"\n🧠 Training {method.upper()} on EfficientNet-B0")
            pipeline = EfficientNetB0SSL(
                method=method, num_classes=self.num_classes, input_size=self.input_size,
                batch_size=self.batch_size, lr=self.lr, weight_decay=self.weight_decay,
                total_epochs=self.total_epochs, warmup_epochs=self.warmup_epochs,
                use_mixed_precision=self.use_mixed_precision, device="cuda",
                verbose=self.verbose, save_dir=os.path.join(self.save_dir, method),
                optimizer_name=self.optimizer_name, **self.kwargs,
            )
            mr = pipeline.pretrain(image_paths, validation_paths, save_interval)
            method_results[method] = {"pipeline": pipeline, **mr}
            pipeline.save_for_transfer()
            try:
                pipeline.generate_report()
            except Exception as e:
                print(f"⚠️ Report {method}: {e}")

        # ✅ Linear Probe best selection
        from src.self_supervised.ProbabilityAlgorithms.linear_probe import build_labeled_eval_loader, evaluate_with_linear_probe

        eval_loader = build_labeled_eval_loader(
            labels_csv="finalReports/labels_extended.csv" if os.path.exists("finalReports/labels_extended.csv") else "finalReports/labels.csv",
            image_dir="finalReports/cropped_extended" if os.path.exists("finalReports/cropped_extended") else "finalReports/cropped",
            input_size=self.input_size, batch_size=32, max_per_class=200,
        )

        lp_results = {}
        for method, res in method_results.items():
            print(f"\n📌 Linear Probe: {method.upper()}")
            enc = res["pipeline"].get_encoder()
            probe = evaluate_with_linear_probe(enc, eval_loader, str(self.device), self.num_classes, 0.3)
            lp_results[method] = probe
            print(f"   ✅ {method.upper()}: {probe['linear_probe_accuracy']:.4f}")

        best_method = max(lp_results.keys(), key=lambda x: lp_results[x]["linear_probe_accuracy"])
        best_loss = method_results[best_method]["best_loss"]
        best_pipeline = method_results[best_method]["pipeline"]

        print(f"\n🏆 Best: {best_method.upper()} (LP Acc={lp_results[best_method]['linear_probe_accuracy']:.4f})")

        results["best_method"] = best_method
        results["best_loss"] = best_loss
        results["linear_probe_results"] = lp_results

        # Transfer to B1
        from src.models.efficientNet.B1 import create_efficientnet_b1

        best_encoder = best_pipeline.get_encoder()
        best_state = best_encoder.state_dict()
        b1 = create_efficientnet_b1(num_classes=self.num_classes, pretrained=False, verbose=False)

        transfer = {}
        matched = 0
        for name, param in b1.state_dict().items():
            if name in best_state and param.shape == best_state[name].shape:
                transfer[name] = best_state[name]
                matched += 1
            else:
                transfer[name] = param
        b1.load_state_dict(transfer, strict=False)
        print(f"✅ Transferred {matched}/{len(b1.state_dict())} layers")

        # Fine-tune B1
        b1_loader = create_ssl_dataloader(
            image_paths, self.batch_size, True,
            self.kwargs.get("num_workers", 4), target_size=240,
            augmentation_method="byol", enhance_ultrasound=self.enhance_ultrasound,
            use_imgaug=self.use_imgaug, use_kornia=self.use_kornia, drop_last=True,
        )
        b1_val = None
        if validation_paths:
            b1_val = create_ssl_dataloader(
                validation_paths, self.batch_size, False,
                self.kwargs.get("num_workers", 4), target_size=240,
                augmentation_method="byol", enhance_ultrasound=self.enhance_ultrasound,
                use_imgaug=self.use_imgaug, use_kornia=self.use_kornia, drop_last=False,
            )

        b1_trainer = create_byol_trainer(
            model=b1, model_name="efficientnet_b1", input_size=240,
            projection_dim=self.kwargs.get("projection_dim", 256),
            hidden_dim=self.kwargs.get("hidden_dim", 4096),
            momentum=self.kwargs.get("momentum", 0.996),
            momentum_schedule="cosine", lr=self.lr * 0.1,
            weight_decay=self.weight_decay, batch_size=self.batch_size,
            use_lars=False, use_mixed_precision=self.use_mixed_precision,
            gradient_accumulation_steps=self.kwargs.get("gradient_accumulation_steps", 1),
            clip_grad_norm=self.kwargs.get("clip_grad_norm", 1.0),
            warmup_epochs=max(2, self.warmup_epochs // 2),
            total_epochs=max(10, self.total_epochs // 2),
            device=str(self.device), verbose=self.verbose,
            augmentation_method="byol",
            save_dir=os.path.join(self.save_dir, "B1_finetuned"),
        )

        b1_results = b1_trainer.train(
            dataloader=b1_loader, epochs=max(10, self.total_epochs // 2),
            save_interval=save_interval, validation_loader=b1_val,
        )

        final_path = os.path.join(FINAL_MODELS_DIR, f"efficientnet_b1_{best_method}_final.pth")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save({
            "encoder_state_dict": b1.state_dict(),
            "method": best_method, "input_size": 240,
            "num_classes": self.num_classes,
            "best_loss": b1_results.get("best_loss", float("inf")),
            "timestamp": datetime.now().isoformat(),
            "config": {"source_model": "EfficientNet-B0", "target_model": "EfficientNet-B1",
                       "transfer_method": best_method},
        }, final_path)

        results["b1_finetuned"] = b1_results
        results["b1_final_model"] = final_path

        print(f"\n✅ Full pipeline complete. Final model: {final_path}")
        return results


# ================================================================
# ✅ FACTORY
# ================================================================

def create_efficientnet_b0_ssl_pipeline(
    method="byol", num_classes=4, input_size=224, batch_size=64,
    lr=1e-4, weight_decay=1e-4, total_epochs=100, warmup_epochs=10,
    use_mixed_precision=True, device="cuda", verbose=True,
    optimizer_name="adamw", use_accelerate=False, use_imgaug=False,
    use_kornia=False, enhance_ultrasound=True, pretrained_path=None, **kwargs,
):
    return EfficientNetB0SSL(
        method=method, num_classes=num_classes, input_size=input_size,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        total_epochs=total_epochs, warmup_epochs=warmup_epochs,
        use_mixed_precision=use_mixed_precision, device=device, verbose=verbose,
        optimizer_name=optimizer_name, use_accelerate=use_accelerate,
        use_imgaug=use_imgaug, use_kornia=use_kornia,
        enhance_ultrasound=enhance_ultrasound, pretrained_path=pretrained_path,
        **kwargs,
    )


# ================================================================
# ✅ TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing EfficientNet-B0 SSL — 2026 ULTIMATE")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        raise SystemExit(1)

    pipeline = create_efficientnet_b0_ssl_pipeline(
        method="byol", batch_size=4, total_epochs=2, warmup_epochs=1,
        verbose=True, device="cuda",
    )
    print(f"✅ Pipeline created: {pipeline}")
    print("=" * 80)