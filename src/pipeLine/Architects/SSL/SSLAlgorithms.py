# SSLAlgorithms.py

"""
Liver AI Pipeline - Main Entry Point (ULTIMATE EDITION)

This is the main entry point for the Liver AI Pipeline.
It provides a comprehensive menu system for:
- Preprocessing
- Fine-Tuning
- Without Fine-Tuning (Traditional)
- Self-Supervised Learning (SSL) on ResNet18
- Self-Supervised Learning (SSL) on EfficientNet-B0

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0 (ULTIMATE EDITION)
"""

import sys
import io
import warnings

# ================================================================
# ✅ AlexNet Fine-Tune + Reporter (NEW)
# ================================================================
try:
    from src.pipeLine.Architects.AlexnetFT.alexnetFineTune import (
        check_best_ssl_checkpoint,
        run_alexnet_finetune_pipeline,
        load_labeled_data,
        FINAL_MODEL_PATH as ALEXNET_FINAL_MODEL_PATH,
        REPORTER_ROOT as ALEXNET_REPORTER_ROOT,
        CV_DIR as ALEXNET_CV_DIR,
        INTERP_DIR as ALEXNET_INTERP_DIR,
        PLOTS_DIR as ALEXNET_PLOTS_DIR,
        ARCH_DIR as ALEXNET_ARCH_DIR,
    )
    from src.pipeLine.Architects.AlexnetFT.alexnetReporter import (
        generate_alexnet_report,
        PDF_PATH as ALEXNET_PDF_PATH,
    )
    HAS_ALEXNET_FINETUNE = True
except ImportError as e:
    HAS_ALEXNET_FINETUNE = False
    ALEXNET_FINAL_MODEL_PATH = None
    ALEXNET_REPORTER_ROOT = None
    ALEXNET_CV_DIR = None
    ALEXNET_INTERP_DIR = None
    ALEXNET_PLOTS_DIR = None
    ALEXNET_ARCH_DIR = None
    ALEXNET_PDF_PATH = None
    print(f"⚠️ AlexNet Fine-Tune modules not available: {e}")

# MobileNetV2 SSL Imports
try:
    from src.pipeLine.Architects.MobileNetV2.mobilenetv2SSL import create_mobilenetv2_ssl_pipeline
    HAS_MOBILENETV2_SSL = True
except ImportError as e:
    HAS_MOBILENETV2_SSL = False
    create_mobilenetv2_ssl_pipeline = None
    print(f"⚠️ MobileNetV2 SSL not available: {e}")

try:
    from src.pipeLine.Architects.MobileNetV2.mobilenetv2FineTune import (
        check_transferred_model as mnv2_check,
        run_mobilenetv2_finetune_pipeline,
        get_transferred_metadata as mnv2_get_meta,
        get_ssl_methods_comparison as mnv2_get_ssl,
    )
    HAS_MOBILENETV2_FINETUNE = True
except ImportError as e:
    HAS_MOBILENETV2_FINETUNE = False
    print(f"⚠️ MobileNetV2 Fine-Tune not available: {e}")

# ================================================================
# ✅ VGG16 SSL Imports (NEW)
# ================================================================
try:
    from src.pipeLine.Architects.VGG16.vgg16SSL import create_vgg16_ssl_pipeline
    HAS_VGG16_SSL = True
except ImportError as e:
    HAS_VGG16_SSL = False
    create_vgg16_ssl_pipeline = None
    print(f"⚠️ VGG16 SSL not available: {e}")

try:
    from src.pipeLine.Architects.VGG16.vgg16FineTune import (
        check_transferred_model as vgg16_check,
        run_vgg16_finetune_pipeline,
        get_transferred_metadata as vgg16_get_meta,
        get_ssl_methods_comparison as vgg16_get_ssl,
    )
    HAS_VGG16_FINETUNE = True
except ImportError as e:
    HAS_VGG16_FINETUNE = False
    print(f"⚠️ VGG16 Fine-Tune not available: {e}")

# ================================================================
# ✅ DenseNet SSL Imports (NEW)
# ================================================================
try:
    from src.pipeLine.Architects.DenseNet.densenetSSL import create_densenet_ssl_pipeline
    HAS_DENSENET_SSL = True
except ImportError as e:
    HAS_DENSENET_SSL = False
    create_densenet_ssl_pipeline = None
    print(f"⚠️ DenseNet SSL not available: {e}")

try:
    from src.pipeLine.Architects.DenseNet.densenetFineTune import (
        check_transferred_model as densenet_check,
        run_densenet_finetune_pipeline,
        get_transferred_metadata as densenet_get_meta,
        get_ssl_methods_comparison as densenet_get_ssl,
    )
    HAS_DENSENET_FINETUNE = True
except ImportError as e:
    HAS_DENSENET_FINETUNE = False
    print(f"⚠️ DenseNet Fine-Tune not available: {e}")

try:
    from src.pipeLine.Architects.DenseNet.densenetDirectFineTune import run_direct_densenet_finetune
    HAS_DENSENET_DIRECT = True
except ImportError as e:
    HAS_DENSENET_DIRECT = False
    print(f"⚠️ DenseNet Direct Fine-Tune not available: {e}")

try:
    from src.pipeLine.ConfigurationAlgorithms.findBestModels import run_find_best_models
    HAS_FIND_BEST = True
except ImportError as e:
    HAS_FIND_BEST = False
    print(f"⚠️ FindBestModels not available: {e}")

# ✅ Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*fitz.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ✅ Suppress unwanted import-time prints
_ORIGINAL_STDOUT = sys.stdout

class _SuppressFilter(io.TextIOBase):
    """Stream that filters out specific unwanted messages."""
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = ""
        self.suppress_patterns = [
            "Custom optimizers not available",
            "Seed set to:",
            "fitz API is deprecated",
            "Use `import pymupdf`",
        ]

    def write(self, message):
        self.buffer += message
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            self.buffer = lines[-1]
            for line in lines[:-1]:
                if not any(p in line for p in self.suppress_patterns):
                    self.original_stream.write(line + "\n")
        return len(message)

    def flush(self):
        if self.buffer:
            if not any(p in self.buffer for p in self.suppress_patterns):
                self.original_stream.write(self.buffer)
            self.buffer = ""
        self.original_stream.flush()

# ✅ Redirect only during import phase
sys.stdout = _SuppressFilter(_ORIGINAL_STDOUT)

# ================================================================
# ✅ All imports below will have their messages filtered
# ================================================================


import json
import os

from datetime import datetime

warnings.filterwarnings("ignore")

os.environ["PYTHONWARNINGS"] = "ignore"


# ================================================================
# ✅ Import existing modules
# ================================================================

from src.pipeLine.resNet18SSL import FINAL_MODELS_DIR
from withoutFineTuner import wFTAlgorithms, RESNET50_REPORTS_DIR

# ✅ Restore normal stdout
sys.stdout = _ORIGINAL_STDOUT

# ================================================================
# ✅ Import SSL modules (NEW)
# ================================================================


# ================================================================
# ✅ Try to import Rich for beautiful UI
# ================================================================

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    print("⚠️ Rich not installed. Install with: pip install rich")

import torch

from src.MassiveClasses.Audio.music import Music

# ✅ Hardware Detector (option 16)
try:
    from src.findDevice.hardwareDetector import _hw_
    HAS_HW_DETECTOR = True
except ImportError as e:
    HAS_HW_DETECTOR = False
    _hw_ = None
    print(f"⚠️ Hardware Detector not available: {e}")

# ✅ B1 Fine-Tune + Reporter modules
try:
    from src.pipeLine.Architects.EfficientNet.B1.b1FineTune import (
        check_transferred_model,
        run_b1_finetune_pipeline,
        load_labeled_data,
        FINAL_MODEL_PATH as B1_FINAL_MODEL_PATH,
        REPORTER_ROOT as B1_REPORTER_ROOT,
        CV_DIR as B1_CV_DIR,
        INTERP_DIR as B1_INTERP_DIR,
        PLOTS_DIR as B1_PLOTS_DIR,
        ARCH_DIR as B1_ARCH_DIR,
    )
    from src.pipeLine.Architects.EfficientNet.B1.b1Reporter import (
        generate_b1_report,
        PDF_PATH as B1_PDF_PATH,
    )
    HAS_B1_FINETUNE = True
except ImportError as e:
    HAS_B1_FINETUNE = False
    B1_FINAL_MODEL_PATH = None
    B1_REPORTER_ROOT = None
    B1_CV_DIR = None
    B1_INTERP_DIR = None
    B1_PLOTS_DIR = None
    B1_ARCH_DIR = None
    B1_PDF_PATH = None
    print(f"⚠️ B1 Fine-Tune modules not available: {e}")

# ✅ Dual Model Reporter
try:
    from src.pipeLine.Reporter.dualModelReporter import (
        run_dual_model_report,
        RESNET50_MODEL_PATH as DUAL_RESNET50_PATH,
        B1_MODEL_PATH as DUAL_B1_PATH,
        PDF_PATH as DUAL_PDF_PATH,
    )
    HAS_DUAL_REPORTER = True
except ImportError as e:
    HAS_DUAL_REPORTER = False
    DUAL_RESNET50_PATH = None
    DUAL_B1_PATH = None
    DUAL_PDF_PATH = None
    print(f"⚠️ Dual Model Reporter not available: {e}")

# ✅ SSL Dual Comparison Reporter
try:
    from src.pipeLine.Architects.SSL.sslDualComparison import (
        run_ssl_dual_comparison,
        MODEL_PATHS as SSL_MODEL_PATHS,
        PDF_PATH as SSL_DUAL_PDF_PATH,
    )
    HAS_SSL_DUAL = True
except ImportError as e:
    HAS_SSL_DUAL = False
    SSL_MODEL_PATHS = {}
    SSL_DUAL_PDF_PATH = None
    print(f"⚠️ SSL Dual Comparison not available: {e}")

# ✅ AlexNet SSL Import (NEW)
try:
    from src.pipeLine.Architects.AlexnetFT.alexnetSSL import create_alexnet_ssl_pipeline
    HAS_ALEXNET_SSL = True
except ImportError as e:
    HAS_ALEXNET_SSL = False
    create_alexnet_ssl_pipeline = None
    print(f"⚠️ AlexNet SSL not available: {e}")


MUSIC_PATH1 = "packages/facilities/assets/musics/music3.mp3"
MUSIC_PATH2 = "packages/facilities/assets/musics/music23.mp3"
MUSIC_PATH3 = "packages/facilities/assets/musics/music32.mp3"
MUSIC_PATH4 = "packages/facilities/assets/musics/music36.mp3"
MUSIC_PATH5 = "packages/facilities/assets/musics/music39.mp3"
MUSIC_PATH6 = "packages/facilities/assets/musics/music41.mp3"

# ================================================================
# ✅ Paths for SSL
# ================================================================

# ResNet weight transformer (ResNet18 → ResNet50)
RESNET_WEIGHT_TRANSFORMER_DIR = os.path.join(
    "../../../../finalReports", "selfSupervisedLearning", "ResNet", "weightTransformer"
)

# EfficientNet weight transformer (B0 → B1)
EFFICIENTNET_WEIGHT_TRANSFORMER_DIR = os.path.join(
    "../../../../finalReports", "selfSupervisedLearning", "EfficientNet", "weightTransformer"
)

RESNET18_SSL_DIR = os.path.join(
    "../../../../finalReports", "selfSupervisedLearning", "ResNet", "18", "SSL_checkpoints"
)

EFFICIENTNET_B0_SSL_DIR = os.path.join(
    "../../../../finalReports", "selfSupervisedLearning", "EfficientNet", "EfficientNet/B0", "SSL_checkpoints"
)

# ================================================================
# ✅ PATHS — ABSOLUTE (FIX for CV model not found)
# ================================================================

PROJECT_ROOT_MAIN = os.path.dirname(os.path.abspath(__file__))

# ResNet
RESNET_WEIGHT_TRANSFORMER_DIR = os.path.join(
    PROJECT_ROOT_MAIN, "../../../../finalReports", "selfSupervisedLearning", "ResNet", "weightTransformer"
)
RESNET18_SSL_DIR = os.path.join(
    PROJECT_ROOT_MAIN, "../../../../finalReports", "selfSupervisedLearning", "ResNet", "18", "SSL_checkpoints"
)

# EfficientNet
EFFICIENTNET_WEIGHT_TRANSFORMER_DIR = os.path.join(
    PROJECT_ROOT_MAIN, "../../../../finalReports", "selfSupervisedLearning", "EfficientNet", "weightTransformer"
)
EFFICIENTNET_B0_SSL_DIR = os.path.join(
    PROJECT_ROOT_MAIN, "../../../../finalReports", "selfSupervisedLearning", "EfficientNet", "EfficientNet/B0", "SSL_checkpoints"
)

# ================================================================
# ✅ Color Codes (Fallback for non-Rich)
# ================================================================


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def clear_python_cache():
    """Clear Python __pycache__ directories."""
    import shutil
    count = 0
    for root, dirs, _ in os.walk('../../../..'):
        for d in dirs:
            if d == '__pycache__':
                try:
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                    count += 1
                except:
                    pass
    if count > 0:
        print(f"🧹 Cleared {count} __pycache__ directories")

# ================================================================
# ✅ Print Header
# ================================================================


def print_header():
    """Print the main header."""
    if HAS_RICH:
        console.print(
            Panel.fit(
                "🧬  Liver AI Pipeline  🧬\n" "Self-Supervised Learning (SSL) - Ultimate Edition",
                border_style="green",
                padding=(1, 4),
            )
        )
    else:
        print(f"\n{Colors.GREEN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.GREEN}🧬  Liver AI Pipeline  🧬{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}Self-Supervised Learning (SSL) - Ultimate Edition{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}\n")


# ================================================================
# ✅ Print Menu
# ================================================================


def print_menu():
    """Print the main menu."""
    if HAS_RICH:
        table = Table(
            title="📋 Main Menu",
            box=box.ROUNDED,
            border_style="bright_blue",
            header_style="bold cyan",
        )
        table.add_column("#", style="bold yellow", width=4)
        table.add_column("Option", style="bold white")
        table.add_column("Description", style="dim white")

        table.add_row("0", "🚪 Exit", "Exit the pipeline")
        table.add_row("1", "🔄 Traditional Pipeline", "Run wFTAlgorithms")
        table.add_row(
            "2",
            "🧠 ResNet18 SSL Full",
            "BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune → ResNet50",
        )
        table.add_row("3", "🧠 ResNet18 within BYOL & Reports", "Run ResNet18 with BYOL only")
        table.add_row("4", "🧠 ResNet18 within SimCLR & Reports", "Run ResNet18 with SimCLR only")
        table.add_row("5", "🧠 ResNet18 within MAE & Reports", "Run ResNet18 with MAE only")
        table.add_row(
            "6",
            "🧠 Find best SSL models & Transfer to ResNet50 & Reports",
            "Compare BYOL/SimCLR/MAE → Transfer best to ResNet50",
        )
        table.add_row(
            "7",
            "🧠 Full-Fine-Tuning Ensemble ResNet Architects & Reports",
            "Complete ResNet ensemble fine-tuning",
        )
        table.add_row(
            "8",
            "⚡ EfficientNet-B0 SSL Full",
            "BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune → B1",
        )
        table.add_row(
            "9",
            "⚡ EfficientNet-B0 within BYOL & Reports",
            "Run EfficientNet-B0 with BYOL only",
        )
        table.add_row(
            "10",
            "⚡ EfficientNet-B0 within SimCLR & Reports",
            "Run EfficientNet-B0 with SimCLR only",
        )
        table.add_row(
            "11",
            "⚡ EfficientNet-B0 within MAE & Reports",
            "Run EfficientNet-B0 with MAE only",
        )
        table.add_row(
            "12",
            "⚡ Find best SSL models & Transfer to B1 & Reports",
            "Compare BYOL/SimCLR/MAE → Transfer best to B1",
        )
        table.add_row(
            "13",
            "⚡ Full-Fine-Tuning Ensemble EfficientNet Architects & Reports",
            "Complete EfficientNet ensemble fine-tuning",
        )
        table.add_row(
            "14",
            "📊 Generate Full SSL Reports",
            "Generate PDF reports for all SSL models",
        )
        table.add_row(
            "15",
            "📈 Compare All SSL Architects",
            "Compare ResNet18 vs EfficientNet-B0 SSL results",
        )
        table.add_row("16", "💻 Depicts Your Own Hardware Device", "CPU - GPU results")

        table.add_row("17", "🔄 Recovery Mode", "Skip FT & CV → Build missing parts")

        # ============================================================
        # ✅ AlexNet SSL (NEW)
        # ============================================================
        table.add_row(
            "18",
            "🎯 AlexNet SSL Full (NEW)",
            "BYOL → SimCLR → MAE → Best on AlexNet",
        )
        table.add_row(
            "19",
            "🎯 AlexNet within BYOL & Reports (NEW)",
            "Run AlexNet with BYOL only",
        )
        table.add_row(
            "20",
            "🎯 AlexNet within SimCLR & Reports (NEW)",
            "Run AlexNet with SimCLR only",
        )
        table.add_row(
            "21",
            "🎯 AlexNet within MAE & Reports (NEW)",
            "Run AlexNet with MAE only",
        )
        table.add_row(
            "22",
            "🎯 AlexNet Find Best SSL (NEW)",
            "Load checkpoints → Linear Probe → select winner",
        )
        table.add_row(
            "23",
            "🏆 AlexNet FFT + TTA + CV5 + CI95% (NEW)",
            "Fine-Tune best SSL → TTA → CV5 → CI95% → PDF",
        )
        # ============================================================
        # ✅ MobileNetV2 SSL (NEW)
        # ============================================================
        table.add_row(
            "24",
            "🎯 MobileNetV2 within BYOL & Reports (NEW)",
            "Run MobileNetV2 with BYOL only",
        )
        table.add_row(
            "25",
            "🎯 MobileNetV2 within SimCLR & Reports (NEW)",
            "Run MobileNetV2 with SimCLR only",
        )
        table.add_row(
            "26",
            "🎯 MobileNetV2 within MAE & Reports (NEW)",
            "Run MobileNetV2 with MAE only",
        )
        table.add_row("27", "🎯 MobileNetV2 SSL Full (NEW)", "BYOL → SimCLR → MAE → Best")
        table.add_row("28", "🎯 MobileNetV2 Find Best SSL (NEW)", "Copy best to weightTransformer/")
        table.add_row("29", "🏆 MobileNetV2 FFT + TTA + CV5 + CI95% (NEW)", "24-page PDF")
        # ============================================================
        # ✅ VGG16 SSL (NEW)
        # ============================================================
        table.add_row(
            "30",
            "🎯 VGG16 SSL Full (NEW)",
            "BYOL → SimCLR → MAE → Best on VGG16",
        )
        table.add_row(
            "31",
            "🎯 VGG16 within BYOL & Reports (NEW)",
            "Run VGG16 with BYOL only",
        )
        table.add_row(
            "32",
            "🎯 VGG16 within SimCLR & Reports (NEW)",
            "Run VGG16 with SimCLR only",
        )
        table.add_row(
            "33",
            "🎯 VGG16 within MAE & Reports (NEW)",
            "Run VGG16 with MAE only",
        )
        table.add_row(
            "34",
            "🎯 VGG16 Find Best SSL (NEW)",
            "Copy best to weightTransformer/",
        )
        table.add_row(
            "35",
            "🏆 VGG16 FFT + TTA + CV5 + CI95% (NEW)",
            "24-page PDF",
        )
        # ============================================================
        # ✅ DenseNet SSL (NEW)
        # ============================================================
        table.add_row("36", "🎯 DenseNet SSL Full (NEW)", "BYOL → SimCLR → MAE → Best")
        table.add_row("37", "🎯 DenseNet within BYOL & Reports (NEW)", "BYOL only")
        table.add_row("38", "🎯 DenseNet within SimCLR & Reports (NEW)", "SimCLR only")
        table.add_row("39", "🎯 DenseNet within MAE & Reports (NEW)", "MAE only")
        table.add_row("40", "🎯 DenseNet Find Best SSL (NEW)", "Copy to weightTransformer/")
        table.add_row("41", "🏆 DenseNet FFT + TTA + CV5 + CI95% (NEW)", "24-page PDF")
        table.add_row("42", "🏆 DenseNet Direct FT (ImageNet)", "Skip SSL → Fine-Tune DenseNet121")
        table.add_row("43", "🏆 Find Best Model Among All 4", "AlexNet/DenseNet/MobileNetV2/VGG16 → best_models.pth")

        console.print(table)
    else:
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 Main Menu{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-'*60}{Colors.ENDC}")
        print(f"{Colors.YELLOW} 0{Colors.ENDC}  🚪 Exit")
        print(f"{Colors.YELLOW} 1{Colors.ENDC}  🔄 Traditional Pipeline (wFTAlgorithms)")
        print(f"{Colors.YELLOW} 2{Colors.ENDC}  🧠 ResNet18 SSL Full")
        print(f"{Colors.YELLOW} 3{Colors.ENDC}  🧠 ResNet18 within BYOL & Reports")
        print(f"{Colors.YELLOW} 4{Colors.ENDC}  🧠 ResNet18 within SimCLR & Reports")
        print(f"{Colors.YELLOW} 5{Colors.ENDC}  🧠 ResNet18 within MAE & Reports")
        print(f"{Colors.YELLOW} 6{Colors.ENDC}  🧠 Find best SSL & Transfer to ResNet50 & Reports")
        print(f"{Colors.YELLOW} 7{Colors.ENDC}  🧠 Full-Fine-Tuning Ensemble ResNet & Reports")
        print(f"{Colors.YELLOW} 8{Colors.ENDC}  ⚡ EfficientNet-B0 SSL Full")
        print(f"{Colors.YELLOW} 9{Colors.ENDC}  ⚡ EfficientNet-B0 within BYOL & Reports")
        print(f"{Colors.YELLOW}10{Colors.ENDC}  ⚡ EfficientNet-B0 within SimCLR & Reports")
        print(f"{Colors.YELLOW}11{Colors.ENDC}  ⚡ EfficientNet-B0 within MAE & Reports")
        print(f"{Colors.YELLOW}12{Colors.ENDC}  ⚡ Find best SSL & Transfer to B1 & Reports")
        print(f"{Colors.YELLOW}13{Colors.ENDC}  ⚡ Full-Fine-Tuning Ensemble EfficientNet & Reports")
        print(f"{Colors.YELLOW}14{Colors.ENDC}  📊 Generate Full SSL Reports")
        print(f"{Colors.YELLOW}15{Colors.ENDC}  📈 Compare All SSL Architects")
        print(f"{Colors.YELLOW}16{Colors.ENDC}  💻 Depicts Your Own Hardware Device")
        print(f"{Colors.YELLOW}17{Colors.ENDC}  🔄 Recovery Mode (skip FT & CV)")
        print(f"{Colors.CYAN}{'-'*40}{Colors.ENDC}")
        print(f"{Colors.YELLOW}18{Colors.ENDC}  🎯 AlexNet SSL Full (NEW)")
        print(f"{Colors.YELLOW}19{Colors.ENDC}  🎯 AlexNet within BYOL & Reports (NEW)")
        print(f"{Colors.YELLOW}20{Colors.ENDC}  🎯 AlexNet within SimCLR & Reports (NEW)")
        print(f"{Colors.YELLOW}21{Colors.ENDC}  🎯 AlexNet within MAE & Reports (NEW)")
        print(f"{Colors.YELLOW}22{Colors.ENDC}  🎯 AlexNet Load Checkpoints (NEW)")
        print(f"{Colors.YELLOW}23{Colors.ENDC}  🎯 AlexNet Full Fine Tune (NEW)")
        print(f"{Colors.YELLOW}24{Colors.ENDC}  🎯 MobileNetV2 within BYOL & Reports (NEW)")
        print(f"{Colors.YELLOW}25{Colors.ENDC}  🎯 MobileNetV2 within SimCLR & Reports (NEW)")
        print(f"{Colors.YELLOW}26{Colors.ENDC}  🎯 MobileNetV2 within MAE & Reports (NEW)")
        print(f"{Colors.YELLOW}27{Colors.ENDC}  🎯 MobileNetV2 SSL Full runtime (NEW)")
        print(f"{Colors.YELLOW}28{Colors.ENDC}  🎯 MobileNetV2 Find Best SSL (NEW)")
        print(f"{Colors.YELLOW}29{Colors.ENDC}  🎯 MobileNetV2 Full Fine Tune (NEW)")
        print(f"{Colors.YELLOW}30{Colors.ENDC}  🎯 VGG16 SSL Full (NEW)")
        print(f"{Colors.YELLOW}31{Colors.ENDC}  🎯 VGG16 within BYOL & Reports (NEW)")
        print(f"{Colors.YELLOW}32{Colors.ENDC}  🎯 VGG16 within SimCLR & Reports (NEW)")
        print(f"{Colors.YELLOW}33{Colors.ENDC}  🎯 VGG16 within MAE & Reports (NEW)")
        print(f"{Colors.YELLOW}34{Colors.ENDC}  🎯 VGG16 Find Best SSL (NEW)")
        print(f"{Colors.YELLOW}35{Colors.ENDC}  🏆 VGG16 FFT + TTA + CV5 + CI95% (NEW)")
        print(f"{Colors.CYAN}{'-'*60}{Colors.ENDC}")
        print(f"{Colors.YELLOW}36{Colors.ENDC}  🎯 DenseNet SSL Full (NEW)")
        print(f"{Colors.YELLOW}37{Colors.ENDC}  🎯 DenseNet within BYOL & Reports (NEW)")
        print(f"{Colors.YELLOW}38{Colors.ENDC}  🎯 DenseNet within SimCLR & Reports (NEW)")
        print(f"{Colors.YELLOW}39{Colors.ENDC}  🎯 DenseNet within MAE & Reports (NEW)")
        print(f"{Colors.YELLOW}40{Colors.ENDC}  🎯 DenseNet Find Best SSL (NEW)")
        print(f"{Colors.YELLOW}41{Colors.ENDC}  🏆 DenseNet FFT + TTA + CV5 + CI95% (NEW)")
        print(f"{Colors.YELLOW}42{Colors.ENDC}  🏆 Direct DenseNet FFT + TTA + CV5 + CI95% (NEW)")
        print(f"{Colors.YELLOW}43{Colors.ENDC}  🏆 Find Best models final for KD (NEW)")
        print(f"{Colors.CYAN}{'-' * 60}{Colors.ENDC}")


# ================================================================
# ✅ Get User Choice
# ================================================================


def get_choice():
    """Get user choice from menu."""
    if HAS_RICH:
        return Prompt.ask(
            "\n[bold cyan]👉 Enter your choice[/bold cyan]",
            choices=[
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
                "32",
                "33",
                "34",
                "35",
                "36",
                "37",
                "38",
                "39",
                "40",
                "41",
                "42",
                "43",
            ],
        )
    else:
        return input(f"\n{Colors.BOLD}👉 Enter your choice (0-43): {Colors.ENDC}")


# ================================================================
# ✅ SSL Pipeline Runner
# ================================================================
# ================================================================
# ✅ SSL Pipeline Runner
# ================================================================
# ================================================================
# ✅ ResNet18 Full SSL Pipeline (BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune)
# ================================================================


def run_resnet18_ssl():
    """
    Run ResNet18 Full SSL Pipeline with user configuration.
    This runs: BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune
    """
    if HAS_RICH:
        console.print(
            Panel.fit(
                "🧠 ResNet18 Full SSL Pipeline",
                subtitle="BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune → ResNet50",
                border_style="green",
            )
        )
    else:
        print(f"\n{Colors.GREEN}🧠 ResNet18 Full SSL Pipeline{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")
        print("📌 BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune → ResNet50")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    # ✅ FIX: استفاده از دیتاست گسترده
    default_image_path = os.path.join("../../../../finalReports", "cropped")
    if not os.path.exists(default_image_path):
        print(f"⚠️ {default_image_path} not found. Using cropped/ instead.")
        default_image_path = os.path.join("../../../../finalReports", "cropped")

    # Get image paths
    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
    else:
        image_path = (
            input(
                f"\n{Colors.BLUE}📁 Enter path to images directory (default={default_image_path}): {Colors.ENDC}"
            )
            or default_image_path
        )

    # Get epochs
    if HAS_RICH:
        total_epochs = int(Prompt.ask("⏳ Number of epochs per SSL method", default="30"))
    else:
        total_epochs = int(
            input(f"\n{Colors.BLUE}⏳ Number of epochs per SSL method (default=30): {Colors.ENDC}")
            or "30"
        )

    # Get batch size
    if HAS_RICH:
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
    else:
        batch_size = int(input(f"{Colors.BLUE}📦 Batch size (default=32): {Colors.ENDC}") or "32")

    # Get optimizer
    if HAS_RICH:
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )
    else:
        print(f"\n{Colors.BLUE}🔧 Select optimizer:{Colors.ENDC}")
        print(f"   {Colors.YELLOW}1{Colors.ENDC}  AdamW (Default)")
        print(f"   {Colors.YELLOW}2{Colors.ENDC}  RAdam")
        print(f"   {Colors.YELLOW}3{Colors.ENDC}  Ranger")
        choice = input(f"{Colors.BOLD}👉 Enter choice (1-3): {Colors.ENDC}")
        opt_map = {"1": "adamw", "2": "radam", "3": "ranger"}
        optimizer_name = opt_map.get(choice, "adamw")

    # Confirm
    if HAS_RICH:
        if not Confirm.ask(
            f"\n🚀 Start ResNet18 Full SSL Pipeline (BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune)?",
            default=True,
        ):
            return
    else:
        confirm = input(
            f"\n{Colors.WARNING}🚀 Start ResNet18 Full SSL Pipeline? (y/n): {Colors.ENDC}"
        )
        if confirm.lower() != "y":
            return

    # Run pipeline
    try:
        # Get image paths
        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            image_paths.extend(
                [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(ext)]
            )

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found in the specified directory![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found in the specified directory!{Colors.ENDC}")
            return

        if HAS_RICH:
            console.print(f"[green]✅ Found {len(image_paths)} images[/green]")
            console.print(
                f"[cyan]🔬 Running Full SSL Pipeline with {optimizer_name.upper()} optimizer...[/cyan]"
            )
            console.print("[dim]📌 This will take time. Please be patient...[/dim]")

        # Create pipeline
        from src.pipeLine.resNet18SSL import create_resnet18_ssl_pipeline

        pipeline = create_resnet18_ssl_pipeline(
            batch_size=batch_size,
            total_epochs=total_epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
            drop_last=True,
            verbose=True,
            optimizer_name=optimizer_name,
            use_imgaug=False,
            use_kornia=True,
            enhance_ultrasound=True,
        )

        # Run full SSL pipeline
        results = pipeline.run_full_ssl_pipeline(image_paths)

        # ✅ RUN CROSS-VALIDATION AFTER FULL PIPELINE
        from src.pipeLine.CV.crossValidationRunner import run_cv_after_model

        if HAS_RICH:
            console.print("\n[cyan]🔬 Running 5-Fold Cross-Validation on final model...[/cyan]")
        else:
            print(
                f"\n{Colors.CYAN}🔬 Running 5-Fold Cross-Validation on final model...{Colors.ENDC}"
            )

        final_model_path = results.get("resnet50_final_model")
        if final_model_path and os.path.exists(final_model_path):
            cv_result = run_cv_after_model(
                model_name="resnet50",
                method=results.get("best_method", "byol"),
                model_path=final_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
            )
            results["cross_validation"] = cv_result
        else:
            if HAS_RICH:
                console.print("[yellow]⚠️ Final model not found. Skipping CV...[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ Final model not found. Skipping CV...{Colors.ENDC}")

        if HAS_RICH:
            console.print("\n[bold green]✅ ResNet18 Full SSL Pipeline Complete![/bold green]")
            console.print(f"   Best Method: {results.get('best_method', 'N/A').upper()}")
            console.print(f"   Best Loss: {results.get('best_loss', 0):.4f}")

            if "resnet50_finetuned" in results and results["resnet50_finetuned"] != "failed":
                resnet50_loss = results["resnet50_finetuned"].get("best_loss", 0)
                console.print(f"   ResNet50 Fine-Tuned Loss: {resnet50_loss:.4f}")

            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                console.print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )

            if "resnet50_final_model" in results:
                console.print(f"   Final Model: {results['resnet50_final_model']}")

            console.print("\n[dim]📊 Reports saved in SSL_Reporter directory[/dim]")
        else:
            print(f"\n{Colors.GREEN}✅ ResNet18 Full SSL Pipeline Complete!{Colors.ENDC}")
            print(f"   Best Method: {results.get('best_method', 'N/A').upper()}")
            print(f"   Best Loss: {results.get('best_loss', 0):.4f}")

            if "resnet50_finetuned" in results and results["resnet50_finetuned"] != "failed":
                resnet50_loss = results["resnet50_finetuned"].get("best_loss", 0)
                print(f"   ResNet50 Fine-Tuned Loss: {resnet50_loss:.4f}")

            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )

            if "resnet50_final_model" in results:
                print(f"   Final Model: {results['resnet50_final_model']}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback

        traceback.print_exc()


# ================================================================
# ✅ EfficientNet-B0 SSL Pipeline (FULL SSL PIPELINE)
# ================================================================
# ================================================================
# ✅ EfficientNet-B0 Full SSL Pipeline (BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune)
# ================================================================


def run_efficientnet_b0_ssl():
    """
    Run EfficientNet-B0 Full SSL Pipeline with user configuration.
    This runs: BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune
    """
    if HAS_RICH:
        console.print(
            Panel.fit(
                "⚡ EfficientNet-B0 Full SSL Pipeline",
                subtitle="BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune → B1",
                border_style="green",
            )
        )
    else:
        print(f"\n{Colors.GREEN}⚡ EfficientNet-B0 Full SSL Pipeline{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")
        print("📌 BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune → B1")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    # ✅ مسیر پیش‌فرض به cropped
    default_image_path = os.path.join("../../../../finalReports", "cropped")

    # Get image paths
    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
    else:
        image_path = (
            input(
                f"\n{Colors.BLUE}📁 Enter path to images directory (default={default_image_path}): {Colors.ENDC}"
            )
            or default_image_path
        )

    # Get epochs
    if HAS_RICH:
        total_epochs = int(Prompt.ask("⏳ Number of epochs per SSL method", default="30"))
    else:
        total_epochs = int(
            input(f"\n{Colors.BLUE}⏳ Number of epochs per SSL method (default=30): {Colors.ENDC}")
            or "30"
        )

    # Get batch size
    if HAS_RICH:
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
    else:
        batch_size = int(input(f"{Colors.BLUE}📦 Batch size (default=32): {Colors.ENDC}") or "32")

    # Get optimizer
    if HAS_RICH:
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )
    else:
        print(f"\n{Colors.BLUE}🔧 Select optimizer:{Colors.ENDC}")
        print(f"   {Colors.YELLOW}1{Colors.ENDC}  AdamW (Default)")
        print(f"   {Colors.YELLOW}2{Colors.ENDC}  RAdam")
        print(f"   {Colors.YELLOW}3{Colors.ENDC}  Ranger")
        choice = input(f"{Colors.BOLD}👉 Enter choice (1-3): {Colors.ENDC}")
        opt_map = {"1": "adamw", "2": "radam", "3": "ranger"}
        optimizer_name = opt_map.get(choice, "adamw")

    # Confirm
    if HAS_RICH:
        if not Confirm.ask(f"\n🚀 Start EfficientNet-B0 Full SSL Pipeline?", default=True):
            return
    else:
        confirm = input(
            f"\n{Colors.WARNING}🚀 Start EfficientNet-B0 Full SSL Pipeline? (y/n): {Colors.ENDC}"
        )
        if confirm.lower() != "y":
            return

    # Run pipeline
    try:
        # Get image paths
        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            image_paths.extend(
                [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(ext)]
            )

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found in the specified directory![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found in the specified directory!{Colors.ENDC}")
            return

        if HAS_RICH:
            console.print(f"[green]✅ Found {len(image_paths)} images[/green]")
            console.print(
                f"[cyan]🔬 Running Full SSL Pipeline with {optimizer_name.upper()} optimizer...[/cyan]"
            )
            console.print("[dim]📌 This will take time. Please be patient...[/dim]")

        # Create pipeline
        from src.pipeLine.Architects.EfficientNet.B0.efficientNetB0SSL import create_efficientnet_b0_ssl_pipeline

        pipeline = create_efficientnet_b0_ssl_pipeline(
            batch_size=batch_size,
            total_epochs=total_epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
            optimizer_name=optimizer_name,
            use_imgaug=False,
            use_kornia=True,
            enhance_ultrasound=True,
        )

        # Run full SSL pipeline
        results = pipeline.run_full_ssl_pipeline(image_paths)

        # ✅ RUN CROSS-VALIDATION AFTER FULL PIPELINE
        from src.pipeLine.CV.crossValidationRunner import run_cv_after_model

        if HAS_RICH:
            console.print("\n[cyan]🔬 Running 5-Fold Cross-Validation on final model...[/cyan]")
        else:
            print(
                f"\n{Colors.CYAN}🔬 Running 5-Fold Cross-Validation on final model...{Colors.ENDC}"
            )

        final_model_path = results.get("b1_final_model")
        if final_model_path and os.path.exists(final_model_path):
            cv_result = run_cv_after_model(
                model_name="efficientnet_b1",
                method=results.get("best_method", "byol"),
                model_path=final_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
            )
            results["cross_validation"] = cv_result
        else:
            if HAS_RICH:
                console.print("[yellow]⚠️ Final model not found. Skipping CV...[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ Final model not found. Skipping CV...{Colors.ENDC}")

        if HAS_RICH:
            console.print(
                "\n[bold green]✅ EfficientNet-B0 Full SSL Pipeline Complete![/bold green]"
            )
            console.print(f"   Best Method: {results.get('best_method', 'N/A').upper()}")
            console.print(f"   Best Loss: {results.get('best_loss', 0):.4f}")

            if "b1_finetuned" in results and results["b1_finetuned"] != "failed":
                b1_loss = results["b1_finetuned"].get("best_loss", 0)
                console.print(f"   B1 Fine-Tuned Loss: {b1_loss:.4f}")

            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                console.print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )

            if "b1_final_model" in results:
                console.print(f"   Final Model: {results['b1_final_model']}")

            console.print("\n[dim]📊 Reports saved in SSL_Reporter directory[/dim]")
        else:
            print(f"\n{Colors.GREEN}✅ EfficientNet-B0 Full SSL Pipeline Complete!{Colors.ENDC}")
            print(f"   Best Method: {results.get('best_method', 'N/A').upper()}")
            print(f"   Best Loss: {results.get('best_loss', 0):.4f}")

            if "b1_finetuned" in results and results["b1_finetuned"] != "failed":
                b1_loss = results["b1_finetuned"].get("best_loss", 0)
                print(f"   B1 Fine-Tuned Loss: {b1_loss:.4f}")

            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )

            if "b1_final_model" in results:
                print(f"   Final Model: {results['b1_final_model']}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback

        traceback.print_exc()

# ================================================================
# ✅ AlexNet Full SSL Pipeline (NEW)
# ================================================================

def run_alexnet_ssl():
    """
    Run AlexNet Full SSL Pipeline with user configuration.

    Flow: BYOL → SimCLR → MAE → Best → (Optional: Transfer to ResNet50)
    """
    if not HAS_ALEXNET_SSL:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ AlexNet SSL module not available![/bold red]\n"
                "[yellow]Make sure src/pipeLine/alexnetSSL.py exists.[/yellow]",
                border_style="red",
            ))
        else:
            print(f"{Colors.FAIL}❌ AlexNet SSL module not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(
            Panel.fit(
                "🧠 AlexNet Full SSL Pipeline",
                subtitle="BYOL → SimCLR → MAE → Best → (Optional Transfer)",
                border_style="green",
            )
        )
    else:
        print(f"\n{Colors.GREEN}🧠 AlexNet Full SSL Pipeline{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")
        print("📌 BYOL → SimCLR → MAE → Best")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    # ✅ Default image path
    default_image_path = os.path.join("../../../../finalReports", "cropped")
    if not os.path.exists(default_image_path):
        print(f"⚠️ {default_image_path} not found. Using cropped/ instead.")
        default_image_path = os.path.join("../../../../finalReports", "cropped")

    # ✅ Get configuration from user
    if HAS_RICH:
        image_path = Prompt.ask(
            "📁 Enter path to images directory",
            default=default_image_path,
        )
        total_epochs = int(Prompt.ask(
            "⏳ Number of epochs per SSL method",
            default="40",
        ))
        batch_size = int(Prompt.ask(
            "📦 Batch size",
            default="32",
        ))
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer",
            choices=["adamw", "radam", "ranger"],
            default="adamw",
        )
        run_transfer = Confirm.ask(
            "🔄 Run weight transfer + fine-tune to ResNet50?",
            default=False,
        )

        if not Confirm.ask(
            "\n🚀 Start AlexNet Full SSL Pipeline (BYOL → SimCLR → MAE → Best)?",
            default=True,
        ):
            return
    else:
        image_path = (
            input(
                f"\n{Colors.BLUE}📁 Enter path to images directory "
                f"(default={default_image_path}): {Colors.ENDC}"
            )
            or default_image_path
        )
        total_epochs = int(
            input(f"{Colors.BLUE}⏳ Number of epochs per SSL method (default=40): {Colors.ENDC}")
            or "40"
        )
        batch_size = int(
            input(f"{Colors.BLUE}📦 Batch size (default=32): {Colors.ENDC}")
            or "32"
        )
        print(f"{Colors.BLUE}🔧 Select optimizer:{Colors.ENDC}")
        print(f"   1. AdamW (Default)\n   2. RAdam\n   3. Ranger")
        opt_choice = input(f"{Colors.BOLD}👉 Enter choice (1-3): {Colors.ENDC}")
        opt_map = {"1": "adamw", "2": "radam", "3": "ranger"}
        optimizer_name = opt_map.get(opt_choice, "adamw")

        transfer_choice = input(
            f"{Colors.BLUE}🔄 Run weight transfer to ResNet50? (y/n) (n): {Colors.ENDC}"
        ) or "n"
        run_transfer = transfer_choice.lower() == "y"

        confirm = input(
            f"\n{Colors.WARNING}🚀 Start AlexNet Full SSL Pipeline? (y/n): {Colors.ENDC}"
        )
        if confirm.lower() != "y":
            return

    # ✅ Collect image paths
    try:
        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            if os.path.exists(image_path):
                image_paths.extend([
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.endswith(ext)
                ])

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found in the specified directory![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found!{Colors.ENDC}")
            return

        if HAS_RICH:
            console.print(f"[green]✅ Found {len(image_paths)} images[/green]")
            console.print(
                f"[cyan]🔬 Running Full SSL Pipeline with {optimizer_name.upper()} optimizer...[/cyan]"
            )
            console.print("[dim]📌 This will take time. Please be patient...[/dim]")
        else:
            print(f"{Colors.GREEN}✅ Found {len(image_paths)} images{Colors.ENDC}")

        # ✅ Create pipeline
        from src.pipeLine.Architects.AlexnetFT.alexnetSSL import create_alexnet_ssl_pipeline

        pipeline = create_alexnet_ssl_pipeline(
            method="byol",
            batch_size=batch_size,
            total_epochs=total_epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
            optimizer_name=optimizer_name,
            use_kornia=True,
            enhance_ultrasound=True,
        )

        # ✅ Run full SSL pipeline
        results = pipeline.run_full_ssl_pipeline(
            image_paths=image_paths,
            validation_paths=None,
            save_interval=10,
        )

        # ✅ Generate PDF report for the best method
        best_method = results.get("best_method", "byol")
        if HAS_RICH:
            console.print(f"\n[cyan]📄 Generating PDF report for best method ({best_method.upper()})...[/cyan]")
        else:
            print(f"\n{Colors.CYAN}📄 Generating PDF report...{Colors.ENDC}")

        try:
            # Switch to best pipeline for reporting
            if "method_results" in results and best_method in results["method_results"]:
                # The pipeline object is inside method_results
                pass

            pdf_path = pipeline.generate_report()
            if HAS_RICH:
                console.print(f"[green]✅ PDF Report: {pdf_path}[/green]")
            else:
                print(f"{Colors.GREEN}✅ PDF Report: {pdf_path}{Colors.ENDC}")
        except Exception as e:
            if HAS_RICH:
                console.print(f"[yellow]⚠️ PDF generation failed: {e}[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ PDF generation failed: {e}{Colors.ENDC}")

        # ✅ Optional: Transfer to ResNet50
        if run_transfer:
            if HAS_RICH:
                console.print("\n[cyan]🔄 Running weight transfer to ResNet50...[/cyan]")
            try:
                from src.pipeLine.Architects.AlexnetFT.alexnetSSL import AlexNetSSL
                # Transfer logic is handled in run_full_ssl_pipeline already
                # For now, we just note it
                if HAS_RICH:
                    console.print("[yellow]⚠️ Transfer is done inside run_full_ssl_pipeline.[/yellow]")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[yellow]⚠️ Transfer failed: {e}[/yellow]")

        # ✅ Final summary
        if HAS_RICH:
            console.print("\n[bold green]✅ AlexNet Full SSL Pipeline Complete![/bold green]")
            console.print(f"   Best Method: {best_method.upper()}")
            console.print(f"   Best LP Accuracy: {results.get('best_linear_probe_accuracy', 0):.4f}")

            # Show all methods comparison
            lp_results = results.get("linear_probe_results", {})
            if lp_results:
                console.print("\n[bold cyan]📊 All Methods Comparison:[/bold cyan]")
                for m, res in lp_results.items():
                    console.print(
                        f"   {m.upper():<8} LP Acc: {res.get('linear_probe_accuracy', 0):.4f}"
                    )

            console.print("\n[dim]📂 Outputs:[/dim]")
            console.print(f"   [dim]Checkpoints: finalReports/selfSupervisedLearning/AlexNet/alexnet/SSL_checkpoints/[/dim]")
            console.print(f"   [dim]PDF Report: finalReports/selfSupervisedLearning/AlexNet/alexnet/SSL_checkpoints/alexnet_ssl_byol_report.pdf[/dim]")
        else:
            print(f"\n{Colors.GREEN}✅ AlexNet Full SSL Pipeline Complete!{Colors.ENDC}")
            print(f"   Best Method: {best_method.upper()}")
            print(f"   Best LP Accuracy: {results.get('best_linear_probe_accuracy', 0):.4f}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ Generate All SSL Reports
# ================================================================


def generate_all_reports():
    """
    ✅ v2.0: Dual Model Comparison Report

    Generates a comprehensive PDF report comparing:
    - ResNet50 Fine-Tuned (from simclr)
    - EfficientNet-B1 Fine-Tuned (from BYOL)
    """
    if not HAS_DUAL_REPORTER:
        if HAS_RICH:
            console.print("[red]❌ Dual Model Reporter not available![/red]")
            console.print("[yellow]   Make sure src/pipeLine/dualModelReporter.py exists[/yellow]")
        else:
            print(f"{Colors.FAIL}❌ Dual Model Reporter not available!{Colors.ENDC}")
            print(f"{Colors.WARNING}   Make sure src/pipeLine/dualModelReporter.py exists{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(
            Panel.fit(
                "📊 Dual Model Comparison Report",
                subtitle="ResNet50 (SimCLR) vs EfficientNet-B1 (BYOL)",
                border_style="cyan",
            )
        )
    else:
        print(f"\n{Colors.CYAN}📊 Dual Model Comparison Report{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}")
        print("📌 ResNet50 (SimCLR) vs EfficientNet-B1 (BYOL)")
        print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}")

    # ✅ Show model paths
    if HAS_RICH:
        console.print("\n[bold cyan]Architects to compare:[/bold cyan]")
        console.print(f"   1️⃣  ResNet50:  [dim]{DUAL_RESNET50_PATH}[/dim]")
        console.print(f"   2️⃣  B1:        [dim]{DUAL_B1_PATH}[/dim]")
        console.print(f"\n[bold cyan]PDF output:[/bold cyan] [dim]{DUAL_PDF_PATH}[/dim]")
    else:
        print(f"\n{Colors.BOLD}Architects to compare:{Colors.ENDC}")
        print(f"   1. ResNet50: {DUAL_RESNET50_PATH}")
        print(f"   2. B1:       {DUAL_B1_PATH}")
        print(f"\nPDF output: {DUAL_PDF_PATH}")

    # ✅ Confirm
    if HAS_RICH:
        if not Confirm.ask("\n🚀 Generate dual model comparison report?", default=True):
            return
    else:
        confirm = input(f"\n{Colors.WARNING}🚀 Generate report? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            return

    # ✅ Run
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = run_dual_model_report(device=device)

        if results and results.get("status") == "success":
            if HAS_RICH:
                console.print("\n[bold green]✅ Dual Model Report generated![/bold green]")
                console.print(f"   📄 PDF: [cyan]{results['pdf_path']}[/cyan]")
                console.print(f"   📊 Metrics: [dim]{results['metrics_json']}[/dim]")
            else:
                print(f"\n{Colors.GREEN}✅ Report generated!{Colors.ENDC}")
                print(f"   PDF: {results['pdf_path']}")
                print(f"   Metrics: {results['metrics_json']}")
        else:
            if HAS_RICH:
                console.print("[yellow]⚠️ Report generation returned no results.[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ Report generation returned no results.{Colors.ENDC}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ Compare All SSL Architects
# ================================================================


def compare_all_ssl_models():
    """
    ✅ v2.0: SSL Dual Comparison — 6 Architects in 3 Pairs

    Compare 6 SSL encoders in 3 pairwise comparisons:
    - Pair 1 (BYOL):   ResNet18-BYOL vs EfficientNet-B0-BYOL
    - Pair 2 (MAE):    ResNet18-MAE  vs EfficientNet-B0-MAE
    - Pair 3 (SimCLR): ResNet18-SimCLR vs EfficientNet-B0-SimCLR
    """
    if not HAS_SSL_DUAL:
        if HAS_RICH:
            console.print("[red]❌ SSL Dual Comparison not available![/red]")
            console.print("[yellow]   Make sure src/pipeLine/sslDualComparison.py exists[/yellow]")
        else:
            print(f"{Colors.FAIL}❌ SSL Dual Comparison not available!{Colors.ENDC}")
            print(f"{Colors.WARNING}   Make sure src/pipeLine/sslDualComparison.py exists{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(
            Panel.fit(
                "📈 SSL Dual Comparison Report",
                subtitle="6 SSL Encoders → 3 Pairwise Comparisons (BYOL, MAE, SimCLR)",
                border_style="magenta",
            )
        )
    else:
        print(f"\n{Colors.MAGENTA}📈 SSL Dual Comparison Report{Colors.ENDC}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.ENDC}")
        print("📌 6 SSL Encoders → 3 Pairwise Comparisons")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.ENDC}")

    # ✅ Show models
    if HAS_RICH:
        console.print("\n[bold cyan]6 Architects to compare (3 pairs):[/bold cyan]")
        console.print("[bold yellow]Pair 1 (BYOL):[/bold yellow]")
        console.print(f"   1️⃣  {SSL_MODEL_PATHS.get('resnet18_byol', 'N/A')}")
        console.print(f"   2️⃣  {SSL_MODEL_PATHS.get('b0_byol', 'N/A')}")
        console.print("[bold yellow]Pair 2 (MAE):[/bold yellow]")
        console.print(f"   3️⃣  {SSL_MODEL_PATHS.get('resnet18_mae', 'N/A')}")
        console.print(f"   4️⃣  {SSL_MODEL_PATHS.get('b0_mae', 'N/A')}")
        console.print("[bold yellow]Pair 3 (SimCLR):[/bold yellow]")
        console.print(f"   5️⃣  {SSL_MODEL_PATHS.get('resnet18_simclr', 'N/A')}")
        console.print(f"   6️⃣  {SSL_MODEL_PATHS.get('b0_simclr', 'N/A')}")
        console.print(f"\n[bold cyan]PDF output:[/bold cyan] [dim]{SSL_DUAL_PDF_PATH}[/dim]")
    else:
        print(f"\n{Colors.BOLD}6 Architects (3 pairs):{Colors.ENDC}")
        print("Pair 1 (BYOL):")
        print(f"   1. {SSL_MODEL_PATHS.get('resnet18_byol', 'N/A')}")
        print(f"   2. {SSL_MODEL_PATHS.get('b0_byol', 'N/A')}")
        print("Pair 2 (MAE):")
        print(f"   3. {SSL_MODEL_PATHS.get('resnet18_mae', 'N/A')}")
        print(f"   4. {SSL_MODEL_PATHS.get('b0_mae', 'N/A')}")
        print("Pair 3 (SimCLR):")
        print(f"   5. {SSL_MODEL_PATHS.get('resnet18_simclr', 'N/A')}")
        print(f"   6. {SSL_MODEL_PATHS.get('b0_simclr', 'N/A')}")

    # ✅ Confirm
    if HAS_RICH:
        if not Confirm.ask("\n🚀 Generate SSL dual comparison report?", default=True):
            return
    else:
        confirm = input(f"\n{Colors.WARNING}🚀 Generate report? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            return

    # ✅ Run
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = run_ssl_dual_comparison(device=device)

        if results and results.get("status") == "success":
            if HAS_RICH:
                console.print("\n[bold green]✅ SSL Dual Comparison generated![/bold green]")
                console.print(f"   📄 PDF: [cyan]{results['pdf_path']}[/cyan]")
                console.print(f"   📊 Metrics: [dim]{results['metrics_json']}[/dim]")
            else:
                print(f"\n{Colors.GREEN}✅ Report generated!{Colors.ENDC}")
                print(f"   PDF: {results['pdf_path']}")
                print(f"   Metrics: {results['metrics_json']}")
        else:
            if HAS_RICH:
                console.print("[yellow]⚠️ Report generation returned no results.[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ Report generation returned no results.{Colors.ENDC}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ RESNET18 SSL - SINGLE METHOD (BYOL / SimCLR / MAE)
# ================================================================


def run_resnet18_ssl_single(method: str = "byol"):
    """
    Run ResNet18 SSL with a single method (BYOL / SimCLR / MAE) and generate report.
    After training, runs 5-Fold Cross-Validation and adds results to report.

    Args:
        method: 'byol', 'simclr', or 'mae'
    """
    if HAS_RICH:
        console.print(Panel.fit(f"🧠 ResNet18 SSL - {method.upper()}", border_style="green"))
    else:
        print(f"\n{Colors.GREEN}🧠 ResNet18 SSL - {method.upper()}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    # ✅ FIX: استفاده از دیتاست گسترده
    default_image_path = os.path.join("../../../../finalReports", "cropped")
    if not os.path.exists(default_image_path):
        print(f"⚠️ {default_image_path} not found. Using cropped/ instead.")
        default_image_path = os.path.join("../../../../finalReports", "cropped")

    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
        total_epochs = int(Prompt.ask("⏳ Number of epochs", default="40"))
        batch_size = int(Prompt.ask("📦 Batch size", default="64"))
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )
        lr = float(Prompt.ask("📊 Learning Rate", default="0.0002"))

        if not Confirm.ask(f"\n🚀 Start ResNet18 SSL with {method.upper()}?", default=True):
            return
    else:
        image_path = (
            input(
                f"\n{Colors.BLUE}📁 Enter path to images directory (default={default_image_path}): {Colors.ENDC}"
            )
            or default_image_path
        )
        total_epochs = int(
            input(f"{Colors.BLUE}⏳ Number of epochs (default=60): {Colors.ENDC}") or "60"
        )
        batch_size = int(input(f"{Colors.BLUE}📦 Batch size (default=32): {Colors.ENDC}") or "32")
        lr = float(input("📊 Learning Rate (default=0.001): ") or "0.001")
        print(f"{Colors.BLUE}🔧 Select optimizer:{Colors.ENDC}")
        print(f"   1. AdamW (Default)\n   2. RAdam\n   3. Ranger")
        opt_choice = input(f"{Colors.BOLD}👉 Enter choice (1-3): {Colors.ENDC}")
        opt_map = {"1": "adamw", "2": "radam", "3": "ranger"}
        optimizer_name = opt_map.get(opt_choice, "adamw")

        confirm = input(
            f"\n{Colors.WARNING}🚀 Start ResNet18 SSL with {method.upper()}? (y/n): {Colors.ENDC}"
        )
        if confirm.lower() != "y":
            return

    try:
        from src.pipeLine.resNet18SSL import create_resnet18_ssl_pipeline

        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            image_paths.extend(
                [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(ext)]
            )

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found!{Colors.ENDC}")
            return

        pipeline = create_resnet18_ssl_pipeline(
            method=method,
            batch_size=batch_size,
            total_epochs=total_epochs,
            lr=lr,
            device="cuda" if torch.cuda.is_available() else "cpu",
            drop_last=True,
            verbose=True,
            optimizer_name=optimizer_name,
            use_mixed_precision=True,
        )

        results = pipeline.pretrain(image_paths)

        # ✅ RUN 5-FOLD CROSS-VALIDATION
        from src.pipeLine.CV.crossValidationRunner import run_cv_after_model

        if HAS_RICH:
            console.print("\n[cyan]🔬 Running 5-Fold Cross-Validation on trained model...[/cyan]")
        else:
            print(
                f"\n{Colors.CYAN}🔬 Running 5-Fold Cross-Validation on trained model...{Colors.ENDC}"
            )

        # ✅ FIX: مسیر درست — مدل‌ها در architecture/{method}/ ذخیره می‌شوند
        model_path = os.path.join(
            RESNET18_SSL_DIR,
            "architecture",
            method,
            f"resnet18_best_update_{method}.pth",
        )

        if os.path.exists(model_path):
            cv_result = run_cv_after_model(
                model_name="resnet18",
                method=method,
                model_path=model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
            )
            results["cross_validation"] = cv_result
        else:
            # ✅ اگر best پیدا نشد، از final استفاده کن
            final_path = os.path.join(
                RESNET18_SSL_DIR,
                "architecture",
                method,
                f"resnet18_final_update_{method}.pth",
            )
            if os.path.exists(final_path):
                if HAS_RICH:
                    console.print(f"[yellow]⚠️ Best model not found, using final: {final_path}[/yellow]")
                cv_result = run_cv_after_model(
                    model_name="resnet18",
                    method=method,
                    model_path=final_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=True,
                )
                results["cross_validation"] = cv_result
            else:
                if HAS_RICH:
                    console.print("[yellow]⚠️ Model not found. Skipping CV...[/yellow]")
                else:
                    print(f"{Colors.WARNING}⚠️ Model not found. Skipping CV...{Colors.ENDC}")

        # ✅ FIX: ساخت labeled_loader قبل از generate_report
        labeled_loader = None
        try:
            from src.self_supervised.ProbabilityAlgorithms.linear_probe import build_labeled_eval_loader

            labeled_loader = build_labeled_eval_loader(
                labels_csv="finalReports/labels.csv",
                image_dir="../../../../finalReports/cropped",
                input_size=pipeline.input_size,
                batch_size=32,
                max_per_class=50,
            )
            print("✅ labeled_loader built for report generation")
        except Exception as e:
            print(f"⚠️ Could not build labeled_loader: {e}")

        # ✅ Generate report with labeled_loader
        report_path = pipeline.generate_report(labeled_loader=labeled_loader)

        if HAS_RICH:
            console.print(f"\n[bold green]✅ ResNet18 {method.upper()} SSL Complete![/bold green]")
            console.print(f"   Best Loss: {results['best_loss']:.4f}")
            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                console.print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )
            console.print(f"   📊 Report: {report_path}")
        else:
            print(f"\n{Colors.GREEN}✅ ResNet18 {method.upper()} SSL Complete!{Colors.ENDC}")
            print(f"   Best Loss: {results['best_loss']:.4f}")
            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )
            print(f"   📊 Report: {report_path}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback

        traceback.print_exc()


# ================================================================
# ✅ RESNET18 - FIND BEST SSL & TRANSFER TO RESNET50
# ================================================================


def run_resnet18_best_transfer():
    """
    Find best SSL method on ResNet18 (FROM EXISTING CHECKPOINTS),
    then transfer weights to a COPY of the trained ResNet50 (from checkpoints),
    and save the transferred model in weightTransformer/.
    """
    if HAS_RICH:
        console.print(
            Panel.fit(
                "🧠 ResNet18 SSL → ResNet50 Weight Transfer",
                subtitle="Load Checkpoints → Compare → Transfer to ResNet50 Copy",
                border_style="cyan",
            )
        )
    else:
        print(f"\n{Colors.CYAN}🧠 ResNet18 SSL → ResNet50 Weight Transfer{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")

    try:
        from src.pipeLine.resNet18SSL import create_resnet18_ssl_pipeline
        from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
            evaluate_with_linear_probe,
            build_labeled_eval_loader,
        )

        # ============================================================
        # ✅ 1. پیدا کردن checkpoint های SSL موجود
        # ============================================================
        methods = ["byol", "simclr", "mae"]
        method_results = {}

        print("\n" + "=" * 80)
        print("🔍 STEP 1: Looking for existing SSL checkpoints...")
        print("=" * 80)

        for method in methods:
            best_path = os.path.join(
                RESNET18_SSL_DIR, "architecture", method,
                f"resnet18_best_update_{method}.pth",
            )
            final_path = os.path.join(
                RESNET18_SSL_DIR, "architecture", method,
                f"resnet18_final_update_{method}.pth",
            )
            old_best_path = os.path.join(
                RESNET18_SSL_DIR, f"resnet18_ssl_{method}_encoder.pth",
            )

            model_path = None
            for p in [best_path, final_path, old_best_path]:
                if os.path.exists(p):
                    model_path = p
                    break

            if model_path is None:
                print(f"⚠️  {method.upper()}: No checkpoint found — SKIPPING")
                continue

            print(f"✅ {method.upper()}: Found: {model_path}")
            method_results[method] = {"model_path": model_path}

        if len(method_results) == 0:
            print("\n❌ No SSL checkpoints found!")
            return

        # ============================================================
        # ✅ 2. Load کردن هر encoder + Linear Probe
        # ============================================================
        print("\n" + "=" * 80)
        print("🔬 STEP 2: Running Linear Probe on each checkpoint...")
        print("=" * 80)

        eval_loader = build_labeled_eval_loader(
            labels_csv="finalReports/labels.csv",
            image_dir="../../../../finalReports/cropped",
            input_size=224,
            batch_size=32,
            max_per_class=200,
        )

        linear_probe_results = {}
        pipelines = {}

        for method, res in method_results.items():
            print(f"\n📌 Loading {method.upper()}...")
            try:
                pipeline = create_resnet18_ssl_pipeline(
                    method=method,
                    verbose=False,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                pipeline.load_model(res["model_path"])
                pipelines[method] = pipeline

                encoder = pipeline.get_encoder()
                probe = evaluate_with_linear_probe(
                    encoder=encoder,
                    labeled_loader=eval_loader,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    num_classes=4,
                    test_size=0.3,
                )
                linear_probe_results[method] = probe
                print(f"   ✅ {method.upper()} LP Acc: {probe['linear_probe_accuracy']:.4f}")
            except Exception as e:
                print(f"   ⚠️ {method.upper()} failed: {e}")
                continue

        if len(linear_probe_results) == 0:
            print("\n❌ No method could be loaded!")
            return

        # ============================================================
        # ✅ 3. انتخاب بهترین method
        # ============================================================
        best_method = max(
            linear_probe_results.keys(),
            key=lambda x: linear_probe_results[x]["linear_probe_accuracy"],
        )
        best_lp_acc = linear_probe_results[best_method]["linear_probe_accuracy"]
        best_pipeline = pipelines[best_method]

        print("\n" + "=" * 80)
        print("🏆 STEP 3: BEST METHOD SELECTED")
        print("=" * 80)
        for m, probe in linear_probe_results.items():
            marker = " ✅ WINNER" if m == best_method else ""
            print(
                f"   {m.upper():<8} LP Acc: {probe['linear_probe_accuracy']:.4f}  "
                f"LP F1: {probe['linear_probe_f1']:.4f}{marker}"
            )
        print(f"\n   ✅ Winner: {best_method.upper()} (LP Acc = {best_lp_acc:.4f})")
        print("=" * 80)

        # ============================================================
        # ✅ 4. پیدا کردن ResNet50 آموزش‌دیده
        # ============================================================
        print("\n" + "=" * 80)
        print("🔍 STEP 4: Looking for trained ResNet50 checkpoint...")
        print("=" * 80)

        resnet50_source = os.path.join(
            "../../../../finalReports", "models", "resnet", "resnet50",
            "checkpoints", "resnet50_best.pth",
        )

        # Fallback در صورت متفاوت بودن case
        if not os.path.exists(resnet50_source):
            alt = resnet50_source.replace("finalReports", "FinalReports")
            if os.path.exists(alt):
                resnet50_source = alt

        if not os.path.exists(resnet50_source):
            print(f"❌ ResNet50 checkpoint NOT FOUND: {resnet50_source}")
            print("   Please make sure the ResNet50 model is trained first.")
            return

        print(f"✅ Found: {resnet50_source}")

        # ============================================================
        # ✅ 5. کپی گرفتن از ResNet50 به weightTransformer/
        # ============================================================
        print("\n" + "=" * 80)
        print("📋 STEP 5: Copying ResNet50 base to weightTransformer/...")
        print("=" * 80)

        os.makedirs(RESNET_WEIGHT_TRANSFORMER_DIR, exist_ok=True)

        base_copy_path = os.path.join(
            RESNET_WEIGHT_TRANSFORMER_DIR,
            "resnet50_base.pth",
        )

        import shutil
        shutil.copy2(resnet50_source, base_copy_path)
        print(f"✅ Base copy saved: {base_copy_path}")

        # ============================================================
        # ✅ 6. Load کردن ResNet50 base
        # ============================================================
        print("\n" + "=" * 80)
        print("🔧 STEP 6: Loading ResNet50 base for weight transfer...")
        print("=" * 80)

        from src.models.resnet.resnet50 import create_resnet50

        resnet50_model = create_resnet50(
            num_classes=4, pretrained=False, verbose=False
        )

        base_checkpoint = torch.load(base_copy_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        if "model_state_dict" in base_checkpoint:
            base_state = base_checkpoint["model_state_dict"]
        elif "encoder_state_dict" in base_checkpoint:
            base_state = base_checkpoint["encoder_state_dict"]
        else:
            base_state = base_checkpoint

        resnet50_model.load_state_dict(base_state, strict=False)
        print(f"✅ ResNet50 base loaded")

        # ============================================================
        # ✅ 7. انتقال وزن‌ها از best SSL encoder (ResNet18) به ResNet50
        # ============================================================
        print("\n" + "=" * 80)
        print(f"🔄 STEP 7: Transferring {best_method.upper()} weights to ResNet50...")
        print("=" * 80)

        best_encoder = best_pipeline.get_encoder()
        ssl_state_dict = best_encoder.state_dict()
        resnet50_state_dict = resnet50_model.state_dict()

        # ✅ انتقال تنها لایه‌هایی که هم اسم و هم شکل یکسان دارن
        transferred = 0
        skipped = 0
        new_state = {}

        for name, param in resnet50_state_dict.items():
            if name in ssl_state_dict and ssl_state_dict[name].shape == param.shape:
                new_state[name] = ssl_state_dict[name]
                transferred += 1
            else:
                new_state[name] = param
                skipped += 1

        resnet50_model.load_state_dict(new_state, strict=False)

        print(f"✅ Weight transfer complete:")
        print(f"   Transferred layers: {transferred}")
        print(f"   Kept original ResNet50 layers: {skipped}")
        print(f"   Total layers: {len(resnet50_state_dict)}")

        # ============================================================
        # ✅ 8. ذخیره‌ی مدل نهایی
        # ============================================================
        print("\n" + "=" * 80)
        print("💾 STEP 8: Saving transferred model...")
        print("=" * 80)

        transfer_path = os.path.join(
            RESNET_WEIGHT_TRANSFORMER_DIR,
            f"resnet50_transferred_from_{best_method}.pth",
        )

        torch.save(
            {
                "encoder_state_dict": resnet50_model.state_dict(),
                "source_method": best_method,
                "source_model": "ResNet18_SSL",
                "target_model": "ResNet50",
                "best_lp_accuracy": best_lp_acc,
                "timestamp": datetime.now().isoformat(),
                "transferred_layers": transferred,
                "total_layers": len(resnet50_state_dict),
            },
            transfer_path,
        )
        print(f"✅ Transferred model saved: {transfer_path}")

        # ✅ همچنین یک summary
        summary_path = os.path.join(
            RESNET_WEIGHT_TRANSFORMER_DIR,
            "transfer_summary.json",
        )
        with open(summary_path, "w") as f:
            json.dump({
                "best_method": best_method,
                "best_lp_accuracy": best_lp_acc,
                "all_methods": {
                    m: {
                        "lp_accuracy": probe["linear_probe_accuracy"],
                        "lp_f1": probe["linear_probe_f1"],
                    }
                    for m, probe in linear_probe_results.items()
                },
                "source_resnet50": resnet50_source,
                "base_copy": base_copy_path,
                "transferred_model": transfer_path,
                "transferred_layers": transferred,
                "total_layers": len(resnet50_state_dict),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4)
        print(f"✅ Summary saved: {summary_path}")

        print("\n" + "=" * 80)
        print("🎉 DONE!")
        print("=" * 80)

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ RESNET18 - ENSEMBLE FULL FINE-TUNE (FROM TRANSFERRED MODEL)
# ================================================================

# ================================================================
# ✅ RECOVERY MODE — از کجا کرش کرد ادامه بده
# ================================================================


def run_recovery_mode():
    """
    ✅ Recovery: ادامه‌ی گزینه 7 از جایی که کرش کرد.
    بدون Fine-Tune و CV دوباره.
    """
    if HAS_RICH:
        console.print(
            Panel.fit(
                "🔄 Recovery Mode — Continue from Crash Point",
                subtitle="Skip FT & CV → Build plots → Interpretability → PDF",
                border_style="yellow",
            )
        )
    else:
        print(f"\n{Colors.WARNING}🔄 Recovery Mode{Colors.ENDC}")
        print(f"{Colors.WARNING}{'='*70}{Colors.ENDC}")
        print("📌 Skip FT & CV → Build plots → Interpretability → PDF")
        print(f"{Colors.WARNING}{'='*70}{Colors.ENDC}")

    import glob

    # ✅ FIX: اول _final.pth رو بگرد (metadata کامل داره)
    final_candidates = sorted(
        glob.glob(os.path.join(FINAL_MODELS_DIR, "resnet50_finetuned_from_*_final.pth")),
        key=os.path.getmtime,
        reverse=True,
    )

    if final_candidates:
        model_path = final_candidates[0]
    else:
        # fallback
        candidates = sorted(
            glob.glob(os.path.join(FINAL_MODELS_DIR, "resnet50_finetuned*.pth")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            if HAS_RICH:
                console.print(f"[red]❌ No fine-tuned model found in:[/red] {FINAL_MODELS_DIR}")
                console.print("[yellow]👉 Please run option 7 first.[/yellow]")
            else:
                print(f"{Colors.FAIL}❌ No fine-tuned model found!{Colors.ENDC}")
            return
        model_path = candidates[0]

    if HAS_RICH:
        console.print(f"\n[green]✅ Found model:[/green] {os.path.basename(model_path)}")
        console.print(f"[dim]   {model_path}[/dim]")
    else:
        print(f"\n{Colors.GREEN}✅ Found model: {os.path.basename(model_path)}{Colors.ENDC}")

    # ✅ Confirm
    if HAS_RICH:
        if not Confirm.ask("\n🚀 Run recovery (skip FT & CV)?", default=True):
            return
    else:
        confirm = input(f"\n{Colors.WARNING}🚀 Run recovery? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            return

    # ✅ Run recovery
    try:
        from src.pipeLine.resnet50FineTune import run_finetune_recovery

        results = run_finetune_recovery(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
        )

        if HAS_RICH:
            console.print("\n[bold green]✅ Recovery Complete![/bold green]")
            console.print(f"   PDF: {results['pdf_path']}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Recovery failed: {e}[/red]")
        else:
            print(f"\n{Colors.FAIL}❌ Recovery failed: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()



def detect_hardware_device():
    """
    ✅ v2.0: Hardware Detection (option 16)

    Detects and displays:
    - System: Brand, Model, Serial, Chassis
    - CPU: Name, Code Name, TDP, Package, Technology, VID, Cache, Cores, Threads
    - Motherboard: Manufacturer, Model, Bus Specs, Chipset, South Bridge
    - BIOS: Brand, Version, Release Date, SMBIOS, ROM
    - Memory: Total RAM, Type, Speed, Manufacturer, Slots, ROM
    - Graphics: All GPUs (integrated + dedicated) with VRAM, Type, Driver
    """
    if not HAS_HW_DETECTOR:
        if HAS_RICH:
            console.print("[red]❌ Hardware Detector not available![/red]")
            console.print("[yellow]   Make sure src/findDevice/hardwareDetector.py exists[/yellow]")
        else:
            print(f"{Colors.FAIL}❌ Hardware Detector not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(
            Panel.fit(
                "💻 Hardware Detection",
                subtitle="Comprehensive system specs",
                border_style="cyan",
            )
        )

    try:
        _hw_()
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()



def run_resnet18_ensemble_finetune():
    """
    ✅ NEW (گزینه 7): Fine-Tune روی مدل Transfer شده.

    این تابع فرض می‌کنه که گزینه 6 قبلاً اجرا شده و مدل Transfer
    در weightTransformer/ ذخیره شده. فقط Fine-Tune به بعد رو انجام می‌ده.

    Steps:
    1. پیدا کردن آخرین مدل Transfer شده
    2. نمایش اطلاعات Transfer
    3. پرسیدن تنظیمات Fine-Tune از کاربر
    4. اجرای Fine-Tune + CV + Plots + Interpretability + PDF
    """
    if HAS_RICH:
        console.print(
            Panel.fit(
                "🧠 ResNet50 Fine-Tune from Pre-Transferred Model",
                subtitle="Load Transferred → FT → CV5 → CI95% → 15-page PDF",
                border_style="magenta",
            )
        )
    else:
        print(f"\n{Colors.MAGENTA}🧠 ResNet50 Fine-Tune from Pre-Transferred Model{Colors.ENDC}")
        print(f"{Colors.MAGENTA}{'=' * 70}{Colors.ENDC}")
        print("📌 Load Transferred → FT → CV5 → CI95% → 15-page PDF")
        print(f"{Colors.MAGENTA}{'=' * 70}{Colors.ENDC}")

    # ================================================================
    # ✅ STEP 1: Find Transferred Architects
    # ================================================================
    import glob

    transferred_pattern = os.path.join(
        RESNET_WEIGHT_TRANSFORMER_DIR,
        "resnet50_transferred_from_*.pth",
    )
    transferred_models = sorted(
        glob.glob(transferred_pattern),
        key=os.path.getmtime,
        reverse=True,  # جدیدترین اول
    )

    if not transferred_models:
        if HAS_RICH:
            console.print("\n[red]❌ No transferred models found![/red]")
            console.print(f"[yellow]📂 Looked in:[/yellow] {RESNET_WEIGHT_TRANSFORMER_DIR}")
            console.print("\n[yellow]👉 Please run option 6 first:[/yellow]")
            console.print("   [cyan]🧠 Find best SSL models & Transfer to ResNet50 & Reports[/cyan]")
        else:
            print(f"\n{Colors.FAIL}❌ No transferred models found!{Colors.ENDC}")
            print(f"{Colors.WARNING}📂 Looked in: {RESNET_WEIGHT_TRANSFORMER_DIR}{Colors.ENDC}")
            print(f"\n{Colors.WARNING}👉 Please run option 6 first:{Colors.ENDC}")
            print(f"   {Colors.CYAN}🧠 Find best SSL models & Transfer to ResNet50{Colors.ENDC}")
        return

    # ================================================================
    # ✅ STEP 2: Display Available Transferred Architects
    # ================================================================
    if HAS_RICH:
        console.print(f"\n[green]✅ Found {len(transferred_models)} transferred model(s):[/green]\n")

        table = Table(
            title="📦 Available Transferred Architects",
            box=box.ROUNDED,
            border_style="cyan",
        )
        table.add_column("#", style="bold yellow", width=4)
        table.add_column("File", style="bold white")
        table.add_column("Source", style="bold cyan")
        table.add_column("Modified", style="dim white")
        table.add_column("Size (MB)", style="dim white", justify="right")

        for idx, m_path in enumerate(transferred_models, 1):
            fname = os.path.basename(m_path)
            # استخراج method از اسم فایل
            source = "unknown"
            if "from_" in fname:
                source = fname.split("from_")[1].replace(".pth", "").upper()

            mtime = datetime.fromtimestamp(os.path.getmtime(m_path))
            size_mb = os.path.getsize(m_path) / 1e6

            table.add_row(
                str(idx),
                fname,
                source,
                mtime.strftime("%Y-%m-%d %H:%M"),
                f"{size_mb:.1f}",
            )

        console.print(table)
    else:
        print(f"\n{Colors.GREEN}✅ Found {len(transferred_models)} transferred model(s):{Colors.ENDC}\n")
        for idx, m_path in enumerate(transferred_models, 1):
            fname = os.path.basename(m_path)
            source = "unknown"
            if "from_" in fname:
                source = fname.split("from_")[1].replace(".pth", "").upper()
            mtime = datetime.fromtimestamp(os.path.getmtime(m_path))
            size_mb = os.path.getsize(m_path) / 1e6
            print(f"   {Colors.YELLOW}{idx}{Colors.ENDC}  {fname}")
            print(f"      Source: {source} | Modified: {mtime.strftime('%Y-%m-%d %H:%M')} | Size: {size_mb:.1f} MB")

    # ================================================================
    # ✅ STEP 3: Select Model
    # ================================================================
    if len(transferred_models) == 1:
        selected_model = transferred_models[0]
        if HAS_RICH:
            console.print(f"\n[green]👉 Auto-selected (only one): {os.path.basename(selected_model)}[/green]")
        else:
            print(f"\n{Colors.GREEN}👉 Auto-selected: {os.path.basename(selected_model)}{Colors.ENDC}")
    else:
        if HAS_RICH:
            choice_str = Prompt.ask(
                f"\n[bold cyan]👉 Select model (1-{len(transferred_models)}, default=1)[/bold cyan]",
                default="1",
            )
        else:
            choice_str = input(
                f"\n{Colors.BOLD}👉 Select model (1-{len(transferred_models)}, default=1): {Colors.ENDC}"
            ) or "1"

        try:
            choice_idx = int(choice_str) - 1
            if choice_idx < 0 or choice_idx >= len(transferred_models):
                raise ValueError
            selected_model = transferred_models[choice_idx]
        except (ValueError, IndexError):
            if HAS_RICH:
                console.print("[yellow]⚠️ Invalid choice. Using first model.[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ Invalid choice. Using first model.{Colors.ENDC}")
            selected_model = transferred_models[0]

    # ================================================================
    # ✅ STEP 4: Load and Display Model Metadata
    # ================================================================
    if HAS_RICH:
        console.print(f"\n[cyan]📂 Loading metadata from:[/cyan] {os.path.basename(selected_model)}")
    else:
        print(f"\n{Colors.CYAN}📂 Loading metadata from: {os.path.basename(selected_model)}{Colors.ENDC}")

    try:
        ckpt = torch.load(selected_model, map_location="cuda" if torch.cuda.is_available() else "cpu")
        source_method = ckpt.get("source_method", "unknown")
        best_lp = ckpt.get("best_lp_accuracy", 0.0)
        transferred_layers = ckpt.get("transferred_layers", 0)
        total_layers = ckpt.get("total_layers", 0)

        if HAS_RICH:
            info_table = Table(
                title="📋 Transfer Metadata",
                box=box.ROUNDED,
                border_style="green",
            )
            info_table.add_column("Property", style="bold cyan")
            info_table.add_column("Value", style="bold white")

            info_table.add_row("Source SSL Method", source_method.upper())
            info_table.add_row("Best LP Accuracy", f"{best_lp:.4f}")
            info_table.add_row("Transferred Layers", f"{transferred_layers}/{total_layers}")
            info_table.add_row("File", os.path.basename(selected_model))
            info_table.add_row("Full Path", selected_model)

            console.print(info_table)
        else:
            print(f"\n{Colors.GREEN}📋 Transfer Metadata:{Colors.ENDC}")
            print(f"   Source SSL Method: {source_method.upper()}")
            print(f"   Best LP Accuracy:  {best_lp:.4f}")
            print(f"   Transferred Layers: {transferred_layers}/{total_layers}")
            print(f"   File: {os.path.basename(selected_model)}")
    except Exception as e:
        if HAS_RICH:
            console.print(f"[yellow]⚠️ Could not read metadata: {e}[/yellow]")
        else:
            print(f"{Colors.WARNING}⚠️ Could not read metadata: {e}{Colors.ENDC}")
        source_method = "unknown"
        best_lp = 0.0

    # ================================================================
    # ✅ STEP 5: Get Fine-Tune Configuration
    # ================================================================
    if HAS_RICH:
        console.print("\n[bold cyan]⚙️  Fine-Tune Configuration[/bold cyan]")

        total_epochs = int(Prompt.ask("⏳ Fine-Tune epochs", default="50"))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        cv_folds = int(Prompt.ask("🔬 CV folds", default="5"))
        cv_epochs_per_fold = int(Prompt.ask("⏳ FT epochs per CV fold", default="10"))

        console.print("\n[bold cyan]📋 Summary:[/bold cyan]")
        console.print(f"   Model:              {os.path.basename(selected_model)}")
        console.print(f"   Source:             {source_method.upper()}")
        console.print(f"   FT Epochs:          {total_epochs}")
        console.print(f"   Batch Size:         {batch_size}")
        console.print(f"   CV Folds:           {cv_folds}")
        console.print(f"   CV Epochs/Fold:     {cv_epochs_per_fold}")

        if not Confirm.ask(f"\n🚀 Start Fine-Tune pipeline?", default=True):
            return
    else:
        print(f"\n{Colors.BOLD}⚙️  Fine-Tune Configuration{Colors.ENDC}")
        total_epochs = int(input("⏳ Fine-Tune epochs (default=50): ") or "50")
        batch_size = int(input("📦 Batch size (default=32): ") or "32")
        cv_folds = int(input("🔬 CV folds (default=5): ") or "5")
        cv_epochs_per_fold = int(input("⏳ FT epochs per CV fold (default=10): ") or "10")

        print(f"\n{Colors.BOLD}📋 Summary:{Colors.ENDC}")
        print(f"   Model:              {os.path.basename(selected_model)}")
        print(f"   Source:             {source_method.upper()}")
        print(f"   FT Epochs:          {total_epochs}")
        print(f"   Batch Size:         {batch_size}")

        confirm = input(f"\n{Colors.WARNING}🚀 Start? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            return

    # ================================================================
    # ✅ STEP 6: Run Fine-Tune Pipeline
    # ================================================================
    try:
        from src.pipeLine.resnet50FineTune import run_finetune_only

        results = run_finetune_only(
            model_path=selected_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            epochs=total_epochs,
            batch_size=batch_size,
            input_size=224,
            cv_folds=cv_folds,
            cv_epochs_per_fold=cv_epochs_per_fold,
            verbose=True,
        )

        # ================================================================
        # ✅ STEP 7: Display Final Summary
        # ================================================================
        if HAS_RICH:
            console.print("\n" + "=" * 80)
            console.print("[bold green]✅ FINE-TUNE PIPELINE COMPLETE![/bold green]")
            console.print("=" * 80)

            summary_table = Table(
                title="🎉 Final Results",
                box=box.ROUNDED,
                border_style="green",
            )
            summary_table.add_column("Metric", style="bold cyan")
            summary_table.add_column("Value", style="bold white")

            summary_table.add_row("Source SSL Method", results["best_method"].upper())
            summary_table.add_row("Source LP Accuracy", f"{results['best_lp_accuracy']:.4f}")

            acc = results["metrics"]["accuracy"]
            f1 = results["metrics"]["f1_score"]
            bal_acc = results["metrics"]["balanced_accuracy"]

            summary_table.add_row("Final Val Accuracy", f"{acc:.4f}")
            summary_table.add_row("Final Val Balanced Acc", f"{bal_acc:.4f}")
            summary_table.add_row("Final Val F1 (weighted)", f"{f1:.4f}")

            cv = results["cv_summary"]
            acc_s = cv["accuracy"]
            summary_table.add_row(
                "CV Mean Accuracy",
                f"{acc_s['mean']:.4f} ± {acc_s['std']:.4f}",
            )
            summary_table.add_row(
                "CV 95% CI",
                f"[{acc_s['ci95_low']:.4f}, {acc_s['ci95_high']:.4f}]",
            )

            summary_table.add_row("Final Model", results["final_model_path"])
            summary_table.add_row("PDF Report", results["pdf_path"])

            console.print(summary_table)

            console.print(f"\n[dim]📂 Reports directory:[/dim] {RESNET50_REPORTS_DIR}")
            console.print(f"[dim]📂 Final models:[/dim] {FINAL_MODELS_DIR}")
        else:
            print(f"\n{'=' * 80}")
            print(f"✅ FINE-TUNE PIPELINE COMPLETE!")
            print(f"{'=' * 80}")
            print(f"   Source SSL Method:    {results['best_method'].upper()}")
            print(f"   Source LP Accuracy:   {results['best_lp_accuracy']:.4f}")
            print(f"   Final Val Accuracy:   {results['metrics']['accuracy']:.4f}")
            print(f"   Final Val F1:         {results['metrics']['f1_score']:.4f}")
            cv = results["cv_summary"]
            acc_s = cv["accuracy"]
            print(f"   CV Mean Accuracy:     {acc_s['mean']:.4f} ± {acc_s['std']:.4f}")
            print(f"   CV 95% CI:            [{acc_s['ci95_low']:.4f}, {acc_s['ci95_high']:.4f}]")
            print(f"   Final Model:          {results['final_model_path']}")
            print(f"   PDF Report:           {results['pdf_path']}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error in Fine-Tune pipeline: {e}[/red]")
        else:
            print(f"\n{Colors.FAIL}❌ Error in Fine-Tune pipeline: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ EFFICIENTNET-B0 SSL - SINGLE METHOD (BYOL / SimCLR / MAE)
# ================================================================


def run_efficientnet_b0_ssl_single(method: str = "byol"):
    """
    Run EfficientNet-B0 SSL with a single method (BYOL / SimCLR / MAE) and generate report.
    After training, runs 5-Fold Cross-Validation and adds results to report.

    Args:
        method: 'byol', 'simclr', or 'mae'
    """
    if HAS_RICH:
        console.print(Panel.fit(f"⚡ EfficientNet-B0 SSL - {method.upper()}", border_style="green"))
    else:
        print(f"\n{Colors.GREEN}⚡ EfficientNet-B0 SSL - {method.upper()}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
        total_epochs = int(Prompt.ask("⏳ Number of epochs", default="30"))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )

        if not Confirm.ask(f"\n🚀 Start EfficientNet-B0 SSL with {method.upper()}?", default=True):
            return
    else:
        image_path = (
            input(
                f"\n{Colors.BLUE}📁 Enter path to images directory (default={default_image_path}): {Colors.ENDC}"
            )
            or default_image_path
        )
        total_epochs = int(
            input(f"{Colors.BLUE}⏳ Number of epochs (default=30): {Colors.ENDC}") or "30"
        )
        batch_size = int(input(f"{Colors.BLUE}📦 Batch size (default=32): {Colors.ENDC}") or "32")
        print(f"{Colors.BLUE}🔧 Select optimizer:{Colors.ENDC}")
        print(f"   1. AdamW (Default)\n   2. RAdam\n   3. Ranger")
        opt_choice = input(f"{Colors.BOLD}👉 Enter choice (1-3): {Colors.ENDC}")
        opt_map = {"1": "adamw", "2": "radam", "3": "ranger"}
        optimizer_name = opt_map.get(opt_choice, "adamw")

        confirm = input(
            f"\n{Colors.WARNING}🚀 Start EfficientNet-B0 SSL with {method.upper()}? (y/n): {Colors.ENDC}"
        )
        if confirm.lower() != "y":
            return

    try:
        from src.pipeLine.Architects.EfficientNet.B0.efficientNetB0SSL import create_efficientnet_b0_ssl_pipeline

        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            image_paths.extend(
                [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(ext)]
            )

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found!{Colors.ENDC}")
            return

        pipeline = create_efficientnet_b0_ssl_pipeline(
            method=method,
            batch_size=batch_size,
            total_epochs=total_epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
            drop_last=True,
            verbose=True,
            optimizer_name=optimizer_name,
            use_kornia=True,
        )

        results = pipeline.pretrain(image_paths)

        # ✅ RUN 5-FOLD CROSS-VALIDATION
        # ✅ RUN 5-FOLD CROSS-VALIDATION
        from src.pipeLine.CV.crossValidationRunner import run_cv_after_model

        if HAS_RICH:
            console.print("\n[cyan]🔬 Running 5-Fold Cross-Validation on trained model...[/cyan]")
        else:
            print(
                f"\n{Colors.CYAN}🔬 Running 5-Fold Cross-Validation on trained model...{Colors.ENDC}"
            )

        # ✅ FIX: چک کردن چند مسیر ممکن — architecture/{method}/ در اولویت
        method_arch_dir = os.path.join(EFFICIENTNET_B0_SSL_DIR, "architecture", method)

        model_candidates = [
            # ✅ اولویت ۱: مسیر جدید architecture/{method}/
            os.path.join(method_arch_dir, f"b0_ssl_{method}_best.pth"),
            os.path.join(method_arch_dir, f"b0_ssl_{method}_final.pth"),
            os.path.join(method_arch_dir, f"b0_ssl_{method}_checkpoint.pth"),
            os.path.join(method_arch_dir, f"b0_ssl_{method}_last.pth"),

            # ✅ اولویت ۲: مسیر قدیمی (backward compat)
            os.path.join(EFFICIENTNET_B0_SSL_DIR, f"b0_ssl_{method}_best.pth"),
            os.path.join(EFFICIENTNET_B0_SSL_DIR, f"b0_ssl_{method}_final.pth"),
            os.path.join(EFFICIENTNET_B0_SSL_DIR, f"efficientnet_b0_ssl_{method}_encoder.pth"),

            # ✅ اولویت ۳: مسیرهای احتمالی دیگه
            os.path.join(
                EFFICIENTNET_B0_SSL_DIR, "architecture", method,
                f"efficientnet_b0_best_update_{method}.pth",
            ),
        ]

        model_path = None
        for p in model_candidates:
            if os.path.exists(p):
                model_path = p
                if HAS_RICH:
                    console.print(f"[green]✅ Found model: {p}[/green]")
                else:
                    print(f"{Colors.GREEN}✅ Found model: {p}{Colors.ENDC}")
                break

        if model_path is not None:
            try:
                cv_result = run_cv_after_model(
                    model_name="efficientnet_b0",
                    method=method,
                    model_path=model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=True,
                    batch_size=32,  # ✅ OPTIONAL
                )
                results["cross_validation"] = cv_result
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[yellow]⚠️ CV failed: {e}[/yellow]")
                else:
                    print(f"{Colors.WARNING}⚠️ CV failed: {e}{Colors.ENDC}")
        else:
            if HAS_RICH:
                console.print("[yellow]⚠️ Model not found in any of the expected paths:[/yellow]")
                for p in model_candidates:
                    console.print(f"   [dim]❌ {p}[/dim]")
            else:
                print(f"{Colors.WARNING}⚠️ Model not found in any of the expected paths:{Colors.ENDC}")
                for p in model_candidates:
                    print(f"   ❌ {p}")

        # Generate report
        report_path = pipeline.generate_report()

        if HAS_RICH:
            console.print(
                f"\n[bold green]✅ EfficientNet-B0 {method.upper()} SSL Complete![/bold green]"
            )
            console.print(f"   Best Loss: {results['best_loss']:.4f}")
            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                console.print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )
            console.print(f"   📊 Report: {report_path}")
        else:
            print(f"\n{Colors.GREEN}✅ EfficientNet-B0 {method.upper()} SSL Complete!{Colors.ENDC}")
            print(f"   Best Loss: {results['best_loss']:.4f}")
            if (
                "cross_validation" in results
                and results["cross_validation"].get("status") == "success"
            ):
                cv = results["cross_validation"]
                print(
                    f"   🔬 5-Fold CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
                )
            print(f"   📊 Report: {report_path}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback

        traceback.print_exc()


# ================================================================
# ✅ EFFICIENTNET-B0 - FIND BEST SSL & TRANSFER TO B1
# ================================================================


def run_efficientnet_b0_best_transfer():
    """
    ✅ FIXED: Find best SSL method on EfficientNet-B0 (FROM EXISTING CHECKPOINTS),
    then transfer weights to a COPY of the trained EfficientNet-B1.

    Steps:
    1. Check that all 4 checkpoints exist (BYOL, SimCLR, MAE, B1)
    2. Copy B1 base to weightTransformer/efficinetnet_b1_best_base.pth
    3. Linear Probe on each of 3 SSL methods
    4. Select best method (highest LP accuracy)
    5. Transfer weights from best SSL to B1 base
    6. Save transferred model: b1_transferred_from_{best_method}.pth
    """
    if HAS_RICH:
        console.print(
            Panel.fit(
                "⚡ EfficientNet-B0 - Find Best SSL & Transfer to B1",
                subtitle="Load Checkpoints → Compare → Transfer to B1 Copy",
                border_style="cyan",
            )
        )
    else:
        print(f"\n{Colors.CYAN}⚡ EfficientNet-B0 - Find Best SSL & Transfer to B1{Colors.ENDC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.ENDC}")
        print("📌 Load Checkpoints → Compare → Transfer to B1 Copy")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.ENDC}")

    # ================================================================
    # ✅ PATHS (ABSOLUTE)
    # ================================================================
    B0_SSL_BASE = os.path.join(
        EFFICIENTNET_B0_SSL_DIR, "architecture"
    )
    B1_SOURCE = os.path.join(
        "../../../../finalReports", "models", "efficientnet", "EfficientNet/B1", "checkpoints", "efficientnet_b1_best.pth"
    )
    B1_BASE_DEST = os.path.join(
        EFFICIENTNET_WEIGHT_TRANSFORMER_DIR, "efficinetnet_b1_best_base.pth"
    )

    # ✅ Ensure directories exist
    os.makedirs(EFFICIENTNET_WEIGHT_TRANSFORMER_DIR, exist_ok=True)

    # ================================================================
    # ✅ STEP 1: Check that all 4 checkpoints exist
    # ================================================================
    print("\n" + "=" * 80)
    print("🔍 STEP 1: Checking all required checkpoints exist...")
    print("=" * 80)

    SSL_PATHS = {
        "byol": os.path.join(B0_SSL_BASE, "byol", "b0_ssl_byol_best.pth"),
        "simclr": os.path.join(B0_SSL_BASE, "simclr", "b0_ssl_simclr_best.pth"),
        "mae": os.path.join(B0_SSL_BASE, "mae", "b0_ssl_mae_best.pth"),
    }

    all_paths = {}
    all_exist = True

    for method, path in SSL_PATHS.items():
        exists = os.path.exists(path)
        icon = "✅" if exists else "❌"
        print(f"   {icon} {method.upper():<8} : {path}")
        all_paths[method] = path
        if not exists:
            all_exist = False

    # B1 check
    b1_exists = os.path.exists(B1_SOURCE)
    icon = "✅" if b1_exists else "❌"
    print(f"   {icon} {'B1':<8} : {B1_SOURCE}")
    if not b1_exists:
        all_exist = False

    if not all_exist:
        print("\n" + "=" * 80)
        print("❌ ERROR: Not all required checkpoints exist!")
        print("=" * 80)
        print("👉 Please make sure you have trained:")
        print("   - EfficientNet-B0 with BYOL   (option 9)")
        print("   - EfficientNet-B0 with SimCLR (option 10)")
        print("   - EfficientNet-B0 with MAE    (option 11)")
        print("   - EfficientNet-B1 base model (from Without Fine-Tuning task)")
        print("=" * 80)
        return {"status": "failed", "error": "Missing checkpoints"}

    print("\n✅ All 4 checkpoints exist!")

    # Confirm with user
    if HAS_RICH:
        if not Confirm.ask("\n🚀 Continue with checkpoint loading and weight transfer?", default=True):
            return
    else:
        confirm = input(f"\n{Colors.WARNING}🚀 Continue? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            return

    # ================================================================
    # ✅ STEP 2: Copy B1 base to weightTransformer/
    # ================================================================
    print("\n" + "=" * 80)
    print("📋 STEP 2: Copying B1 base to weightTransformer/...")
    print("=" * 80)

    try:
        import shutil
        shutil.copy2(B1_SOURCE, B1_BASE_DEST)
        print(f"✅ B1 base copied to: {B1_BASE_DEST}")
    except Exception as e:
        print(f"❌ Failed to copy B1 base: {e}")
        return {"status": "failed", "error": str(e)}

    # ================================================================
    # ✅ STEP 3: Load each SSL encoder + Linear Probe
    # ================================================================
    print("\n" + "=" * 80)
    print("🔬 STEP 3: Running Linear Probe on each SSL checkpoint...")
    print("=" * 80)

    from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
        evaluate_with_linear_probe,
        build_labeled_eval_loader,
    )
    from src.pipeLine.Architects.EfficientNet.B0.efficientNetB0SSL import create_efficientnet_b0_ssl_pipeline

    # Build labeled eval loader
    print("\n📊 Building labeled eval loader...")
    eval_loader = build_labeled_eval_loader(
        labels_csv="finalReports/labels.csv",
        image_dir="../../../../finalReports/cropped",
        input_size=224,  # B0 input size
        batch_size=32,
        max_per_class=200,
    )

    # Evaluate each method
    linear_probe_results = {}
    pipelines = {}

    for method, path in all_paths.items():
        print(f"\n📌 Loading {method.upper()}...")
        try:
            # Create pipeline and load model
            pipeline = create_efficientnet_b0_ssl_pipeline(
                method=method,
                verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            pipeline.load_model(path)
            pipelines[method] = pipeline

            # Extract encoder and run Linear Probe
            encoder = pipeline.get_encoder()
            probe = evaluate_with_linear_probe(
                encoder=encoder,
                labeled_loader=eval_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_classes=4,
                test_size=0.3,
            )
            linear_probe_results[method] = probe
            print(f"   ✅ {method.upper()} LP Acc: {probe['linear_probe_accuracy']:.4f}  "
                  f"| LP F1: {probe['linear_probe_f1']:.4f}")

        except Exception as e:
            print(f"   ⚠️ {method.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(linear_probe_results) == 0:
        print("\n❌ No SSL method could be loaded and evaluated!")
        return {"status": "failed", "error": "No method loaded"}

    # ================================================================
    # ✅ STEP 4: Select best method
    # ================================================================
    print("\n" + "=" * 80)
    print("🏆 STEP 4: BEST METHOD SELECTED")
    print("=" * 80)

    best_method = max(
        linear_probe_results.keys(),
        key=lambda x: linear_probe_results[x]["linear_probe_accuracy"],
    )
    best_lp_acc = linear_probe_results[best_method]["linear_probe_accuracy"]
    best_lp_f1 = linear_probe_results[best_method]["linear_probe_f1"]

    for m, probe in linear_probe_results.items():
        marker = " ✅ WINNER" if m == best_method else ""
        print(f"   {m.upper():<8} LP Acc: {probe['linear_probe_accuracy']:.4f}  "
              f"LP F1: {probe['linear_probe_f1']:.4f}{marker}")

    print(f"\n   ✅ Winner: {best_method.upper()} (LP Acc = {best_lp_acc:.4f})")
    print("=" * 80)

    # ================================================================
    # ✅ STEP 5: Load B1 base and transfer weights
    # ================================================================
    print("\n" + "=" * 80)
    print("🔧 STEP 5: Loading B1 base and transferring weights...")
    print("=" * 80)

    from src.models.efficientNet.B1 import create_efficientnet_b1

    # Create fresh B1 model
    b1_model = create_efficientnet_b1(
        num_classes=4, pretrained=False, verbose=False
    )

    # Load B1 base weights
    try:
        b1_ckpt = torch.load(B1_BASE_DEST, map_location="cpu")
        if "model_state_dict" in b1_ckpt:
            b1_state = b1_ckpt["model_state_dict"]
        elif "encoder_state_dict" in b1_ckpt:
            b1_state = b1_ckpt["encoder_state_dict"]
        else:
            b1_state = b1_ckpt

        missing, unexpected = b1_model.load_state_dict(b1_state, strict=False)
        print(f"✅ B1 base loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    except Exception as e:
        print(f"❌ Failed to load B1 base: {e}")
        return {"status": "failed", "error": str(e)}

    # Get best SSL encoder state dict
    best_pipeline = pipelines[best_method]
    best_encoder = best_pipeline.get_encoder()
    ssl_state_dict = best_encoder.state_dict()
    b1_state_dict = b1_model.state_dict()

    print(f"\n🔄 Transferring {best_method.upper()} weights to B1...")
    print(f"   SSL encoder layers: {len(ssl_state_dict)}")
    print(f"   B1 model layers:    {len(b1_state_dict)}")

    # Transfer only matching layers
    transferred = 0
    skipped = 0
    new_state = {}

    for name, param in b1_state_dict.items():
        if name in ssl_state_dict and ssl_state_dict[name].shape == param.shape:
            new_state[name] = ssl_state_dict[name]
            transferred += 1
        else:
            new_state[name] = param
            skipped += 1

    b1_model.load_state_dict(new_state, strict=False)

    print(f"\n✅ Weight transfer complete:")
    print(f"   Transferred layers: {transferred}")
    print(f"   Kept original:      {skipped}")
    print(f"   Total:              {len(b1_state_dict)}")
    print(f"   Transfer ratio:     {100 * transferred / len(b1_state_dict):.1f}%")

    # ================================================================
    # ✅ STEP 6: Save transferred model
    # ================================================================
    print("\n" + "=" * 80)
    print("💾 STEP 6: Saving transferred model...")
    print("=" * 80)

    transfer_path = os.path.join(
        EFFICIENTNET_WEIGHT_TRANSFORMER_DIR,
        f"b1_transferred_from_{best_method}.pth",
    )

    torch.save({
        "encoder_state_dict": b1_model.state_dict(),
        "source_method": best_method,
        "source_model": "EfficientNet-B0_SSL",
        "target_model": "EfficientNet-B1",
        "best_lp_accuracy": best_lp_acc,
        "best_lp_f1": best_lp_f1,
        "all_methods_lp": {
            m: {
                "lp_accuracy": probe["linear_probe_accuracy"],
                "lp_f1": probe["linear_probe_f1"],
            }
            for m, probe in linear_probe_results.items()
        },
        "transferred_layers": transferred,
        "total_layers": len(b1_state_dict),
        "transfer_ratio": transferred / len(b1_state_dict),
        "timestamp": datetime.now().isoformat(),
        "b1_source": B1_SOURCE,
        "b1_base_copy": B1_BASE_DEST,
    }, transfer_path)

    print(f"✅ Transferred model saved: {transfer_path}")

    # ================================================================
    # ✅ STEP 7: Save summary JSON
    # ================================================================
    summary_path = os.path.join(
        EFFICIENTNET_WEIGHT_TRANSFORMER_DIR,
        "transfer_summary.json",
    )
    with open(summary_path, "w") as f:
        json.dump({
            "best_method": best_method,
            "best_lp_accuracy": best_lp_acc,
            "best_lp_f1": best_lp_f1,
            "all_methods": {
                m: {
                    "lp_accuracy": probe["linear_probe_accuracy"],
                    "lp_f1": probe["linear_probe_f1"],
                }
                for m, probe in linear_probe_results.items()
            },
            "source_b1": B1_SOURCE,
            "b1_base_copy": B1_BASE_DEST,
            "transferred_model": transfer_path,
            "transferred_layers": transferred,
            "total_layers": len(b1_state_dict),
            "transfer_ratio": transferred / len(b1_state_dict),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=4)

    print(f"✅ Summary saved: {summary_path}")

    # ================================================================
    # ✅ DONE
    # ================================================================
    print("\n" + "=" * 80)
    print("🎉 WEIGHT TRANSFER COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Best SSL Method:  {best_method.upper()}")
    print(f"   Best LP Accuracy: {best_lp_acc:.4f}")
    print(f"   Transferred:      {transferred}/{len(b1_state_dict)} layers "
          f"({100 * transferred / len(b1_state_dict):.1f}%)")
    print(f"\n📂 Files:")
    print(f"   Base copy:         {B1_BASE_DEST}")
    print(f"   Transferred model: {transfer_path}")
    print(f"   Summary:           {summary_path}")
    print("=" * 80)

    return {
        "status": "success",
        "best_method": best_method,
        "best_lp_accuracy": best_lp_acc,
        "transferred_model": transfer_path,
        "transferred_layers": transferred,
        "total_layers": len(b1_state_dict),
    }


# ================================================================
# ✅ EFFICIENTNET - B1 COMPLETE FINE-TUNE (OPTION 13)
# ================================================================


def run_efficientnet_ensemble_finetune():
    """
    ✅ OPTION 13: B1 Complete Fine-Tune Pipeline

    Complete flow:
    1. Check transferred model (from B0 SSL)
    2. Load labeled data (labels.csv + cropped/)
    3. Fine-Tune on labeled data
    4. 5-Fold Cross-Validation
    5. 95% Confidence Interval (Bootstrap, 1000 samples)
    6. Generate 23-page colorful PDF report

    Outputs:
    - PDF: SSL_Reporter/b1_finetune/b1_finetune_report.pdf
    - Plots: SSL_Reporter/b1_finetune/plots/
    - Interpretability: SSL_Reporter/b1_finetune/interpretability/
    - CV: SSL_Reporter/b1_finetune/cv/
    - Architecture: SSL_Reporter/b1_finetune/architecture/
    - Final Model: SSL_Reporter/b1_finetune/b1_finetuned_final.pth
    - Metrics: SSL_Reporter/b1_finetune/metrics.json
    """
    # ✅ Check modules
    if not HAS_B1_FINETUNE:
        if HAS_RICH:
            console.print("[red]❌ B1 Fine-Tune modules not available![/red]")
            console.print("[yellow]   Make sure these files exist:[/yellow]")
            console.print("   - src/pipeLine/b1FineTune.py")
            console.print("   - src/pipeLine/b1Reporter.py")
        else:
            print(f"{Colors.FAIL}❌ B1 Fine-Tune modules not available!{Colors.ENDC}")
            print(f"{Colors.WARNING}   Make sure these files exist:{Colors.ENDC}")
            print("   - src/pipeLine/b1FineTune.py")
            print("   - src/pipeLine/b1Reporter.py")
        return

    # ✅ Header
    if HAS_RICH:
        console.print(
            Panel.fit(
                "⚡ EfficientNet-B1 Complete Fine-Tune",
                subtitle="Fine-Tune → CV5 → CI95% → 23-page Colorful PDF Report",
                border_style="magenta",
            )
        )
    else:
        print(f"\n{Colors.MAGENTA}⚡ EfficientNet-B1 Complete Fine-Tune{Colors.ENDC}")
        print(f"{Colors.MAGENTA}{'=' * 70}{Colors.ENDC}")
        print("📌 Fine-Tune → CV5 → CI95% → 23-page Colorful PDF Report")
        print(f"{Colors.MAGENTA}{'=' * 70}{Colors.ENDC}")

    # ================================================================
    # ✅ STEP 1: Check transferred model exists
    # ================================================================
    print("\n🔍 Step 1: Checking for transferred model...")

    try:
        exists, transferred_path = check_transferred_model()
    except Exception as e:
        print(f"❌ Error checking transferred model: {e}")
        import traceback
        traceback.print_exc()
        return

    if not exists:
        print(f"\n❌ Transferred model NOT FOUND!")
        print(f"   Looked for: b1_transferred_from_*.pth")
        print(f"   In: {EFFICIENTNET_WEIGHT_TRANSFORMER_DIR}")
        print(f"\n👉 Please run option 12 first to transfer weights from B0 SSL")
        return

    print(f"✅ Found transferred model:")
    print(f"   {transferred_path}")

    # ================================================================
    # ✅ STEP 2: Ask for hyperparameters
    # ================================================================
    print("\n⚙️  Step 2: Configuration")
    print("=" * 70)

    if HAS_RICH:
        epochs_final = int(Prompt.ask("⏳ Fine-Tune epochs (final)", default="100"))
        epochs_cv = int(Prompt.ask("⏳ Fine-Tune epochs per CV fold", default="50"))
        lr_final = float(Prompt.ask("📊 Learning Rate (final)", default="0.000005"))  # ✅ 5e-6
        lr_cv = float(Prompt.ask("📊 Learning Rate (CV)", default="0.000005"))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        patience_final = int(Prompt.ask("🛑 Early stopping patience (final)", default="10"))
        patience_cv = int(Prompt.ask("🛑 Early stopping patience (CV)", default="10"))
        n_bootstrap = int(Prompt.ask("🎲 Bootstrap samples (CI 95%)", default="1000"))
        run_interpretability = Confirm.ask("🔬 Run interpretability (SHAP, Captum, LIME)?", default=True)
    else:
        print(f"\n{Colors.BOLD}Configuration (press Enter for defaults):{Colors.ENDC}")
        epochs_final = int(input("⏳ Fine-Tune epochs (final) [100]: ") or "100")
        epochs_cv = int(input("⏳ Fine-Tune epochs per CV fold [50]: ") or "50")
        lr_final = float(input("📊 Learning Rate (final) [1e-5]: ") or "1e-5")
        lr_cv = float(input("📊 Learning Rate (CV) [1e-5]: ") or "1e-5")
        batch_size = int(input("📦 Batch size [32]: ") or "32")
        patience_final = int(input("🛑 Early stopping patience (final) [10]: ") or "10")
        patience_cv = int(input("🛑 Early stopping patience (CV) [10]: ") or "10")
        n_bootstrap = int(input("🎲 Bootstrap samples (CI 95%) [1000]: ") or "1000")
        confirm = input("🔬 Run interpretability (SHAP, Captum, LIME)? [y/n] (y): ") or "y"
        run_interpretability = confirm.lower() == "y"

    # Summary
    print("\n" + "=" * 70)
    print("📋 Configuration Summary:")
    print("=" * 70)
    print(f"   Transferred model:  {os.path.basename(transferred_path)}")
    print(f"   Fine-Tune epochs:   {epochs_final} (final) / {epochs_cv} (CV)")
    print(f"   Learning Rate:      {lr_final} (final) / {lr_cv} (CV)")
    print(f"   Batch size:         {batch_size}")
    print(f"   Patience:           {patience_final} (final) / {patience_cv} (CV)")
    print(f"   Bootstrap samples:  {n_bootstrap}")
    print(f"   Interpretability:   {'✅ Yes' if run_interpretability else '❌ No'}")
    print("=" * 70)

    # Confirm
    if HAS_RICH:
        if not Confirm.ask("\n🚀 Start full B1 Fine-Tune pipeline?", default=True):
            return
    else:
        confirm = input(f"\n{Colors.WARNING}🚀 Start? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            return

    # ================================================================
    # ✅ STEP 3: Run Fine-Tune + CV + CI
    # ================================================================
    print("\n" + "=" * 80)
    print("🧠 Step 3: Running Fine-Tune + CV + CI")
    print("=" * 80)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = run_b1_finetune_pipeline(
            transferred_path=transferred_path,
            epochs_final=epochs_final,
            epochs_cv=epochs_cv,
            lr_final=lr_final,
            lr_cv=lr_cv,
            head_lr=1e-3,
            batch_size=batch_size,
            patience_final=patience_final,
            patience_cv=patience_cv,
            n_bootstrap=n_bootstrap,
            device=device,
            verbose=True,
        )

        if results.get("status") != "success":
            print(f"\n❌ Fine-Tune pipeline failed!")
            return

        # Extract results
        final_metrics = results["final_metrics"]
        cv_summary = results["cv_summary"]
        ci95 = results["ci95"]
        history = results["history"]

        print(f"\n✅ Fine-Tune + CV + CI complete!")

    except Exception as e:
        print(f"\n❌ Error in Fine-Tune pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================================================================
    # ✅ STEP 4: Load final model for reporting
    # ================================================================
    print("\n🔍 Step 4: Loading final model for reporting...")

    final_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from src.models.efficientNet.B1 import create_efficientnet_b1

        final_model = create_efficientnet_b1(
            num_classes=4, pretrained=False, verbose=False,
        )

        if B1_FINAL_MODEL_PATH and os.path.exists(B1_FINAL_MODEL_PATH):
            ckpt = torch.load(B1_FINAL_MODEL_PATH, map_location="cpu")
            if "model_state_dict" in ckpt:
                final_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                final_model.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded final model from: {B1_FINAL_MODEL_PATH}")
        else:
            print(f"⚠️ Final model file not found at: {B1_FINAL_MODEL_PATH}")
            print("   Skipping interpretability generation...")

        final_model = final_model.to(device)
        final_model.eval()

    except Exception as e:
        print(f"⚠️ Could not reload final model: {e}")
        import traceback
        traceback.print_exc()
        final_model = None

    # ================================================================
    # ✅ STEP 5: Load TEST SET for interpretability (FIXED)
    # ================================================================
    print("\n📊 Step 5: Preparing data for interpretability...")

    image_paths, labels = [], []

    if final_model is not None and run_interpretability:
        # ✅ FIX: Use TEST SET from split_info (not full dataset)
        split_info = results.get("split_info", {})
        test_paths = split_info.get("test_paths", [])
        test_labels = split_info.get("test_labels", [])

        if len(test_paths) > 0:
            image_paths = test_paths
            labels = test_labels
            print(f"✅ Using TEST SET for interpretability: {len(image_paths)} images")
            print(f"   (This is the held-out set, never seen during training)")
        else:
            # Fallback: use full dataset
            print("⚠️ Test set not in results — falling back to full dataset")
            try:
                image_paths, labels = load_labeled_data()
                print(f"✅ Loaded {len(image_paths)} images (full dataset)")
            except Exception as e:
                print(f"⚠️ Could not load labeled data: {e}")
                image_paths, labels = [], []
    else:
        if final_model is None:
            print("⚠️ Model not available — skipping")
        elif not run_interpretability:
            print("ℹ️  Interpretability disabled by user")

    # ================================================================
    # ✅ STEP 6: Generate 23-page PDF report
    # ================================================================
    print("\n" + "=" * 80)
    print("📄 Step 6: Generating 23-page PDF report")
    print("=" * 80)

    pdf_path = None

    if final_model is not None:
        try:
            # ✅ v8.0: Pass ensemble info to reporter
            final_metrics_for_report = dict(final_metrics)
            final_metrics_for_report["ensemble_seeds"] = results.get("ensemble_seeds", [])
            final_metrics_for_report["per_seed_metrics"] = results.get("ensemble_results", [])
            final_metrics_for_report["test_paths"] = results.get("test_paths", [])
            final_metrics_for_report["test_labels"] = results.get("test_labels", [])

            pdf_path = generate_b1_report(
                final_model=final_model,
                metrics=final_metrics_for_report,   # ← اینجا changed
                history=history,
                cv_summary=cv_summary,
                ci95=ci95,
                image_paths=image_paths,
                labels=labels,
                transferred_path=transferred_path,
                device=device,
                generate_interpretability_flag=run_interpretability,
                verbose=True,
            )

            if pdf_path:
                print(f"\n✅ Report generated: {pdf_path}")
            else:
                print(f"\n⚠️ Report generation returned None")

        except Exception as e:
            print(f"\n❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            pdf_path = None
    else:
        print("⚠️ Skipping report generation (model not loaded)")

    # ================================================================
    # ✅ STEP 7: Display Final Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("🎉 B1 FINE-TUNE COMPLETE!")
    print("=" * 80)

    if HAS_RICH:
        # Rich summary table
        summary_table = Table(
            title="🎉 Final Results",
            box=box.ROUNDED,
            border_style="green",
        )
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold white")

        summary_table.add_row("Transferred Model", os.path.basename(transferred_path))
        summary_table.add_row("Final Accuracy", f"{final_metrics['accuracy']:.4f}")
        summary_table.add_row("Final F1-Score", f"{final_metrics['f1_score']:.4f}")
        summary_table.add_row("Final AUC", f"{final_metrics['roc_auc_mean']:.4f}")
        summary_table.add_row("Balanced Accuracy", f"{final_metrics['balanced_accuracy']:.4f}")
        summary_table.add_row(
            "CV Mean Accuracy",
            f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}",
        )
        summary_table.add_row(
            "CV Mean F1",
            f"{cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}",
        )
        summary_table.add_row(
            "CV Mean AUC",
            f"{cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}",
        )
        summary_table.add_row(
            "CI 95% Accuracy",
            f"[{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]",
        )
        summary_table.add_row(
            "CI 95% F1",
            f"[{ci95['f1_score']['ci95_low']:.4f}, {ci95['f1_score']['ci95_high']:.4f}]",
        )

        if pdf_path:
            summary_table.add_row("PDF Report", os.path.basename(pdf_path))

        console.print(summary_table)

        # Output files
        console.print("\n[bold green]📂 Output Files:[/bold green]")
        if B1_PDF_PATH:
            console.print(f"   📄 PDF Report:       {B1_PDF_PATH}")
        if B1_PLOTS_DIR:
            console.print(f"   🎨 Plots:            {B1_PLOTS_DIR}")
        if B1_INTERP_DIR:
            console.print(f"   🔬 Interpretability: {B1_INTERP_DIR}")
        if B1_CV_DIR:
            console.print(f"   📊 CV:               {B1_CV_DIR}")
        if B1_ARCH_DIR:
            console.print(f"   🏗️  Architecture:     {B1_ARCH_DIR}")
        if B1_FINAL_MODEL_PATH:
            console.print(f"   🎯 Final Model:      {B1_FINAL_MODEL_PATH}")
        if B1_REPORTER_ROOT:
            console.print(f"   📈 Metrics JSON:     {os.path.join(B1_REPORTER_ROOT, 'metrics.json')}")

    else:
        # Plain summary
        print(f"\n📊 Final Results:")
        print(f"   Transferred Model:  {os.path.basename(transferred_path)}")
        print(f"   Final Accuracy:     {final_metrics['accuracy']:.4f}")
        print(f"   Final F1-Score:     {final_metrics['f1_score']:.4f}")
        print(f"   Final AUC:          {final_metrics['roc_auc_mean']:.4f}")
        print(f"   Balanced Accuracy:  {final_metrics['balanced_accuracy']:.4f}")
        print(f"\n   5-Fold CV:")
        print(f"   Mean Accuracy:      {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"   Mean F1:            {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        print(f"   Mean AUC:           {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}")
        print(f"\n   95% CI (Bootstrap):")
        print(f"   Accuracy:           [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
        print(f"   F1:                 [{ci95['f1_score']['ci95_low']:.4f}, {ci95['f1_score']['ci95_high']:.4f}]")

        print(f"\n📂 Output Files:")
        if B1_PDF_PATH:
            print(f"   📄 PDF Report:       {B1_PDF_PATH}")
        if B1_PLOTS_DIR:
            print(f"   🎨 Plots:            {B1_PLOTS_DIR}")
        if B1_INTERP_DIR:
            print(f"   🔬 Interpretability: {B1_INTERP_DIR}")
        if B1_CV_DIR:
            print(f"   📊 CV:               {B1_CV_DIR}")

    print("=" * 80)
    print("✅ Option 13 complete!")
    print("=" * 80)

# ================================================================
# ✅ AlexNet SSL - SINGLE METHOD (BYOL / SimCLR / MAE)
# ================================================================

def run_alexnet_ssl_single(method: str = "byol"):
    """
    Run AlexNet SSL with a single method (BYOL / SimCLR / MAE) and generate report.

    ✅ FIXED: LR recommended based on method:
    - SimCLR: 5e-4 (needs higher LR for NT-Xent)
    - BYOL: 1e-4
    - MAE: 1e-4

    Args:
        method: 'byol', 'simclr', or 'mae'
    """
    if not HAS_ALEXNET_SSL:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ AlexNet SSL module not available![/bold red]\n"
                "[yellow]Make sure src/pipeLine/alexnetSSL.py exists.[/yellow]",
                border_style="red",
            ))
        else:
            print(f"{Colors.FAIL}❌ AlexNet SSL module not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(
            Panel.fit(f"🧠 AlexNet SSL - {method.upper()}", border_style="green")
        )
    else:
        print(f"\n{Colors.GREEN}🧠 AlexNet SSL - {method.upper()}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    #default_image_path = os.path.join("finalReports", "cropped")

    # ✅ LR پیشنهادی بر اساس method
    #if method == "simclr":
        #default_lr = "0.002"
        #default_epochs = "80"    # SimCLR needs more epochs
    #elif method == "byol":
        #default_lr = "0.0001"    # 1e-4
        #default_epochs = "40"
    #else:  # mae
        #default_lr = "0.0001"
        #default_epochs = "60"

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    # ✅ FIX: اول batch_size را بگیر، بعد default_lr را محاسبه کن
    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))

        # ✅ حالا LR را بر اساس batch_size و method محاسبه کن
        # BYOL paper: lr = 0.2 * batch_size / 256
        # SimCLR paper: lr = 0.3 * batch_size / 256
        if method == "byol":
            default_lr = f"{0.2 * batch_size / 256:.6f}"
            default_epochs = "40"
        elif method == "simclr":
            default_lr = f"{0.3 * batch_size / 256:.6f}"
            default_epochs = "30"
        else:  # mae
            default_lr = "0.00015"
            default_epochs = "40"

        total_epochs = int(Prompt.ask("⏳ Number of epochs", default=default_epochs))
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )
        lr = float(Prompt.ask(f"📊 Learning Rate (for {method.upper()})", default=default_lr))

        if not Confirm.ask(f"\n🚀 Start AlexNet SSL with {method.upper()}?", default=True):
            return
    else:
        image_path = (
            input(f"\n📁 Enter path (default={default_image_path}): ")
            or default_image_path
        )
        batch_size = int(input("📦 Batch size (32): ") or "32")

        # ✅ محاسبه LR بر اساس batch_size
        if method == "byol":
            default_lr = 0.2 * batch_size / 256
            default_epochs = 40
        elif method == "simclr":
            default_lr = 0.3 * batch_size / 256
            default_epochs = 30
        else:
            default_lr = 0.00015
            default_epochs = 40

        total_epochs = int(input(f"⏳ Number of epochs (default={default_epochs}): ") or str(default_epochs))
        optimizer_name = "adamw"
        lr = float(input(f"📊 Learning Rate (default={default_lr:.6f}): ") or str(default_lr))

        confirm = input(f"\n🚀 Start? (y/n): ")
        if confirm.lower() != "y":
            return

    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
        total_epochs = int(Prompt.ask("⏳ Number of epochs", default=default_epochs))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )
        lr = float(Prompt.ask(f"📊 Learning Rate (for {method.upper()})", default=default_lr))

        if not Confirm.ask(f"\n🚀 Start AlexNet SSL with {method.upper()}?", default=True):
            return
    else:
        image_path = (
            input(
                f"\n{Colors.BLUE}📁 Enter path to images directory "
                f"(default={default_image_path}): {Colors.ENDC}"
            )
            or default_image_path
        )
        total_epochs = int(
            input(f"{Colors.BLUE}⏳ Number of epochs (default={default_epochs}): {Colors.ENDC}")
            or default_epochs
        )
        batch_size = int(
            input(f"{Colors.BLUE}📦 Batch size (default=32): {Colors.ENDC}")
            or "32"
        )
        print(f"{Colors.BLUE}🔧 Select optimizer:{Colors.ENDC}")
        print(f"   1. AdamW (Default)\n   2. RAdam\n   3. Ranger")
        opt_choice = input(f"{Colors.BOLD}👉 Enter choice (1-3): {Colors.ENDC}")
        opt_map = {"1": "adamw", "2": "radam", "3": "ranger"}
        optimizer_name = opt_map.get(opt_choice, "adamw")

        lr = float(
            input(f"{Colors.BLUE}📊 Learning Rate (default={default_lr}): {Colors.ENDC}")
            or default_lr
        )

        confirm = input(
            f"\n{Colors.WARNING}🚀 Start AlexNet SSL with {method.upper()}? (y/n): {Colors.ENDC}"
        )
        if confirm.lower() != "y":
            return

    try:
        from src.pipeLine.Architects.AlexnetFT.alexnetSSL import create_alexnet_ssl_pipeline

        image_paths = []
        if os.path.exists(image_path):
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
                image_paths.extend([
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.endswith(ext)
                ])

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found!{Colors.ENDC}")
            return

        if HAS_RICH:
            console.print(f"[green]✅ Found {len(image_paths)} images[/green]")
            console.print(
                f"[cyan]🔬 Method: {method.upper()} | LR: {lr} | Epochs: {total_epochs}[/cyan]"
            )

        # ✅ Create pipeline with method-specific settings
        pipeline = create_alexnet_ssl_pipeline(
            method=method,
            batch_size=batch_size,
            total_epochs=total_epochs,
            lr=lr,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
            optimizer_name=optimizer_name,
            use_kornia=True,
            enhance_ultrasound=True,
            num_workers=4,  # ✅ FIX: از 0 به 4
        )

        # ✅ Run pretrain
        results = pipeline.pretrain(image_paths)

        # ✅ Generate report
        if HAS_RICH:
            console.print("\n[cyan]📄 Generating PDF report...[/cyan]")
        try:
            report_path = pipeline.generate_report()
            if HAS_RICH:
                console.print(f"[green]✅ PDF Report: {report_path}[/green]")
        except Exception as e:
            if HAS_RICH:
                console.print(f"[yellow]⚠️ PDF generation failed: {e}[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ PDF generation failed: {e}{Colors.ENDC}")

        # ✅ Final summary
        if HAS_RICH:
            console.print(f"\n[bold green]✅ AlexNet {method.upper()} SSL Complete![/bold green]")
            console.print(f"   Best Loss: {results.get('best_loss', 0):.4f}")
            console.print(f"   Best Epoch: {results.get('best_epoch', 0)}")
            console.print(f"   Total Epochs: {results.get('total_epochs', 0)}")

            # ✅ Metrics (test_accuracy + all)
            metrics = results.get("metrics", {})
            if metrics and "accuracy" in metrics:
                console.print(f"\n[bold cyan]📊 Test Metrics:[/bold cyan]")
                console.print(f"   Test Accuracy:  {metrics.get('accuracy', 0):.4f}")
                console.print(f"   Test Precision: {metrics.get('precision', 0):.4f}")
                console.print(f"   Test Recall:    {metrics.get('recall', 0):.4f}")
                console.print(f"   Test F1-Score:  {metrics.get('f1_score', 0):.4f}")
                console.print(f"   Test AUC:       {metrics.get('auc', 0):.4f}")

                # Per-class recall
                if "per_class_recall" in metrics:
                    console.print("\n[bold cyan]📊 Per-Class Recall:[/bold cyan]")
                    for i, r in enumerate(metrics["per_class_recall"]):
                        cls_name = ["NL", "G1", "G2", "G3"][i]
                        console.print(f"   {cls_name}: {r:.4f}")

            # Show output paths
            if "saved_paths" in results:
                paths = results["saved_paths"]
                console.print("\n[dim]📂 Saved checkpoints:[/dim]")
                console.print(f"   [dim]Best: {paths.get('best', 'N/A')}[/dim]")
                console.print(f"   [dim]Final: {paths.get('final', 'N/A')}[/dim]")
        else:
            print(f"\n{Colors.GREEN}✅ AlexNet {method.upper()} SSL Complete!{Colors.ENDC}")
            print(f"   Best Loss: {results.get('best_loss', 0):.4f}")
            print(f"   Total Epochs: {results.get('total_epochs', 0)}")

            metrics = results.get("metrics", {})
            if metrics and "accuracy" in metrics:
                print(f"\n📊 Test Metrics:")
                print(f"   Test Accuracy:  {metrics.get('accuracy', 0):.4f}")
                print(f"   Test Precision: {metrics.get('precision', 0):.4f}")
                print(f"   Test Recall:    {metrics.get('recall', 0):.4f}")
                print(f"   Test F1-Score:  {metrics.get('f1_score', 0):.4f}")
                print(f"   Test AUC:       {metrics.get('auc', 0):.4f}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

# ================================================================
# ✅ AlexNet: Find Best SSL Method (g22) — mirror of g6
# ================================================================

# ================================================================
# ✅ AlexNet: Find Best SSL → Transfer to weightTransformer/ (g22)
# ================================================================

def run_alexnet_best_transfer():
    """
    🎯 AlexNet — Find Best SSL Method and Copy to weightTransformer/

    Steps:
    1. Check that BYOL / SimCLR / MAE checkpoints exist
    2. Load each encoder + Linear Probe
    3. Select best method (highest LP accuracy)
    4. Copy best .pth to:
         finalReports/selfSupervisedLearning/AlexNet/weightTransformer/
         alexnet_transferred_from_{best_method}.pth
    5. Save best_method_summary.json

    Output:
        - alexnet_transferred_from_{best_method}.pth
        - best_method_summary.json
    """
    if not HAS_ALEXNET_SSL:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ AlexNet SSL module not available![/bold red]",
                border_style="red",
            ))
        else:
            print(f"{Colors.FAIL}❌ AlexNet SSL not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(Panel.fit(
            "🎯 AlexNet — Find Best SSL & Transfer",
            subtitle="Load Checkpoints → Linear Probe → Copy Best to weightTransformer/",
            border_style="cyan",
        ))
    else:
        print(f"\n{Colors.CYAN}🎯 AlexNet — Find Best SSL & Transfer{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}")

    try:
        import shutil
        from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
            evaluate_with_linear_probe,
            build_labeled_eval_loader,
        )
        from src.pipeLine.Architects.AlexnetFT.alexnetSSL import (
            ALEXNET_ARCHITECTURE_DIR,
            ALEXNET_SSL_DIR,
            create_alexnet_ssl_pipeline,
            ALEXNET_PRETRAINED_PATH,
        )

        # ✅ weightTransformer directory
        ALEXNET_WEIGHT_TRANSFORMER_DIR = os.path.join(
            PROJECT_ROOT_MAIN,
            "../../../../finalReports", "selfSupervisedLearning",
            "AlexNet", "weightTransformer",
        )
        os.makedirs(ALEXNET_WEIGHT_TRANSFORMER_DIR, exist_ok=True)

        # ============================================================
        # ✅ STEP 1: Find SSL checkpoints
        # ============================================================
        print("\n" + "=" * 80)
        print("🔍 STEP 1: Looking for AlexNet SSL checkpoints...")
        print("=" * 80)

        methods = ["byol", "simclr", "mae"]
        method_results = {}

        for method in methods:
            candidates = [
                os.path.join(ALEXNET_ARCHITECTURE_DIR, method,
                             f"alexnet_best_update_{method}.pth"),
                os.path.join(ALEXNET_ARCHITECTURE_DIR, method,
                             f"alexnet_final_update_{method}.pth"),
                os.path.join(ALEXNET_SSL_DIR,
                             f"alexnet_ssl_{method}_for_transfer.pth"),
            ]
            model_path = None
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break

            if model_path is None:
                print(f"⚠️  {method.upper()}: No checkpoint found — SKIPPING")
                continue

            print(f"✅ {method.upper()}: {model_path}")
            method_results[method] = {"model_path": model_path}

        if len(method_results) == 0:
            print("\n❌ No AlexNet SSL checkpoints found!")
            print("👉 Run options 19/20/21 first (AlexNet BYOL/SimCLR/MAE).")
            return {"status": "failed", "error": "No checkpoints"}

        # ============================================================
        # ✅ STEP 2: Linear Probe on each method
        # ============================================================
        print("\n" + "=" * 80)
        print("🔬 STEP 2: Running Linear Probe on each checkpoint...")
        print("=" * 80)

        eval_loader = build_labeled_eval_loader(
            labels_csv="finalReports/labels.csv",
            image_dir="../../../../finalReports/cropped",
            input_size=224,
            batch_size=32,
            max_per_class=200,
        )

        linear_probe_results = {}

        for method, res in method_results.items():
            print(f"\n📌 Loading {method.upper()}...")
            try:
                pipeline = create_alexnet_ssl_pipeline(
                    method=method,
                    input_size=224,
                    batch_size=32,
                    total_epochs=1,
                    lr=1e-4,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    save_dir=os.path.join(ALEXNET_SSL_DIR, method),
                    pretrained_path=res["model_path"],  # ← بارگذاری checkpoint از قبل
                )

                encoder = pipeline.get_encoder()
                probe = evaluate_with_linear_probe(
                    encoder=encoder,
                    labeled_loader=eval_loader,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    num_classes=4,
                    test_size=0.3,
                )
                linear_probe_results[method] = probe
                print(f"   ✅ {method.upper()} LP Acc: {probe['linear_probe_accuracy']:.4f} "
                      f"| LP F1: {probe['linear_probe_f1']:.4f}")
            except Exception as e:
                print(f"   ⚠️ {method.upper()} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(linear_probe_results) == 0:
            print("\n❌ No method could be loaded!")
            return {"status": "failed", "error": "No method loaded"}

        # ============================================================
        # ✅ STEP 3: Select best method
        # ============================================================
        best_method = max(
            linear_probe_results.keys(),
            key=lambda x: linear_probe_results[x]["linear_probe_accuracy"],
        )
        best_lp_acc = linear_probe_results[best_method]["linear_probe_accuracy"]
        best_lp_f1 = linear_probe_results[best_method]["linear_probe_f1"]
        best_source_path = method_results[best_method]["model_path"]

        print("\n" + "=" * 80)
        print("🏆 STEP 3: BEST METHOD SELECTED")
        print("=" * 80)
        for m, probe in linear_probe_results.items():
            marker = " ✅ WINNER" if m == best_method else ""
            print(f"   {m.upper():<8} LP Acc: {probe['linear_probe_accuracy']:.4f}  "
                  f"LP F1: {probe['linear_probe_f1']:.4f}{marker}")
        print(f"\n   ✅ Winner: {best_method.upper()} (LP Acc = {best_lp_acc:.4f})")
        print("=" * 80)

        # ============================================================
        # ✅ STEP 4: Copy best .pth to weightTransformer/
        # ============================================================
        print("\n" + "=" * 80)
        print("📋 STEP 4: Copying best checkpoint to weightTransformer/...")
        print("=" * 80)

        # ✅ name like: alexnet_transferred_from_byol.pth
        transfer_path = os.path.join(
            ALEXNET_WEIGHT_TRANSFORMER_DIR,
            f"alexnet_transferred_from_{best_method}.pth",
        )

        shutil.copy2(best_source_path, transfer_path)
        print(f"✅ Copied: {best_source_path}")
        print(f"     → {transfer_path}")

        # ✅ Also copy the encoder weights with metadata (safer format)
        # so option 23 can read metadata (source_method, lp_acc, etc.)
        try:
            ckpt = torch.load(best_source_path, map_location="cpu", weights_only=False)

            # Extract encoder state
            if "encoder_state_dict" in ckpt:
                encoder_state = ckpt["encoder_state_dict"]
            elif "model_state_dict" in ckpt:
                encoder_state = ckpt["model_state_dict"]
            else:
                encoder_state = ckpt

            # Get original best_loss if present
            original_loss = ckpt.get("best_loss", None)

            # ✅ FIX: Use integer for transferred_layers (not string)
            total_layers_count = len(encoder_state)
            transferred_layers_count = total_layers_count  # same architecture → all transferred

            # Save with metadata
            torch.save(
                {
                    "encoder_state_dict": encoder_state,
                    "source_method": best_method,
                    "source_model": "AlexNet_SSL",
                    "target_model": "AlexNet",
                    "best_lp_accuracy": best_lp_acc,
                    "best_lp_f1": best_lp_f1,
                    "original_best_loss": original_loss,
                    "transferred_layers": transferred_layers_count,  # ✅ int, not string
                    "total_layers": total_layers_count,
                    "transfer_note": "all layers (same architecture)",
                    "all_methods_lp": {
                        m: {
                            "lp_accuracy": p["linear_probe_accuracy"],
                            "lp_f1": p["linear_probe_f1"],
                        }
                        for m, p in linear_probe_results.items()
                    },
                    "source_checkpoint": best_source_path,
                    "timestamp": datetime.now().isoformat(),
                },
                transfer_path,
            )
            print(f"✅ Metadata-enriched model saved")
        except Exception as e:
            print(f"⚠️ Could not enrich with metadata: {e}")
            # Already copied; OK to continue

        # ============================================================
        # ✅ STEP 5: Save summary JSON
        # ============================================================
        print("\n" + "=" * 80)
        print("💾 STEP 5: Saving best-method metadata...")
        print("=" * 80)

        summary = {
            "best_method": best_method,
            "best_lp_accuracy": best_lp_acc,
            "best_lp_f1": best_lp_f1,
            "best_checkpoint_source": best_source_path,
            "transferred_checkpoint": transfer_path,
            "all_methods": {
                m: {
                    "lp_accuracy": p["linear_probe_accuracy"],
                    "lp_f1": p["linear_probe_f1"],
                    "checkpoint": method_results[m]["model_path"],
                }
                for m, p in linear_probe_results.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = os.path.join(ALEXNET_WEIGHT_TRANSFORMER_DIR, "best_method_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"✅ Summary saved: {summary_path}")

        # ============================================================
        # ✅ DONE
        # ============================================================
        print("\n" + "=" * 80)
        print("🎉 DONE!")
        print("=" * 80)

        if HAS_RICH:
            console.print(f"\n[bold green]✅ Best Method: {best_method.upper()}[/bold green]")
            console.print(f"   LP Accuracy: {best_lp_acc:.4f}")
            console.print(f"   LP F1: {best_lp_f1:.4f}")
            console.print(f"\n[cyan]📂 Transferred checkpoint:[/cyan]")
            console.print(f"   {transfer_path}")
            console.print(f"\n[yellow]👉 Next: run [bold]option 23[/bold] to fine-tune this model.[/yellow]")
        else:
            print(f"\n✅ Best Method: {best_method.upper()}")
            print(f"   LP Accuracy: {best_lp_acc:.4f}")
            print(f"   LP F1: {best_lp_f1:.4f}")
            print(f"\n📂 Transferred checkpoint:")
            print(f"   {transfer_path}")
            print(f"\n👉 Next: run option 23 to fine-tune.")

        return {
            "status": "success",
            "best_method": best_method,
            "best_lp_accuracy": best_lp_acc,
            "transferred_checkpoint": transfer_path,
            "summary_path": summary_path,
        }

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

# ================================================================
# ✅ AlexNet: Complete FFT + TTA + CV5 + CI95 (g23) — mirror of g7
# ================================================================

# ================================================================
# ✅ AlexNet: Full FFT on transferred model (g23)
# ================================================================

def run_alexnet_ensemble_finetune():
    """
    OPTION 23: AlexNet Complete Fine-Tune Pipeline.

    Uses the model in:
        finalReports/selfSupervisedLearning/AlexNet/weightTransformer/
        alexnet_transferred_from_*.pth
    (copied by option 22)

    Flow:
    1. Load transferred model from weightTransformer/
    2. Fine-Tune on labeled data
    3. 5-Fold CV
    4. Bootstrap 95% CI
    5. TTA (16 augmentations)
    6. Generate 23-page PDF report
    """
    if not HAS_ALEXNET_FINETUNE:
        if HAS_RICH:
            console.print("[red]❌ AlexNet Fine-Tune modules not available![/red]")
            console.print("[yellow]   Make sure these files exist:[/yellow]")
            console.print("   - src/pipeLine/alexnetFineTune.py")
            console.print("   - src/pipeLine/alexnetReporter.py")
        else:
            print(f"{Colors.FAIL}❌ AlexNet Fine-Tune modules not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(Panel.fit(
            "🏆 AlexNet Complete Fine-Tune",
            subtitle="Load from weightTransformer/ → FFT → TTA → CV5 → CI95% → 23-page PDF",
            border_style="magenta",
        ))
    else:
        print(f"\n{Colors.MAGENTA}🏆 AlexNet Complete Fine-Tune{Colors.ENDC}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.ENDC}")

    # ============================================================
    # ✅ STEP 1: Find transferred model in weightTransformer/
    # ============================================================
    print("\n🔍 Step 1: Finding transferred model in weightTransformer/...")

    from src.pipeLine.Architects.AlexnetFT.alexnetFineTune import (
        check_transferred_model,
        get_transferred_metadata,
    )

    try:
        exists, transferred_path = check_transferred_model()
    except Exception as e:
        print(f"❌ Error checking transferred model: {e}")
        import traceback
        traceback.print_exc()
        return

    if not exists or transferred_path is None:
        print(f"\n❌ Transferred model NOT FOUND!")
        print(f"   Looked for: alexnet_transferred_from_*.pth")
        print(f"   In: finalReports/selfSupervisedLearning/AlexNet/weightTransformer/")
        print(f"\n👉 Please run option 22 first to select and copy the best SSL model.")
        return

    print(f"✅ Found: {transferred_path}")

    # ============================================================
    # ✅ STEP 2: Read metadata
    # ============================================================
    print("\n📋 Step 2: Reading metadata...")
    meta = get_transferred_metadata(transferred_path)

    if "error" in meta:
        print(f"⚠️ Could not read metadata: {meta['error']}")
        source_method = "unknown"
        best_lp = 0.0
        best_lp_f1 = 0.0
    else:
        source_method = meta["source_method"]
        best_lp = meta["best_lp_accuracy"]
        best_lp_f1 = meta["best_lp_f1"]

        if HAS_RICH:
            info_table = Table(
                title="📋 Transferred Model Metadata",
                box=box.ROUNDED,
                border_style="green",
            )
            info_table.add_column("Property", style="bold cyan")
            info_table.add_column("Value", style="bold white")

            info_table.add_row("Source SSL Method", source_method.upper())
            info_table.add_row("Best LP Accuracy", f"{best_lp:.4f}")
            info_table.add_row("Best LP F1", f"{best_lp_f1:.4f}")
            info_table.add_row("Timestamp", meta.get("timestamp", "N/A"))
            info_table.add_row("File", os.path.basename(transferred_path))
            info_table.add_row("Full Path", transferred_path)

            # Show all methods comparison if available
            all_methods = meta.get("all_methods", {})
            if all_methods:
                for m, r in all_methods.items():
                    marker = " 🏆" if m == source_method else ""
                    info_table.add_row(
                        f"  {m.upper()} LP Acc",
                        f"{r.get('lp_accuracy', 0):.4f}{marker}",
                    )

            console.print(info_table)
        else:
            print(f"   Source SSL Method: {source_method.upper()}")
            print(f"   Best LP Accuracy: {best_lp:.4f}")
            print(f"   Best LP F1: {best_lp_f1:.4f}")
            print(f"   File: {os.path.basename(transferred_path)}")

    # ============================================================
    # ✅ STEP 3: Ask for hyperparameters
    # ============================================================
    print("\n⚙️  Step 3: Configuration")
    print("=" * 70)

    if HAS_RICH:
        epochs_final = int(Prompt.ask("⏳ Fine-Tune epochs (final)", default="100"))
        epochs_cv = int(Prompt.ask("⏳ Fine-Tune epochs per CV fold", default="30"))
        lr_final = float(Prompt.ask("📊 Learning Rate (final)", default="0.0001"))
        lr_cv = float(Prompt.ask("📊 Learning Rate (CV)", default="0.0001"))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        patience_final = int(Prompt.ask("🛑 Early stopping patience (final)", default="50"))
        patience_cv = int(Prompt.ask("🛑 Early stopping patience (CV)", default="20"))
        n_bootstrap = int(Prompt.ask("🎲 Bootstrap samples (CI 95%)", default="2000"))
        n_tta = int(Prompt.ask("🔄 TTA augmentations", default="32"))
        cv_folds = int(Prompt.ask("🔬 CV folds", default="5"))
        run_interpretability = Confirm.ask(
            "🔬 Run interpretability (SHAP, Captum, LIME)?", default=True
        )
    else:
        print(f"\n{Colors.BOLD}Configuration (press Enter for defaults):{Colors.ENDC}")
        epochs_final = int(input("⏳ FT epochs (final) [100]: ") or "100")
        epochs_cv = int(input("⏳ FT epochs per CV fold [30]: ") or "30")
        lr_final = float(input("📊 LR (final) [1e-4]: ") or "1e-4")
        lr_cv = float(input("📊 LR (CV) [1e-4]: ") or "15e-4")
        batch_size = int(input("📦 Batch size [32]: ") or "32")
        patience_final = int(input("🛑 Patience (final) [50]: ") or "50")
        patience_cv = int(input("🛑 Patience (CV) [20]: ") or "20")
        n_bootstrap = int(input("🎲 Bootstrap [2000]: ") or "2000")
        n_tta = int(input("🔄 TTA augs [32]: ") or "32")
        cv_folds = int(input("🔬 CV folds [5]: ") or "5")
        confirm = input("🔬 Run interpretability? [y/n] (y): ") or "y"
        run_interpretability = confirm.lower() == "y"

    print("\n" + "=" * 70)
    print("📋 Summary:")
    print("=" * 70)
    print(f"   Transferred model:  {os.path.basename(transferred_path)}")
    print(f"   Source method:      {source_method.upper()}")
    print(f"   FT epochs:          {epochs_final} (final) / {epochs_cv} (CV)")
    print(f"   LR:                 {lr_final} (final) / {lr_cv} (CV)")
    print(f"   Batch size:         {batch_size}")
    print(f"   TTA augmentations:  {n_tta}")
    print(f"   CV folds:           {cv_folds}")
    print(f"   Bootstrap samples:  {n_bootstrap}")
    print(f"   Interpretability:   {'✅ Yes' if run_interpretability else '❌ No'}")
    print("=" * 70)

    if HAS_RICH:
        if not Confirm.ask("\n🚀 Start AlexNet Fine-Tune pipeline?", default=True):
            return
    else:
        confirm = input(f"\n{Colors.WARNING}🚀 Start? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            return

    # ============================================================
    # ✅ STEP 4: Run Fine-Tune + CV + CI + TTA
    # ============================================================
    print("\n" + "=" * 80)
    print("🧠 Step 4: Running Fine-Tune + CV + CI + TTA")
    print("=" * 80)

    try:
        from src.pipeLine.Architects.AlexnetFT.alexnetFineTune import run_alexnet_finetune_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = run_alexnet_finetune_pipeline(
            ssl_checkpoint_path=transferred_path,   # ← از weightTransformer/
            ssl_method=source_method,
            epochs_final=epochs_final,
            epochs_cv=epochs_cv,
            lr_final=lr_final,
            lr_cv=lr_cv,
            batch_size=batch_size,
            patience_final=patience_final,
            patience_cv=patience_cv,
            n_bootstrap=n_bootstrap,
            n_tta=n_tta,
            cv_folds=cv_folds,
            device=device,
            verbose=True,
        )

        if results.get("status") != "success":
            print(f"\n❌ Fine-Tune pipeline failed!")
            return

        final_metrics = results["final_metrics"]
        cv_summary = results["cv_summary"]
        ci95 = results["ci95"]
        history = results["history"]

        print(f"\n✅ Fine-Tune + CV + CI complete!")

    except Exception as e:
        print(f"\n❌ Error in Fine-Tune pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================
    # ✅ STEP 5: Load final model
    # ============================================================
    print("\n🔍 Step 5: Loading final model for reporting...")

    final_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from src.models.AlexNet import create_alexnet

        final_model = create_alexnet(
            num_classes=4, width_mult=0.5, fc6_features=1024,
            fc7_features=512, dropout_fc=0.4,
            use_batch_norm=True, input_size=224, verbose=False,
        )

        if ALEXNET_FINAL_MODEL_PATH and os.path.exists(ALEXNET_FINAL_MODEL_PATH):
            ckpt = torch.load(ALEXNET_FINAL_MODEL_PATH, map_location="cpu")
            if "model_state_dict" in ckpt:
                final_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                final_model.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded: {ALEXNET_FINAL_MODEL_PATH}")
        else:
            print(f"⚠️ Final model not found: {ALEXNET_FINAL_MODEL_PATH}")

        final_model = final_model.to(device)
        final_model.eval()

    except Exception as e:
        print(f"⚠️ Could not reload final model: {e}")
        import traceback
        traceback.print_exc()
        final_model = None

    # ============================================================
    # ✅ STEP 6: Prepare test set for interpretability
    # ============================================================
    print("\n📊 Step 6: Preparing test set for interpretability...")

    image_paths, labels_list = [], []

    if final_model is not None and run_interpretability:
        split_info = results.get("split_info", {})
        test_paths = split_info.get("test_paths", [])
        test_labels = split_info.get("test_labels", [])

        if len(test_paths) > 0:
            image_paths = test_paths
            labels_list = test_labels
            print(f"✅ Using TEST SET: {len(image_paths)} images")
        else:
            print("⚠️ Test set not in results — using full dataset")
            try:
                from src.pipeLine.Architects.AlexnetFT.alexnetFineTune import load_labeled_data
                image_paths, labels_list = load_labeled_data()
                print(f"✅ Loaded {len(image_paths)} images (full dataset)")
            except Exception as e:
                print(f"⚠️ Could not load labeled data: {e}")

    # ============================================================
    # ✅ STEP 7: Generate 24-page PDF
    # ============================================================
    print("\n" + "=" * 80)
    print("📄 Step 7: Generating 24-page PDF report")
    print("=" * 80)

    pdf_path = None

    if final_model is not None:
        try:
            from src.pipeLine.Architects.AlexnetFT.alexnetReporter import generate_alexnet_report
            from src.pipeLine.Architects.AlexnetFT.alexnetFineTune import get_ssl_methods_comparison

            # ✅ Load SSL methods comparison
            ssl_comparison = get_ssl_methods_comparison()
            if ssl_comparison.get("available"):
                print(f"✅ Loaded SSL comparison: {len(ssl_comparison.get('all_methods', {}))} methods")
            else:
                print("⚠️ SSL comparison not available — section will show 'not available'")

            # ✅ Prepare configs
            config_dict = {
                "input_size": 224,
                "epochs_final": epochs_final,
                "epochs_cv": epochs_cv,
                "lr_final": lr_final,
                "lr_cv": lr_cv,
                "batch_size": batch_size,
                "patience_final": patience_final,
                "patience_cv": patience_cv,
                "cv_folds": cv_folds,
                "n_bootstrap": n_bootstrap,
                "device": str(device),
            }

            data_split_dict = {
                "train": results.get("num_train", 0),
                "val": results.get("num_val", 0),
                "test": results.get("num_test", 0),
                "total": results.get("num_train", 0)
                         + results.get("num_val", 0)
                         + results.get("num_test", 0),
            }

            tta_info_dict = {"n_aug": n_tta}

            pdf_path = generate_alexnet_report(
                final_model=final_model,
                metrics=final_metrics,
                history=history,
                cv_summary=cv_summary,
                ci95=ci95,
                image_paths=image_paths,
                labels=labels_list,
                ssl_checkpoint_path=transferred_path,
                ssl_method=source_method,
                config=config_dict,
                data_split=data_split_dict,
                tta_info=tta_info_dict,
                ssl_comparison=ssl_comparison,  # 🆕 NEW
                device=device,
                generate_interpretability_flag=run_interpretability,
                verbose=True,
            )

            if pdf_path:
                print(f"\n✅ 24-page PDF: {pdf_path}")

        except Exception as e:
            print(f"\n❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            pdf_path = None

    # ============================================================
    # ✅ STEP 8: Display Final Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("🎉 ALEXNET FINE-TUNE COMPLETE!")
    print("=" * 80)

    if HAS_RICH:
        summary_table = Table(title="🎉 Final Results", box=box.ROUNDED, border_style="green")
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold white")

        summary_table.add_row("Transferred Model", os.path.basename(transferred_path))
        summary_table.add_row("Source Method", source_method.upper())
        summary_table.add_row("Final Accuracy", f"{final_metrics['accuracy']:.4f}")
        summary_table.add_row("Final F1", f"{final_metrics['f1_score']:.4f}")
        summary_table.add_row("Final AUC", f"{final_metrics['roc_auc_mean']:.4f}")
        summary_table.add_row("Balanced Accuracy", f"{final_metrics['balanced_accuracy']:.4f}")
        summary_table.add_row(
            "CV Mean Accuracy",
            f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}",
        )
        summary_table.add_row(
            "CI 95% Accuracy",
            f"[{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]",
        )
        if pdf_path:
            summary_table.add_row("PDF Report", os.path.basename(pdf_path))

        console.print(summary_table)
    else:
        print(f"\n📊 Results:")
        print(f"   Transferred:       {os.path.basename(transferred_path)}")
        print(f"   Source method:     {source_method.upper()}")
        print(f"   Final Accuracy:    {final_metrics['accuracy']:.4f}")
        print(f"   Final F1:          {final_metrics['f1_score']:.4f}")
        print(f"   Final AUC:         {final_metrics['roc_auc_mean']:.4f}")
        print(f"   Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
        print(f"   CV Mean Acc:       {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"   CI 95% Acc:        [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
        if pdf_path:
            print(f"   PDF Report:        {pdf_path}")

    print("=" * 80)
    print("✅ Option 23 complete!")
    print("=" * 80)

# ================================================================
# ✅ MobileNetV2 SSL - SINGLE METHOD (BYOL / SimCLR / MAE)
# ================================================================

def run_mobilenetv2_ssl_single(method: str = "byol"):
    """
    Run MobileNetV2 SSL with a single method (BYOL / SimCLR / MAE) and generate report.

    Args:
        method: 'byol', 'simclr', or 'mae'
    """
    # ✅ Check availability
    try:
        from src.pipeLine.Architects.MobileNetV2.mobilenetv2SSL import create_mobilenetv2_ssl_pipeline
        HAS_MNV2_SSL = True
    except ImportError as e:
        HAS_MNV2_SSL = False
        if HAS_RICH:
            console.print(Panel.fit(
                f"[bold red]❌ MobileNetV2 SSL module not available![/bold red]\n"
                f"[yellow]Error: {e}[/yellow]\n"
                f"[yellow]Make sure src/pipeLine/mobilenetv2SSL.py exists.[/yellow]",
                border_style="red",
            ))
        else:
            print(f"{Colors.FAIL}❌ MobileNetV2 SSL not available: {e}{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(
            Panel.fit(f"🧠 MobileNetV2 SSL - {method.upper()}", border_style="green")
        )
    else:
        print(f"\n{Colors.GREEN}🧠 MobileNetV2 SSL - {method.upper()}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    # ✅ LR پیشنهادی بر اساس method
    if method == "simclr":
        default_lr = "0.002"
        default_epochs = "80"
    elif method == "byol":
        default_lr = "0.0001"
        default_epochs = "40"
    else:  # mae
        default_lr = "0.0001"
        default_epochs = "60"

    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
        total_epochs = int(Prompt.ask("⏳ Number of epochs", default=default_epochs))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )
        lr = float(Prompt.ask(f"📊 Learning Rate (for {method.upper()})", default=default_lr))

        if not Confirm.ask(f"\n🚀 Start MobileNetV2 SSL with {method.upper()}?", default=True):
            return
    else:
        image_path = (
            input(
                f"\n{Colors.BLUE}📁 Enter path to images directory "
                f"(default={default_image_path}): {Colors.ENDC}"
            )
            or default_image_path
        )
        total_epochs = int(
            input(f"{Colors.BLUE}⏳ Number of epochs (default={default_epochs}): {Colors.ENDC}")
            or default_epochs
        )
        batch_size = int(
            input(f"{Colors.BLUE}📦 Batch size (default=32): {Colors.ENDC}")
            or "32"
        )
        print(f"{Colors.BLUE}🔧 Select optimizer:{Colors.ENDC}")
        print(f"   1. AdamW (Default)\n   2. RAdam\n   3. Ranger")
        opt_choice = input(f"{Colors.BOLD}👉 Enter choice (1-3): {Colors.ENDC}")
        opt_map = {"1": "adamw", "2": "radam", "3": "ranger"}
        optimizer_name = opt_map.get(opt_choice, "adamw")

        lr = float(
            input(f"{Colors.BLUE}📊 Learning Rate (default={default_lr}): {Colors.ENDC}")
            or default_lr
        )

        confirm = input(
            f"\n{Colors.WARNING}🚀 Start MobileNetV2 SSL with {method.upper()}? (y/n): {Colors.ENDC}"
        )
        if confirm.lower() != "y":
            return

    try:
        # ✅ Collect image paths
        image_paths = []
        if os.path.exists(image_path):
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
                image_paths.extend([
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.endswith(ext)
                ])

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found!{Colors.ENDC}")
            return

        if HAS_RICH:
            console.print(f"[green]✅ Found {len(image_paths)} images[/green]")
            console.print(
                f"[cyan]🔬 Method: {method.upper()} | LR: {lr} | Epochs: {total_epochs}[/cyan]"
            )
        else:
            print(f"{Colors.GREEN}✅ Found {len(image_paths)} images{Colors.ENDC}")

        # ✅ Create pipeline
        pipeline = create_mobilenetv2_ssl_pipeline(
            method=method,
            batch_size=batch_size,
            total_epochs=total_epochs,
            lr=lr,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
            optimizer_name=optimizer_name,
            use_kornia=True,
            enhance_ultrasound=True,
        )

        # ✅ Run pretrain
        results = pipeline.pretrain(image_paths)

        # ✅ Generate report
        if HAS_RICH:
            console.print("\n[cyan]📄 Generating PDF report...[/cyan]")
        try:
            report_path = pipeline.generate_report()
            if HAS_RICH:
                console.print(f"[green]✅ PDF Report: {report_path}[/green]")
        except Exception as e:
            if HAS_RICH:
                console.print(f"[yellow]⚠️ PDF generation failed: {e}[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ PDF generation failed: {e}{Colors.ENDC}")

        # ✅ Final summary
        if HAS_RICH:
            console.print(f"\n[bold green]✅ MobileNetV2 {method.upper()} SSL Complete![/bold green]")
            console.print(f"   Best Loss: {results.get('best_loss', 0):.4f}")
            console.print(f"   Best Epoch: {results.get('best_epoch', 0)}")
            console.print(f"   Total Epochs: {results.get('total_epochs', 0)}")

            metrics = results.get("metrics", {})
            if metrics and "accuracy" in metrics:
                console.print(f"\n[bold cyan]📊 Test Metrics:[/bold cyan]")
                console.print(f"   Test Accuracy:  {metrics.get('accuracy', 0):.4f}")
                console.print(f"   Test Precision: {metrics.get('precision', 0):.4f}")
                console.print(f"   Test Recall:    {metrics.get('recall', 0):.4f}")
                console.print(f"   Test F1-Score:  {metrics.get('f1_score', 0):.4f}")
                console.print(f"   Test AUC:       {metrics.get('auc', 0):.4f}")

                if "per_class_recall" in metrics:
                    console.print("\n[bold cyan]📊 Per-Class Recall:[/bold cyan]")
                    for i, r in enumerate(metrics["per_class_recall"]):
                        cls_name = ["NL", "G1", "G2", "G3"][i]
                        console.print(f"   {cls_name}: {r:.4f}")

            if "saved_paths" in results:
                paths = results["saved_paths"]
                console.print("\n[dim]📂 Saved checkpoints:[/dim]")
                console.print(f"   [dim]Best: {paths.get('best', 'N/A')}[/dim]")
                console.print(f"   [dim]Final: {paths.get('final', 'N/A')}[/dim]")
        else:
            print(f"\n{Colors.GREEN}✅ MobileNetV2 {method.upper()} SSL Complete!{Colors.ENDC}")
            print(f"   Best Loss: {results.get('best_loss', 0):.4f}")
            print(f"   Total Epochs: {results.get('total_epochs', 0)}")

            metrics = results.get("metrics", {})
            if metrics and "accuracy" in metrics:
                print(f"\n📊 Test Metrics:")
                print(f"   Test Accuracy:  {metrics.get('accuracy', 0):.4f}")
                print(f"   Test Precision: {metrics.get('precision', 0):.4f}")
                print(f"   Test Recall:    {metrics.get('recall', 0):.4f}")
                print(f"   Test F1-Score:  {metrics.get('f1_score', 0):.4f}")
                print(f"   Test AUC:       {metrics.get('auc', 0):.4f}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

# ================================================================
# ✅ MobileNetV2 SSL Full (g27)
# ================================================================

def run_mobilenetv2_ssl():
    """MobileNetV2 SSL Full Pipeline: BYOL → SimCLR → MAE"""
    if not HAS_MOBILENETV2_SSL:
        if HAS_RICH:
            console.print("[red]❌ MobileNetV2 SSL not available![/red]")
        else:
            print(f"{Colors.FAIL}❌ MobileNetV2 SSL not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(Panel.fit(
            "🎯 MobileNetV2 SSL Full",
            subtitle="BYOL → SimCLR → MAE",
            border_style="green",
        ))

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    if HAS_RICH:
        image_path = Prompt.ask("📁 Path to images", default=default_image_path)
        total_epochs = int(Prompt.ask("⏳ Epochs per method", default="40"))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        lr = float(Prompt.ask("📊 LR", default="0.0001"))
        if not Confirm.ask("\n🚀 Start?", default=True):
            return
    else:
        image_path = input(f"📁 Path [{default_image_path}]: ") or default_image_path
        total_epochs = int(input("⏳ Epochs [40]: ") or "40")
        batch_size = int(input("📦 Batch [32]: ") or "32")
        lr = float(input("📊 LR [0.0001]: ") or "0.0001")
        if input("🚀 Start? (y/n): ").lower() != "y":
            return

    try:
        from src.pipeLine.Architects.MobileNetV2.mobilenetv2SSL import create_mobilenetv2_ssl_pipeline

        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            if os.path.exists(image_path):
                image_paths.extend([
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.endswith(ext)
                ])

        if not image_paths:
            print("❌ No images found!")
            return

        print(f"✅ Found {len(image_paths)} images")
        print("🔬 Running 3 methods: BYOL → SimCLR → MAE")

        # ✅ Run each method separately
        all_results = {}
        for method in ["byol", "simclr", "mae"]:
            print(f"\n{'='*80}")
            print(f"🧠 Training {method.upper()} on MobileNetV2")
            print(f"{'='*80}")

            pipeline = create_mobilenetv2_ssl_pipeline(
                method=method,
                batch_size=batch_size,
                total_epochs=total_epochs,
                lr=lr,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
                use_kornia=True,
            )
            results = pipeline.pretrain(image_paths)
            all_results[method] = results

            # Save for transfer
            try:
                pipeline.save_for_transfer()
            except Exception as e:
                print(f"⚠️ save_for_transfer failed: {e}")

        print(f"\n✅ All 3 methods complete!")
        print(f"   BYOL Loss:   {all_results.get('byol', {}).get('best_loss', 0):.4f}")
        print(f"   SimCLR Loss: {all_results.get('simclr', {}).get('best_loss', 0):.4f}")
        print(f"   MAE Loss:    {all_results.get('mae', {}).get('best_loss', 0):.4f}")
        print(f"\n👉 Run option 28 to select best & copy to weightTransformer/")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ MobileNetV2 Find Best SSL (g28)
# ================================================================

def run_mobilenetv2_best_transfer():
    """MobileNetV2 - Find best SSL via Linear Probe & copy to weightTransformer/"""
    if not HAS_MOBILENETV2_SSL:
        print("❌ MobileNetV2 SSL not available!")
        return

    try:
        import shutil
        from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
            evaluate_with_linear_probe,
            build_labeled_eval_loader,
        )
        from src.pipeLine.Architects.MobileNetV2.mobilenetv2SSL import (
            MOBILENETV2_ARCHITECTURE_DIR,
            MOBILENETV2_SSL_DIR,
            create_mobilenetv2_ssl_pipeline,
        )

        # ✅ weightTransformer directory
        MNV2_WEIGHT_TRANSFORMER_DIR = os.path.join(
            PROJECT_ROOT_MAIN,
            "../../../../finalReports", "selfSupervisedLearning",
            "MobileNetV2", "weightTransformer",
        )
        os.makedirs(MNV2_WEIGHT_TRANSFORMER_DIR, exist_ok=True)

        # ============================================================
        # ✅ STEP 1: Find SSL checkpoints
        # ============================================================
        print("\n" + "=" * 80)
        print("🔍 STEP 1: Looking for MobileNetV2 SSL checkpoints...")
        print("=" * 80)

        methods = ["byol", "simclr", "mae"]
        method_results = {}

        for method in methods:
            candidates = [
                os.path.join(MOBILENETV2_ARCHITECTURE_DIR, method,
                             f"mobilenetv2_best_update_{method}.pth"),
                os.path.join(MOBILENETV2_ARCHITECTURE_DIR, method,
                             f"mobilenetv2_final_update_{method}.pth"),
                os.path.join(MOBILENETV2_SSL_DIR,
                             f"mobilenetv2_ssl_{method}_for_transfer.pth"),
            ]
            model_path = None
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break

            if model_path is None:
                print(f"⚠️  {method.upper()}: No checkpoint found — SKIPPING")
                continue

            print(f"✅ {method.upper()}: {model_path}")
            method_results[method] = {"model_path": model_path}

        if len(method_results) == 0:
            print("\n❌ No MobileNetV2 SSL checkpoints found!")
            print("👉 Run options 24/25/26 first.")
            return {"status": "failed"}

        # ============================================================
        # ✅ STEP 2: Linear Probe on each method
        # ============================================================
        print("\n" + "=" * 80)
        print("🔬 STEP 2: Running Linear Probe on each checkpoint...")
        print("=" * 80)

        eval_loader = build_labeled_eval_loader(
            labels_csv="finalReports/labels.csv",
            image_dir="../../../../finalReports/cropped",
            input_size=224,
            batch_size=32,
            max_per_class=200,
        )

        linear_probe_results = {}

        for method, res in method_results.items():
            print(f"\n📌 Loading {method.upper()}...")
            try:
                pipeline = create_mobilenetv2_ssl_pipeline(
                    method=method,
                    input_size=224,
                    batch_size=32,
                    total_epochs=1,
                    lr=1e-4,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    save_dir=os.path.join(MOBILENETV2_SSL_DIR, method),
                    pretrained_path=res["model_path"],
                )

                encoder = pipeline.get_encoder()
                probe = evaluate_with_linear_probe(
                    encoder=encoder,
                    labeled_loader=eval_loader,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    num_classes=4,
                    test_size=0.3,
                )
                linear_probe_results[method] = probe
                print(f"   ✅ {method.upper()} LP Acc: {probe['linear_probe_accuracy']:.4f} "
                      f"| LP F1: {probe['linear_probe_f1']:.4f}")
            except Exception as e:
                print(f"   ⚠️ {method.upper()} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(linear_probe_results) == 0:
            print("\n❌ No method could be loaded!")
            return {"status": "failed"}

        # ============================================================
        # ✅ STEP 3: Select best
        # ============================================================
        best_method = max(
            linear_probe_results.keys(),
            key=lambda x: linear_probe_results[x]["linear_probe_accuracy"],
        )
        best_lp_acc = linear_probe_results[best_method]["linear_probe_accuracy"]
        best_lp_f1 = linear_probe_results[best_method]["linear_probe_f1"]
        best_source_path = method_results[best_method]["model_path"]

        print("\n" + "=" * 80)
        print("🏆 STEP 3: BEST METHOD SELECTED")
        print("=" * 80)
        for m, probe in linear_probe_results.items():
            marker = " ✅ WINNER" if m == best_method else ""
            print(f"   {m.upper():<8} LP Acc: {probe['linear_probe_accuracy']:.4f}  "
                  f"LP F1: {probe['linear_probe_f1']:.4f}{marker}")
        print(f"\n   ✅ Winner: {best_method.upper()} (LP Acc = {best_lp_acc:.4f})")
        print("=" * 80)

        # ============================================================
        # ✅ STEP 4: Copy to weightTransformer/ with metadata
        # ============================================================
        print("\n" + "=" * 80)
        print("📋 STEP 4: Copying best checkpoint to weightTransformer/...")
        print("=" * 80)

        transfer_path = os.path.join(
            MNV2_WEIGHT_TRANSFORMER_DIR,
            f"mobilenetv2_transferred_from_{best_method}.pth",
        )

        # Load source checkpoint
        ckpt = torch.load(best_source_path, map_location="cpu", weights_only=False)

        if "encoder_state_dict" in ckpt:
            encoder_state = ckpt["encoder_state_dict"]
        elif "model_state_dict" in ckpt:
            encoder_state = ckpt["model_state_dict"]
        else:
            encoder_state = ckpt

        original_loss = ckpt.get("best_loss", None)
        total_layers_count = len(encoder_state)

        # Save with metadata
        torch.save(
            {
                "encoder_state_dict": encoder_state,
                "source_method": best_method,
                "source_model": "MobileNetV2_SSL",
                "target_model": "MobileNetV2",
                "best_lp_accuracy": best_lp_acc,
                "best_lp_f1": best_lp_f1,
                "original_best_loss": original_loss,
                "transferred_layers": total_layers_count,
                "total_layers": total_layers_count,
                "transfer_note": "all layers (same architecture)",
                "all_methods_lp": {
                    m: {
                        "lp_accuracy": p["linear_probe_accuracy"],
                        "lp_f1": p["linear_probe_f1"],
                    }
                    for m, p in linear_probe_results.items()
                },
                "source_checkpoint": best_source_path,
                "timestamp": datetime.now().isoformat(),
            },
            transfer_path,
        )
        print(f"✅ Transferred model saved: {transfer_path}")

        # ============================================================
        # ✅ STEP 5: Save summary JSON
        # ============================================================
        summary_path = os.path.join(MNV2_WEIGHT_TRANSFORMER_DIR, "best_method_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "best_method": best_method,
                "best_lp_accuracy": best_lp_acc,
                "best_lp_f1": best_lp_f1,
                "best_checkpoint_source": best_source_path,
                "transferred_checkpoint": transfer_path,
                "all_methods": {
                    m: {
                        "lp_accuracy": p["linear_probe_accuracy"],
                        "lp_f1": p["linear_probe_f1"],
                        "checkpoint": method_results[m]["model_path"],
                    }
                    for m, p in linear_probe_results.items()
                },
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4)
        print(f"✅ Summary saved: {summary_path}")

        # ============================================================
        # ✅ DONE
        # ============================================================
        print("\n" + "=" * 80)
        print("🎉 DONE!")
        print("=" * 80)
        print(f"\n✅ Best Method: {best_method.upper()}")
        print(f"   LP Accuracy: {best_lp_acc:.4f}")
        print(f"   LP F1: {best_lp_f1:.4f}")
        print(f"\n📂 Transferred checkpoint:")
        print(f"   {transfer_path}")
        print(f"\n👉 Next: run option 29 to fine-tune.")

        return {"status": "success", "best_method": best_method}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed"}


# ================================================================
# ✅ MobileNetV2 Complete Fine-Tune (g29)
# ================================================================

def run_mobilenetv2_ensemble_finetune():
    """MobileNetV2 FFT + TTA + CV5 + CI95 + 24-page PDF"""
    if not HAS_MOBILENETV2_FINETUNE:
        print("❌ MobileNetV2 Fine-Tune not available!")
        return

    from src.pipeLine.Architects.MobileNetV2.mobilenetv2FineTune import (
        check_transferred_model,
        run_mobilenetv2_finetune_pipeline,
        get_transferred_metadata,
        get_ssl_methods_comparison,
        FINAL_MODEL_PATH as MNV2_FINAL_MODEL_PATH,
    )

    # ============================================================
    # ✅ STEP 1: Find transferred model
    # ============================================================
    print("\n🔍 Step 1: Finding transferred model...")
    exists, transferred_path = check_transferred_model()

    if not exists:
        print("❌ Transferred model NOT FOUND!")
        print("👉 Run option 28 first.")
        return

    print(f"✅ Found: {transferred_path}")

    # ============================================================
    # ✅ STEP 2: Metadata
    # ============================================================
    print("\n📋 Step 2: Reading metadata...")
    meta = get_transferred_metadata(transferred_path)
    source_method = meta.get("source_method", "byol")
    best_lp = meta.get("best_lp_accuracy", 0.0)
    best_lp_f1 = meta.get("best_lp_f1", 0.0)

    if HAS_RICH:
        info_table = Table(title="📋 Metadata", box=box.ROUNDED, border_style="green")
        info_table.add_column("Property", style="bold cyan")
        info_table.add_column("Value", style="bold white")
        info_table.add_row("Source Method", source_method.upper())
        info_table.add_row("LP Accuracy", f"{best_lp:.4f}")
        info_table.add_row("LP F1", f"{best_lp_f1:.4f}")
        info_table.add_row("File", os.path.basename(transferred_path))
        console.print(info_table)

    # ============================================================
    # ✅ STEP 3: Configuration
    # ============================================================
    print("\n⚙️  Step 3: Configuration")
    print("=" * 70)

    if HAS_RICH:
        epochs_final = int(Prompt.ask("⏳ FT epochs (final)", default="100"))
        epochs_cv = int(Prompt.ask("⏳ FT epochs (CV fold)", default="30"))
        lr_final = float(Prompt.ask("📊 LR (final)", default="0.0001"))
        lr_cv = float(Prompt.ask("📊 LR (CV)", default="0.0001"))
        batch_size = int(Prompt.ask("📦 Batch size", default="32"))
        patience_final = int(Prompt.ask("🛑 Patience (final)", default="30"))
        patience_cv = int(Prompt.ask("🛑 Patience (CV)", default="15"))
        n_bootstrap = int(Prompt.ask("🎲 Bootstrap", default="2000"))
        n_tta = int(Prompt.ask("🔄 TTA augmentations", default="32"))
        cv_folds = int(Prompt.ask("🔬 CV folds", default="5"))
        run_interp = Confirm.ask("🔬 Interpretability?", default=True)
        if not Confirm.ask("\n🚀 Start MobileNetV2 Fine-Tune?", default=True):
            return
    else:
        epochs_final = int(input("⏳ FT final [100]: ") or "100")
        epochs_cv = int(input("⏳ FT CV [30]: ") or "30")
        lr_final = float(input("📊 LR final [1e-4]: ") or "1e-4")
        lr_cv = float(input("📊 LR CV [1e-4]: ") or "1e-4")
        batch_size = int(input("📦 Batch [32]: ") or "32")
        patience_final = int(input("🛑 Patience final [30]: ") or "30")
        patience_cv = int(input("🛑 Patience CV [15]: ") or "15")
        n_bootstrap = int(input("🎲 Bootstrap [2000]: ") or "2000")
        n_tta = int(input("🔄 TTA [32]: ") or "32")
        cv_folds = int(input("🔬 CV folds [5]: ") or "5")
        run_interp = input("🔬 Interpretability? (y/n) (y): ").lower() != "n"
        if input("🚀 Start? (y/n): ").lower() != "y":
            return

    print("\n" + "=" * 70)
    print("📋 Summary:")
    print("=" * 70)
    print(f"   Transferred:        {os.path.basename(transferred_path)}")
    print(f"   Source method:      {source_method.upper()}")
    print(f"   FT epochs:          {epochs_final} (final) / {epochs_cv} (CV)")
    print(f"   LR:                 {lr_final} (final) / {lr_cv} (CV)")
    print(f"   Batch size:         {batch_size}")
    print(f"   TTA augs:           {n_tta}")
    print(f"   CV folds:           {cv_folds}")
    print(f"   Bootstrap:          {n_bootstrap}")
    print(f"   Interpretability:   {'✅' if run_interp else '❌'}")
    print("=" * 70)

    # ============================================================
    # ✅ STEP 4: Run Fine-Tune pipeline
    # ============================================================
    print("\n" + "=" * 80)
    print("🧠 Step 4: Running Fine-Tune + CV + CI + TTA")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        results = run_mobilenetv2_finetune_pipeline(
            ssl_checkpoint_path=transferred_path,
            ssl_method=source_method,
            epochs_final=epochs_final,
            epochs_cv=epochs_cv,
            lr_final=lr_final,
            lr_cv=lr_cv,
            batch_size=batch_size,
            patience_final=patience_final,
            patience_cv=patience_cv,
            n_bootstrap=n_bootstrap,
            n_tta=n_tta,
            cv_folds=cv_folds,
            device=device,
            verbose=True,
        )

        if results.get("status") != "success":
            print("❌ Pipeline failed")
            return

        final_metrics = results["final_metrics"]
        cv_summary = results["cv_summary"]
        ci95 = results["ci95"]
        history = results["history"]

        print(f"\n✅ Fine-Tune + CV + CI complete!")

    except Exception as e:
        print(f"❌ Error in Fine-Tune pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================
    # ✅ STEP 5: Load final model
    # ============================================================
    print("\n🔍 Step 5: Loading final model...")
    final_model = None

    try:
        from torchvision.models import mobilenet_v2
        final_model = mobilenet_v2(num_classes=4, width_mult=1.0)

        if MNV2_FINAL_MODEL_PATH and os.path.exists(MNV2_FINAL_MODEL_PATH):
            ckpt = torch.load(MNV2_FINAL_MODEL_PATH, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                final_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                final_model.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded: {MNV2_FINAL_MODEL_PATH}")
        else:
            print(f"⚠️ Final model not found: {MNV2_FINAL_MODEL_PATH}")

        final_model = final_model.to(device).eval()

    except Exception as e:
        print(f"⚠️ Could not reload final model: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # ✅ STEP 6: Prepare test set
    # ============================================================
    print("\n📊 Step 6: Preparing test set for interpretability...")
    image_paths, labels_list = [], []

    if final_model is not None and run_interp:
        split_info = results.get("split_info", {})
        test_paths = split_info.get("test_paths", [])
        test_labels = split_info.get("test_labels", [])

        if len(test_paths) > 0:
            image_paths = test_paths
            labels_list = test_labels
            print(f"✅ Using TEST SET: {len(image_paths)} images")
        else:
            print("⚠️ Test set not in results — using full dataset")
            try:
                from src.pipeLine.Architects.MobileNetV2.mobilenetv2FineTune import load_labeled_data
                image_paths, labels_list = load_labeled_data()
                print(f"✅ Loaded {len(image_paths)} images (full dataset)")
            except Exception as e:
                print(f"⚠️ Could not load labeled data: {e}")

    # ============================================================
    # ✅ STEP 7: Generate 24-page PDF
    # ============================================================
    print("\n" + "=" * 80)
    print("📄 Step 7: Generating 24-page PDF report")
    print("=" * 80)

    pdf_path = None

    if final_model is not None:
        try:
            from src.pipeLine.Architects.MobileNetV2.mobilenetv2Reporter import generate_mobilenetv2_report

            # Load SSL comparison
            ssl_comparison = get_ssl_methods_comparison()
            if ssl_comparison.get("available"):
                print(f"✅ Loaded SSL comparison: {len(ssl_comparison.get('all_methods', {}))} methods")
            else:
                print("⚠️ SSL comparison not available")

            # Prepare config dicts
            config_dict = {
                "input_size": 224,
                "epochs_final": epochs_final,
                "epochs_cv": epochs_cv,
                "lr_final": lr_final,
                "lr_cv": lr_cv,
                "batch_size": batch_size,
                "patience_final": patience_final,
                "patience_cv": patience_cv,
                "cv_folds": cv_folds,
                "n_bootstrap": n_bootstrap,
                "device": str(device),
            }

            data_split_dict = {
                "train": results.get("num_train", 0),
                "val": results.get("num_val", 0),
                "test": results.get("num_test", 0),
                "total": results.get("num_train", 0)
                         + results.get("num_val", 0)
                         + results.get("num_test", 0),
            }

            tta_info_dict = {"n_aug": n_tta}

            pdf_path = generate_mobilenetv2_report(
                final_model=final_model,
                metrics=final_metrics,
                history=history,
                cv_summary=cv_summary,
                ci95=ci95,
                image_paths=image_paths,
                labels=labels_list,
                ssl_checkpoint_path=transferred_path,
                ssl_method=source_method,
                config=config_dict,
                data_split=data_split_dict,
                tta_info=tta_info_dict,
                ssl_comparison=ssl_comparison,
                device=device,
                generate_interpretability_flag=run_interp,
                verbose=True,
            )

            if pdf_path:
                print(f"\n✅ 24-page PDF: {pdf_path}")

        except Exception as e:
            print(f"\n❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            pdf_path = None

    # ============================================================
    # ✅ STEP 8: Display Final Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("🎉 MOBILENETV2 FINE-TUNE COMPLETE!")
    print("=" * 80)

    if HAS_RICH:
        summary_table = Table(title="🎉 Final Results", box=box.ROUNDED, border_style="green")
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold white")

        summary_table.add_row("Transferred Model", os.path.basename(transferred_path))
        summary_table.add_row("Source Method", source_method.upper())
        summary_table.add_row("Final Accuracy", f"{final_metrics['accuracy']:.4f}")
        summary_table.add_row("Final F1", f"{final_metrics['f1_score']:.4f}")
        summary_table.add_row("Final AUC", f"{final_metrics['roc_auc_mean']:.4f}")
        summary_table.add_row("Balanced Accuracy", f"{final_metrics['balanced_accuracy']:.4f}")
        summary_table.add_row(
            "CV Mean Accuracy",
            f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}",
        )
        summary_table.add_row(
            "CI 95% Accuracy",
            f"[{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]",
        )
        if pdf_path:
            summary_table.add_row("PDF Report", os.path.basename(pdf_path))

        console.print(summary_table)
    else:
        print(f"\n📊 Results:")
        print(f"   Transferred:       {os.path.basename(transferred_path)}")
        print(f"   Source method:     {source_method.upper()}")
        print(f"   Final Accuracy:    {final_metrics['accuracy']:.4f}")
        print(f"   Final F1:          {final_metrics['f1_score']:.4f}")
        print(f"   Final AUC:         {final_metrics['roc_auc_mean']:.4f}")
        print(f"   Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
        print(f"   CV Mean Acc:       {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"   CI 95% Acc:        [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
        if pdf_path:
            print(f"   PDF Report:        {pdf_path}")

    print("=" * 80)
    print("✅ Option 29 complete!")
    print("=" * 80)

# ================================================================
# ✅ VGG16 SSL - SINGLE METHOD (BYOL / SimCLR / MAE)
# ================================================================

def run_vgg16_ssl_single(method: str = "byol"):
    """
    Run VGG16 SSL with a single method (BYOL / SimCLR / MAE).
    LIGHT config: batch=2, grad_accum=16, image=160.

    Args:
        method: 'byol', 'simclr', or 'mae'
    """
    if not HAS_VGG16_SSL:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ VGG16 SSL module not available![/bold red]\n"
                "[yellow]Make sure src/pipeLine/vgg16SSL.py exists.[/yellow]",
                border_style="red",
            ))
        else:
            print(f"{Colors.FAIL}❌ VGG16 SSL not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(Panel.fit(
            f"🧠 VGG16 SSL - {method.upper()} (LIGHT)",
            border_style="green",
        ))
    else:
        print(f"\n{Colors.GREEN}🧠 VGG16 SSL - {method.upper()}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    if HAS_RICH:
        image_path = Prompt.ask("📁 Enter path to images directory", default=default_image_path)
        batch_size = int(Prompt.ask("📦 Batch size (LIGHT: 2)", default="2"))
        grad_accum = int(Prompt.ask("🔁 Gradient accumulation (LIGHT: 16)", default="16"))

        # ✅ LR محاسبه‌شده بر اساس effective_batch = batch_size × grad_accum
        effective_batch = batch_size * grad_accum
        if method == "byol":
            default_lr = f"{0.2 * effective_batch / 256:.6f}"  # ≈ 0.025 برای effective=32
            default_epochs = "6"
        elif method == "simclr":
            default_lr = f"{0.3 * effective_batch / 256:.6f}"  # ≈ 0.0375
            default_epochs = "7"
        else:  # mae
            default_lr = "0.00015"
            default_epochs = "7"

        total_epochs = int(Prompt.ask("⏳ Number of epochs", default=default_epochs))
        optimizer_name = Prompt.ask(
            "🔧 Select optimizer", choices=["adamw", "radam", "ranger"], default="adamw"
        )
        lr = float(Prompt.ask(f"📊 Learning Rate (for {method.upper()})", default=default_lr))

        if not Confirm.ask(f"\n🚀 Start VGG16 SSL with {method.upper()}?", default=True):
            return
    else:
        image_path = (
                input(f"\n📁 Path [{default_image_path}]: ") or default_image_path
        )
        batch_size = int(input("📦 Batch size [2]: ") or "2")
        grad_accum = int(input("🔁 Grad accumulation [16]: ") or "16")

        effective_batch = batch_size * grad_accum
        if method == "byol":
            default_lr = 0.2 * effective_batch / 256
            default_epochs = 6
        elif method == "simclr":
            default_lr = 0.3 * effective_batch / 256
            default_epochs = 7
        else:
            default_lr = 0.00015
            default_epochs = 7

        total_epochs = int(input(f"⏳ Epochs [{default_epochs}]: ") or str(default_epochs))
        print("🔧 Optimizer: 1.AdamW 2.RAdam 3.Ranger")
        opt = input("👉 [1]: ") or "1"
        optimizer_name = {"1": "adamw", "2": "radam", "3": "ranger"}.get(opt, "adamw")
        lr = float(input(f"📊 LR [{default_lr:.6f}]: ") or str(default_lr))

        if input(f"\n🚀 Start? (y/n): ").lower() != "y":
            return

        if not Confirm.ask(f"\n🚀 Start VGG16 SSL with {method.upper()}?", default=True):
            return


    try:
        from src.pipeLine.Architects.VGG16.vgg16SSL import create_vgg16_ssl_pipeline

        image_paths = []
        if os.path.exists(image_path):
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
                image_paths.extend([
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.endswith(ext)
                ])

        if len(image_paths) == 0:
            if HAS_RICH:
                console.print("[red]❌ No images found![/red]")
            else:
                print(f"{Colors.FAIL}❌ No images found!{Colors.ENDC}")
            return

        if HAS_RICH:
            console.print(f"[green]✅ Found {len(image_paths)} images[/green]")
            console.print(f"[cyan]🔬 Method: {method.upper()} | LR: {lr} | Epochs: {total_epochs}[/cyan]")
            console.print(f"[dim]⚠️ LIGHT: batch={batch_size}, grad_accum={grad_accum}, img_size=160[/dim]")

        pipeline = create_vgg16_ssl_pipeline(
            method=method,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            total_epochs=total_epochs,
            lr=lr,
            input_size=240,  # ✅ LIGHT
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
            optimizer_name=optimizer_name,
            use_kornia=True,
            enhance_ultrasound=True,
            num_workers=4,
            patch_size=16,  # ← 8 یا 16 (نه 1)
            mask_ratio=0.5,  # یا 0.75
        )

        results = pipeline.pretrain(image_paths)

        # ✅ Generate report
        if HAS_RICH:
            console.print("\n[cyan]📄 Generating PDF report...[/cyan]")
        try:
            report_path = pipeline.generate_report()
            if HAS_RICH:
                console.print(f"[green]✅ PDF Report: {report_path}[/green]")
        except Exception as e:
            if HAS_RICH:
                console.print(f"[yellow]⚠️ PDF generation failed: {e}[/yellow]")
            else:
                print(f"{Colors.WARNING}⚠️ PDF failed: {e}{Colors.ENDC}")

        # ✅ Final summary
        if HAS_RICH:
            console.print(f"\n[bold green]✅ VGG16 {method.upper()} SSL Complete![/bold green]")
            console.print(f"   Best Loss: {results.get('best_loss', 0):.4f}")
            console.print(f"   Best Epoch: {results.get('best_epoch', 0)}")

            metrics = results.get("metrics", {})
            if metrics and "accuracy" in metrics:
                console.print(f"\n[bold cyan]📊 Test Metrics:[/bold cyan]")
                console.print(f"   Accuracy:  {metrics.get('accuracy', 0):.4f}")
                console.print(f"   Balanced:  {metrics.get('balanced_accuracy', 0):.4f}")
                console.print(f"   F1:        {metrics.get('f1_score', 0):.4f}")
                console.print(f"   AUC:       {metrics.get('auc', 0):.4f}")

                if "per_class_recall" in metrics:
                    console.print("\n[bold cyan]📊 Per-Class Recall:[/bold cyan]")
                    for i, r in enumerate(metrics["per_class_recall"]):
                        cls_name = ["NL", "G1", "G2", "G3"][i]
                        console.print(f"   {cls_name}: {r:.4f}")

            if "saved_paths" in results:
                paths = results["saved_paths"]
                console.print("\n[dim]📂 Saved checkpoints:[/dim]")
                console.print(f"   Best: {paths.get('best', 'N/A')}")
                console.print(f"   Final: {paths.get('final', 'N/A')}")
        else:
            print(f"\n{Colors.GREEN}✅ VGG16 {method.upper()} SSL Complete!{Colors.ENDC}")
            print(f"   Best Loss: {results.get('best_loss', 0):.4f}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ VGG16 SSL Full (g30)
# ================================================================

def run_vgg16_ssl():
    """VGG16 SSL Full Pipeline: BYOL → SimCLR → MAE (LIGHT config)"""
    if not HAS_VGG16_SSL:
        if HAS_RICH:
            console.print("[red]❌ VGG16 SSL not available![/red]")
        else:
            print(f"{Colors.FAIL}❌ VGG16 SSL not available!{Colors.ENDC}")
        return

    if HAS_RICH:
        console.print(Panel.fit(
            "🎯 VGG16 SSL Full (LIGHT)",
            subtitle="BYOL → SimCLR → MAE (batch=2, grad_accum=16, img=160)",
            border_style="green",
        ))

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    if HAS_RICH:
        image_path = Prompt.ask("📁 Path to images", default=default_image_path)
        batch_size = int(Prompt.ask("📦 Batch size (LIGHT: 2)", default="2"))
        grad_accum = int(Prompt.ask("🔁 Grad accumulation (LIGHT: 16)", default="16"))
        if not Confirm.ask("\n🚀 Start full pipeline (3 methods)?", default=True):
            return
    else:
        image_path = input(f"📁 Path [{default_image_path}]: ") or default_image_path
        batch_size = int(input("📦 Batch [2]: ") or "2")
        grad_accum = int(input("🔁 Grad accum [16]: ") or "16")
        if input("🚀 Start? (y/n): ").lower() != "y":
            return

    try:
        from src.pipeLine.Architects.VGG16.vgg16SSL import create_vgg16_ssl_pipeline

        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            if os.path.exists(image_path):
                image_paths.extend([
                    os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(ext)
                ])

        if not image_paths:
            print("❌ No images found!")
            return

        print(f"✅ Found {len(image_paths)} images")
        print(f"⚠️  LIGHT: batch={batch_size}, grad_accum={grad_accum}, img=160")
        print("🔬 Running 3 methods: BYOL → SimCLR → MAE (this will take 10-15 hours)")

        # ✅ Method-specific defaults
        methods_config = {
            "byol": {"epochs": 15, "lr": 1e-4},
            "simclr": {"epochs": 25, "lr": 2e-3},
            "mae": {"epochs": 20, "lr": 1e-4},
        }

        all_results = {}
        for method, cfg in methods_config.items():
            print(f"\n{'='*80}")
            print(f"🧠 Training {method.upper()} on VGG16 ({cfg['epochs']} epochs, LR={cfg['lr']})")
            print(f"{'='*80}")

            pipeline = create_vgg16_ssl_pipeline(
                method=method,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                total_epochs=cfg["epochs"],
                lr=cfg["lr"],
                input_size=160,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
                use_kornia=True,
            )
            results = pipeline.pretrain(image_paths)
            all_results[method] = results

            try:
                pipeline.save_for_transfer()
            except Exception as e:
                print(f"⚠️ save_for_transfer failed: {e}")

        print(f"\n✅ All 3 methods complete!")
        for m, r in all_results.items():
            print(f"   {m.upper()}: Loss={r.get('best_loss', 0):.4f}")
        print(f"\n👉 Run option 34 to select best & copy to weightTransformer/")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ VGG16 Find Best SSL (g34)
# ================================================================

def run_vgg16_best_transfer():
    """VGG16 - Find best SSL via Linear Probe & copy to weightTransformer/"""
    if not HAS_VGG16_SSL:
        print("❌ VGG16 SSL not available!")
        return

    try:
        import shutil
        from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
            evaluate_with_linear_probe, build_labeled_eval_loader,
        )
        from src.pipeLine.Architects.VGG16.vgg16SSL import (
            VGG16_ARCHITECTURE_DIR, VGG16_SSL_DIR,
            create_vgg16_ssl_pipeline,
        )

        VGG16_WEIGHT_TRANSFORMER_DIR = os.path.join(
            PROJECT_ROOT_MAIN, "../../../../finalReports", "selfSupervisedLearning",
            "VGG16", "weightTransformer",
        )
        os.makedirs(VGG16_WEIGHT_TRANSFORMER_DIR, exist_ok=True)

        print("\n" + "=" * 80)
        print("🔍 STEP 1: Looking for VGG16 SSL checkpoints...")
        print("=" * 80)

        methods = ["byol", "simclr", "mae"]
        method_results = {}

        for method in methods:
            candidates = [
                os.path.join(VGG16_ARCHITECTURE_DIR, method,
                             f"vgg16_best_update_{method}.pth"),
                os.path.join(VGG16_ARCHITECTURE_DIR, method,
                             f"vgg16_final_update_{method}.pth"),
                os.path.join(VGG16_SSL_DIR,
                             f"vgg16_ssl_{method}_for_transfer.pth"),
            ]
            model_path = None
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break

            if model_path is None:
                print(f"⚠️  {method.upper()}: No checkpoint found — SKIPPING")
                continue

            print(f"✅ {method.upper()}: {model_path}")
            method_results[method] = {"model_path": model_path}

        if len(method_results) == 0:
            print("\n❌ No VGG16 SSL checkpoints found!")
            print("👉 Run options 31/32/33 first.")
            return {"status": "failed"}

        print("\n" + "=" * 80)
        print("🔬 STEP 2: Running Linear Probe on each checkpoint...")
        print("=" * 80)

        eval_loader = build_labeled_eval_loader(
            labels_csv="finalReports/labels.csv",
            image_dir="../../../../finalReports/cropped",
            input_size=160,  # ✅ LIGHT
            batch_size=4,
            max_per_class=150,  # ✅ LIGHT
        )

        linear_probe_results = {}

        for method, res in method_results.items():
            print(f"\n📌 Loading {method.upper()}...")
            try:
                pipeline = create_vgg16_ssl_pipeline(
                    method=method,
                    input_size=160,
                    batch_size=4,
                    total_epochs=1,
                    lr=1e-4,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    save_dir=os.path.join(VGG16_SSL_DIR, method),
                    pretrained_path=res["model_path"],
                )

                encoder = pipeline.get_encoder()
                probe = evaluate_with_linear_probe(
                    encoder=encoder,
                    labeled_loader=eval_loader,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    num_classes=4,
                    test_size=0.3,
                )
                linear_probe_results[method] = probe
                print(f"   ✅ {method.upper()} LP Acc: {probe['linear_probe_accuracy']:.4f} "
                      f"| LP F1: {probe['linear_probe_f1']:.4f}")
            except Exception as e:
                print(f"   ⚠️ {method.upper()} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(linear_probe_results) == 0:
            print("\n❌ No method could be loaded!")
            return {"status": "failed"}

        best_method = max(
            linear_probe_results.keys(),
            key=lambda x: linear_probe_results[x]["linear_probe_accuracy"],
        )
        best_lp_acc = linear_probe_results[best_method]["linear_probe_accuracy"]
        best_lp_f1 = linear_probe_results[best_method]["linear_probe_f1"]
        best_source_path = method_results[best_method]["model_path"]

        print("\n" + "=" * 80)
        print("🏆 STEP 3: BEST METHOD SELECTED")
        print("=" * 80)
        for m, probe in linear_probe_results.items():
            marker = " ✅ WINNER" if m == best_method else ""
            print(f"   {m.upper():<8} LP Acc: {probe['linear_probe_accuracy']:.4f}  "
                  f"LP F1: {probe['linear_probe_f1']:.4f}{marker}")
        print(f"\n   ✅ Winner: {best_method.upper()} (LP Acc = {best_lp_acc:.4f})")

        # Copy
        print("\n" + "=" * 80)
        print("📋 STEP 4: Copying to weightTransformer/...")
        print("=" * 80)

        transfer_path = os.path.join(
            VGG16_WEIGHT_TRANSFORMER_DIR,
            f"vgg16_transferred_from_{best_method}.pth",
        )

        # Load source
        ckpt = torch.load(best_source_path, map_location="cpu", weights_only=False)
        if "encoder_state_dict" in ckpt:
            encoder_state = ckpt["encoder_state_dict"]
        elif "model_state_dict" in ckpt:
            encoder_state = ckpt["model_state_dict"]
        else:
            encoder_state = ckpt

        total_layers_count = len(encoder_state)

        torch.save(
            {
                "encoder_state_dict": encoder_state,
                "source_method": best_method,
                "source_model": "VGG16_SSL",
                "target_model": "VGG16",
                "best_lp_accuracy": best_lp_acc,
                "best_lp_f1": best_lp_f1,
                "transferred_layers": total_layers_count,  # ✅ int
                "total_layers": total_layers_count,
                "transfer_note": "all layers (same architecture)",
                "all_methods_lp": {
                    m: {
                        "lp_accuracy": p["linear_probe_accuracy"],
                        "lp_f1": p["linear_probe_f1"],
                    }
                    for m, p in linear_probe_results.items()
                },
                "source_checkpoint": best_source_path,
                "timestamp": datetime.now().isoformat(),
            },
            transfer_path,
        )
        print(f"✅ Saved: {transfer_path}")

        summary_path = os.path.join(VGG16_WEIGHT_TRANSFORMER_DIR, "best_method_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "best_method": best_method,
                "best_lp_accuracy": best_lp_acc,
                "best_lp_f1": best_lp_f1,
                "best_checkpoint_source": best_source_path,
                "transferred_checkpoint": transfer_path,
                "all_methods": {
                    m: {
                        "lp_accuracy": p["linear_probe_accuracy"],
                        "lp_f1": p["linear_probe_f1"],
                        "checkpoint": method_results[m]["model_path"],
                    }
                    for m, p in linear_probe_results.items()
                },
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4)
        print(f"✅ Summary: {summary_path}")

        print("\n🎉 DONE!")
        print(f"   ✅ Winner: {best_method.upper()}")
        print(f"   ✅ LP Acc: {best_lp_acc:.4f}")
        print(f"   📂 {transfer_path}")
        print(f"\n👉 Next: run option 35 to fine-tune.")

        return {"status": "success", "best_method": best_method}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed"}


# ================================================================
# ✅ VGG16 Fine-Tune (g35)
# ================================================================

def run_vgg16_ensemble_finetune():
    """VGG16 FFT + TTA + CV5 + CI95 + 24-page PDF (LIGHT)"""
    if not HAS_VGG16_FINETUNE:
        print("❌ VGG16 Fine-Tune not available!")
        print("👉 Make sure src/pipeLine/vgg16FineTune.py exists")
        return

    from src.pipeLine.Architects.VGG16.vgg16FineTune import (
        check_transferred_model,
        run_vgg16_finetune_pipeline,
        get_transferred_metadata,
        get_ssl_methods_comparison,
        FINAL_MODEL_PATH as VGG16_FINAL_MODEL_PATH,
    )

    # Step 1: Find transferred
    print("\n🔍 Step 1: Finding transferred model...")
    exists, transferred_path = check_transferred_model()

    if not exists:
        print("❌ Transferred model NOT FOUND!")
        print("👉 Run option 34 first.")
        return

    print(f"✅ Found: {transferred_path}")

    # Step 2: Metadata
    print("\n📋 Step 2: Reading metadata...")
    meta = get_transferred_metadata(transferred_path)
    source_method = meta.get("source_method", "byol")
    best_lp = meta.get("best_lp_accuracy", 0.0)

    if HAS_RICH:
        info_table = Table(title="📋 Metadata", box=box.ROUNDED, border_style="green")
        info_table.add_column("Property", style="bold cyan")
        info_table.add_column("Value", style="bold white")
        info_table.add_row("Source Method", source_method.upper())
        info_table.add_row("LP Accuracy", f"{best_lp:.4f}")
        info_table.add_row("File", os.path.basename(transferred_path))
        console.print(info_table)

    # Step 3: LIGHT config
    print("\n⚙️  Step 3: Configuration (LIGHT)")
    print("=" * 70)

    if HAS_RICH:
        epochs_final = int(Prompt.ask("⏳ FT epochs (final) [LIGHT: 30]", default="30"))
        epochs_cv = int(Prompt.ask("⏳ FT epochs (CV) [LIGHT: 10]", default="10"))
        lr_final = float(Prompt.ask("📊 LR (final)", default="0.0001"))
        lr_cv = float(Prompt.ask("📊 LR (CV)", default="0.0001"))
        batch_size = int(Prompt.ask("📦 Batch size [LIGHT: 2]", default="2"))
        grad_accum = int(Prompt.ask("🔁 Grad accum [LIGHT: 16]", default="16"))
        patience_final = int(Prompt.ask("🛑 Patience (final) [10]", default="10"))
        patience_cv = int(Prompt.ask("🛑 Patience (CV) [5]", default="5"))
        n_bootstrap = int(Prompt.ask("🎲 Bootstrap [LIGHT: 500]", default="500"))
        n_tta = int(Prompt.ask("🔄 TTA [LIGHT: 8]", default="8"))
        cv_folds = int(Prompt.ask("🔬 CV folds [5]", default="5"))
        run_interp = Confirm.ask("🔬 Interpretability?", default=True)
        if not Confirm.ask("\n🚀 Start VGG16 Fine-Tune (LIGHT)?", default=True):
            return
    else:
        epochs_final = int(input("⏳ FT final [30]: ") or "30")
        epochs_cv = int(input("⏳ FT CV [10]: ") or "10")
        lr_final = float(input("📊 LR final [1e-4]: ") or "1e-4")
        lr_cv = float(input("📊 LR CV [1e-4]: ") or "1e-4")
        batch_size = int(input("📦 Batch [2]: ") or "2")
        grad_accum = int(input("🔁 Grad accum [16]: ") or "16")
        patience_final = int(input("🛑 Patience final [10]: ") or "10")
        patience_cv = int(input("🛑 Patience CV [5]: ") or "5")
        n_bootstrap = int(input("🎲 Bootstrap [500]: ") or "500")
        n_tta = int(input("🔄 TTA [8]: ") or "8")
        cv_folds = int(input("🔬 CV folds [5]: ") or "5")
        run_interp = input("🔬 Interpretability? (y/n) (y): ").lower() != "n"
        if input("🚀 Start? (y/n): ").lower() != "y":
            return

    # Summary
    print("\n" + "=" * 70)
    print("📋 Summary (LIGHT):")
    print("=" * 70)
    print(f"   Transferred:        {os.path.basename(transferred_path)}")
    print(f"   Source method:      {source_method.upper()}")
    print(f"   FT epochs:          {epochs_final} (final) / {epochs_cv} (CV)")
    print(f"   Batch × Grad:       {batch_size} × {grad_accum} = {batch_size * grad_accum}")
    print(f"   Image size:         160×160")
    print(f"   LR:                 {lr_final} (final) / {lr_cv} (CV)")
    print(f"   TTA augs:           {n_tta}")
    print(f"   CV folds:           {cv_folds}")
    print(f"   Bootstrap:          {n_bootstrap}")
    print(f"   Interpretability:   {'✅' if run_interp else '❌'}")
    print("=" * 70)

    # Step 4: Run
    print("\n" + "=" * 80)
    print("🧠 Step 4: Running Fine-Tune + CV + CI + TTA (LIGHT)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        results = run_vgg16_finetune_pipeline(
            ssl_checkpoint_path=transferred_path,
            ssl_method=source_method,
            epochs_final=epochs_final,
            epochs_cv=epochs_cv,
            lr_final=lr_final,
            lr_cv=lr_cv,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            patience_final=patience_final,
            patience_cv=patience_cv,
            n_bootstrap=n_bootstrap,
            n_tta=n_tta,
            cv_folds=cv_folds,
            input_size=160,  # ✅ LIGHT
            device=device,
            verbose=True,
        )

        if results.get("status") != "success":
            print("❌ Pipeline failed")
            return

        final_metrics = results["final_metrics"]
        cv_summary = results["cv_summary"]
        ci95 = results["ci95"]
        history = results["history"]

        print(f"\n✅ Fine-Tune + CV + CI complete!")

    except Exception as e:
        print(f"❌ Error in Fine-Tune pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Load final model
    print("\n🔍 Step 5: Loading final model...")
    final_model = None

    try:
        from torchvision.models import vgg16
        final_model = vgg16(num_classes=4, weights=None)

        if VGG16_FINAL_MODEL_PATH and os.path.exists(VGG16_FINAL_MODEL_PATH):
            ckpt = torch.load(VGG16_FINAL_MODEL_PATH, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                final_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                final_model.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded: {VGG16_FINAL_MODEL_PATH}")
        else:
            print(f"⚠️ Final model not found: {VGG16_FINAL_MODEL_PATH}")

        final_model = final_model.to(device).eval()

    except Exception as e:
        print(f"⚠️ Could not reload final model: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Prepare test set
    print("\n📊 Step 6: Preparing test set...")
    image_paths, labels_list = [], []

    if final_model is not None and run_interp:
        split_info = results.get("split_info", {})
        test_paths = split_info.get("test_paths", [])
        test_labels = split_info.get("test_labels", [])

        if len(test_paths) > 0:
            image_paths = test_paths
            labels_list = test_labels
            print(f"✅ Using TEST SET: {len(image_paths)} images")
        else:
            print("⚠️ Test set not in results — using full dataset")
            try:
                from src.pipeLine.Architects.VGG16.vgg16FineTune import load_labeled_data
                image_paths, labels_list = load_labeled_data()
                print(f"✅ Loaded {len(image_paths)} images")
            except Exception as e:
                print(f"⚠️ Could not load: {e}")

    # Step 7: Generate PDF
    print("\n" + "=" * 80)
    print("📄 Step 7: Generating 24-page PDF (LIGHT)")
    print("=" * 80)

    pdf_path = None

    if final_model is not None:
        try:
            from src.pipeLine.Architects.VGG16.vgg16Reporter import generate_vgg16_report

            ssl_comparison = get_ssl_methods_comparison()
            if ssl_comparison.get("available"):
                print(f"✅ SSL comparison: {len(ssl_comparison.get('all_methods', {}))} methods")
            else:
                print("⚠️ SSL comparison not available")

            config_dict = {
                "input_size": 160,
                "epochs_final": epochs_final,
                "epochs_cv": epochs_cv,
                "lr_final": lr_final,
                "lr_cv": lr_cv,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "patience_final": patience_final,
                "patience_cv": patience_cv,
                "cv_folds": cv_folds,
                "n_bootstrap": n_bootstrap,
                "device": str(device),
            }

            data_split_dict = {
                "train": results.get("num_train", 0),
                "val": results.get("num_val", 0),
                "test": results.get("num_test", 0),
                "total": results.get("num_train", 0)
                         + results.get("num_val", 0)
                         + results.get("num_test", 0),
            }

            tta_info_dict = {"n_aug": n_tta}

            pdf_path = generate_vgg16_report(
                final_model=final_model,
                metrics=final_metrics,
                history=history,
                cv_summary=cv_summary,
                ci95=ci95,
                image_paths=image_paths,
                labels=labels_list,
                ssl_checkpoint_path=transferred_path,
                ssl_method=source_method,
                config=config_dict,
                data_split=data_split_dict,
                tta_info=tta_info_dict,
                ssl_comparison=ssl_comparison,
                device=device,
                generate_interpretability_flag=run_interp,
                verbose=True,
            )

            if pdf_path:
                print(f"\n✅ 24-page PDF: {pdf_path}")

        except Exception as e:
            print(f"\n❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            pdf_path = None

    # Step 8: Display Summary
    print("\n" + "=" * 80)
    print("🎉 VGG16 FINE-TUNE COMPLETE!")
    print("=" * 80)

    if HAS_RICH:
        summary_table = Table(title="🎉 Final Results (LIGHT)", box=box.ROUNDED, border_style="green")
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold white")

        summary_table.add_row("Transferred Model", os.path.basename(transferred_path))
        summary_table.add_row("Source Method", source_method.upper())
        summary_table.add_row("Final Accuracy", f"{final_metrics['accuracy']:.4f}")
        summary_table.add_row("Final F1", f"{final_metrics['f1_score']:.4f}")
        summary_table.add_row("Final AUC", f"{final_metrics['roc_auc_mean']:.4f}")
        summary_table.add_row("Balanced Accuracy", f"{final_metrics['balanced_accuracy']:.4f}")
        summary_table.add_row(
            "CV Mean Accuracy",
            f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}",
        )
        summary_table.add_row(
            "CI 95% Accuracy",
            f"[{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]",
        )
        if pdf_path:
            summary_table.add_row("PDF Report", os.path.basename(pdf_path))

        console.print(summary_table)
    else:
        print(f"\n📊 Results:")
        print(f"   Transferred:       {os.path.basename(transferred_path)}")
        print(f"   Source method:     {source_method.upper()}")
        print(f"   Final Accuracy:    {final_metrics['accuracy']:.4f}")
        print(f"   Final F1:          {final_metrics['f1_score']:.4f}")
        print(f"   Final AUC:         {final_metrics['roc_auc_mean']:.4f}")
        print(f"   Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
        print(f"   CV Mean Acc:       {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"   CI 95% Acc:        [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
        if pdf_path:
            print(f"   PDF Report:        {pdf_path}")

    print("=" * 80)
    print("✅ Option 35 complete!")
    print("=" * 80)

# ================================================================
# ✅ DenseNet SSL - SINGLE METHOD
# ================================================================

def run_densenet_ssl_single(method: str = "byol"):
    """Run DenseNet SSL with a single method (BYOL / SimCLR / MAE)."""
    if not HAS_DENSENET_SSL:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ DenseNet SSL not available![/bold red]",
                border_style="red",
            ))
        return

    if HAS_RICH:
        console.print(Panel.fit(f"🧠 DenseNet SSL - {method.upper()}", border_style="green"))

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    if method == "simclr":
        default_lr, default_epochs = "0.002", "30"
    elif method == "byol":
        default_lr, default_epochs = "0.0001", "20"
    else:
        default_lr, default_epochs = "0.0001", "25"

    if HAS_RICH:
        image_path = Prompt.ask("📁 Path to images", default=default_image_path)
        total_epochs = int(Prompt.ask("⏳ Epochs", default=default_epochs))
        batch_size = int(Prompt.ask("📦 Batch size (DenseNet: 8)", default="8"))
        grad_accum = int(Prompt.ask("🔁 Grad accumulation (DenseNet: 4)", default="4"))
        lr = float(Prompt.ask(f"📊 LR", default=default_lr))
        if not Confirm.ask(f"\n🚀 Start DenseNet SSL {method.upper()}?", default=True):
            return
    else:
        image_path = input(f"📁 Path [{default_image_path}]: ") or default_image_path
        total_epochs = int(input(f"⏳ Epochs [{default_epochs}]: ") or default_epochs)
        batch_size = int(input("📦 Batch [8]: ") or "8")
        grad_accum = int(input("🔁 Grad accum [4]: ") or "4")
        lr = float(input(f"📊 LR [{default_lr}]: ") or default_lr)
        if input("🚀 Start? (y/n): ").lower() != "y":
            return

    try:
        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            if os.path.exists(image_path):
                image_paths.extend([
                    os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(ext)
                ])

        if not image_paths:
            print("❌ No images!")
            return

        print(f"✅ Found {len(image_paths)} images")

        pipeline = create_densenet_ssl_pipeline(
            method=method,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            total_epochs=total_epochs,
            lr=lr,
            input_size=224,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
            use_kornia=True,
            num_workers=4,
        )
        results = pipeline.pretrain(image_paths)

        try:
            report_path = pipeline.generate_report()
            print(f"✅ PDF: {report_path}")
        except Exception as e:
            print(f"⚠️ PDF failed: {e}")

        print(f"\n✅ DenseNet {method.upper()} complete!")
        print(f"   Best Loss: {results.get('best_loss', 0):.4f}")
        if "saved_paths" in results:
            print(f"   Best: {results['saved_paths'].get('best', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ DenseNet SSL Full (g36)
# ================================================================

def run_densenet_ssl():
    """DenseNet SSL Full Pipeline: BYOL → SimCLR → MAE"""
    if not HAS_DENSENET_SSL:
        print("❌ DenseNet SSL not available!")
        return

    if HAS_RICH:
        console.print(Panel.fit(
            "🎯 DenseNet SSL Full",
            subtitle="BYOL → SimCLR → MAE (batch=8, grad_accum=4)",
            border_style="green",
        ))

    default_image_path = os.path.join("../../../../finalReports", "cropped")

    if HAS_RICH:
        image_path = Prompt.ask("📁 Path", default=default_image_path)
        batch_size = int(Prompt.ask("📦 Batch (8)", default="8"))
        grad_accum = int(Prompt.ask("🔁 Grad accum (4)", default="4"))
        if not Confirm.ask("\n🚀 Start full pipeline?", default=True):
            return
    else:
        image_path = input(f"📁 Path [{default_image_path}]: ") or default_image_path
        batch_size = int(input("📦 Batch [8]: ") or "8")
        grad_accum = int(input("🔁 Grad accum [4]: ") or "4")
        if input("🚀 Start? (y/n): ").lower() != "y":
            return

    try:
        image_paths = []
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".dcm"]:
            if os.path.exists(image_path):
                image_paths.extend([
                    os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(ext)
                ])

        if not image_paths:
            print("❌ No images!")
            return

        print(f"✅ Found {len(image_paths)} images")

        methods_config = {
            "byol": {"epochs": 20, "lr": 1e-4},
            "simclr": {"epochs": 30, "lr": 2e-3},
            "mae": {"epochs": 25, "lr": 1e-4},
        }

        all_results = {}
        for method, cfg in methods_config.items():
            print(f"\n{'='*80}")
            print(f"🧠 Training {method.upper()} ({cfg['epochs']} epochs, LR={cfg['lr']})")
            print(f"{'='*80}")

            pipeline = create_densenet_ssl_pipeline(
                method=method,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                total_epochs=cfg["epochs"],
                lr=cfg["lr"],
                input_size=224,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
            )
            results = pipeline.pretrain(image_paths)
            all_results[method] = results

            try:
                pipeline.save_for_transfer()
            except Exception as e:
                print(f"⚠️ save_for_transfer failed: {e}")

        print(f"\n✅ All 3 methods complete!")
        for m, r in all_results.items():
            print(f"   {m.upper()}: Loss={r.get('best_loss', 0):.4f}")
        print(f"\n👉 Run option 40 to select best & copy.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ DenseNet Find Best SSL (g40)
# ================================================================

def run_densenet_best_transfer():
    """DenseNet - Find best SSL & copy to weightTransformer/"""
    if not HAS_DENSENET_SSL:
        print("❌ DenseNet SSL not available!")
        return

    try:
        import shutil
        from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
            evaluate_with_linear_probe, build_labeled_eval_loader,
        )
        from src.pipeLine.Architects.DenseNet.densenetSSL import (
            DENSENET_ARCHITECTURE_DIR, DENSENET_SSL_DIR,
            create_densenet_ssl_pipeline,
        )

        DENSENET_WEIGHT_TRANSFORMER_DIR = os.path.join(
            PROJECT_ROOT_MAIN, "../../../../finalReports", "selfSupervisedLearning",
            "DenseNet", "weightTransformer",
        )
        os.makedirs(DENSENET_WEIGHT_TRANSFORMER_DIR, exist_ok=True)

        print("\n" + "=" * 80)
        print("🔍 STEP 1: Looking for DenseNet SSL checkpoints...")
        print("=" * 80)

        methods = ["byol", "simclr", "mae"]
        method_results = {}

        for method in methods:
            candidates = [
                os.path.join(DENSENET_ARCHITECTURE_DIR, method,
                             f"densenet_best_update_{method}.pth"),
                os.path.join(DENSENET_ARCHITECTURE_DIR, method,
                             f"densenet_final_update_{method}.pth"),
                os.path.join(DENSENET_SSL_DIR,
                             f"densenet_ssl_{method}_for_transfer.pth"),
            ]
            model_path = None
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break

            if model_path is None:
                print(f"⚠️  {method.upper()}: No checkpoint")
                continue

            print(f"✅ {method.upper()}: {model_path}")
            method_results[method] = {"model_path": model_path}

        if not method_results:
            print("\n❌ No checkpoints found!")
            print("👉 Run options 37/38/39 first.")
            return {"status": "failed"}

        print("\n" + "=" * 80)
        print("🔬 STEP 2: Running Linear Probe...")
        print("=" * 80)

        eval_loader = build_labeled_eval_loader(
            labels_csv="finalReports/labels.csv",
            image_dir="../../../../finalReports/cropped",
            input_size=224,
            batch_size=16,
            max_per_class=200,
        )

        linear_probe_results = {}

        for method, res in method_results.items():
            print(f"\n📌 Loading {method.upper()}...")
            try:
                pipeline = create_densenet_ssl_pipeline(
                    method=method,
                    input_size=224,
                    batch_size=8,
                    total_epochs=1,
                    lr=1e-4,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    save_dir=os.path.join(DENSENET_SSL_DIR, method),
                    pretrained_path=res["model_path"],
                )

                encoder = pipeline.get_encoder()
                probe = evaluate_with_linear_probe(
                    encoder=encoder,
                    labeled_loader=eval_loader,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    num_classes=4,
                    test_size=0.3,
                )
                linear_probe_results[method] = probe
                print(f"   ✅ {method.upper()} LP Acc: {probe['linear_probe_accuracy']:.4f}")
            except Exception as e:
                print(f"   ⚠️ {method.upper()} failed: {e}")
                continue

        if not linear_probe_results:
            return {"status": "failed"}

        best_method = max(linear_probe_results.keys(),
                          key=lambda x: linear_probe_results[x]["linear_probe_accuracy"])
        best_lp_acc = linear_probe_results[best_method]["linear_probe_accuracy"]
        best_lp_f1 = linear_probe_results[best_method]["linear_probe_f1"]
        best_source = method_results[best_method]["model_path"]

        print(f"\n🏆 WINNER: {best_method.upper()} (LP Acc = {best_lp_acc:.4f})")

        transfer_path = os.path.join(
            DENSENET_WEIGHT_TRANSFORMER_DIR,
            f"densenet_transferred_from_{best_method}.pth",
        )

        ckpt = torch.load(best_source, map_location="cpu", weights_only=False)
        if "encoder_state_dict" in ckpt:
            encoder_state = ckpt["encoder_state_dict"]
        elif "model_state_dict" in ckpt:
            encoder_state = ckpt["model_state_dict"]
        else:
            encoder_state = ckpt

        total_layers_count = len(encoder_state)

        torch.save({
            "encoder_state_dict": encoder_state,
            "source_method": best_method,
            "source_model": "DenseNet_SSL",
            "target_model": "DenseNet",
            "best_lp_accuracy": best_lp_acc,
            "best_lp_f1": best_lp_f1,
            "transferred_layers": total_layers_count,
            "total_layers": total_layers_count,
            "transfer_note": "all layers (same architecture)",
            "all_methods_lp": {
                m: {"lp_accuracy": p["linear_probe_accuracy"],
                    "lp_f1": p["linear_probe_f1"]}
                for m, p in linear_probe_results.items()
            },
            "source_checkpoint": best_source,
            "timestamp": datetime.now().isoformat(),
        }, transfer_path)

        summary_path = os.path.join(DENSENET_WEIGHT_TRANSFORMER_DIR, "best_method_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "best_method": best_method,
                "best_lp_accuracy": best_lp_acc,
                "best_lp_f1": best_lp_f1,
                "best_checkpoint_source": best_source,
                "transferred_checkpoint": transfer_path,
                "all_methods": {
                    m: {"lp_accuracy": p["linear_probe_accuracy"],
                        "lp_f1": p["linear_probe_f1"],
                        "checkpoint": method_results[m]["model_path"]}
                    for m, p in linear_probe_results.items()
                },
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4)

        print(f"✅ Copied: {transfer_path}")
        print(f"✅ Summary: {summary_path}")
        print(f"\n👉 Next: run option 41 to fine-tune.")

        return {"status": "success", "best_method": best_method}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed"}


# ================================================================
# ✅ DenseNet Fine-Tune (g41)
# ================================================================

def run_densenet_ensemble_finetune():
    """DenseNet FFT + TTA + CV5 + CI95 + 24-page PDF"""
    if not HAS_DENSENET_FINETUNE:
        print("❌ DenseNet Fine-Tune not available!")
        print("👉 Make sure src/pipeLine/densenetFineTune.py exists")
        return

    from src.pipeLine.Architects.DenseNet.densenetFineTune import (
        check_transferred_model,
        run_densenet_finetune_pipeline,
        get_transferred_metadata,
        get_ssl_methods_comparison,
        FINAL_MODEL_PATH as DENSENET_FINAL_MODEL_PATH,
    )

    # Step 1: Find transferred
    print("\n🔍 Step 1: Finding transferred model...")
    exists, transferred_path = check_transferred_model()

    if not exists:
        print("❌ Transferred model NOT FOUND!")
        print("👉 Run option 40 first.")
        return

    print(f"✅ Found: {transferred_path}")

    # Step 2: Metadata
    print("\n📋 Step 2: Reading metadata...")
    meta = get_transferred_metadata(transferred_path)
    source_method = meta.get("source_method", "byol")
    best_lp = meta.get("best_lp_accuracy", 0.0)

    if HAS_RICH:
        info_table = Table(title="📋 Metadata", box=box.ROUNDED, border_style="green")
        info_table.add_column("Property", style="bold cyan")
        info_table.add_column("Value", style="bold white")
        info_table.add_row("Source Method", source_method.upper())
        info_table.add_row("LP Accuracy", f"{best_lp:.4f}")
        info_table.add_row("File", os.path.basename(transferred_path))
        console.print(info_table)

    # Step 3: Config
    print("\n⚙️  Step 3: Configuration (LIGHT)")
    print("=" * 70)

    if HAS_RICH:
        epochs_final = int(Prompt.ask("⏳ FT epochs (final) [30]", default="30"))
        epochs_cv = int(Prompt.ask("⏳ FT epochs (CV) [15]", default="15"))
        lr_final = float(Prompt.ask("📊 LR (final) [0.0005]", default="0.0005"))
        lr_cv = float(Prompt.ask("📊 LR (CV) [0.0005]", default="0.0005"))
        batch_size = int(Prompt.ask("📦 Batch size [8]", default="8"))
        grad_accum = int(Prompt.ask("🔁 Grad accum [4]", default="4"))
        patience_final = int(Prompt.ask("🛑 Patience (final) [10]", default="10"))
        patience_cv = int(Prompt.ask("🛑 Patience (CV) [6]", default="6"))
        n_bootstrap = int(Prompt.ask("🎲 Bootstrap [1000]", default="1000"))
        n_tta = int(Prompt.ask("🔄 TTA [16]", default="16"))
        cv_folds = int(Prompt.ask("🔬 CV folds [5]", default="5"))
        run_interp = Confirm.ask("🔬 Interpretability?", default=True)
        if not Confirm.ask("\n🚀 Start?", default=True):
            return
    else:
        epochs_final = int(input("⏳ FT final [40]: ") or "40")
        epochs_cv = int(input("⏳ FT CV [15]: ") or "15")
        lr_final = float(input("📊 LR final [1e-4]: ") or "1e-4")
        lr_cv = float(input("📊 LR CV [1e-4]: ") or "1e-4")
        batch_size = int(input("📦 Batch [8]: ") or "8")
        grad_accum = int(input("🔁 Grad accum [4]: ") or "4")
        patience_final = int(input("🛑 Patience final [15]: ") or "15")
        patience_cv = int(input("🛑 Patience CV [8]: ") or "8")
        n_bootstrap = int(input("🎲 Bootstrap [1000]: ") or "1000")
        n_tta = int(input("🔄 TTA [16]: ") or "16")
        cv_folds = int(input("🔬 CV folds [5]: ") or "5")
        run_interp = input("🔬 Interpretability? (y/n) (y): ").lower() != "n"
        if input("🚀 Start? (y/n): ").lower() != "y":
            return

    # Step 4: Run
    print("\n" + "=" * 80)
    print("🧠 Step 4: Running Fine-Tune + CV + CI + TTA")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        results = run_densenet_finetune_pipeline(
            ssl_checkpoint_path=transferred_path,
            ssl_method=source_method,
            epochs_final=epochs_final,
            epochs_cv=epochs_cv,
            lr_final=lr_final,
            lr_cv=lr_cv,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            patience_final=patience_final,
            patience_cv=patience_cv,
            n_bootstrap=n_bootstrap,
            n_tta=n_tta,
            cv_folds=cv_folds,
            input_size=224,
            device=device,
            verbose=True,
        )

        if results.get("status") != "success":
            print("❌ Pipeline failed")
            return

        final_metrics = results["final_metrics"]
        cv_summary = results["cv_summary"]
        ci95 = results["ci95"]
        history = results["history"]

        print(f"\n✅ Fine-Tune complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Load final model
    print("\n🔍 Step 5: Loading final model...")
    final_model = None

    try:
        from torchvision.models import densenet121
        final_model = densenet121(num_classes=4, weights=None)

        if DENSENET_FINAL_MODEL_PATH and os.path.exists(DENSENET_FINAL_MODEL_PATH):
            ckpt = torch.load(DENSENET_FINAL_MODEL_PATH, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                final_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                final_model.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded: {DENSENET_FINAL_MODEL_PATH}")
        else:
            print(f"⚠️ Final model not found: {DENSENET_FINAL_MODEL_PATH}")

        final_model = final_model.to(device).eval()

    except Exception as e:
        print(f"⚠️ Could not reload: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Prepare test set
    print("\n📊 Step 6: Preparing test set...")
    image_paths, labels_list = [], []

    if final_model is not None and run_interp:
        split_info = results.get("split_info", {})
        test_paths = split_info.get("test_paths", [])
        test_labels = split_info.get("test_labels", [])

        if len(test_paths) > 0:
            image_paths = test_paths
            labels_list = test_labels
            print(f"✅ Using TEST SET: {len(image_paths)} images")
        else:
            print("⚠️ Test set not in results")
            try:
                from src.pipeLine.Architects.DenseNet.densenetFineTune import load_labeled_data
                image_paths, labels_list = load_labeled_data()
            except Exception:
                pass

    # Step 7: Generate PDF
    print("\n" + "=" * 80)
    print("📄 Step 7: Generating 24-page PDF")
    print("=" * 80)

    pdf_path = None

    if final_model is not None:
        try:
            from src.pipeLine.Architects.DenseNet.densenetReporter import generate_densenet_report

            ssl_comparison = get_ssl_methods_comparison()
            if ssl_comparison.get("available"):
                print(f"✅ SSL comparison: {len(ssl_comparison.get('all_methods', {}))} methods")

            config_dict = {
                "input_size": 224,
                "epochs_final": epochs_final,
                "epochs_cv": epochs_cv,
                "lr_final": lr_final,
                "lr_cv": lr_cv,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "patience_final": patience_final,
                "patience_cv": patience_cv,
                "cv_folds": cv_folds,
                "n_bootstrap": n_bootstrap,
                "device": str(device),
            }

            data_split_dict = {
                "train": results.get("num_train", 0),
                "val": results.get("num_val", 0),
                "test": results.get("num_test", 0),
                "total": results.get("num_train", 0)
                         + results.get("num_val", 0)
                         + results.get("num_test", 0),
            }

            tta_info_dict = {"n_aug": n_tta}

            pdf_path = generate_densenet_report(
                final_model=final_model,
                metrics=final_metrics,
                history=history,
                cv_summary=cv_summary,
                ci95=ci95,
                image_paths=image_paths,
                labels=labels_list,
                ssl_checkpoint_path=transferred_path,
                ssl_method=source_method,
                config=config_dict,
                data_split=data_split_dict,
                tta_info=tta_info_dict,
                ssl_comparison=ssl_comparison,
                device=device,
                generate_interpretability_flag=run_interp,
                verbose=True,
            )

            if pdf_path:
                print(f"\n✅ 24-page PDF: {pdf_path}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Step 8: Summary
    print("\n" + "=" * 80)
    print("🎉 DENSENET FINE-TUNE COMPLETE!")
    print("=" * 80)

    if HAS_RICH:
        summary_table = Table(title="🎉 Final Results", box=box.ROUNDED, border_style="green")
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold white")
        summary_table.add_row("Transferred", os.path.basename(transferred_path))
        summary_table.add_row("Source Method", source_method.upper())
        summary_table.add_row("Accuracy", f"{final_metrics['accuracy']:.4f}")
        summary_table.add_row("F1", f"{final_metrics['f1_score']:.4f}")
        summary_table.add_row("AUC", f"{final_metrics['roc_auc_mean']:.4f}")
        summary_table.add_row("Balanced Acc", f"{final_metrics['balanced_accuracy']:.4f}")
        summary_table.add_row("CV Mean Acc",
                               f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        summary_table.add_row("CI 95% Acc",
                               f"[{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
        if pdf_path:
            summary_table.add_row("PDF", os.path.basename(pdf_path))
        console.print(summary_table)

    print("=" * 80)
    print("✅ Option 41 complete!")
    print("=" * 80)

# ================================================================
# ✅ Main Function
# ================================================================


def main():
    """
    Main entry point for the Liver AI Pipeline.
    """
    clear_python_cache()
    print_header()

    while True:
        print_menu()
        choice = get_choice()

        if choice == "0":
            if HAS_RICH:
                console.print(
                    "\n[bold yellow]👋 Goodbye! Thanks for using Liver AI Pipeline.[/bold yellow]"
                )
            else:
                print(
                    f"\n{Colors.WARNING}👋 Goodbye! Thanks for using Liver AI Pipeline.{Colors.ENDC}"
                )
            break

        elif choice == "1":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH1)
            musicPlayer.play_music()
            try:
                wFTAlgorithms()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in wFTAlgorithms: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in wFTAlgorithms: {e}{Colors.ENDC}")
            musicPlayer.stop_music()

        elif choice == "2":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH2)
            musicPlayer.play_music()
            run_resnet18_ssl()  # Full SSL Pipeline (BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune)
            musicPlayer.stop_music()

        elif choice == "3":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            run_resnet18_ssl_single("byol")
            musicPlayer.stop_music()

        elif choice == "4":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            run_resnet18_ssl_single("simclr")
            musicPlayer.stop_music()

        elif choice == "5":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            run_resnet18_ssl_single("mae")
            musicPlayer.stop_music()

        elif choice == "6":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            run_resnet18_best_transfer()
            musicPlayer.stop_music()

        elif choice == "7":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            run_resnet18_ensemble_finetune()
            musicPlayer.stop_music()

        elif choice == "8":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH2)
            musicPlayer.play_music()
            run_efficientnet_b0_ssl()  # Full SSL Pipeline (BYOL → SimCLR → MAE → Best → Transfer → Fine-Tune)
            musicPlayer.stop_music()

        elif choice == "9":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            run_efficientnet_b0_ssl_single("byol")
            musicPlayer.stop_music()

        elif choice == "10":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            run_efficientnet_b0_ssl_single("simclr")
            musicPlayer.stop_music()

        elif choice == "11":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            run_efficientnet_b0_ssl_single("mae")
            musicPlayer.stop_music()

        elif choice == "12":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            run_efficientnet_b0_best_transfer()
            musicPlayer.stop_music()

        elif choice == "13":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            run_efficientnet_ensemble_finetune()
            musicPlayer.stop_music()

        elif choice == "14":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH5)
            musicPlayer.play_music()
            generate_all_reports()
            musicPlayer.stop_music()

        elif choice == "15":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH5)
            musicPlayer.play_music()
            compare_all_ssl_models()
            musicPlayer.stop_music()

        elif choice == "16":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH6)
            musicPlayer.play_music()
            detect_hardware_device()
            musicPlayer.stop_music()

        elif choice == "17":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            run_recovery_mode()
            musicPlayer.stop_music()

        # ============================================================
        # ✅ AlexNet SSL (NEW)
        # ============================================================

        elif choice == "18":
            # 🎯 AlexNet Full SSL Pipeline (BYOL → SimCLR → MAE → Best)
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH2)
            musicPlayer.play_music()
            try:
                run_alexnet_ssl()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in AlexNet SSL Full: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in AlexNet SSL Full: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "19":
            # 🎯 AlexNet + BYOL only
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_alexnet_ssl_single("byol")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in AlexNet BYOL: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in AlexNet BYOL: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "20":
            # 🎯 AlexNet + SimCLR only
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_alexnet_ssl_single("simclr")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in AlexNet SimCLR: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in AlexNet SimCLR: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "21":
            # 🎯 AlexNet + MAE only
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_alexnet_ssl_single("mae")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in AlexNet MAE: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in AlexNet MAE: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "22":
            # 🎯 AlexNet: Find Best SSL
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_alexnet_best_transfer()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in AlexNet Best SSL: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "23":
            # 🏆 AlexNet: Full FFT + TTA + CV5 + CI95
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_alexnet_ensemble_finetune()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in AlexNet FFT: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        # ============================================================
        # ✅ MobileNetV2 SSL (NEW)
        # ============================================================

        elif choice == "24":
            # 🎯 MobileNetV2 + BYOL only
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_mobilenetv2_ssl_single("byol")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in MobileNetV2 BYOL: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in MobileNetV2 BYOL: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "25":
            # 🎯 MobileNetV2 + SimCLR only
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_mobilenetv2_ssl_single("simclr")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in MobileNetV2 SimCLR: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in MobileNetV2 SimCLR: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "26":
            # 🎯 MobileNetV2 + MAE only
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_mobilenetv2_ssl_single("mae")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in MobileNetV2 MAE: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error in MobileNetV2 MAE: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "27":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH2)
            musicPlayer.play_music()
            try:
                run_mobilenetv2_ssl()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "28":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_mobilenetv2_best_transfer()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "29":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_mobilenetv2_ensemble_finetune()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        # ============================================================
        # ✅ VGG16 SSL (NEW)
        # ============================================================

        elif choice == "30":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH2)
            musicPlayer.play_music()
            try:
                run_vgg16_ssl()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in VGG16 SSL Full: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "31":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_vgg16_ssl_single("byol")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in VGG16 BYOL: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "32":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_vgg16_ssl_single("simclr")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in VGG16 SimCLR: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "33":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_vgg16_ssl_single("mae")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in VGG16 MAE: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "34":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_vgg16_best_transfer()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in VGG16 Best SSL: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "35":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_vgg16_ensemble_finetune()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in VGG16 FFT: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        # ============================================================
        # ✅ DenseNet SSL (NEW)
        # ============================================================

        elif choice == "36":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH2)
            musicPlayer.play_music()
            try:
                run_densenet_ssl()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "37":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_densenet_ssl_single("byol")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "38":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_densenet_ssl_single("simclr")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "39":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH3)
            musicPlayer.play_music()
            try:
                run_densenet_ssl_single("mae")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "40":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_densenet_best_transfer()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "41":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_densenet_ensemble_finetune()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "42":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH4)
            musicPlayer.play_music()
            try:
                run_direct_densenet_finetune()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        elif choice == "43":
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH5)
            musicPlayer.play_music()
            try:
                if not HAS_FIND_BEST:
                    if HAS_RICH:
                        console.print("[red]❌ FindBestModels module not available![/red]")
                    else:
                        print(f"{Colors.FAIL}❌ FindBestModels not available!{Colors.ENDC}")
                else:
                    run_find_best_models()
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]❌ Error in Find Best Architects: {e}[/red]")
                else:
                    print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
            musicPlayer.stop_music()

        else:
            if HAS_RICH:
                console.print(
                    "[red]❌ Invalid choice! Please enter a number between 0 and 21.[/red]"
                )
            else:
                print(
                    f"{Colors.FAIL}❌ Invalid choice! Please enter a number between 0 and 21.{Colors.ENDC}"
                )

        # Pause before showing menu again
        if HAS_RICH:
            input("\nPress Enter to continue...")
        else:
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")


# ================================================================
# ✅ Entry Point
# ================================================================

def runSSLAlgorithm():
    main()

if __name__ == "__main__":
    runSSLAlgorithm()
