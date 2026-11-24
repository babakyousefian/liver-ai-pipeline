# main.py
# -*- coding: utf-8 -*-

"""
Liver AI Pipeline — Main Entry Point (ULTIMATE EDITION v1.1)

Beautiful menu system with Rich UI:
- 0) Exit
- 1) Check Requirements Package
- 2) Run SSL Algorithms
- 3) KD Ensemble
- 4) KD Ensemble new Models     ← ✅ NEW

Author: Babak Yousefian
Version: 1.1.0
"""

import os
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Rich UI
# ================================================================

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.live import Live
    from rich.layout import Layout
    from rich.markdown import Markdown
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    print("⚠️ Rich not installed")
    print("Install: pip install rich")


# ================================================================
# ✅ Safe Imports (do not crash if not available)
# ================================================================

# SSL Algorithms (renamed from main.py — should always exist)
try:
    from src.pipeLine.Architects.SSL.SSLAlgorithms import runSSLAlgorithm
    HAS_SSL = True
except ImportError as e:
    HAS_SSL = False
    runSSLAlgorithm = None
    print(f"⚠️ SSLAlgorithms not available: {e}")

# KD Ensemble (classic)
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.knowledgeDistillationEnsemble import (
        run_kd_ensemble_pipeline,
        FINAL_MODEL_PATH as KD_FINAL_MODEL,
        KD_PDF_PATH,
    )
    HAS_KD = True
except ImportError as e:
    HAS_KD = False
    run_kd_ensemble_pipeline = None
    KD_FINAL_MODEL = None
    KD_PDF_PATH = None
    print(f"⚠️ KD Ensemble not available: {e}")

# ✅ NEW — KD Ensemble new Models (DenseNet → AlexNet/MobileNetV2/VGG16)
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.newModels import (
        run_new_models_pipeline, regenerate_pdf_for_student,
)
    HAS_KD_NEW = True
except ImportError as e:
    HAS_KD_NEW = False
    run_new_models_pipeline = None
    print(f"⚠️ KD Ensemble new Models not available: {e}")

# Requirements Checker
try:
    from src.requirements_packages.checkReqPac import run_check_requirements
    HAS_REQ_CHECK = True
except ImportError as e:
    HAS_REQ_CHECK = False
    run_check_requirements = None
    print(f"⚠️ checkReqPac not available: {e}")

# ✅ NEW — Swin Transformer Report Regeneration
try:
    from src.pipeLine.ConfigurationAlgorithms.resNetEfficientNetModels import (
        generate_report_swin_transformer,
    )
    HAS_SWIN_REPORT = True
except ImportError as e:
    HAS_SWIN_REPORT = False
    generate_report_swin_transformer = None
    print(f"⚠️ Swin report regeneration not available: {e}")

# ✅ NEW — ViT Report Regeneration
try:
    from src.pipeLine.ConfigurationAlgorithms.resNetEfficientNetModels import (
        generate_report_visio_transformer,
    )
    HAS_VIT_REPORT = True
except ImportError as e:
    HAS_VIT_REPORT = False
    generate_report_visio_transformer = None
    print(f"⚠️ ViT report regeneration not available: {e}")

# ✅ NEW — Select Best Models (Pretrained vs Scratch)
try:
    from src.pipeLine.ConfigurationAlgorithms.SelectBestModels import select_best_models
    HAS_SELECT_BEST = True
except ImportError as e:
    HAS_SELECT_BEST = False
    select_best_models = None
    print(f"⚠️ Select Best Models not available: {e}")

# ✅ NEW — Final KDE Algorithms (Best Model Selector)
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.finalKDEalgorithms import (
        run_menu_loop as run_final_kde_menu,
    )
    HAS_FINAL_KDE = True
except ImportError as e:
    HAS_FINAL_KDE = False
    run_final_kde_menu = None
    print(f"⚠️ Final KDE algorithms not available: {e}")

# ✅ NEW — ConvNeXt Boosted Ensemble
try:
    from src.AI.ConvNext.convnext_boosted import run_convnext_boosted
    HAS_CONVNEXT_BOOSTED = True
except ImportError as e:
    HAS_CONVNEXT_BOOSTED = False
    run_convnext_boosted = None
    print(f"⚠️ ConvNeXt Boosted Ensemble not available: {e}")
# ✅ NEW — ConvNeXt Boundary Refinement (G1/G2 targeted)
try:
    from src.AI.ConvNext.convnext_boundary_refine import (
        run_convnext_boosted as run_convnext_boundary_refine,
    )
    HAS_CONVNEXT_BOUNDARY = True
except ImportError as e:
    HAS_CONVNEXT_BOUNDARY = False
    run_convnext_boundary_refine = None
    print(f"⚠️ ConvNeXt Boundary Refinement not available: {e}")

# ✅ NEW — ConvNeXt Balanced Fine-Tune (replaces boundary refine)
try:
    from src.AI.ConvNext.convnext_balanced_finetune import (
        run_convnext_boosted as run_convnext_balanced_finetune,
    )
    HAS_CONVNEXT_BOUNDARY = True
except ImportError as e:
    HAS_CONVNEXT_BOUNDARY = False
    run_convnext_balanced_finetune = None
    print(f"⚠️ ConvNeXt Balanced Fine-Tune not available: {e}")

# ✅ NEW — ConvNeXt Corrector G1
try:
    from src.AI.ConvNext.convnext_corrector_g1 import (
        run_convnext_boosted as run_convnext_corrector_g1,
    )
    HAS_CONVNEXT_CORRECTOR_G1 = True
except ImportError as e:
    HAS_CONVNEXT_CORRECTOR_G1 = False
    run_convnext_corrector_g1 = None
    print(f"⚠️ ConvNeXt Corrector G1 not available: {e}")

try:
    from src.AI.ConvNext.convnext_g1_vs_nl_specialist import (
        run_convnext_boosted as run_convnext_g1_specialist,
    )
    HAS_G1_SPECIALIST = True
except ImportError as e:
    HAS_G1_SPECIALIST = False
    print(f"⚠️ G1 Specialist not available: {e}")

# ✅ NEW — ConvNeXt Targeted Fine-Tune (G1/G2)
try:
    from src.AI.ConvNext.convnext_targeted_finetune import (
        run_convnext_boosted as run_convnext_targeted_finetune,
    )
    HAS_TARGETED_FT = True
except ImportError as e:
    HAS_TARGETED_FT = False
    run_convnext_targeted_finetune = None
    print(f"⚠️ ConvNeXt Targeted Fine-Tune not available: {e}")


# ================================================================
# ✅ ASCII Art / Logo
# ================================================================

LOGO = r"""
    ██╗     ██╗██╗   ██╗███████╗██████╗      █████╗ ██╗
    ██║     ██║██║   ██║██╔════╝██╔══██╗    ██╔══██╗██║
    ██║     ██║██║   ██║█████╗  ██████╔╝    ███████║██║
    ██║     ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗    ██╔══██║██║
    ███████╗██║ ╚████╔╝ ███████╗██║  ██║    ██║  ██║██║
    ╚══════╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
"""

DEVELOPER = r"""
 ██████╗ ███████╗██╗   ██╗███████╗██╗      ██████╗ ██████╗ ███████╗██████╗
 ██╔══██╗██╔════╝██║   ██║██╔════╝██║     ██╔═══██╗██╔══██╗██╔════╝██╔══██╗
 ██║  ██║█████╗  ██║   ██║█████╗  ██║     ██║   ██║██████╔╝█████╗  ██║  ██║
 ██║  ██║██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║     ██║   ██║██╔═══╝ ██╔══╝  ██║  ██║
 ██████╔╝███████╗ ╚████╔╝ ███████╗███████╗╚██████╔╝██║     ███████╗██████╔╝
 ╚═════╝ ╚══════╝  ╚═══╝  ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═════╝

                ██████╗  █████╗ ██████╗  █████╗ ██╗  ██╗
                ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║ ██╔╝
                ██████╔╝███████║██████╔╝███████║█████╔╝
                ██╔══██╗██╔══██║██╔══██╗██╔══██║██╔═██╗
                ██████╔╝██║  ██║██████╔╝██║  ██║██║  ██╗
                ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
"""

# ================================================================
# ✅ Header
# ================================================================

import shutil
import time

def printLogo():
    if HAS_RICH:
        console.clear()
        console.print()

        terminal_width = shutil.get_terminal_size().columns

        logo_lines = LOGO.strip("\n").splitlines()
        developer_lines = DEVELOPER.strip("\n").splitlines()

        # Calculate total design width
        design_width = max(
            max(len(line) for line in logo_lines),
            max(len(line) for line in developer_lines)
        )

        # Center the complete design
        for line in logo_lines:
            console.print(line.center(terminal_width))

        for line in developer_lines:
            console.print(line.center(terminal_width))

        # Center footer relative to the complete design
        footer = "developed by Babak"
        console.print(
            footer.center(terminal_width),
            style="bold green"
        )

        time.sleep(9)

    else:
        pass

def print_header():
    """Print the main header with Rich."""
    if HAS_RICH:
        console.clear()
        console.print()
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]╔════════════════════════════════════════════════════════════════╗[/bold cyan]"
            )
        ))
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]║[/bold cyan]                                                                [bold cyan]║[/bold cyan]"
            )
        ))
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]║[/bold cyan]  [bold green]🧬  LIVER AI PIPELINE  🧬[/bold green]                              [bold cyan]║[/bold cyan]"
            )
        ))
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]║[/bold cyan]  [dim italic]Self-Supervised Learning · Knowledge Distillation[/dim italic]        [bold cyan]║[/bold cyan]"
            )
        ))
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]║[/bold cyan]                                                                [bold cyan]║[/bold cyan]"
            )
        ))
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]║[/bold cyan]  [bold yellow]Version 1.1.0[/bold yellow]  ·  [dim]Powered by PyTorch + Rich[/dim]              [bold cyan]║[/bold cyan]"
            )
        ))
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]║[/bold cyan]                                                                [bold cyan]║[/bold cyan]"
            )
        ))
        console.print(Align.center(
            Text.from_markup(
                "[bold cyan]╚════════════════════════════════════════════════════════════════╝[/bold cyan]"
            )
        ))
        console.print()


# ================================================================
# ✅ Menu
# ================================================================

def print_menu():
    """Print the main menu with Rich."""
    if HAS_RICH:
        table = Table(
            title="[bold cyan]📋 Main Menu[/bold cyan]",
            box=box.ROUNDED,
            border_style="bright_blue",
            header_style="bold magenta",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("#", style="bold yellow", width=4, justify="center")
        table.add_column("Option", style="bold white", width=32)
        table.add_column("Description", style="dim white", width=45)

        table.add_row("0", "🚪 Exit", "Goodbye!")
        table.add_row("1", "📦 Check Requirements Package", "Verify all packages are installed")
        table.add_row("2", "🧠 Run SSL Algorithms", "Self-Supervised Learning pipeline")
        table.add_row("3", "🎯 KD Ensemble", "Knowledge Distillation → single .pth")
        table.add_row(
            "4",
            "🎯 KD Ensemble new Models",
            "DenseNet → AlexNet / MobileNetV2 / VGG16 (3 seeds)",
        )
        table.add_row(
            "5",
            "📄 Regenerate ConvNeXt Reports",
            "Rebuild PDF + plots + JSON from existing .pth (no retraining)",
        )
        table.add_row(
            "6",
            "📄 Regenerate DenseNet Reports",
            "Rebuild PDF + plots + JSON from existing .pth (no retraining)",
        )
        table.add_row(
            "7",
            "📄 Regenerate Swin Transformer Reports",
            "Rebuild PDF + plots + JSON + CSV from existing history/config (no retraining)",
        )
        table.add_row(
            "8",
            "📄 Regenerate VGG16 Reports",
            "Rebuild PDF + plots + JSON + CSV from existing history/config (no retraining)",
        )
        # ✅ NEW: Option 9
        table.add_row(
            "9",
            "📄 Regenerate Vision Transformer Reports",
            "Rebuild PDF + plots + JSON + CSV from existing history/config (no retraining)",
        )
        # ✅ NEW: Option 10
        table.add_row(
            "10",
            "🏆 Select Best Models (Pretrained vs Scratch)",
            "Compare 7 pairs → copy winners to bestModels/<Model>/<name>BEST.pth",
        )
        # ✅ NEW: Option 11
        table.add_row(
            "11",
            "🧠 Final KDE Algorithms",
            "Best-Model Selector for Knowledge Distillation Ensemble (Typer + Rich)",
        )

        console.print(table)
    else:
        print("\n" + "=" * 80)
        print("  📋 Main Menu")
        print("=" * 80)
        print("  0) 🚪 Exit")
        print("  1) 📦 Check Requirements Package")
        print("  2) 🧠 Run SSL Algorithms")
        print("  3) 🎯 KD Ensemble")
        print("  4) 🎯 KD Ensemble new Models")
        print("  5) 📄 Regenerate ConvNeXt Reports")
        print("  6) 📄 Regenerate DenseNet Reports")
        print("  7) 📄 Regenerate Swin Transformer Reports")
        print("  8) 📄 Regenerate VGG16 Reports")
        print("  9) 📄 Regenerate Vision Transformer Reports")
        print(" 10) 🏆 Select Best Models (Pretrained vs Scratch)")
        print(" 11) 🧠 Final KDE Algorithms")
        print("=" * 80)


# ================================================================
# ✅ Handlers
# ================================================================

def handle_exit():
    """Handle exit."""
    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold yellow]👋 Goodbye! Thanks for using Liver AI Pipeline.[/bold yellow]\n"
            "[dim italic]See you next time! 🚀[/dim italic]",
            border_style="yellow",
            padding=(1, 4),
        ))
        console.print()
    else:
        print("\n👋 Goodbye!")


def handle_check_requirements():
    """Handle requirements check."""
    if not HAS_REQ_CHECK:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ checkReqPac module not available![/bold red]\n"
                "[yellow]Make sure checkReqPac.py exists in the project root.[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ checkReqPac not available")
        return

    try:
        results = run_check_requirements(
            req_path="requirements_gpu_RTX_2050.txt",
            verbose=True,
            save_script=True,
        )
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def handle_run_ssl():
    """Handle SSL algorithms."""
    if not HAS_SSL:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ SSLAlgorithms module not available![/bold red]\n"
                "[yellow]Make sure SSLAlgorithms.py exists in the project root.[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ SSLAlgorithms not available")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🧠 Self-Supervised Learning Pipeline[/bold cyan]\n"
            "[dim]Running SSL algorithms...[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

    if HAS_RICH:
        if not Confirm.ask("\n[bold yellow]🚀 Start SSL pipeline?[/bold yellow]", default=True):
            return
    else:
        confirm = input("\n🚀 Start SSL pipeline? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        runSSLAlgorithm()
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")

        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def handle_kd_ensemble():
    """Handle KD Ensemble pipeline."""
    if not HAS_KD:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ KD Ensemble module not available![/bold red]\n"
                "[yellow]Make sure src/pipeLine/knowledgeDistillationEnsemble.py exists.[/yellow]",
                border_style="red",
            ))

        else:
            print("❌ KD Ensemble not available")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold magenta]🎯 Knowledge Distillation Ensemble[/bold magenta]\n"
            "[dim]ResNet50 + EfficientNet-B1 → Student ResNet50 (.pth)[/dim]",
            border_style="magenta",
            padding=(1, 4),
        ))


        # Show config
        config_table = Table(
            title="[bold cyan]⚙️  Pipeline Configuration (v9.0)[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
        )
        config_table.add_column("Parameter", style="bold yellow")
        config_table.add_column("Value", style="bold white")

        config_table.add_row("KD Epochs", "100")
        config_table.add_row("Polish Epochs", "ResNet50")
        config_table.add_row("Batch Size", "16")
        config_table.add_row("Backbone LR", "1e-5")
        config_table.add_row("Head LR", "1e-3")
        config_table.add_row("Polish LR", "5e-6")
        config_table.add_row("KD Patience", "80")
        config_table.add_row("Polish Patience", "30")
        config_table.add_row("KD Alpha", "0.5")
        config_table.add_row("KD Temperature", "2.0")
        config_table.add_row("TTA Augmentations", "32")
        config_table.add_row("Bootstrap Samples", "1000")
        config_table.add_row("CV Folds", "5")

        console.print()
        console.print(config_table)

        console.print(f"\n[dim]📁 Final model output: {KD_FINAL_MODEL}[/dim]")
        console.print(f"[dim]📄 PDF report: {KD_PDF_PATH}[/dim]\n")

        if not Confirm.ask("[bold yellow]🚀 Start KD Ensemble pipeline?[/bold yellow]", default=True):
            return
    else:
        print("\n🎯 KD Ensemble")
        print("Starting...")
        confirm = input("🚀 Start? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ v9.1: Balanced config (not too low, not too high)
        kd_epochs = 100
        polish_epochs = 50
        batch_size = 16
        backbone_lr = 1e-5          # ✅ برگشت به 1e-5
        head_lr = 1e-3              # ✅ برگشت به 1e-3
        polish_lr = 5e-6            # ✅ نگه دار
        patience = 80
        polish_patience = 30
        alpha = 0.5
        temperature = 2.0           # ✅ نگه دار
        n_aug_tta = 32
        n_bootstrap = 1000
        cv_folds = 5

        # ✅ Print actual config BEFORE running
        if HAS_RICH:
            console.print("\n[bold yellow]🔍 ACTUAL CONFIG BEING PASSED:[/bold yellow]")
            console.print(f"   KD Epochs:       [cyan]{kd_epochs}[/cyan]")
            console.print(f"   Polish Epochs:   [cyan]{polish_epochs}[/cyan]")
            console.print(f"   Backbone LR:     [cyan]{backbone_lr}[/cyan]")
            console.print(f"   Head LR:         [cyan]{head_lr}[/cyan]")
            console.print(f"   Polish LR:       [cyan]{polish_lr}[/cyan]")
            console.print(f"   KD Patience:     [cyan]{patience}[/cyan]")
            console.print(f"   Polish Patience: [cyan]{polish_patience}[/cyan]")
            console.print(f"   Alpha:           [cyan]{alpha}[/cyan]")
            console.print(f"   Temperature:     [cyan]{temperature}[/cyan]")

        # ✅ Run pipeline
        results = run_kd_ensemble_pipeline(
            device=device,
            kd_epochs=kd_epochs,
            polish_epochs=polish_epochs,
            batch_size=batch_size,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            polish_lr=polish_lr,
            patience=patience,
            polish_patience=polish_patience,
            alpha=alpha,
            temperature=temperature,
            n_aug_tta=n_aug_tta,
            n_bootstrap=n_bootstrap,
            cv_folds=cv_folds,
            verbose=True,
        )

        if results and results.get("status") == "success":
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    "[bold green]🎉 KD Ensemble Complete![/bold green]\n"
                    f"[cyan]📁 Final model:[/cyan] {results['final_model_path']}\n"
                    f"[cyan]📄 PDF:[/cyan] {results['pdf_path']}\n"
                    f"[cyan]📊 Metrics:[/cyan] {results['metrics_json']}",
                    border_style="green",
                    padding=(1, 4),
                ))
            else:
                print("\n🎉 KD Ensemble Complete!")
                print(f"   Final model: {results['final_model_path']}")
                print(f"   PDF: {results['pdf_path']}")

        if results and results.get("status") == "success":
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    "[bold green]🎉 KD Ensemble Complete![/bold green]\n"
                    f"[cyan]📁 Final model:[/cyan] {results['final_model_path']}\n"
                    f"[cyan]📄 PDF:[/cyan] {results['pdf_path']}\n"
                    f"[cyan]📊 Metrics:[/cyan] {results['metrics_json']}",
                    border_style="green",
                    padding=(1, 4),
                ))
            else:
                print("\n🎉 KD Ensemble Complete!")
                print(f"   Final model: {results['final_model_path']}")
                print(f"   PDF: {results['pdf_path']}")

    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")

        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ NEW — Handle KD Ensemble new Models (Option 4)
# ================================================================
def handle_kd_new_models():
    """
    Handle KD Ensemble new Models pipeline.
    Delegates to newModels.run_new_models_pipeline() which shows its own
    internal menu (0/1/2/3) for DenseNet → AlexNet/MobileNetV2/VGG16.
    """
    if not HAS_KD_NEW:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ KD Ensemble new Models not available![/bold red]\n"
                "[yellow]Make sure "
                "src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/newModels.py "
                "exists.[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ KD Ensemble new Models not available")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🎯 KD Ensemble — New Models[/bold cyan]\n"
            "[dim]DenseNet Teacher → AlexNet / MobileNetV2 / VGG16[/dim]\n"
            "[dim]3 Seeds · Full TTA · CV5 · CI95 · 24-page PDF[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

    if HAS_RICH:
        if not Confirm.ask(
            "\n[bold yellow]🚀 Start KD Ensemble new Models?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Start KD Ensemble new Models? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        run_new_models_pipeline()
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# ✅ Get User Choice
# ================================================================

def get_choice():
    """Get user choice."""
    if HAS_RICH:
        return Prompt.ask(
            "\n[bold cyan]👉 Enter your choice[/bold cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
            default="0",
        )
    else:
        return input("\n👉 Enter your choice (0-11): ").strip()

def handle_regenerate_convnext_reports():
    """Handle ConvNeXt report regeneration (no retraining)."""
    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]📄 Regenerate ConvNeXt Reports[/bold cyan]\n"
            "[dim]Reads .pth + history + config → rebuilds PDF + plots + JSON[/dim]\n"
            "[dim]Output: finalReports/AI_Comparisons/ConvNext/reports/[/dim]\n"
            "[dim]No retraining. ~2-3 minutes.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Regenerate ConvNeXt reports?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Regenerate ConvNeXt reports? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        from src.AI.ConvNext.convNext import regenerate_convnext_reports
        result = regenerate_convnext_reports()

        if result:
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ ConvNeXt Reports Regenerated![/bold green]\n"
                    f"[cyan]📄 PDF:[/cyan] {result}\n"
                    f"[cyan]📁 Reports dir:[/cyan] "
                    f"{os.path.dirname(result)}",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Report regeneration failed.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_regenerate_densnet_reports():
    """Handle DenseNet report regeneration (no retraining)."""
    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]📄 Regenerate DenseNet Reports[/bold cyan]\n"
            "[dim]Reads .pth + history + config → rebuilds PDF + plots + JSON[/dim]\n"
            "[dim]Output: finalReports/AI_Comparisons/DensNet/reports/[/dim]\n"
            "[dim]No retraining. ~2-3 minutes.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Regenerate DenseNet reports?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Regenerate DenseNet reports? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        # ✅ استفاده از تابع generate_report_densnet از resNetEfficientNetModels
        from src.pipeLine.ConfigurationAlgorithms.resNetEfficientNetModels import generate_report_densnet
        result = generate_report_densnet()

        if result:
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ DenseNet Reports Regenerated![/bold green]\n"
                    f"[cyan]📄 PDF:[/cyan] {result}\n"
                    f"[cyan]📁 Reports dir:[/cyan] "
                    f"{os.path.dirname(result)}",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Report regeneration failed.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_regenerate_swin_reports():
    """Handle Swin Transformer report regeneration (no retraining)."""
    if not HAS_SWIN_REPORT:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ Swin report regeneration not available![/bold red]\n"
                "[yellow]Make sure "
                "src/pipeLine/ConfigurationAlgorithms/resNetEfficientNetModels.py "
                "exists and has generate_report_swin_transformer().[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ Swin report regeneration not available")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]📄 Regenerate Swin Transformer Reports[/bold cyan]\n"
            "[dim]Reads history + config + meta → rebuilds PDF + plots + JSON + CSV[/dim]\n"
            "[dim]Input:  finalReports/AI_Comparisons/Swin_Transformer/Swin_Transformer/[/dim]\n"
            "[dim]Output: finalReports/AI_Comparisons/Swin_Transformer/reports/[/dim]\n"
            "[dim]No retraining. ~2-3 minutes.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Regenerate Swin Transformer reports?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Regenerate Swin Transformer reports? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        result = generate_report_swin_transformer()

        if result:
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ Swin Transformer Reports Regenerated![/bold green]\n"
                    f"[cyan]📄 PDF:[/cyan] {result}\n"
                    f"[cyan]📁 Reports dir:[/cyan] "
                    f"{os.path.dirname(result)}",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Report regeneration failed.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_regenerate_vgg16_reports():
    """Handle VGG16 report regeneration (no retraining)."""
    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]📄 Regenerate VGG16 Reports[/bold cyan]\n"
            "[dim]Reads history + config + meta → rebuilds PDF + plots + JSON + CSV[/dim]\n"
            "[dim]Input:  finalReports/AI_Comparisons/VGG16/vgg16/[/dim]\n"
            "[dim]Output: finalReports/AI_Comparisons/VGG16/reports/[/dim]\n"
            "[dim]No retraining. ~2-3 minutes.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Regenerate VGG16 reports?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Regenerate VGG16 reports? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        from src.pipeLine.ConfigurationAlgorithms.resNetEfficientNetModels import (
            generate_report_vgg16,
        )
        result = generate_report_vgg16()

        if result:
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ VGG16 Reports Regenerated![/bold green]\n"
                    f"[cyan]📄 PDF:[/cyan] {result}\n"
                    f"[cyan]📁 Reports dir:[/cyan] "
                    f"{os.path.dirname(result)}",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Report regeneration failed.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_regenerate_vit_reports():
    """Handle Vision Transformer report regeneration (no retraining)."""
    if not HAS_VIT_REPORT:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ ViT report regeneration not available![/bold red]\n"
                "[yellow]Make sure "
                "src/pipeLine/ConfigurationAlgorithms/resNetEfficientNetModels.py "
                "exists and has generate_report_visio_transformer().[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ ViT report regeneration not available")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]📄 Regenerate Vision Transformer Reports[/bold cyan]\n"
            "[dim]Reads history + config + meta → rebuilds PDF + plots + JSON + CSV[/dim]\n"
            "[dim]Input:  finalReports/AI_Comparisons/visioTransformer/visioTransformer/[/dim]\n"
            "[dim]Output: finalReports/AI_Comparisons/visioTransformer/reports/[/dim]\n"
            "[dim]No retraining. ~2-3 minutes.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Regenerate Vision Transformer reports?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Regenerate Vision Transformer reports? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        result = generate_report_visio_transformer()

        if result:
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ Vision Transformer Reports Regenerated![/bold green]\n"
                    f"[cyan]📄 PDF:[/cyan] {result}\n"
                    f"[cyan]📁 Reports dir:[/cyan] "
                    f"{os.path.dirname(result)}",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Report regeneration failed.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_select_best_models():
    """Handle selecting best models between pretrained and scratch."""
    if not HAS_SELECT_BEST:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ Select Best Models module not available![/bold red]\n"
                "[yellow]Make sure "
                "src/pipeLine/SelectBestModels.py exists and has select_best_models().[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ Select Best Models not available")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🏆 Select Best Models (Pretrained vs Scratch)[/bold cyan]\n"
            "[dim]Compares 7 architectures:</dim]\n"
            "[dim]  ConvNext · DenseNet · VGG16 · VisionTransformer[/dim]\n"
            "[dim]  AlexNet · CNNLSTM · MobileNetV2[/dim]\n\n"
            "[dim]Winner → bestModels/<Model>/<name>BEST.pth[/dim]\n"
            "[dim]Output: finalReports/bestModels/[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Run best-model selection?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Run best-model selection? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        result = select_best_models()

        if result and result.get("pairs"):
            n_ok = sum(
                1 for info in result["pairs"].values()
                if info.get("status") == "copied"
            )
            n_total = len(result["pairs"])
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ Best-Model Selection Complete![/bold green]\n"
                    f"[cyan]📊 Copied:[/cyan] {n_ok}/{n_total} models\n"
                    f"[cyan]📁 Output:[/cyan] {result['best_models_dir']}\n"
                    f"[cyan]📄 Summary:[/cyan] summary.json + summary.txt",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Selection returned no results.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_final_kde():
    """Handle Final KDE Algorithms (best-model selector)."""
    if not HAS_FINAL_KDE:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ Final KDE module not available![/bold red]\n"
                "[yellow]Make sure "
                "src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/finalKDEalgorithms.py "
                "exists and exports run_menu_loop().[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ Final KDE module not available")
        return

    try:
        run_final_kde_menu()
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_convnext_boosted():
    """Handle ConvNeXt Boosted Ensemble (AdaBoost.SAMME)."""
    if not HAS_CONVNEXT_BOOSTED:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ ConvNeXt Boosted Ensemble not available![/bold red]\n"
                "[yellow]Make sure src/AI/ConvNext/convnext_boosted.py exists.[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ ConvNeXt Boosted Ensemble not available")
        return

    convnext_ckpt = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNext\convnextBEST.pth"
    )
    output_dir = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNextBoosted"
    )

    if not os.path.exists(convnext_ckpt):
        if HAS_RICH:
            console.print(Panel.fit(
                f"[bold red]❌ ConvNeXt checkpoint not found:[/bold red]\n"
                f"[yellow]{convnext_ckpt}[/yellow]\n\n"
                f"[dim]Run option 10 (Select Best Models) first.[/dim]",
                border_style="red",
            ))
        else:
            print(f"❌ Checkpoint not found: {convnext_ckpt}")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🚀 ConvNeXt Boosted Ensemble[/bold cyan]\n\n"
            f"[cyan]ConvNeXt checkpoint:[/cyan]\n"
            f"[dim]{convnext_ckpt}[/dim]\n\n"
            f"[cyan]Output dir:[/cyan]\n"
            f"[dim]{output_dir}[/dim]\n\n"
            f"[dim]Pipeline:[/dim]\n"
            f"[dim]  1. Extract deep features (768-dim) from ConvNeXt[/dim]\n"
            f"[dim]  2. Train AdaBoost.SAMME (300 trees, depth=10)[/dim]\n"
            f"[dim]  3. Tune class-specific thresholds for <3% error[/dim]\n"
            f"[dim]  4. Evaluate on test set[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Run ConvNeXt Boosted Ensemble?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Run ConvNeXt Boosted Ensemble? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        result = run_convnext_boosted(
            convnext_checkpoint=convnext_ckpt,
            output_dir=output_dir,
        )

        if result:
            if HAS_RICH:
                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ Boosted Ensemble Complete![/bold green]\n\n"
                    f"[cyan]Baseline Acc:[/cyan] {result.get('baseline_acc', 0):.4f}\n"
                    f"[cyan]Boosted Acc:[/cyan]  "
                    f"[bold green]{result.get('boosted_acc', 0):.4f}[/bold green]\n\n"
                    f"[cyan]Per-class error rates:[/cyan]\n" +
                    "\n".join([
                        f"  {name}: {err*100:.2f}%"
                        for name, err in zip(
                            ["NL", "G1", "G2", "G3"],
                            result.get("per_class_error", [0, 0, 0, 0]),
                        )
                    ]) +
                    f"\n\n[cyan]Saved to:[/cyan] {output_dir}",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Boosted Ensemble returned no results.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_convnext_boundary_refine():
    """Handle ConvNeXt Boundary Refinement (G1/G2 targeted)."""
    if not HAS_CONVNEXT_BOUNDARY:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ ConvNeXt Boundary Refinement not available![/bold red]\n"
                "[yellow]Make sure src/AI/ConvNext/convnext_boundary_refine.py exists.[/yellow]",
                border_style="red",
            ))
        else:
            print("❌ ConvNeXt Boundary Refinement not available")
        return

    convnext_ckpt = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNext\convnextBEST.pth"
    )
    output_dir = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNextBoundaryRefined"
    )

    if not os.path.exists(convnext_ckpt):
        if HAS_RICH:
            console.print(Panel.fit(
                f"[bold red]❌ ConvNeXt checkpoint not found:[/bold red]\n"
                f"[yellow]{convnext_ckpt}[/yellow]\n\n"
                f"[dim]Run option 10 (Select Best Models) first.[/dim]",
                border_style="red",
            ))
        else:
            print(f"❌ Checkpoint not found: {convnext_ckpt}")
        return

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🎯 ConvNeXt Boundary Refinement[/bold cyan]\n"
            "[dim]Targeted G1/G2 error reduction — NL untouched[/dim]\n\n"
            f"[cyan]ConvNeXt checkpoint:[/cyan]\n"
            f"[dim]{convnext_ckpt}[/dim]\n\n"
            f"[cyan]Output dir:[/cyan]\n"
            f"[dim]{output_dir}[/dim]\n\n"
            f"[dim]Pipeline:[/dim]\n"
            f"[dim]  1. ConvNeXt (91.32%) = PRIMARY classifier[/dim]\n"
            f"[dim]  2. NL protected (never touched if confident)[/dim]\n"
            f"[dim]  3. Advisor 1: XGBoost G1 vs NL  → fixes G1→NL errors[/dim]\n"
            f"[dim]  4. Advisor 2: XGBoost G1 vs G2  → fixes G1→G2 errors[/dim]\n"
            f"[dim]  5. Advisor 3: kNN G2 detector   → fixes G2 errors[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Run ConvNeXt Boundary Refinement?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Run ConvNeXt Boundary Refinement? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        result = run_convnext_boundary_refine(
            convnext_checkpoint=convnext_ckpt,
            output_dir=output_dir,
        )

        if result:
            if HAS_RICH:
                baseline = result.get("convnext_acc", 0)
                refined = result.get("refined_acc", 0)
                delta = refined - baseline
                delta_str = f"+{delta*100:.2f}%" if delta >= 0 else f"{delta*100:.2f}%"

                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ Boundary Refinement Complete![/bold green]\n\n"
                    f"[cyan]ConvNeXt baseline:[/cyan] {baseline:.4f}\n"
                    f"[cyan]Refined:[/cyan]          "
                    f"[bold green]{refined:.4f}[/bold green]  ({delta_str})\n\n"
                    f"[cyan]Per-class error rates:[/cyan]\n" +
                    "\n".join([
                        f"  {name}: {err*100:.2f}%"
                        for name, err in zip(
                            ["NL", "G1", "G2", "G3"],
                            result.get("per_class_error_refined", [0, 0, 0, 0]),
                        )
                    ]) +
                    f"\n\n[cyan]Saved to:[/cyan] {output_dir}",
                    border_style="green",
                    padding=(1, 4),
                ))
        else:
            if HAS_RICH:
                console.print(Panel.fit(
                    "[bold red]❌ Boundary Refinement returned no results.[/bold red]\n"
                    "[yellow]Check the log above for details.[/yellow]",
                    border_style="red",
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def handle_convnext_boundary_refine():
    """Handle ConvNeXt Balanced Fine-Tune."""
    if not HAS_CONVNEXT_BOUNDARY:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ ConvNeXt Balanced Fine-Tune not available![/bold red]\n"
                "[yellow]Make sure src/AI/ConvNext/convnext_balanced_finetune.py exists.[/yellow]",
                border_style="red",
            ))
        return

    convnext_ckpt = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNext\convnextBEST.pth"
    )
    output_dir = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNextBalanced"
    )

    if not os.path.exists(convnext_ckpt):
        if HAS_RICH:
            console.print(Panel.fit(
                f"[bold red]❌ ConvNeXt checkpoint not found:[/bold red]\n"
                f"[yellow]{convnext_ckpt}[/yellow]\n\n"
                f"[dim]Run option 10 (Select Best Models) first.[/dim]",
                border_style="red",
            ))
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🎯 ConvNeXt Balanced Fine-Tune[/bold cyan]\n"
            "[dim]Class-Balanced Weighted Loss · PDF Report · GPU Accelerated[/dim]\n\n"
            f"[cyan]Checkpoint:[/cyan]\n[dim]{convnext_ckpt}[/dim]\n\n"
            f"[cyan]Output:[/cyan]\n[dim]{output_dir}[/dim]\n\n"
            f"[cyan]Device:[/cyan] [bold yellow]{device.upper()}[/bold yellow]\n" +
            (f"[dim]GPU: {torch.cuda.get_device_name(0)}[/dim]\n" if device == "cuda"
             else "[bold red]⚠️ CUDA not available — CPU mode (very slow!)[/bold red]\n") +
            f"\n[dim]Pipeline:[/dim]\n"
            f"[dim]  1. Load ConvNeXt checkpoint (91.32%)[/dim]\n"
            f"[dim]  2. Compute Class-Balanced weights (β=0.999)[/dim]\n"
            f"[dim]  3. Fine-tune with Weighted CE + Mixup + CutMix[/dim]\n"
            f"[dim]  4. Early stop on balanced accuracy[/dim]\n"
            f"[dim]  5. Generate PDF report with all plots[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if device == "cpu":
            console.print(
                "\n[bold red]⚠️ CPU mode will take 1-2 HOURS. GPU strongly recommended.[/bold red]"
            )
            if not Confirm.ask("\n[bold yellow]Continue on CPU?[/bold yellow]", default=False):
                return

        if not Confirm.ask("\n[bold yellow]🚀 Run ConvNeXt Balanced Fine-Tune?[/bold yellow]", default=True):
            return
    else:
        confirm = input("\n🚀 Run ConvNeXt Balanced Fine-Tune? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        result = run_convnext_balanced_finetune(
            convnext_checkpoint=convnext_ckpt,
            output_dir=output_dir,
            device=device,
        )

        if result:
            if HAS_RICH:
                base_acc = result.get("baseline_acc", 0)
                ft_acc = result.get("finetuned_acc", 0)
                delta = ft_acc - base_acc
                delta_str = f"+{delta*100:.2f}%" if delta >= 0 else f"{delta*100:.2f}%"

                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ Balanced Fine-Tune Complete![/bold green]\n\n"
                    f"[cyan]Baseline acc:[/cyan]   {base_acc:.4f}\n"
                    f"[cyan]Fine-tuned acc:[/cyan] [bold green]{ft_acc:.4f}[/bold green]  ({delta_str})\n\n"
                    f"[cyan]Per-class error rates (fine-tuned):[/cyan]\n" +
                    "\n".join([
                        f"  {name}: {err*100:.2f}%"
                        for name, err in zip(
                            ["NL", "G1", "G2", "G3"],
                            result.get("per_class_error_finetuned", [0, 0, 0, 0]),
                        )
                    ]) +
                    f"\n\n[cyan]📄 PDF:[/cyan] {output_dir}\\ConvNext_Balanced_Report.pdf\n"
                    f"[cyan]📁 Output:[/cyan] {output_dir}",
                    border_style="green",
                    padding=(1, 4),
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()

def handle_convnext_corrector_g1():
    """Handle ConvNeXt Corrector G1."""
    if not HAS_CONVNEXT_CORRECTOR_G1:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ ConvNeXt Corrector G1 not available![/bold red]\n"
                "[yellow]Make sure src/AI/ConvNext/convnext_corrector_g1.py exists.[/yellow]",
                border_style="red",
            ))
        return

    convnext_ckpt = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNext\convnextBEST.pth"
    )
    output_dir = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNextCorrectorG1"
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🎯 ConvNeXt Corrector G1[/bold cyan]\n"
            "[dim]Zero impact on NL/G2/G3 — only fixes G1[/dim]\n\n"
            f"[cyan]Checkpoint:[/cyan]\n[dim]{convnext_ckpt}[/dim]\n\n"
            f"[cyan]Output:[/cyan]\n[dim]{output_dir}[/dim]\n\n"
            f"[cyan]Device:[/cyan] [bold yellow]{device.upper()}[/bold yellow]\n\n"
            f"[dim]Pipeline:[/dim]\n"
            f"[dim]  1. Load ConvNeXt baseline (FROZEN)[/dim]\n"
            f"[dim]  2. Extract 768-d features + probs[/dim]\n"
            f"[dim]  3. Train binary classifier: G1 vs {{NL, G2}} (G3 EXCLUDED)[/dim]\n"
            f"[dim]  4. Apply with 3 safety rules (NL/G2 protected)[/dim]\n"
            f"[dim]  5. Rollback if NL worsens[/dim]\n"
            f"[dim]  6. Save as convnext_corrector_g1.pkl[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Run ConvNeXt Corrector G1?[/bold yellow]",
            default=True,
        ):
            return
    else:
        confirm = input("\n🚀 Run ConvNeXt Corrector G1? (y/n): ")
        if confirm.lower() != "y":
            return

    try:
        result = run_convnext_corrector_g1(
            convnext_checkpoint=convnext_ckpt,
            output_dir=output_dir,
            device=device,
        )

        if result:
            if HAS_RICH:
                base_acc = result.get("baseline_acc", 0)
                corr_acc = result.get("corrected_acc", 0)
                delta = corr_acc - base_acc
                delta_str = f"+{delta*100:.2f}%" if delta >= 0 else f"{delta*100:.2f}%"

                console.print()
                console.print(Panel.fit(
                    f"[bold green]✅ Corrector G1 Complete![/bold green]\n\n"
                    f"[cyan]Baseline acc:[/cyan]   {base_acc:.4f}\n"
                    f"[cyan]Corrected acc:[/cyan] [bold green]{corr_acc:.4f}[/bold green] "
                    f"({delta_str})\n\n"
                    f"[cyan]Per-class error rates (corrected):[/cyan]\n" +
                    "\n".join([
                        f"  {name}: {err*100:.2f}%"
                        for name, err in zip(
                            ["NL", "G1", "G2", "G3"],
                            result.get("per_class_error_corrected", [0, 0, 0, 0]),
                        )
                    ]) +
                    f"\n\n[cyan]Stats:[/cyan] {result.get('corrector_stats', {})}\n"
                    f"[cyan]Rollback:[/cyan] {result.get('rollback_triggered', False)}\n\n"
                    f"[cyan]📁 Output:[/cyan] {output_dir}",
                    border_style="green",
                    padding=(1, 4),
                ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()

def handle_g1_specialist():
    convnext_ckpt = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNext\convnextBEST.pth"
    )
    output_dir = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNextG1Specialist"
    )
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]🎯 ConvNeXt G1 vs NL Specialist[/bold cyan]\n"
            "[dim]Train SEPARATE ConvNeXt ONLY on NL vs G1[/dim]\n\n"
            f"[cyan]Baseline:[/cyan]\n[dim]{convnext_ckpt}[/dim]\n\n"
            f"[cyan]Output:[/cyan]\n[dim]{output_dir}[/dim]\n\n"
            f"[cyan]Device:[/cyan] [bold yellow]{device.upper()}[/bold yellow]",
            border_style="cyan",
        ))
        if not Confirm.ask("\n[bold yellow]🚀 Run?[/bold yellow]", default=True):
            return

    try:
        result = run_convnext_g1_specialist(
            convnext_checkpoint=convnext_ckpt,
            output_dir=output_dir,
            device=device,
        )
        if HAS_RICH and result:
            console.print(Panel.fit(
                f"[bold green]✅ Specialist Complete![/bold green]\n\n"
                f"[cyan]Baseline:[/cyan]   {result['baseline_acc']:.4f}\n"
                f"[cyan]Final:[/cyan]     {result['final_acc']:.4f}\n"
                f"[cyan]Specialist val_acc:[/cyan] {result['specialist_val_acc']:.4f}",
                border_style="green",
            ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()

def handle_convnext_targeted_finetune():
    """Handle ConvNeXt Targeted Fine-Tune (G1/G2 only)."""
    if not HAS_TARGETED_FT:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold red]❌ ConvNeXt Targeted Fine-Tune not available![/bold red]\n"
                "[yellow]Make sure src/AI/ConvNext/convnext_targeted_finetune.py exists.[/yellow]",
                border_style="red",
            ))
        return

    convnext_ckpt = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNext\convnextBEST.pth"
    )
    output_dir = (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\bestModels\ConvNextTargeted"
    )

    if not os.path.exists(convnext_ckpt):
        if HAS_RICH:
            console.print(f"[bold red]❌ Checkpoint not found: {convnext_ckpt}[/bold red]")
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🎯 ConvNeXt Targeted Fine-Tune[/bold cyan]\n"
            "[dim]G1/G2 only — head fine-tune, backbone frozen[/dim]\n\n"
            f"[cyan]Baseline:[/cyan]\n[dim]{convnext_ckpt}[/dim]\n\n"
            f"[cyan]Output:[/cyan]\n[dim]{output_dir}[/dim]\n\n"
            f"[cyan]Device:[/cyan] [bold yellow]{device.upper()}[/bold yellow]\n\n"
            f"[dim]Pipeline:[/dim]\n"
            f"[dim]  1. Load ConvNeXt baseline (FROZEN)[/dim]\n"
            f"[dim]  2. Weights: NL=1.0, G1=2.5, G2=1.5, G3=1.0[/dim]\n"
            f"[dim]  3. Fine-tune HEAD only (no Mixup/CutMix)[/dim]\n"
            f"[dim]  4. Early stop on balanced_acc - 0.5*max_err[/dim]\n"
            f"[dim]  5. Save as convnext_targeted.pth[/dim]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if not Confirm.ask(
            "\n[bold yellow]🚀 Run ConvNeXt Targeted Fine-Tune?[/bold yellow]",
            default=True,
        ):
            return

    try:
        result = run_convnext_targeted_finetune(
            baseline_ckpt=convnext_ckpt,
            output_dir=output_dir,
            device=device,
        )

        if result and HAS_RICH:
            base_acc = result["baseline_acc"]
            ft_acc = result["finetuned_acc"]
            delta = ft_acc - base_acc
            delta_str = f"+{delta*100:.2f}%" if delta >= 0 else f"{delta*100:.2f}%"

            console.print()
            console.print(Panel.fit(
                f"[bold green]✅ Targeted Fine-Tune Complete![/bold green]\n\n"
                f"[cyan]Baseline:[/cyan]   {base_acc:.4f}\n"
                f"[cyan]Fine-tuned:[/cyan] [bold green]{ft_acc:.4f}[/bold green]  ({delta_str})\n\n"
                f"[cyan]Per-class error (fine-tuned):[/cyan]\n" +
                "\n".join([
                    f"  {name}: {err*100:.2f}%"
                    for name, err in zip(
                        ["NL", "G1", "G2", "G3"],
                        result["per_class_error_finetuned"],
                    )
                ]) +
                f"\n\n[cyan]📁 Output:[/cyan] {output_dir}",
                border_style="green",
                padding=(1, 4),
            ))
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()

# ================================================================
# ✅ Main Loop
# ================================================================

def main():
    """Main entry point."""
    while True:
        printLogo()
        print_header()
        print_menu()

        try:
            choice = get_choice()
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "0":
            handle_exit()
            break
        elif choice == "1":
            handle_check_requirements()
        elif choice == "2":
            handle_run_ssl()
        elif choice == "3":
            handle_kd_ensemble()
        elif choice == "4":
            handle_kd_new_models()
        elif choice == "5":
            handle_regenerate_convnext_reports()
        elif choice == "6":
            handle_regenerate_densnet_reports()
        elif choice == "7":
            handle_regenerate_swin_reports()
        elif choice == "8":
            handle_regenerate_vgg16_reports()
        elif choice == "9":
            handle_regenerate_vit_reports()
        elif choice == "10":
            handle_select_best_models()
        elif choice == "11":
            handle_final_kde()
        else:
            if HAS_RICH:
                console.print(f"[red]❌ Invalid choice: {choice}[/red]")
            else:
                print(f"❌ Invalid choice: {choice}")

        # Pause before returning to menu
        if choice != "0":
            if HAS_RICH:
                console.print()
                Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
            else:
                input("\nPress Enter to continue...")


# ================================================================
# ✅ Entry
# ================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print("\n\n[bold yellow]⚠️  Interrupted by user. Goodbye![/bold yellow]\n")

        else:
            print("\n\n⚠️  Interrupted. Goodbye!")
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[bold red]❌ Fatal error: {e}[/bold red]")
        else:
            print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()