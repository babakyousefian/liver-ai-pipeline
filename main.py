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
        run_new_models_pipeline,
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


# ================================================================
# ✅ Header
# ================================================================

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
        # ✅ NEW
        table.add_row(
            "4",
            "🎯 KD Ensemble new Models",
            "DenseNet → AlexNet / MobileNetV2 / VGG16 (3 seeds)",
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
        print("  4) 🎯 KD Ensemble new Models")  # ✅ NEW
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
        config_table.add_row("Polish Epochs", "50")
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
            choices=["0", "1", "2", "3", "4"],  # ✅ "4" اضافه شد
            default="0",
        )
    else:
        return input("\n👉 Enter your choice (0-4): ").strip()  # ✅ (0-4)


# ================================================================
# ✅ Main Loop
# ================================================================

def main():
    """Main entry point."""
    while True:
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
            handle_kd_new_models()  # ✅ NEW

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