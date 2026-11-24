# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/finalKDEalgorithms.py
# -*- coding: utf-8 -*-

"""
Final KDE Algorithms — Best Model Selector
============================================
Interactive CLI (Typer + Rich) that:

  1) Scans `finalReports/bestModels/<Model>/*.pth`
  2) Reads each model's `test_accuracy` from:
       - `finalReports/bestModels/summary.json`   (preferred)
       - or from a per-model metrics file, if present
  3) Picks the BEST model (highest test_accuracy)
  4) Copies it to:
       finalReports/bestModels/bestKDEnsemble/bestKDE.pth

Author: Babak Yousefian
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.align import Align
from rich.text import Text
from rich.rule import Rule
from rich import box

# ✅ NEW — KD Swin pipeline
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.kd_convnext_swin import (
        run_kd_convnext_swin,
    )

    HAS_KD_SWIN = True
except ImportError as e:
    HAS_KD_SWIN = False
    run_kd_convnext_swin = None
    print(f"⚠️ KD Swin pipeline not available: {e}")

# ✅ NEW — Regenerate KD reports
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.regenerate_kd_report import (
        regenerate_densenet_report,
        regenerate_swin_report,
    )

    HAS_KD_REPORT_REGEN = True
except ImportError as e:
    HAS_KD_REPORT_REGEN = False
    regenerate_densenet_report = None
    regenerate_swin_report = None
    print(f"⚠️ KD report regeneration not available: {e}")

# ✅ NEW — for Swin KD report regeneration
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.kd_convnext_swin import (
        STUDENT_BEST, KD_OUTPUT_DIR,
    )

    HAS_SWIN_CKPT = True
except ImportError as e:
    HAS_SWIN_CKPT = False
    STUDENT_BEST = None
    KD_OUTPUT_DIR = None
    print(f"⚠️ Swin KD paths not available: {e}")

# ✅ NEW — ConvNeXt → ViT KD
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.convnext_vit_kd import (
        run_convnext_vit_kd,
    )
    HAS_CONVNEXT_VIT_KD = True
except ImportError as e:
    HAS_CONVNEXT_VIT_KD = False
    run_convnext_vit_kd = None
    print(f"⚠️ ConvNeXt→ViT KD not available: {e}")

# =====================================================
# 🎵 MUSIC PLAYER
# =====================================================

try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    pygame = None


def play_music(music_path: Optional[str] = None, loop: bool = True) -> bool:
    """
    Play background music in a non-blocking way.
    If pygame is not installed, silently skip.
    """
    if not HAS_PYGAME:
        return False

    # Default path — user can change
    if music_path is None:
        music_path = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\assets\music\kd_theme.mp3"

    if not os.path.exists(music_path):
        console.print(
            f"[dim]🎵 Music file not found (skipping):[/dim] {music_path}",
            highlight=False,
        )
        return False

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.set_volume(0.3)
        pygame.mixer.music.play(-1 if loop else 0)
        console.print(f"[dim]🎵 Playing background music:[/dim] {os.path.basename(music_path)}")
        return True
    except Exception as e:
        console.print(f"[dim]⚠️ Music playback failed: {e}[/dim]")
        return False


def stop_music():
    """Stop background music if playing."""
    if not HAS_PYGAME:
        return
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    except Exception:
        pass


# ✅ NEW — KD pipeline
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.kd_convnext_densenet import (
        run_kd_convnext_densenet,
    )

    HAS_KD_PIPELINE = True
except ImportError as e:
    HAS_KD_PIPELINE = False
    run_kd_convnext_densenet = None
    print(f"⚠️ KD pipeline not available: {e}")

# ✅ NEW — ConvNeXt → VGG16 KD Ensemble
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.convnext_vgg16_kd import (
        run_convnext_vgg16_kd,
    )

    HAS_CONVNEXT_VGG16_KD = True
except ImportError as e:
    HAS_CONVNEXT_VGG16_KD = False
    run_convnext_vgg16_kd = None
    print(f"⚠️ ConvNeXt→VGG16 KD not available: {e}")

try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.convnext_vit_kd import (
        run_convnext_vit_kd,
        run_advanced_only_vit,
    )
    HAS_CONVNEXT_VIT_KD = True
    HAS_ADVANCED_ONLY_VIT = True
except ImportError as e:
    HAS_CONVNEXT_VIT_KD = False
    HAS_ADVANCED_ONLY_VIT = False
    run_convnext_vit_kd = None
    run_advanced_only_vit = None
    print(f"⚠️ ConvNeXt→ViT KD not available: {e}")

# ✅ NEW — ConvNeXt → AlexNet KD
try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.convnext_alexnet_kd import (
        run_convnext_alexnet_kd,
        run_advanced_only_alexnet,
    )
    HAS_CONVNEXT_ALEXNET_KD = True
    HAS_ADVANCED_ONLY_ALEXNET = True
except ImportError as e:
    HAS_CONVNEXT_ALEXNET_KD = False
    HAS_ADVANCED_ONLY_ALEXNET = False
    run_convnext_alexnet_kd = None
    run_advanced_only_alexnet = None
    print(f"⚠️ ConvNeXt→AlexNet KD not available: {e}")

try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.kd_convnext_cnnlstm import (
        run_kd_convnext_cnnlstm,
    )
    HAS_KD_CNNLSTM = True
except ImportError as e:
    HAS_KD_CNNLSTM = False
    run_kd_convnext_cnnlstm = None
    print(f"⚠️ KD CNNLSTM pipeline not available: {e}")

# =====================================================
# CONSTANTS
# =====================================================

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
BEST_MODELS_DIR = os.path.join(FINAL_REPORTS_DIR, "bestModels")

# ── Teacher folder (SINGULAR) ──
KD_ENSEMBLE_DIR = os.path.join(BEST_MODELS_DIR, "bestKDEnsemble")
KD_OUTPUT_NAME = "bestKDE.pth"
TEACHER_CKPT = os.path.join(KD_ENSEMBLE_DIR, KD_OUTPUT_NAME)

# ── Students folder (PLURAL) ──
KD_STUDENTS_DIR = os.path.join(BEST_MODELS_DIR, "bestKDEnsembles")
STUDENT_NAME = "DenseNet121"
KD_OUTPUT_DIR = os.path.join(KD_STUDENTS_DIR, STUDENT_NAME)

SUMMARY_JSON = os.path.join(BEST_MODELS_DIR, "summary.json")

# Models we expect inside `bestModels/`
EXPECTED_MODELS = [
    "ConvNext",
    "DenseNet",
    "VGG16",
    "VisionTransformer",
    "AlexNet",
    "CNNLSTM",
    "MobileNetV2",
]

# =====================================================
# RICH CONSOLE
# =====================================================

console = Console()

app = typer.Typer(
    name="final-kde",
    help="Final Knowledge Distillation Ensemble — Best Model Selector",
    add_completion=False,
    rich_markup_mode="rich",
)

# =====================================================
# BANNER
# =====================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ███████╗██╗███╗   ██╗ █████╗ ██╗         ██╗  ██╗██████╗ ███████╗  ║
║   ██╔════╝██║████╗  ██║██╔══██╗██║         ██║ ██╔╝██╔══██╗██╔════╝  ║
║   █████╗  ██║██╔██╗ ██║███████║██║         █████╔╝ ██║  ██║█████╗    ║
║   ██╔══╝  ██║██║╚██╗██║██╔══██║██║         ██╔═██╗ ██║  ██║██╔══╝    ║
║   ██║     ██║██║ ╚████║██║  ██║███████╗    ██║  ██╗██████╔╝███████╗  ║
║   ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═════╝ ╚══════╝  ║
║                                                                      ║
║          🧠  Knowledge Distillation Ensemble Selector  🧠            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Print the main banner."""
    console.clear()
    console.print()
    console.print(Align.center(Text.from_markup(f"[bold cyan]{BANNER}[/bold cyan]")))
    console.print()


def print_header():
    """Print a compact header."""
    console.print()
    console.print(Panel.fit(
        "[bold green]🧬  FINAL KDE ALGORITHMS  🧬[/bold green]\n"
        "[dim italic]Best-Model Selector for Knowledge Distillation Ensemble[/dim italic]",
        border_style="bright_cyan",
        padding=(1, 4),
    ))


# =====================================================
# HELPERS — JSON / METRICS
# =====================================================

def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_test_acc_from_dict(data: Dict[str, Any]) -> Optional[float]:
    """Extract test_accuracy from metrics.json (multiple layouts)."""
    if not isinstance(data, dict):
        return None

    # A) config.test_accuracy
    cfg = data.get("config")
    if isinstance(cfg, dict):
        for key in ("test_accuracy", "test_acc"):
            if key in cfg and cfg[key] is not None:
                try:
                    return float(cfg[key])
                except (TypeError, ValueError):
                    pass

    # B) metrics.accuracy / metrics.test_accuracy
    m = data.get("metrics")
    if isinstance(m, dict):
        for key in ("test_accuracy", "test_acc", "accuracy"):
            if key in m and m[key] is not None:
                try:
                    return float(m[key])
                except (TypeError, ValueError):
                    pass

    # C) Top-level
    for key in ("test_accuracy", "test_acc", "accuracy"):
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                pass

    return None


def action_regen_swin_report():
    """Regenerate Swin KD report from existing checkpoint (no retraining)."""
    print_banner()

    # ✅ مسیر صریح Swin — نه از KD_OUTPUT_DIR قدیمی
    SWIN_DIR = os.path.join(BEST_MODELS_DIR, "bestKDEnsembles", "SwinTransformer")
    SWIN_CKPT = os.path.join(SWIN_DIR, "kd_swin_best.pth")

    console.print(Rule("[bold cyan]🔄 Regenerate Swin KD Report[/bold cyan]"))
    console.print()

    if not os.path.exists(SWIN_CKPT):
        console.print(Panel.fit(
            f"[bold red]❌ Swin checkpoint not found:[/bold red]\n"
            f"[yellow]{SWIN_CKPT}[/yellow]\n\n"
            f"[dim]Run option 3 (KD ConvNext → Swin) first.[/dim]",
            border_style="red",
        ))
        return

    console.print(Panel.fit(
        "[bold cyan]⚙️  What this does[/bold cyan]\n\n"
        "1. Loads kd_swin_best.pth\n"
        "2. Rebuilds history from training log (76 epochs)\n"
        "3. Runs test eval + CV5 + CI95 + TTA32\n"
        "4. Saves all JSONs, plots, predictions.csv\n"
        "5. Generates 24-page PDF report\n\n"
        f"[cyan]Checkpoint:[/cyan]\n[dim]{SWIN_CKPT}[/dim]\n\n"
        f"[cyan]Output dir:[/cyan]\n[dim]{SWIN_DIR}[/dim]",
        border_style="cyan", padding=(1, 4),
    ))

    if not Confirm.ask("\n[bold yellow]🚀 Run?[/bold yellow]", default=True):
        return

    try:
        from .regen_swin_kd_report import regen_swin_kd_report
        result = regen_swin_kd_report()
        console.print(Panel.fit(
            f"[bold green]✅ Report regenerated![/bold green]\n"
            f"[cyan]📄 PDF:[/cyan] {result['pdf_path']}\n"
            f"[cyan]📊 Test Acc:[/cyan] {result['test_acc']:.4f}",
            border_style="green", padding=(1, 4),
        ))
    except Exception as e:
        console.print(Panel.fit(f"[bold red]❌ Error:[/bold red] {e}", border_style="red"))
        import traceback;
        traceback.print_exc()


def _collect_models() -> List[Dict[str, Any]]:
    """
    Scan `bestModels/` for `<Model>/*.pth` checkpoints and try to
    read `test_accuracy` from:
      - summary.json (preferred)
      - or per-model metrics.json if present in sub-folder
    """
    results: List[Dict[str, Any]] = []

    if not os.path.isdir(BEST_MODELS_DIR):
        return results

    # 1) Load summary.json once (if exists)
    summary = _load_json(SUMMARY_JSON) or {}
    summary_pairs = summary.get("pairs", {}) if isinstance(summary, dict) else {}

    for model_name in EXPECTED_MODELS:
        model_dir = os.path.join(BEST_MODELS_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Find .pth files inside
        pth_files = [
            f for f in os.listdir(model_dir)
            if f.lower().endswith(".pth")
        ]
        if not pth_files:
            continue

        ckpt_path = os.path.join(model_dir, pth_files[0])

        # Determine score
        score: Optional[float] = None
        source = "unknown"

        # a) summary.json
        if model_name in summary_pairs:
            pair = summary_pairs[model_name]
            s = pair.get("winner_score")
            if s is not None:
                try:
                    score = float(s)
                    source = "summary.json"
                except (TypeError, ValueError):
                    pass

        # b) try metrics.json in same folder or parent
        if score is None:
            for candidate in [
                os.path.join(model_dir, "metrics.json"),
                os.path.join(BEST_MODELS_DIR, model_name, "metrics.json"),
            ]:
                data = _load_json(candidate)
                if data:
                    s = _extract_test_acc_from_dict(data)
                    if s is not None:
                        score = s
                        source = os.path.basename(candidate)
                        break

        results.append({
            "model": model_name,
            "ckpt_path": ckpt_path,
            "filename": pth_files[0],
            "size_mb": os.path.getsize(ckpt_path) / (1024 * 1024),
            "test_acc": score,
            "score_source": source,
        })

    # Sort: models with score first, descending
    results.sort(
        key=lambda r: (r["test_acc"] is not None, r["test_acc"] or 0.0),
        reverse=True,
    )
    return results


# =====================================================
# ACTIONS
# =====================================================

def action_find_best_model():
    """
    Scan bestModels/, pick the best (highest test_accuracy),
    and copy it to bestModels/bestKDEnsemble/bestKDE.pth
    """
    print_banner()

    console.print(Rule("[bold cyan]📂 Scanning bestModels/[/bold cyan]"))
    console.print()

    if not os.path.isdir(BEST_MODELS_DIR):
        console.print(Panel.fit(
            f"[bold red]❌ bestModels directory not found:[/bold red]\n"
            f"[yellow]{BEST_MODELS_DIR}[/yellow]\n\n"
            f"[dim]Run option 10 in main.py first to generate it.[/dim]",
            border_style="red",
        ))
        return

    with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="cyan"),
            console=console,
    ) as progress:
        task = progress.add_task("Scanning models...", total=None)
        models = _collect_models()
        progress.update(task, completed=True)

    if not models:
        console.print(Panel.fit(
            "[bold red]❌ No .pth models found in bestModels/[/bold red]\n"
            "[yellow]Make sure the bestModels/ folder contains model sub-folders.[/yellow]",
            border_style="red",
        ))
        return

    # ---- Display table ----
    table = Table(
        title="[bold cyan]📊 Found Models in bestModels/[/bold cyan]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("#", style="bold yellow", width=4, justify="center")
    table.add_column("Model", style="bold white", width=22)
    table.add_column("Test Acc", style="bold green", width=12, justify="right")
    table.add_column("Size (MB)", style="dim white", width=12, justify="right")
    table.add_column("Source", style="dim cyan", width=18)
    table.add_column("Checkpoint", style="dim white", width=32)

    for i, m in enumerate(models, 1):
        acc_str = f"{m['test_acc']:.4f}" if m["test_acc"] is not None else "N/A"
        acc_style = "bold green" if m["test_acc"] is not None else "red"
        table.add_row(
            str(i),
            m["model"],
            f"[{acc_style}]{acc_str}[/{acc_style}]",
            f"{m['size_mb']:.1f}",
            m["score_source"],
            m["filename"],
        )

    console.print(table)

    # ---- Pick best ----
    best = models[0]
    if best["test_acc"] is None:
        console.print(Panel.fit(
            "[bold red]❌ No model has a valid test_accuracy[/bold red]\n"
            "[yellow]Check summary.json or per-model metrics.json.[/yellow]",
            border_style="red",
        ))
        return

    console.print()
    console.print(Panel.fit(
        f"[bold green]🏆 BEST MODEL[/bold green]\n\n"
        f"[cyan]Model:[/cyan] [bold white]{best['model']}[/bold white]\n"
        f"[cyan]Test Acc:[/cyan] [bold green]{best['test_acc']:.4f}[/bold green]\n"
        f"[cyan]Checkpoint:[/cyan] [dim]{best['ckpt_path']}[/dim]",
        border_style="green",
        padding=(1, 4),
    ))

    # ---- Confirm ----
    if not Confirm.ask(
            f"\n[bold yellow]📥 Copy this model to "
            f"bestModels/bestKDEnsemble/{KD_OUTPUT_NAME}?[/bold yellow]",
            default=True,
    ):
        console.print("[dim]Cancelled.[/dim]")
        return

    # ---- Copy ----
    os.makedirs(KD_ENSEMBLE_DIR, exist_ok=True)
    out_path = os.path.join(KD_ENSEMBLE_DIR, KD_OUTPUT_NAME)

    try:
        shutil.copy2(best["ckpt_path"], out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        # ---- Write metadata ----
        meta = {
            "timestamp": datetime.now().isoformat(),
            "selected_model": best["model"],
            "test_accuracy": best["test_acc"],
            "source_checkpoint": best["ckpt_path"],
            "output_checkpoint": out_path,
            "size_mb": size_mb,
            "candidates": [
                {
                    "model": m["model"],
                    "test_accuracy": m["test_acc"],
                    "score_source": m["score_source"],
                    "ckpt_path": m["ckpt_path"],
                }
                for m in models
            ],
        }
        meta_path = os.path.join(KD_ENSEMBLE_DIR, "selection_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False, default=str)

        console.print()
        console.print(Panel.fit(
            f"[bold green]✅ Best KDE Model Copied![/bold green]\n\n"
            f"[cyan]📄 Output:[/cyan] {out_path}\n"
            f"[cyan]📊 Size:[/cyan] {size_mb:.2f} MB\n"
            f"[cyan]🧾 Metadata:[/cyan] {meta_path}",
            border_style="green",
            padding=(1, 4),
        ))
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Copy failed:[/bold red] {e}",
            border_style="red",
        ))


# =====================================================
# MENU
# =====================================================

def show_menu():
    """Display the main menu."""
    table = Table(
        title="[bold cyan]📋 Final KDE — Main Menu[/bold cyan]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("#", style="bold yellow", width=4, justify="center")
    table.add_column("Action", style="bold white", width=50)
    table.add_column("Description", style="dim white", width=40)

    table.add_row(
        "0", "🚪 Exit", "Leave the Final KDE menu",
    )
    table.add_row(
        "1", "🏆 Find Best Model for KDE",
        "Scan bestModels/ → pick highest test_acc → copy to bestKDEnsemble/",
    )
    table.add_row(
        "2", "🧠 Run KD: ConvNeXt (Teacher) → DenseNet (Student)",
        "Full KD pipeline → kd_densenet_best.pth (target: test_acc > 90%, good fit)",
    )
    table.add_row(
        "3", "🧠 Run KD: ConvNeXt (Teacher) → Swin Transformer (Student)",
        "KD + CV5 + CI95 + TTA32 + 24-page PDF (LIME/SHAP/Captum/Netron)",
    )
    table.add_row(
        "4", "📄 Regenerate KD Reports (no retraining)",
        "Rebuild 24-page PDF for existing DenseNet121 / SwinTransformer artifacts",
    )
    table.add_row(
        "5", "🔄 Regenerate Swin KD Report (no retraining)",
        "Load kd_swin_best.pth → rebuild history from log → eval → 24-page PDF",
    )
    table.add_row(
        "6",
        "🔗 ConvNeXt → VGG16 KD",
        "Teacher: ConvNeXt, Student: VGG16",
    )
    table.add_row(
        "7",
        "🔗 ConvNeXt → ViT KD",
        "Teacher: ConvNeXt, Student: ViT",
    )
    table.add_row(
        "8",
        "📄 Regenerate ViT KD Report (no retraining)",
        "Load best ViT → eval + CI95 + TTA32 → 24-page PDF",
    )

    table.add_row(
        "9",
        "🔗 ConvNeXt → AlexNet KD",
        "Teacher: ConvNeXt, Student: AlexNet",
    )

    table.add_row(
        "10",
        "🔗 ConvNeXt → AlexNet KD Reports",
        "Fully Reports",
    )

    table.add_row(
        "11", "🎯 KD: ConvNeXt → CNNLSTM (3-stage, ALL classes)",
        "KD + Full FT (all classes) + Threshold Tuning + TTA32 + 24-page PDF",
    )
    console.print(table)


def action_run_kd():
    """Run KD: ConvNeXt (Teacher) → DenseNet (Student)."""
    print_banner()

    console.print(Rule("[bold magenta]🧠 Knowledge Distillation Pipeline[/bold magenta]"))
    console.print()

    if not HAS_KD_PIPELINE:
        console.print(Panel.fit(
            "[bold red]❌ KD pipeline not available![/bold red]\n"
            "[yellow]Make sure kd_convnext_densenet.py exists.[/yellow]",
            border_style="red",
        ))
        return

    if not os.path.exists(TEACHER_CKPT):
        console.print(Panel.fit(
            f"[bold red]❌ Teacher checkpoint not found:[/bold red]\n"
            f"[yellow]{TEACHER_CKPT}[/yellow]\n\n"
            f"[dim]⚠️  Run option 1 (Find Best Model for KDE) FIRST[/dim]\n"
            f"[dim]   to create bestKDEnsemble/bestKDE.pth[/dim]",
            border_style="red",
        ))
        return

    # Show which teacher was selected (from bestKDEnsemble/ — SINGULAR)
    teacher_name = "?"
    teacher_acc = 0.0
    meta_path = os.path.join(KD_ENSEMBLE_DIR, "selection_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            teacher_name = meta.get("selected_model", "?")
            teacher_acc = float(meta.get("test_accuracy", 0.0))
            console.print(Panel.fit(
                f"[bold green]🏆 Teacher Selected[/bold green]\n\n"
                f"[cyan]Model:[/cyan] [bold white]{teacher_name}[/bold white]\n"
                f"[cyan]Test Acc:[/cyan] [bold green]{teacher_acc:.4f}[/bold green]\n"
                f"[cyan]File:[/cyan] [dim]{meta.get('output_checkpoint', TEACHER_CKPT)}[/dim]",
                border_style="green",
                padding=(1, 4),
            ))
        except Exception:
            pass

    if not Confirm.ask(
            "\n[bold yellow]🚀 Start KD training now?[/bold yellow]",
            default=True,
    ):
        return

    # 🎵 Start music
    music_on = play_music()

    try:
        result = run_kd_convnext_densenet()
    finally:
        # 🎵 Stop music even if error
        if music_on:
            stop_music()

    console.print()
    console.print(Panel.fit(
        f"[bold green]✅ KD Training Complete![/bold green]\n\n"
        f"[cyan]Best Val Acc:[/cyan] {result['best_val_acc']:.4f}\n"
        f"[cyan]Test Acc:[/cyan]     [bold green]{result['test_accuracy']:.4f}[/bold green]\n"
        f"[cyan]Test AUC:[/cyan]     {result['test_auc']:.4f}\n\n"
        f"[cyan]Student:[/cyan] {result['student_path']}\n"
        f"[cyan]Metrics:[/cyan] {result['metrics_path']}\n"
        f"[cyan]Summary:[/cyan] {result['summary_path']}\n"
        f"[cyan]Gap plot:[/cyan] {result['gap_plot']}",
        border_style="green",
        padding=(1, 4),
    ))


def action_run_kd_swin():
    """Run KD: ConvNeXt (Teacher) → Swin Transformer (Student)."""
    print_banner()

    console.print(Rule("[bold cyan]🧠 KD: ConvNext → Swin Transformer[/bold cyan]"))
    console.print()

    if not HAS_KD_SWIN:
        console.print(Panel.fit(
            "[bold red]❌ KD Swin pipeline not available![/bold red]\n"
            "[yellow]Make sure kd_convnext_swin.py exists.[/yellow]",
            border_style="red",
        ))
        return

    if not os.path.exists(TEACHER_CKPT):
        console.print(Panel.fit(
            f"[bold red]❌ Teacher checkpoint not found:[/bold red]\n"
            f"[yellow]{TEACHER_CKPT}[/yellow]\n\n"
            f"[dim]Run option 1 first to create bestKDE.pth[/dim]",
            border_style="red",
        ))
        return

    teacher_name = "?"
    teacher_acc = 0.0
    meta_path = os.path.join(KD_ENSEMBLE_DIR, "selection_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            teacher_name = meta.get("selected_model", "?")
            teacher_acc = float(meta.get("test_accuracy", 0.0))
        except Exception:
            pass

    console.print(Panel.fit(
        "[bold cyan]⚙️  Configuration[/bold cyan]\n\n"
        f"[cyan]Teacher:[/cyan]  [bold white]{teacher_name}[/bold white]"
        f"  [dim](test_acc={teacher_acc:.4f})[/dim]\n"
        f"[cyan]Student:[/cyan]  [bold white]Swin Transformer (Tiny)[/bold white]\n\n"
        f"[cyan]Teacher file:[/cyan]  [dim]{TEACHER_CKPT}[/dim]\n"
        f"[cyan]Student dir:[/cyan]  [dim]{os.path.join(KD_STUDENTS_DIR, 'SwinTransformer')}[/dim]\n\n"
        f"[dim]Temperature = 4.0 · Alpha = 0.65 · Epochs = 80 · Patience = 25[/dim]\n"
        f"[dim]Advanced: CV5 + CI95(2000) + TTA32 → 24-page PDF[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    if not Confirm.ask(
            "\n[bold yellow]🚀 Start KD (ConvNext → Swin) training?[/bold yellow]",
            default=True,
    ):
        return

    music_on = play_music()
    try:
        result = run_kd_convnext_swin()
    finally:
        if music_on:
            stop_music()

    console.print()
    console.print(Panel.fit(
        f"[bold green]✅ KD (ConvNext → Swin) Complete![/bold green]\n\n"
        f"[cyan]Best Val Acc:[/cyan] {result['best_val_acc']:.4f}\n"
        f"[cyan]Test Acc:[/cyan]     [bold green]{result['test_accuracy']:.4f}[/bold green]\n"
        f"[cyan]Test AUC:[/cyan]     {result['test_auc']:.4f}\n"
        f"[cyan]CV5 Mean:[/cyan]     "
        f"{result['advanced_eval'].get('cv5', {}).get('mean_acc', 0):.4f}\n"
        f"[cyan]TTA-32:[/cyan]       "
        f"{result['advanced_eval'].get('tta32', {}).get('acc', 0):.4f}\n\n"
        f"[cyan]Student:[/cyan] {result['student_path']}\n"
        f"[cyan]Report:[/cyan]  {result['pdf_report']}",
        border_style="green",
        padding=(1, 4),
    ))


def action_regenerate_kd_reports():
    """Regenerate KD reports for DenseNet121 / SwinTransformer from existing artifacts."""
    print_header()

    console.print(Rule("[bold cyan]📄 Regenerate KD Reports[/bold cyan]"))
    console.print()

    if not HAS_KD_REPORT_REGEN:
        console.print(Panel.fit(
            "[bold red]❌ KD report regeneration not available![/bold red]\n"
            "[yellow]Make sure regenerate_kd_report.py exists.[/yellow]",
            border_style="red",
        ))
        return

    dense_dir = os.path.join(KD_STUDENTS_DIR, "DenseNet121")
    swin_dir = os.path.join(KD_STUDENTS_DIR, "SwinTransformer")

    console.print(Panel.fit(
        "[bold cyan]📄 Regenerate KD Reports[/bold cyan]\n\n"
        f"[cyan]DenseNet121 dir:[/cyan]  [dim]{dense_dir}[/dim]\n"
        f"[cyan]SwinTransformer dir:[/cyan]  [dim]{swin_dir}[/dim]\n\n"
        f"[dim]Rebuilds 24-page PDF with all plots + XAI (no retraining)[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # Show sub-menu
    sub_table = Table(
        title="[bold cyan]Choose which report to regenerate[/bold cyan]",
        box=box.ROUNDED, border_style="bright_blue",
        header_style="bold magenta", show_lines=True,
    )
    sub_table.add_column("#", style="bold yellow", width=4, justify="center")
    sub_table.add_column("Target", style="bold white", width=25)
    sub_table.add_column("Status", style="dim white", width=40)

    dense_exists = os.path.isdir(dense_dir) and any(
        f.startswith("kd_densenet_best") for f in os.listdir(dense_dir)
    ) if os.path.isdir(dense_dir) else False
    swin_exists = os.path.isdir(swin_dir) and any(
        f.startswith("kd_swin_best") for f in os.listdir(swin_dir)
    ) if os.path.isdir(swin_dir) else False

    sub_table.add_row("1", "DenseNet121 KD",
                      "✅ Ready" if dense_exists else "❌ Artifacts missing")
    sub_table.add_row("2", "SwinTransformer KD",
                      "✅ Ready" if swin_exists else "❌ Artifacts missing")
    sub_table.add_row("3", "BOTH (regenerate both)",
                      "Run both sequentially")
    sub_table.add_row("0", "🔙 Back to main menu", "Cancel")

    console.print(sub_table)

    sub_choice = Prompt.ask(
        "\n[bold cyan]👉 Choose[/bold cyan]",
        choices=["0", "1", "2", "3"],
        default="0",
    )

    if sub_choice == "0":
        return

    try:
        results = []
        if sub_choice in ("1", "3") and dense_exists:
            console.print("\n[bold cyan]▶️  Regenerating DenseNet121 KD report...[/bold cyan]")
            p = regenerate_densenet_report(dense_dir)
            if p:
                results.append(("DenseNet121", p))
                console.print(f"[green]✅ DenseNet121 → {p}[/green]")

        if sub_choice in ("2", "3") and swin_exists:
            console.print("\n[bold cyan]▶️  Regenerating SwinTransformer KD report...[/bold cyan]")
            p = regenerate_swin_report(swin_dir)
            if p:
                results.append(("SwinTransformer", p))
                console.print(f"[green]✅ SwinTransformer → {p}[/green]")

        if results:
            lines = "\n".join([f"[cyan]{name}:[/cyan] {path}"
                               for name, path in results])
            console.print(Panel.fit(
                f"[bold green]✅ Report(s) Regenerated![/bold green]\n\n{lines}",
                border_style="green", padding=(1, 4),
            ))
        else:
            console.print(Panel.fit(
                "[bold red]❌ No report was regenerated.[/bold red]\n"
                "[yellow]Check that artifacts exist in the chosen dirs.[/yellow]",
                border_style="red",
            ))
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Error:[/bold red] {e}",
            border_style="red",
        ))
        import traceback
        traceback.print_exc()


def handle_convnext_vgg16_kd():
    """Handle ConvNeXt→VGG16 KD Ensemble."""
    if not HAS_CONVNEXT_VGG16_KD:
        console.print(Panel.fit(
            "[bold red]❌ ConvNeXt→VGG16 KD not available![/bold red]\n"
            "[yellow]Make sure src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/convnext_vgg16_kd.py exists.[/yellow]",
            border_style="red",
        ))
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔗 ConvNeXt → VGG16 KD Ensemble[/bold cyan]\n"
        "[dim]Teacher: ConvNeXt (bestKDE.pth)[/dim]\n"
        "[dim]Student: VGG16 (vgg16BEST.pth)[/dim]\n"
        "[dim]Goal: Match or beat teacher accuracy, reduce G1 error < 7%[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    if not Confirm.ask("\n[bold yellow]🚀 Start KD pipeline?[/bold yellow]", default=True):
        return

    # 🎵 Start music
    music_on = play_music()

    try:
        result = run_convnext_vgg16_kd()
        if result:
            console.print(Panel.fit(
                f"[bold green]✅ KD Complete![/bold green]\n"
                f"[cyan]📁 Model:[/cyan] {result['model_path']}\n"
                f"[cyan]📄 Report:[/cyan] {result['report_path']}",
                border_style="green",
            ))
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        # 🎵 Stop music even if error
        if music_on:
            stop_music()

def handle_convnext_vit_kd():
    """Handle ConvNeXt→ViT KD Ensemble."""
    if not HAS_CONVNEXT_VIT_KD:
        console.print(Panel.fit(
            "[bold red]❌ ConvNeXt→ViT KD not available![/bold red]\n"
            "[yellow]Make sure src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/convnext_vit_kd.py exists.[/yellow]",
            border_style="red",
        ))
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔗 ConvNeXt → ViT KD Ensemble[/bold cyan]\n"
        "[dim]Teacher: ConvNeXt (bestKDE.pth)[/dim]\n"
        "[dim]Student: ViT (visiontransformerBEST.pth)[/dim]\n"
        "[dim]Goal: Match or beat teacher accuracy[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    if not Confirm.ask("\n[bold yellow]🚀 Start KD pipeline?[/bold yellow]", default=True):
        return

    music_on = play_music()
    try:
        result = run_convnext_vit_kd()
        if result:
            console.print(Panel.fit(
                f"[bold green]✅ KD Complete![/bold green]\n"
                f"[cyan]📁 Model:[/cyan] {result['model_path']}\n"
                f"[cyan]📄 Report:[/cyan] {result['report_path']}",
                border_style="green",
            ))
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        if music_on:
            stop_music()

def handle_advanced_only_vit():
    """Handle ViT KD report regeneration (no retraining, no CV5)."""
    if not HAS_ADVANCED_ONLY_VIT:
        console.print(Panel.fit(
            "[bold red]❌ Advanced-only ViT not available![/bold red]\n"
            "[yellow]Make sure convnext_vit_kd.py has run_advanced_only_vit().[/yellow]",
            border_style="red",
        ))
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]📄 Regenerate ViT KD Report[/bold cyan]\n"
        "[dim]Loads convnext_to_vit_kd_best.pth[/dim]\n"
        "[dim]Runs: eval + CI95 + TTA8 (NO CV5)[/dim]\n"
        "[dim]Generates 24-page PDF[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    if not Confirm.ask("\n[bold yellow]🚀 Run?[/bold yellow]", default=True):
        return

    music_on = play_music()
    try:
        console.print("\n[bold cyan]▶️  Step 1/3: Loading model and running eval...[/bold cyan]")
        result = run_advanced_only_vit()
        if result:
            console.print(Panel.fit(
                f"[bold green]✅ Report Regenerated![/bold green]\n"
                f"[cyan]📄 Report:[/cyan] {result['report_path']}\n"
                f"[cyan]📊 Test Acc:[/cyan] {result['metrics']['accuracy']:.4f}",
                border_style="green",
            ))
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        if music_on:
            stop_music()

def handle_convnext_alexnet_kd():
    """Handle ConvNeXt→AlexNet KD Ensemble."""
    if not HAS_CONVNEXT_ALEXNET_KD:
        console.print(Panel.fit(
            "[bold red]❌ ConvNeXt→AlexNet KD not available![/bold red]\n"
            "[yellow]Make sure convnext_alexnet_kd.py exists.[/yellow]",
            border_style="red",
        ))
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔗 ConvNeXt → AlexNet KD Ensemble[/bold cyan]\n"
        "[dim]Teacher: ConvNeXt (bestKDE.pth)[/dim]\n"
        "[dim]Student: AlexNet (alexnetBEST.pth)[/dim]\n"
        "[dim]Goal: Match or beat teacher accuracy[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    if not Confirm.ask("\n[bold yellow]🚀 Start KD pipeline?[/bold yellow]", default=True):
        return

    music_on = play_music()
    try:
        result = run_convnext_alexnet_kd()
        if result:
            console.print(Panel.fit(
                f"[bold green]✅ KD Complete![/bold green]\n"
                f"[cyan]📁 Model:[/cyan] {result['model_path']}\n"
                f"[cyan]📄 Report:[/cyan] {result['report_path']}",
                border_style="green",
            ))
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        if music_on:
            stop_music()

def handle_advanced_only_alexnet():
    """Handle AlexNet KD report regeneration (no retraining, no CV5)."""
    if not HAS_ADVANCED_ONLY_ALEXNET:
        console.print(Panel.fit(
            "[bold red]❌ Advanced-only AlexNet not available![/bold red]\n"
            "[yellow]Make sure convnext_alexnet_kd.py has run_advanced_only_alexnet().[/yellow]",
            border_style="red",
        ))
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]📄 Regenerate AlexNet KD Report[/bold cyan]\n"
        "[dim]Loads convnext_to_alexnet_kd_best.pth[/dim]\n"
        "[dim]Runs: eval + CI95 + TTA8 (NO CV5)[/dim]\n"
        "[dim]Generates 24-page PDF[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    if not Confirm.ask("\n[bold yellow]🚀 Run?[/bold yellow]", default=True):
        return

    music_on = play_music()
    try:
        console.print("\n[bold cyan]▶️  Step 1/3: Loading model and running eval...[/bold cyan]")
        result = run_advanced_only_alexnet()
        if result:
            console.print(Panel.fit(
                f"[bold green]✅ Report Regenerated![/bold green]\n"
                f"[cyan]📄 Report:[/cyan] {result['report_path']}\n"
                f"[cyan]📊 Test Acc:[/cyan] {result['metrics']['accuracy']:.4f}",
                border_style="green",
            ))
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        if music_on:
            stop_music()

def action_run_kd_cnnlstm():
    print_banner()
    console.print(Rule("[bold magenta]🧠 KD: ConvNeXt (Teacher) → CNNLSTM (Student)[/bold magenta]"))
    console.print()

    if not HAS_KD_CNNLSTM:
        console.print(Panel.fit(
            "[bold red]❌ KD CNNLSTM pipeline not available![/bold red]\n"
            "[yellow]Make sure kd_convnext_cnnlstm.py exists.[/yellow]",
            border_style="red",
        ))
        return

    console.print(Panel.fit(
        "[bold cyan]⚙️  3-Stage Pipeline[/bold cyan]\n\n"
        "[bold]Stage 1:[/bold] KD training (α·T²·KLDiv + CE) — anti-overfit\n"
        "[bold]Stage 2:[/bold] Targeted FT on G1 & G3 (Focal + Mixup/CutMix)\n"
        "[bold]Stage 3:[/bold] Per-class threshold tuning + TTA-32\n\n"
        "[bold]CNNLSTM architecture:[/bold]\n"
        "  • ResNet18 backbone (no maxpool/avgpool/fc)\n"
        "  • LSTM(input=512, hidden=256, 2 layers)\n"
        "  • Classifier: Linear(256→128) → BN → Dropout → Linear(128→4)\n\n"
        f"[cyan]Teacher:[/cyan] ConvNeXt (bestKDE.pth)\n"
        f"[cyan]Student:[/cyan] CNNLSTM (cnnlstmBEST.pth)\n"
        f"[cyan]Output:[/cyan] ...\\bestKDEnsembles\\CNNLSTM\\",
        border_style="magenta", padding=(1, 4),
    ))

    if not Confirm.ask("\n[bold yellow]🚀 Run KD (ConvNeXt → CNNLSTM)?[/bold yellow]", default=True):
        return

    music_on = play_music()
    try:
        result = run_kd_convnext_cnnlstm()
    finally:
        if music_on:
            stop_music()

    if result:
        per_class = result.get("per_class", {})
        lines = []
        for name, m in per_class.items():
            err = m["error_rate"] * 100
            flag = "✅" if err < 7 else "⚠️"
            lines.append(f"  {flag} {name}: error={err:.2f}% | F1={m['f1']:.4f}")

        console.print()
        console.print(Panel.fit(
            "[bold cyan]⚙️  3-Stage Pipeline (ALL CLASSES)[/bold cyan]\n\n"
            "[bold]Stage 1:[/bold] KD training on ALL classes (α·T²·KLDiv + CE)\n"
            "[bold]Stage 2:[/bold] Full fine-tune on ALL classes (Focal + Mixup/CutMix)\n"
            "[bold]Stage 3:[/bold] Per-class threshold tuning + TTA-32\n\n"
            "[bold]Targets:[/bold]\n"
            "  • Test Acc ≥ Teacher (91.32%)\n"
            "  • Per-class error < 7% for ALL 4 classes\n"
            "  • Avg |Gap| ≤ 0.05 → Good Fit\n\n"
            "[bold]CNNLSTM architecture:[/bold]\n"
            "  • ResNet18 backbone\n"
            "  • LSTM(input=512, hidden=256, 2 layers)\n"
            "  • Classifier: Linear(256→128) → BN → Dropout → Linear(128→4)\n\n"
            f"[cyan]Teacher:[/cyan] ConvNeXt (bestKDE.pth)\n"
            f"[cyan]Student:[/cyan] CNNLSTM (cnnlstmBEST.pth)\n"
            f"[cyan]Output:[/cyan] ...\\bestKDEnsembles\\CNNLSTM\\",
            border_style="magenta", padding=(1, 4),
        ))

def run_menu_loop():
    """Interactive menu loop."""
    first_run = True
    while True:
        # ═══════════════════════════════════════════════════════════════
        # ✅ Clear console ONLY at the start (not after every action)
        # ═══════════════════════════════════════════════════════════════
        if first_run:
            print_banner()
            print_header()
            first_run = False
        else:
            # Just clear (no header, no banner — the menu title says everything)
            console.clear()

        console.print()
        show_menu()

        try:
            choice = Prompt.ask(
                "\n[bold cyan]👉 Enter your choice[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
                default="0",
            )
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "0":
            console.print()
            console.print(Panel.fit(
                "[bold yellow]👋 Goodbye! Thanks for using Final KDE.[/bold yellow]\n"
                "[dim italic]See you next time! 🚀[/dim italic]",
                border_style="yellow",
                padding=(1, 4),
            ))
            console.print()
            break

        elif choice == "1":
            try:
                action_find_best_model()
            except Exception as e:
                console.print(Panel.fit(
                    f"[bold red]❌ Error:[/bold red] {e}",
                    border_style="red",
                ))
                import traceback
                traceback.print_exc()

        elif choice == "2":
            try:
                action_run_kd()
            except Exception as e:
                console.print(Panel.fit(
                    f"[bold red]❌ Error:[/bold red] {e}",
                    border_style="red",
                ))
                import traceback
                traceback.print_exc()

        elif choice == "3":
            try:
                action_run_kd_swin()
            except Exception as e:
                console.print(Panel.fit(
                    f"[bold red]❌ Error:[/bold red] {e}",
                    border_style="red",
                ))
                import traceback
                traceback.print_exc()

        elif choice == "4":
            try:
                action_regenerate_kd_reports()
            except Exception as e:
                console.print(Panel.fit(
                    f"[bold red]❌ Error:[/bold red] {e}",
                    border_style="red",
                ))
                import traceback
                traceback.print_exc()

        elif choice == "5":
            try:
                action_regen_swin_report()
            except Exception as e:
                console.print(Panel.fit(f"[bold red]❌ Error:[/bold red] {e}", border_style="red"))
                import traceback
                traceback.print_exc()

        elif choice == "6":
            handle_convnext_vgg16_kd()

        elif choice == "7":
            handle_convnext_vit_kd()

        elif choice == "8":
            handle_advanced_only_vit()

        elif choice == "9":
            handle_convnext_alexnet_kd()

        elif choice == "10":
            handle_advanced_only_alexnet()

        elif choice == "11":
            try:
                action_run_kd_cnnlstm()
            except Exception as e:
                console.print(Panel.fit(f"[bold red]❌ Error:[/bold red] {e}", border_style="red"))
                import traceback;
                traceback.print_exc()

        # ═══════════════════════════════════════════════════════════════
        # ✅ Pause once, then loop (menu will be re-rendered cleanly)
        # ═══════════════════════════════════════════════════════════════
        console.print()
        Prompt.ask("[dim]Press Enter to continue[/dim]", default="")


# =====================================================
# TYPER COMMANDS
# =====================================================

@app.command("menu")
def cmd_menu():
    """Run the interactive menu loop."""
    run_menu_loop()


@app.command("find-best")
def cmd_find_best():
    """Non-interactive: find best model and copy it."""
    action_find_best_model()


# =====================================================
# ENTRY POINT
# =====================================================

def main():
    """Entry point — run the interactive menu by default."""
    run_menu_loop()


if __name__ == "__main__":
    app()