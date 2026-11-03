# checkReqPac.py
# -*- coding: utf-8 -*-

"""
Requirements Checker with Rich UI + Typer CLI (v4.0)
====================================================
Fixed: no more duplicate rows in Live display.

NEW in v4.0:
- ✅ Sliding window (last N packages only)
- ✅ Live progress bar with spinner
- ✅ No duplicate rendering
- ✅ Shows per-package version
- ✅ N/A if version not detected
- ✅ Counts comments/empty separately
"""

import importlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Typer (CLI)
# ================================================================
try:
    import typer
    from typer import Option
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False

# ================================================================
# ✅ Rich (UI)
# ================================================================
try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn, Progress, SpinnerColumn, TaskProgressColumn,
        TextColumn, TimeElapsedColumn,
    )
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("⚠️ rich not installed. Install: pip install rich")
    sys.exit(1)


# ================================================================
# ✅ Theme
# ================================================================

CUSTOM_THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "header": "bold bright_cyan",
    "version": "dim cyan",
})
console = Console(theme=CUSTOM_THEME)


# ================================================================
# ✅ Categories
# ================================================================

CORE_PACKAGES = {
    "torch", "torchvision", "tensorboard", "numpy", "scipy", "pandas",
    "scikit-learn", "pillow", "opencv-python", "opencv-python-headless",
    "albumentations", "albucore", "kornia", "kornia_rs",
    "matplotlib", "seaborn", "tqdm", "pyyaml", "packaging", "requests",
}
OPTIONAL_PACKAGES = {
    "captum", "lime", "shap", "grad-cam", "reportlab", "pymupdf",
    "optuna", "alembic", "sqlalchemy", "mako", "netron", "onnx",
    "onnx-ir", "onnxscript", "accelerate", "einops", "pygame",
}
MEDICAL_PACKAGES = {
    "medpy", "itk", "itk-core", "itk-io", "itk-filtering",
    "itk-numerics", "itk-registration", "itk-segmentation",
    "pydicom", "nibabel", "simpleitk", "monai", "scikit-image",
}
DEV_PACKAGES = {
    "pytest", "pytest-cov", "pytest-xdist", "black", "isort",
    "autopep8", "pycodestyle", "pre-commit", "mypy-extensions",
    "build", "pyinstaller", "pyinstaller-hooks-contrib",
    "virtualenv", "identify", "nodeenv", "cfgv", "distlib",
    "filelock", "platformdirs", "line-profiler", "memory-profiler",
    "coverage", "iniconfig", "pluggy", "typer",
    "typing-extensions", "typing-inspection",
}


def get_category(pkg: str) -> str:
    p = pkg.lower().replace("-", "_")
    for s, c in [(CORE_PACKAGES, "core"), (MEDICAL_PACKAGES, "medical"),
                 (OPTIONAL_PACKAGES, "optional"), (DEV_PACKAGES, "dev")]:
        for item in s:
            if item.replace("-", "_") == p:
                return c
    return "other"


# ================================================================
# ✅ Data
# ================================================================

@dataclass
class PackageStatus:
    name: str
    required_version: str
    installed_version: Optional[str]
    status: str  # installed, missing, mismatch, skipped, pending
    category: str = "other"
    reason: str = ""


@dataclass
class CheckResults:
    total_lines: int = 0
    total_packages: int = 0
    total_skipped: int = 0
    installed: List[PackageStatus] = field(default_factory=list)
    missing: List[PackageStatus] = field(default_factory=list)
    mismatch: List[PackageStatus] = field(default_factory=list)
    skipped: List[PackageStatus] = field(default_factory=list)
    all_packages: List[PackageStatus] = field(default_factory=list)
    by_category: Dict[str, Dict[str, List[PackageStatus]]] = field(
        default_factory=lambda: {
            "core": {"ok": [], "missing": []},
            "medical": {"ok": [], "missing": []},
            "optional": {"ok": [], "missing": []},
            "dev": {"ok": [], "missing": []},
            "other": {"ok": [], "missing": []},
        }
    )


# ================================================================
# ✅ Parse
# ================================================================

REQ_PATTERN = re.compile(
    r"^\s*([a-zA-Z0-9_\-\.]+)\s*(?:([=<>!~]+)\s*(.+?))?\s*(?:#.*)?$"
)


def parse_requirements_full(file_path: str) -> Tuple[List[PackageStatus], int]:
    path = Path(file_path)
    if not path.exists():
        return [], 0

    lines = []
    for enc in ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.read().splitlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    parsed: List[PackageStatus] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            parsed.append(PackageStatus(
                name="", required_version="", installed_version=None,
                status="skipped", reason="empty",
            ))
            continue
        if stripped.startswith("#"):
            parsed.append(PackageStatus(
                name=stripped, required_version="", installed_version=None,
                status="skipped", reason="comment",
            ))
            continue
        if "git+" in stripped or stripped.startswith("http"):
            parsed.append(PackageStatus(
                name=stripped, required_version="", installed_version=None,
                status="skipped", reason="url",
            ))
            continue
        match = REQ_PATTERN.match(stripped)
        if match:
            name = match.group(1)
            op = match.group(2) or ""
            ver = match.group(3) or ""
            parsed.append(PackageStatus(
                name=name,
                required_version=f"{op}{ver}".strip("=") or "any",
                installed_version=None,
                status="pending",
                category=get_category(name),
            ))
        else:
            parsed.append(PackageStatus(
                name=stripped, required_version="", installed_version=None,
                status="skipped", reason="unknown",
            ))

    return parsed, len(lines)


# ================================================================
# ✅ Import mapping
# ================================================================

IMPORT_NAME_MAP = {
    "opencv-python": "cv2", "opencv-python-headless": "cv2",
    "pillow": "PIL", "scikit-learn": "sklearn", "scikit-image": "skimage",
    "pyyaml": "yaml", "pytorch-ranger": "pytorch_ranger",
    "torch-optimizer": "torch_optimizer",
    "typing-extensions": "typing_extensions",
    "pyinstaller": "PyInstaller",
    "pyinstaller-hooks-contrib": "pyinstaller_hooks_contrib",
    "python-dateutil": "dateutil", "python-bidi": "bidi",
    "python-magic": "magic",
    "itk-core": "itk", "itk-io": "itk", "itk-filtering": "itk",
    "itk-numerics": "itk", "itk-registration": "itk", "itk-segmentation": "itk",
    "simpleitk": "SimpleITK", "medpy": "medpy",
    "numpy-typing-compat": "numpy", "scipy-stubs": "scipy",
    "pandas-stubs": "pandas", "tensorboard-data-server": "tensorboard",
    "googleapis-common-protos": "googleapis_common_protos",
    "opentelemetry-exporter-otlp-proto-common": "opentelemetry",
    "opentelemetry-exporter-otlp-proto-http": "opentelemetry",
    "opentelemetry-semantic-conventions": "opentelemetry",
    "opentelemetry-api": "opentelemetry", "opentelemetry-sdk": "opentelemetry",
    "opentelemetry-proto": "opentelemetry",
    "pydantic-core": "pydantic_core", "typing-inspection": "typing_inspection",
    "annotated-types": "annotated_types", "annotated-doc": "annotated_doc",
    "markdown-it-py": "markdown_it",
    "imageio": "imageio", "pywin32": "win32api",
    "pywin32-ctypes": "win32ctypes", "hf-xet": "hf_xet",
    "ml_dtypes": "ml_dtypes", "beautifulsoup4": "bs4",
    "sentry-sdk": "sentry_sdk", "huggingface-hub": "huggingface_hub",
    "easyocr": "easyocr", "grad-cam": "pytorch_grad_cam",
    "pytorch-grad-cam": "pytorch_grad_cam", "lr-scheduler": "LR_scheduler",
    "protobuf": "google.protobuf", "pymupdf": "fitz",
    "line-profiler": "line_profiler", "memory-profiler": "memory_profiler",
    "wmi": "wmi", "grpcio": "grpc", "pytest-cov": "pytest_cov",
    "pytest-xdist": "xdist", "pre-commit": "pre_commit", "absl-py": "absl",
}


def get_import_name(pkg: str) -> str:
    key = pkg.lower().replace("-", "_")
    for k, v in IMPORT_NAME_MAP.items():
        if k.replace("-", "_") == key:
            return v
    return key


def get_installed_version(pkg: str) -> Optional[str]:
    try:
        from importlib.metadata import version as get_version
        v = get_version(pkg)
        if v:
            return str(v)
    except Exception:
        pass
    try:
        import_name = get_import_name(pkg)
        mod = importlib.import_module(import_name)
        v = getattr(mod, "__version__", None)
        if v is not None:
            return str(v)
        return "unknown"
    except Exception:
        return None


def compare_versions(installed: str, required: str, operator: str) -> bool:
    try:
        from packaging.version import Version
        inst_v, req_v = Version(installed), Version(required)
        if operator in ("==", "==="):
            return inst_v == req_v
        if operator == ">=":
            return inst_v >= req_v
        if operator == "<=":
            return inst_v <= req_v
        if operator == ">":
            return inst_v > req_v
        if operator == "<":
            return inst_v < req_v
        if operator == "!=":
            return inst_v != req_v
        return True
    except Exception:
        return True


def check_package_live(pkg_status: PackageStatus) -> PackageStatus:
    if pkg_status.status == "skipped":
        return pkg_status

    installed = get_installed_version(pkg_status.name)

    if installed is None:
        pkg_status.status = "missing"
        pkg_status.installed_version = None
        return pkg_status

    pkg_status.installed_version = installed

    if pkg_status.required_version and pkg_status.required_version != "any":
        op_match = re.match(r"^([=<>!~]+)(.+)$", pkg_status.required_version)
        if op_match:
            op, req = op_match.group(1), op_match.group(2)
            if not compare_versions(installed, req, op):
                pkg_status.status = "mismatch"
                return pkg_status

    pkg_status.status = "installed"
    return pkg_status


# ================================================================
# ✅ SLIDING WINDOW UI (🔥 THE FIX)
# ================================================================

def build_window_table(
    recent_pkgs: deque,
    idx: int,
    total: int,
    status_counts: Dict[str, int],
    window_size: int = 12,
) -> Table:
    """
    Build a table that shows ONLY the last N packages (sliding window).
    This prevents Rich from duplicating rows.
    """
    table = Table(
        title=(
            f"[bold bright_cyan]🔍 Checking packages[/bold bright_cyan] "
            f"[dim]({idx}/{total})[/dim]"
        ),
        box=box.ROUNDED,
        border_style="bright_cyan",
        header_style="bold bright_white",
        show_lines=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=5, justify="right")
    table.add_column("Status", width=3, justify="center")
    table.add_column("Package", style="bold bright_white", width=28)
    table.add_column("Required", style="yellow", width=14)
    table.add_column("Installed", style="cyan", width=14)
    table.add_column("Category", style="magenta", width=11)
    table.add_column("Note", style="dim white", width=15)

    for pkg in list(recent_pkgs)[-window_size:]:
        if pkg.status == "installed":
            icon = "✅"
            req = pkg.required_version or "any"
            inst = pkg.installed_version or "N/A"
            note = ""
            cat = pkg.category
        elif pkg.status == "missing":
            icon = "❌"
            req = pkg.required_version or "any"
            inst = "N/A"
            note = "not installed"
            cat = pkg.category
        elif pkg.status == "mismatch":
            icon = "⚠️"
            req = pkg.required_version
            inst = pkg.installed_version or "N/A"
            note = "version mismatch"
            cat = pkg.category
        elif pkg.status == "skipped":
            icon = "⏭️"
            req = "-"
            inst = "-"
            note = pkg.reason
            cat = "-"
        else:
            icon = "⏳"
            req = pkg.required_version or "any"
            inst = "-"
            note = "checking..."
            cat = pkg.category

        num = getattr(pkg, "_num", "?")
        table.add_row(str(num), icon, pkg.name[:28], req[:14],
                      inst[:14], cat[:11], note)

    # Footer with counts
    footer = (
        f"[green]✅ {status_counts.get('installed', 0)}[/green]  "
        f"[red]❌ {status_counts.get('missing', 0)}[/red]  "
        f"[yellow]⚠️ {status_counts.get('mismatch', 0)}[/yellow]  "
        f"[dim]⏭️ {status_counts.get('skipped', 0)}[/dim]  "
        f"[dim]⏳ {status_counts.get('pending', 0)}[/dim]"
    )
    table.caption = footer
    table.caption_justify = "center"

    return table


def render_live_check(all_packages: List[PackageStatus], speed: float = 0.7):
    """
    ✅ NEW: Print EACH package one-by-one with delay.
    No table, no duplication. Just clean line-by-line output.

    Args:
        all_packages: List of all PackageStatus
        speed: Delay per package (seconds). Default 0.7s.
    """
    results = CheckResults()
    results.total_lines = len(all_packages)

    real_pkgs = [p for p in all_packages if p.status != "skipped"]
    skipped_pkgs = [p for p in all_packages if p.status == "skipped"]
    results.total_packages = len(real_pkgs)
    results.total_skipped = len(skipped_pkgs)

    # Counters
    status_counts = {
        "installed": 0, "missing": 0, "mismatch": 0,
        "skipped": len(skipped_pkgs), "pending": len(real_pkgs),
    }

    # Header line
    console.print()
    console.print(Rule(
        f"[bold bright_cyan]🔍 Checking {len(real_pkgs)} packages "
        f"(+ {len(skipped_pkgs)} skipped lines)[/bold bright_cyan]"
    ))
    console.print(
        f"[dim]Format: [white]#num[/white]  [green]✅/❌/⚠️/⏭️[/green]  "
        f"[white]name[/white]  [yellow]required[/yellow]  "
        f"[cyan]installed[/cyan]  [magenta]category[/magenta][/dim]"
    )
    console.print()

    # Print each package ONE BY ONE with delay
    real_idx = 0
    skipped_idx = 0

    for i, pkg in enumerate(all_packages):
        # Handle skipped lines
        if pkg.status == "skipped":
            skipped_idx += 1
            # Show skipped with dim style
            if pkg.reason == "comment":
                preview = pkg.name[:60] if pkg.name else ""
                console.print(
                    f"[dim]{'⏭️':<3} [white]skipped[/white]  "
                    f"[dim]comment  {preview}[/dim][/dim]"
                )
            elif pkg.reason == "empty":
                console.print(f"[dim]⏭️  [white]skipped[/white]  [dim]empty line[/dim][/dim]")
            else:
                console.print(f"[dim]⏭️  [white]skipped[/white]  [dim]{pkg.reason}[/dim][/dim]")
            time.sleep(speed * 0.3)  # faster for skipped
            continue

        # Check real package
        real_idx += 1
        pkg = check_package_live(pkg)
        status_counts[pkg.status] = status_counts.get(pkg.status, 0) + 1
        status_counts["pending"] = max(0, status_counts["pending"] - 1)

        # Determine icon and styles
        if pkg.status == "installed":
            icon = "✅"
            name_style = "bold bright_green"
            req_str = pkg.required_version or "any"
            inst_str = pkg.installed_version or "N/A"
            req_color = "yellow"
            inst_color = "bright_green"
        elif pkg.status == "missing":
            icon = "❌"
            name_style = "bold bright_red"
            req_str = pkg.required_version or "any"
            inst_str = "N/A"
            req_color = "yellow"
            inst_color = "bright_red"
        elif pkg.status == "mismatch":
            icon = "⚠️"
            name_style = "bold bright_yellow"
            req_str = pkg.required_version
            inst_str = pkg.installed_version or "N/A"
            req_color = "yellow"
            inst_color = "bright_yellow"
        else:
            icon = "⏳"
            name_style = "white"
            req_str = pkg.required_version or "any"
            inst_str = "-"
            req_color = "dim"
            inst_color = "dim"

        # Category color
        cat_colors = {
            "core": "bright_red",
            "medical": "bright_magenta",
            "optional": "bright_blue",
            "dev": "bright_white",
            "other": "dim white",
        }
        cat_color = cat_colors.get(pkg.category, "white")

        # Build single line
        # Format: #num  icon  name (padding)  required  installed  category
        line = (
            f"[dim]{real_idx:>3}[/dim]  "
            f"{icon}  "
            f"[{name_style}]{pkg.name:<30}[/{name_style}]  "
            f"[{req_color}]{req_str:>14}[/{req_color}]  "
            f"[{inst_color}]{inst_str:>14}[/{inst_color}]  "
            f"[{cat_color}]{pkg.category:<11}[/{cat_color}]"
        )
        console.print(line)

        # Add warning note if mismatch/missing
        if pkg.status == "missing":
            console.print(
                f"           [dim red]└─ not installed → "
                f"[cyan]pip install {pkg.name}[/cyan][/dim red]"
            )
        elif pkg.status == "mismatch":
            console.print(
                f"           [dim yellow]└─ version mismatch: "
                f"installed {pkg.installed_version}, required {pkg.required_version}[/dim yellow]"
            )

        # Delay
        time.sleep(speed)

    # Finalize
    for pkg in all_packages:
        if pkg.status == "installed":
            results.installed.append(pkg)
            results.by_category[pkg.category]["ok"].append(pkg)
        elif pkg.status == "missing":
            results.missing.append(pkg)
            results.by_category[pkg.category]["missing"].append(pkg)
        elif pkg.status == "mismatch":
            results.mismatch.append(pkg)
            results.by_category[pkg.category]["ok"].append(pkg)
        elif pkg.status == "skipped":
            results.skipped.append(pkg)

    results.all_packages = all_packages
    return results

# ================================================================
# ✅ Rendering helpers (for summary after check)
# ================================================================

def _make_bar(current: int, total: int, color: str) -> str:
    if total == 0:
        return ""
    pct = current / total
    width = 22
    filled = int(pct * width)
    empty = width - filled
    color_code = {"green": "green", "red": "red",
                  "yellow": "yellow", "cyan": "cyan"}.get(color, "white")
    return f"[{color_code}]{'█' * filled}[/{color_code}][dim]{'░' * empty}[/dim]"


def render_summary(results: CheckResults):
    """Render summary panel after all checks."""
    console.print()
    n_ok = len(results.installed) + len(results.mismatch)
    n_miss = len(results.missing)
    n_skip = results.total_skipped
    total = results.total_packages

    if n_miss == 0:
        title = "🎉  ALL PACKAGES INSTALLED  🎉"
        border = "bright_green"
        style = "bold bright_green"
    elif n_miss <= 5:
        title = f"✅  MOSTLY OK ({n_miss} missing)  ✅"
        border = "green"
        style = "bold green"
    elif n_miss <= 20:
        title = f"⚠️  {n_miss} PACKAGES MISSING  ⚠️"
        border = "yellow"
        style = "bold yellow"
    else:
        title = f"🚨  MANY PACKAGES MISSING ({n_miss})  🚨"
        border = "red"
        style = "bold red"

    stats = Table(box=box.ROUNDED, border_style=border,
                  show_header=False, padding=(0, 2))
    stats.add_column("Metric", style="bold cyan", width=24)
    stats.add_column("Value", style="bold white", width=10, justify="right")
    stats.add_column("Bar", width=24)

    stats.add_row("✅ Installed",
                  f"[bright_green]{n_ok}[/bright_green]",
                  _make_bar(n_ok, total, "green"))
    stats.add_row("❌ Missing",
                  f"[bright_red]{n_miss}[/bright_red]",
                  _make_bar(n_miss, total, "red"))
    if results.mismatch:
        stats.add_row("⚠️ Mismatch",
                      f"[bright_yellow]{len(results.mismatch)}[/bright_yellow]",
                      "")
    if n_skip:
        stats.add_row("⏭️  Skipped (comments/empty)",
                      f"[dim]{n_skip}[/dim]", "")
    stats.add_row("📦 Packages", f"[cyan]{total}[/cyan]", "")
    stats.add_row("📄 Total lines", f"[dim]{results.total_lines}[/dim]", "")

    console.print(Panel(
        Group(Align.center(Text(title, style=style)), Text(""), stats),
        title="[bold bright_white]📊 SUMMARY[/bold bright_white]",
        border_style=border, padding=(1, 2),
    ))


def render_category_breakdown(results: CheckResults):
    """Render per-category breakdown."""
    console.print()
    console.print(Rule("[bold bright_cyan]📦 Categories[/bold bright_cyan]"))

    icons = {"core": "🔥", "medical": "🏥", "optional": "🎨",
             "dev": "🛠️", "other": "📦"}
    names = {"core": "Core (training)", "medical": "Medical imaging",
             "optional": "Optional (interpretability)",
             "dev": "Development", "other": "Other"}

    for cat in ["core", "medical", "optional", "dev", "other"]:
        data = results.by_category[cat]
        ok = len(data["ok"])
        miss = len(data["missing"])
        tot = ok + miss
        if tot == 0:
            continue
        if miss == 0:
            border = "green"
        elif ok == 0:
            border = "red"
        else:
            border = "yellow"
        console.print(Panel(
            f"[bold]{icons[cat]} {names[cat]}[/bold]\n\n"
            f"  [green]✅ {ok}/{tot}[/green]  "
            f"[red]❌ {miss}/{tot}[/red]",
            border_style=border, padding=(0, 1),
        ))


def render_missing(results: CheckResults):
    """Render detailed list of missing packages."""
    if not results.missing:
        return
    console.print()
    console.print(Rule(f"[bold red]❌ Missing Packages ({len(results.missing)})[/bold red]"))

    table = Table(box=box.ROUNDED, border_style="red",
                  header_style="bold bright_red")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Package", style="bold bright_white", width=30)
    table.add_column("Required", style="yellow", width=15)
    table.add_column("Category", style="cyan", width=15)
    table.add_column("Install Command", style="dim white", width=40)

    for i, pkg in enumerate(results.missing, 1):
        table.add_row(
            str(i), pkg.name, pkg.required_version or "any",
            pkg.category, f"pip install {pkg.name}"
        )
    console.print(table)

    # One-line install
    if results.missing:
        all_pkgs = " ".join(p.name for p in results.missing)
        console.print()
        console.print(Panel(
            f"[bold bright_white]💡 Install all at once:[/bold bright_white]\n\n"
            f"[cyan]pip install {all_pkgs}[/cyan]",
            border_style="cyan", padding=(1, 2),
        ))


def render_mismatch(results: CheckResults):
    """Render version mismatches."""
    if not results.mismatch:
        return
    console.print()
    console.print(Rule(f"[bold yellow]⚠️ Version Mismatches ({len(results.mismatch)})[/bold yellow]"))
    table = Table(box=box.ROUNDED, border_style="yellow",
                  header_style="bold bright_yellow")
    table.add_column("Package", style="bold bright_white", width=25)
    table.add_column("Installed", style="cyan", width=15)
    table.add_column("Required", style="yellow", width=15)
    table.add_column("Fix Command", style="dim white", width=40)
    for pkg in results.mismatch:
        table.add_row(
            pkg.name, pkg.installed_version or "N/A",
            pkg.required_version,
            f"pip install --upgrade {pkg.name}=={pkg.required_version}"
        )
    console.print(table)


def render_header():
    """Render header banner."""
    console.print()
    logo = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🔍  REQUIREMENTS CHECKER  🔍                               ║
    ║                                                              ║
    ║   Live, per-package verification with Rich UI                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Align.center(Text(logo, style="bold bright_cyan")))
    console.print(Align.center(Text(
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim"
    )))
    console.print()


# ================================================================
# ✅ Install missing (with prompt)
# ================================================================

def install_missing_packages(results: CheckResults):
    """Install missing packages with progress."""
    if not results.missing:
        console.print(Panel("[bold green]✅ Nothing to install![/bold green]",
                            border_style="green"))
        return

    console.print()
    console.print(Panel(
        f"[bold bright_yellow]📦 Installing {len(results.missing)} packages[/bold bright_yellow]\n\n"
        f"[dim]Press Ctrl+C to cancel...[/dim]",
        border_style="yellow",
    ))

    try:
        for i in range(3, 0, -1):
            console.print(Align.center(
                Text(f"🚀 Starting in {i}...", style="bold bright_yellow")))
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("[yellow]⏸️ Cancelled[/yellow]")
        return

    ok = 0
    fail = 0

    for pkg in results.missing:
        console.print(f"\n[cyan]📦 Installing [bold]{pkg.name}[/bold]...[/cyan]")
        try:
            cmd = [sys.executable, "-m", "pip", "install", pkg.name]
            if pkg.required_version and pkg.required_version != "any":
                if pkg.required_version.startswith("=="):
                    ver = pkg.required_version[2:]
                    cmd = [sys.executable, "-m", "pip", "install",
                           f"{pkg.name}=={ver}"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                console.print(f"[green]✅ {pkg.name} installed[/green]")
                ok += 1
            else:
                console.print(f"[red]❌ {pkg.name} failed[/red]")
                fail += 1
        except Exception as e:
            console.print(f"[red]❌ {pkg.name} error: {e}[/red]")
            fail += 1

    console.print()
    if fail == 0:
        console.print(Panel(
            f"[bold green]🎉 All {ok} packages installed![/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠️ {ok} installed, {fail} failed[/bold yellow]",
            border_style="yellow",
        ))


def _ask_install(results: CheckResults) -> bool:
    """Ask user if they want to auto-install missing packages."""
    if not results.missing:
        return False
    console.print()
    console.print(Panel(
        f"[bold yellow]❓ {len(results.missing)} packages missing.[/bold yellow]\n\n"
        f"[dim]Auto-install them now?[/dim]",
        border_style="yellow",
    ))
    try:
        ans = console.input(
            "\n[bold cyan]👉 Install missing packages? [y/n] (n): [/bold cyan]"
        ).strip().lower()
        return ans in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


def save_json(results: CheckResults, path: str):
    """Save JSON report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_lines": results.total_lines,
            "total_packages": results.total_packages,
            "installed": len(results.installed),
            "missing": len(results.missing),
            "mismatch": len(results.mismatch),
            "skipped": results.total_skipped,
        },
        "packages": [
            {
                "name": p.name,
                "required": p.required_version,
                "installed": p.installed_version or "N/A",
                "status": p.status,
                "category": p.category,
            }
            for p in results.all_packages
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    console.print(Panel(
        f"[bold green]✅ JSON report saved:[/bold green]\n\n[cyan]{path}[/cyan]",
        border_style="green",
    ))


# ================================================================
# ✅ MAIN — run_check_requirements (called from main.py)
# ================================================================

def run_check_requirements(
    req_path: str = None,
    verbose: bool = True,
    save_script: bool = True,   # kept for backward compat
    auto_install_prompt: bool = True,
    save_report: bool = False,
    report_path: str = "requirements_check_report.json",
    speed: float = 0.7,
) -> Dict:
    """
    Main entry — called from main.py.

    Steps:
        1. Parse requirements file
        2. Print EACH package one-by-one (0.7s delay)
        3. Show summary
        4. Ask user to auto-install missing (optional)
        5. Save JSON report (optional)

    Args:
        req_path: Path to requirements file
        verbose: Print detailed output
        save_script: Legacy arg (ignored)
        auto_install_prompt: If True, ask user to install missing
        save_report: If True, save JSON report
        report_path: Path for JSON report
        speed: Delay per package (seconds). Default 0.7.

    Returns:
        Dict with installed, missing, warnings, total.
    """
    # ============================================================
    # 1. Resolve file path
    # ============================================================
    if req_path is None:
        candidates = [
            "requirements_gpu_RTX_2050.txt",
            "requirements.txt",
            os.path.join("../../..", "requirements_gpu_RTX_2050.txt"),
            os.path.join(os.path.dirname(__file__), "../../requirements_gpu_RTX_2050.txt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                req_path = c
                break
        if req_path is None:
            req_path = "../../requirements_gpu_RTX_2050.txt"

    # ============================================================
    # 2. Header
    # ============================================================
    render_header()

    if not os.path.exists(req_path):
        console.print(Panel(
            f"[bold red]❌ File not found: {req_path}[/bold red]",
            border_style="red",
        ))
        return {"installed": [], "missing": [], "warnings": [], "total": 0}

    file_size = os.path.getsize(req_path) / 1024
    console.print(Panel(
        f"[bold cyan]📄 File:[/bold cyan] [white]{req_path}[/white]\n"
        f"[bold cyan]📦 Size:[/bold cyan] [white]{file_size:.1f} KB[/white]",
        border_style="cyan",
    ))

    # ============================================================
    # 3. Parse
    # ============================================================
    console.print()
    with console.status("[bold cyan]🔍 Parsing requirements...", spinner="dots"):
        all_packages, total_lines = parse_requirements_full(req_path)

    real_pkgs = [p for p in all_packages if p.status != "skipped"]
    skipped_pkgs = [p for p in all_packages if p.status == "skipped"]

    if not real_pkgs:
        console.print(Panel(
            "[bold red]❌ No packages found in file![/bold red]",
            border_style="red",
        ))
        return {"installed": [], "missing": [], "warnings": [], "total": 0}

    console.print(
        f"[green]✅ Parsed {len(real_pkgs)} packages[/green] "
        f"[dim](+ {len(skipped_pkgs)} skipped lines, {total_lines} total)[/dim]"
    )

    # ============================================================
    # 4. Live per-package check (0.7s per package)
    # ============================================================
    results = render_live_check(all_packages, speed=speed)

    # ============================================================
    # 5. Summary
    # ============================================================
    render_summary(results)
    render_category_breakdown(results)
    render_missing(results)
    render_mismatch(results)

    # ============================================================
    # 6. JSON report
    # ============================================================
    if save_report:
        save_json(results, report_path)

    # ============================================================
    # 7. Interactive install prompt
    # ============================================================
    if auto_install_prompt and results.missing:
        if _ask_install(results):
            install_missing_packages(results)
        else:
            all_pkgs = " ".join(p.name for p in results.missing)
            console.print()
            console.print(Panel(
                f"[bold bright_white]💡 Install manually:[/bold bright_white]\n\n"
                f"[cyan]pip install {all_pkgs}[/cyan]\n\n"
                f"[dim]Or:[/dim]\n"
                f"[cyan]python -m pip install -r {req_path}[/cyan]",
                border_style="cyan", padding=(1, 2),
            ))

    # ============================================================
    # 8. Final footer
    # ============================================================
    console.print()
    if results.missing:
        console.print(Panel(
            Align.center(Text(
                f"⚠️ {len(results.missing)} packages missing. Fix and re-run.",
                style="bold bright_yellow")),
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            Align.center(Text(
                "🎉 All packages installed! Ready to run SSL pipeline.",
                style="bold bright_green")),
            border_style="green",
        ))

    # ============================================================
    # 9. Return (backward-compatible dict)
    # ============================================================
    return {
        "installed": [(p.name, p.installed_version or "N/A") for p in results.installed],
        "missing": [(p.name, p.name, "not installed") for p in results.missing],
        "warnings": [],
        "total": results.total_packages,
    }


# ================================================================
# ✅ Typer CLI (optional)
# ================================================================

if HAS_TYPER:
    app = typer.Typer(
        help="🔍 Beautiful requirements checker",
        add_completion=False,
        no_args_is_help=True,
    )

    @app.command()
    def run(
        file: str = Option(
            "requirements_gpu_RTX_2050.txt", "--file", "-f",
            help="Path to requirements file",
        ),
        install: bool = Option(
            False, "--install", "-i",
            help="Auto-install missing packages (no prompt)",
        ),
        report: bool = Option(
            False, "--report", "-r",
            help="Save JSON report",
        ),
        report_path: str = Option(
            "requirements_check_report.json", "--report-path",
            help="Path for JSON report",
        ),
        no_prompt: bool = Option(
            False, "--no-prompt",
            help="Skip interactive install prompt",
        ),
        speed: float = Option(
            0.7, "--speed",
            help="Delay per package (seconds)",
        ),
    ):
        """Check all packages one-by-one with delay."""
        run_check_requirements(
            req_path=file,
            verbose=True,
            save_script=False,
            auto_install_prompt=not (install or no_prompt),
            save_report=report,
            report_path=report_path,
            speed=speed,
        )

        if install:
            all_packages, _ = parse_requirements_full(file)
            res = render_live_check(all_packages, speed=0.01)
            install_missing_packages(res)

    if __name__ == "__main__":
        app()
else:
    if __name__ == "__main__":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("-f", "--file", default="requirements_gpu_RTX_2050.txt")
        p.add_argument("-i", "--install", action="store_true")
        p.add_argument("-r", "--report", action="store_true")
        p.add_argument("--report-path", default="requirements_check_report.json")
        p.add_argument("--no-prompt", action="store_true")
        p.add_argument("--speed", type=float, default=0.7)
        a = p.parse_args()

        run_check_requirements(
            req_path=a.file,
            save_script=False,
            auto_install_prompt=not (a.install or a.no_prompt),
            save_report=a.report,
            report_path=a.report_path,
            speed=a.speed,
        )

