# src/utils/memory_monitor.py

import gc
import os

import psutil
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def get_memory_usage() -> dict:
    """Get detailed memory usage."""
    memory_info = {}

    # CPU Memory
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_info["cpu_rss"] = mem_info.rss / (1024**3)  # GB
    memory_info["cpu_vms"] = mem_info.vms / (1024**3)  # GB
    memory_info["cpu_percent"] = process.memory_percent()

    # System Memory
    system_mem = psutil.virtual_memory()
    memory_info["system_total"] = system_mem.total / (1024**3)
    memory_info["system_available"] = system_mem.available / (1024**3)
    memory_info["system_used"] = system_mem.used / (1024**3)
    memory_info["system_percent"] = system_mem.percent

    # GPU Memory (if available)
    if torch.cuda.is_available():
        memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / (1024**3)
        memory_info["gpu_reserved"] = torch.cuda.memory_reserved() / (1024**3)
        memory_info["gpu_max_allocated"] = torch.cuda.max_memory_allocated() / (1024**3)

    return memory_info


def print_memory_status(title: str = "Memory Status"):
    """Print beautiful memory status table."""

    mem = get_memory_usage()

    table = Table(
        title=title,
        box=box.ROUNDED,
        border_style="bright_cyan",
        title_style="bold yellow",
    )

    table.add_column("Component", style="bold white", width=20)
    table.add_column("Usage", style="white", width=25)
    table.add_column("Percent", style="white", width=12, justify="center")

    # CPU
    table.add_row(
        "💻 CPU Memory (RSS)",
        f"{mem.get('cpu_rss', 0):.3f} GB",
        f"{mem.get('cpu_percent', 0):.1f}%",
    )

    # System
    table.add_row(
        "🖥️ System Memory",
        f"{mem.get('system_used', 0):.3f} / {mem.get('system_total', 0):.3f} GB",
        f"{mem.get('system_percent', 0):.1f}%",
    )

    # GPU
    if "gpu_allocated" in mem:
        table.add_row("🖥️ GPU Allocated", f"{mem.get('gpu_allocated', 0):.3f} GB", "-")
        table.add_row("🖥️ GPU Reserved", f"{mem.get('gpu_reserved', 0):.3f} GB", "-")

    console.print(table)

    return mem


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
