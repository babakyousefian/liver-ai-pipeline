# src/optimization/hardware_detector.py
"""
Hardware Detection for GA Optimization.

Detects CPU, GPU, RAM, Disk, OS and computes derived constraints.
"""

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict

import psutil
import torch


@dataclass
class HardwareProfile:
    """Complete hardware description."""
    # OS / Software
    os_name: str = "Unknown"
    os_version: str = "Unknown"
    python_version: str = "Unknown"
    torch_version: str = "Unknown"
    cuda_version: str = "N/A"
    cudnn_version: str = "N/A"

    # CPU
    cpu_name: str = "Unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_frequency_mhz: float = 0.0

    # RAM
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU
    gpu_available: bool = False
    gpu_name: str = "Unknown"
    gpu_count: int = 0
    gpu_vram_total_gb: float = 0.0
    gpu_vram_available_gb: float = 0.0
    gpu_compute_capability: float = 0.0

    # Disk
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0

    # Derived constraints
    max_batch_size: int = 32
    max_num_workers: int = 4
    force_mixed_precision: bool = True
    max_ram_gb: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "🖥️  HARDWARE PROFILE",
            "=" * 70,
            f"   OS:     {self.os_name} {self.os_version}",
            f"   Python: {self.python_version}",
            f"   PyTorch:{self.torch_version}  CUDA: {self.cuda_version}  cuDNN: {self.cudnn_version}",
            f"   CPU:    {self.cpu_name}",
            f"           Physical: {self.cpu_cores_physical} | Logical: {self.cpu_cores_logical}",
            f"           Freq: {self.cpu_frequency_mhz:.0f} MHz",
            f"   RAM:    {self.ram_total_gb:.2f} GB total | {self.ram_available_gb:.2f} GB free",
            f"   Disk:   {self.disk_total_gb:.2f} GB total | {self.disk_free_gb:.2f} GB free",
        ]
        if self.gpu_available:
            lines.extend([
                f"   GPU:    {self.gpu_name} (x{self.gpu_count})",
                f"           VRAM: {self.gpu_vram_total_gb:.2f} GB | Free: {self.gpu_vram_available_gb:.2f} GB",
                f"           Compute: {self.gpu_compute_capability}",
            ])
        else:
            lines.append("   GPU:    ❌ Not available")
        lines.extend([
            "-" * 70,
            "🎯 DERIVED CONSTRAINTS",
            f"   max_batch_size:           {self.max_batch_size}",
            f"   max_num_workers:          {self.max_num_workers}",
            f"   force_mixed_precision:    {self.force_mixed_precision}",
            f"   max_ram_gb:               {self.max_ram_gb}",
            "=" * 70,
        ])
        return "\n".join(lines)


class HardwareDetector:
    """One-time hardware detection."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.profile = self._detect()
        if verbose:
            print(self.profile.summary())

    def _detect(self) -> HardwareProfile:
        p = HardwareProfile()

        # OS / software
        p.os_name = platform.system()
        p.os_version = platform.release()
        p.python_version = sys.version.split()[0]
        p.torch_version = torch.__version__
        p.cuda_version = torch.version.cuda or "N/A"
        try:
            p.cudnn_version = str(torch.backends.cudnn.version())
        except Exception:
            pass

        # CPU
        p.cpu_name = self._get_cpu_name()
        p.cpu_cores_physical = psutil.cpu_count(logical=False) or 0
        p.cpu_cores_logical = psutil.cpu_count(logical=True) or 0
        try:
            f = psutil.cpu_freq()
            p.cpu_frequency_mhz = f.max if f else 0.0
        except Exception:
            pass

        # RAM
        vm = psutil.virtual_memory()
        p.ram_total_gb = vm.total / (1024 ** 3)
        p.ram_available_gb = vm.available / (1024 ** 3)

        # Disk (project root)
        try:
            du = psutil.disk_usage(os.path.abspath(os.sep))
            p.disk_total_gb = du.total / (1024 ** 3)
            p.disk_free_gb = du.free / (1024 ** 3)
        except Exception:
            pass

        # GPU
        if torch.cuda.is_available():
            p.gpu_available = True
            p.gpu_count = torch.cuda.device_count()
            p.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            p.gpu_vram_total_gb = props.total_memory / (1024 ** 3)
            p.gpu_compute_capability = float(f"{props.major}.{props.minor}")
            try:
                free, total = torch.cuda.mem_get_info(0)
                p.gpu_vram_available_gb = free / (1024 ** 3)
            except Exception:
                p.gpu_vram_available_gb = p.gpu_vram_total_gb

        # Derived constraints
        p.max_batch_size = self._decide_batch(p)
        p.max_num_workers = self._decide_workers(p)
        p.force_mixed_precision = (
            p.gpu_available and p.gpu_compute_capability >= 7.0
        )
        p.max_ram_gb = min(p.ram_available_gb * 0.7, 8.0)

        return p

    def _decide_batch(self, p: HardwareProfile) -> int:
        if not p.gpu_available:
            return 8
        v = p.gpu_vram_total_gb
        if v < 3:
            return 8
        if v < 5:
            return 32
        if v < 8:
            return 64
        if v < 12:
            return 96
        return 128

    def _decide_workers(self, p: HardwareProfile) -> int:
        cores = p.cpu_cores_physical or p.cpu_cores_logical or 4
        return max(2, min(cores - 1, 6))

    def _get_cpu_name(self) -> str:
        try:
            if platform.system() == "Windows":
                out = subprocess.check_output(
                    ["wmic", "cpu", "get", "name"],
                    stderr=subprocess.DEVNULL, timeout=3,
                ).decode()
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                if len(lines) >= 2:
                    return lines[1]
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            elif platform.system() == "Darwin":
                return subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3,
                ).decode().strip()
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"