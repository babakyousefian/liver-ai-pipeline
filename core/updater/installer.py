# core/updater/installer.py
# -*- coding: utf-8 -*-
"""
Installer: applies the downloaded update.

Two modes:
  - FROZEN (PyInstaller): write a small updater script + relaunch
  - SOURCE (python): unzip into project dir with backup + rollback

Health-check integration:
  - `install_source()` creates `_update_pending.flag` BEFORE extracting.
  - On next app start, `main()` checks the flag and rolls back if needed.
  - After successful UI load, `confirm_update_success()` removes the flag.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from typing import Callable, Optional

from .health import create_update_flag, get_project_root


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_app_dir() -> str:
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(sys.argv[0]))


# ============================================================
# FROZEN mode (PyInstaller EXE)
# ============================================================

_WIN_UPDATER_BAT = r"""@echo off
REM Liver AI - Auto Updater
REM Usage: updater.bat <new_exe> <old_exe> <pid>
setlocal
set NEW=%~1
set OLD=%~2
set PID=%~3

echo Waiting for app (PID %PID%) to exit...
:wait
tasklist /FI "PID eq %PID%" 2>NUL | find "%PID%" >NUL
if not errorlevel 1 (
    timeout /t 1 /nobreak >NUL
    goto wait
)

echo Backing up old exe...
if exist "%OLD%.bak" del "%OLD%.bak"
move "%OLD%" "%OLD%.bak" >NUL

echo Installing new exe...
move "%NEW%" "%OLD%" >NUL

if errorlevel 1 (
    echo Update failed. Rolling back...
    move "%OLD%.bak" "%OLD%" >NUL
    pause
    exit /b 1
)

echo Update successful. Restarting...
start "" "%OLD%"
exit /b 0
"""


def install_frozen(new_exe_path: str, target_version: str = "") -> bool:
    """
    Replace the running EXE on Windows via a helper .bat.

    Note: FROZEN mode has limited rollback — only if the file move fails.
    If the new exe crashes on startup, there's no automatic rollback.
    For full rollback in FROZEN mode, a separate launcher is needed.
    """
    if not is_frozen():
        print("⚠️ install_frozen called but app is not frozen")
        return False
    if not os.path.isfile(new_exe_path):
        print(f"⚠️ new exe not found: {new_exe_path}")
        return False

    app_dir = get_app_dir()
    current_exe = sys.executable
    pid = os.getpid()

    # ✅ Create the health flag BEFORE launching the updater
    # (the .bat backs up + replaces; if new exe crashes, health flag triggers rollback)
    backup_dir = os.path.join(app_dir, f"_backup_exe_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        os.makedirs(backup_dir, exist_ok=True)
        # Backup the current exe manually (the .bat will also back it up, but we want a copy)
        try:
            shutil.copy2(current_exe, os.path.join(backup_dir, os.path.basename(current_exe)))
        except Exception as e:
            print(f"⚠️ failed to copy current exe to backup: {e}")
    except Exception as e:
        print(f"⚠️ failed to create backup dir: {e}")
        backup_dir = ""

    create_update_flag(target_version=target_version, backup_dir=backup_dir)

    # Write the bat in temp
    bat_path = os.path.join(tempfile.gettempdir(), "liverai_updater.bat")
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(_WIN_UPDATER_BAT)

    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", bat_path,
             new_exe_path, current_exe, str(pid)],
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0,
        )
        return True
    except Exception as e:
        print(f"⚠️ failed to launch updater: {e}")
        return False


# ============================================================
# SOURCE mode (zip of .py files)
# ============================================================

def install_source(
    zip_path: str,
    project_root: str,
    progress_cb: Optional[Callable[[int], None]] = None,
    target_version: str = "",
) -> bool:
    """
    Unzip into project_root with:
      - backup/rollback
      - path traversal protection
      - health-check flag integration

    Returns True on success, False on failure (with automatic rollback).
    """
    if not os.path.isfile(zip_path):
        print(f"⚠️ zip not found: {zip_path}")
        return False
    if not zipfile.is_zipfile(zip_path):
        print(f"⚠️ not a zip: {zip_path}")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(project_root, f"_backup_{timestamp}")

    try:
        os.makedirs(backup_dir, exist_ok=True)

        # ✅ Create health flag ONCE, before extracting anything
        create_update_flag(target_version=target_version, backup_dir=backup_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            # ✅ Filter: reject absolute paths and '..'
            safe_names = []
            for n in names:
                if n.startswith("/") or ".." in n.split("/"):
                    print(f"⚠️ skipping unsafe path: {n}")
                    continue
                safe_names.append(n)

            total = max(1, len(safe_names))
            for i, n in enumerate(safe_names):
                if n.endswith("/"):
                    continue  # directory entry
                target = os.path.join(project_root, n)

                # ✅ Backup existing file BEFORE overwriting
                if os.path.isfile(target):
                    backup_target = os.path.join(backup_dir, n)
                    os.makedirs(os.path.dirname(backup_target), exist_ok=True)
                    try:
                        shutil.copy2(target, backup_target)
                    except Exception as e:
                        print(f"⚠️ backup failed for {n}: {e}")
                        raise

                # ✅ Extract this file
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                with zf.open(n) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                if progress_cb:
                    try:
                        progress_cb(int((i + 1) * 100 / total))
                    except Exception:
                        pass

        return True

    except Exception as e:
        print(f"❌ install_source failed: {e}")
        # ✅ Rollback on any error during extraction
        try:
            _rollback(project_root, backup_dir)
            # Also remove the health flag because we've rolled back
            from .health import clear_update_flag
            clear_update_flag()
        except Exception as re:
            print(f"❌ rollback also failed: {re}")
        return False


def _rollback(project_root: str, backup_dir: str):
    """Restore every file from backup_dir into project_root."""
    if not os.path.isdir(backup_dir):
        return
    for root, _dirs, files in os.walk(backup_dir):
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, backup_dir)
            dst = os.path.join(project_root, rel)
            try:
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                shutil.copy2(src, dst)
                print(f"↩️ rolled back: {rel}")
            except Exception as e:
                print(f"⚠️ rollback failed for {rel}: {e}")