# core/updater/health.py
# -*- coding: utf-8 -*-
"""
Health-check + auto-rollback after update install.

Flow:
1. Before install: create `_update_pending.flag` + backup dir
2. After install: app restarts
3. In main():
   - If `_update_pending.flag` exists → start 90s timer
   - If UI loads OK before timer → delete flag (install confirmed)
   - If timer fires and flag still exists → auto-rollback
"""

import os
import sys
import shutil
from datetime import datetime


FLAG_FILENAME = "_update_pending.flag"
HEALTH_FILENAME = "_update_health.flag"


def get_project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def create_update_flag(target_version: str = "", backup_dir: str = "") -> str:
    """
    Called by installer BEFORE applying update.
    Creates a flag file with metadata.
    """
    root = get_project_root()
    flag_path = os.path.join(root, FLAG_FILENAME)
    try:
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write(f"target_version={target_version}\n")
            f.write(f"backup_dir={backup_dir}\n")
            f.write(f"created_at={datetime.now().isoformat()}\n")
        print(f"✅ Update flag created: {flag_path}")
        return flag_path
    except Exception as e:
        print(f"⚠️ Failed to create update flag: {e}")
        return ""


def clear_update_flag() -> bool:
    """Remove the pending-update flag (after successful install or rollback)."""
    root = get_project_root()
    flag_path = os.path.join(root, FLAG_FILENAME)
    try:
        if os.path.isfile(flag_path):
            os.remove(flag_path)
            print("✅ Update flag cleared")
            return True
    except Exception as e:
        print(f"⚠️ Failed to clear update flag: {e}")
    return False


def check_update_pending() -> dict:
    """
    Called on app startup.
    Returns dict if update is pending, empty dict otherwise.
    """
    root = get_project_root()
    flag_path = os.path.join(root, FLAG_FILENAME)
    if not os.path.isfile(flag_path):
        return {}

    try:
        data = {}
        with open(flag_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    data[k.strip()] = v.strip()
        return data
    except Exception as e:
        print(f"⚠️ Failed to read update flag: {e}")
        return {}


def confirm_update_success():
    """
    Called after UI is fully loaded.
    Deletes the pending flag → update is confirmed.
    """
    root = get_project_root()
    flag_path = os.path.join(root, FLAG_FILENAME)
    health_path = os.path.join(root, HEALTH_FILENAME)

    try:
        if os.path.isfile(flag_path):
            os.remove(flag_path)
            print("✅ Update confirmed, flag removed")
        with open(health_path, "w", encoding="utf-8") as f:
            f.write(f"ok_at={datetime.now().isoformat()}\n")
    except Exception as e:
        print(f"⚠️ Failed to confirm update: {e}")


def rollback_if_needed(pending_data: dict) -> bool:
    """
    Called from main() BEFORE UI is shown, if update was pending.
    Rolls back to the backup if the app crashes or doesn't confirm in time.

    Returns True if rollback was performed, False otherwise.
    """
    if not pending_data:
        return False

    backup_dir = pending_data.get("backup_dir", "")
    if not backup_dir or not os.path.isdir(backup_dir):
        print(f"⚠️ Rollback: backup dir not found: {backup_dir}")
        # Still clear the flag so we don't loop
        clear_update_flag()
        return False

    root = get_project_root()
    restored = 0
    failed = 0

    try:
        for dirpath, _dirnames, filenames in os.walk(backup_dir):
            for fname in filenames:
                src = os.path.join(dirpath, fname)
                rel = os.path.relpath(src, backup_dir)
                dst = os.path.join(root, rel)
                try:
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    shutil.copy2(src, dst)
                    restored += 1
                except Exception as e:
                    print(f"⚠️ Rollback failed for {rel}: {e}")
                    failed += 1

        # ✅ Clear the flag after rollback
        clear_update_flag()

        print(f"↩️ Rollback complete: {restored} restored, {failed} failed")
        return restored > 0
    except Exception as e:
        print(f"❌ Rollback failed: {e}")
        return False

