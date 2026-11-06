# test_download.py
# -*- coding: utf-8 -*-
"""
Standalone test for UpdateDownloader.
Run: python test_download.py
"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

app = QApplication(sys.argv)

# ✅ Test 1: Check if URL is considered safe
from core.updater.source import _is_safe_url
URL = "https://github.com/babakyousefian/liver-ai-pipeline/releases/download/v3.2.0/update_v3.2.0.zip"

print("=" * 60)
print("TEST 1: _is_safe_url")
print("=" * 60)
print(f"URL: {URL}")
safe = _is_safe_url(URL)
print(f"Result: {'✅ SAFE' if safe else '❌ UNSAFE (this is the bug!)'}")

if not safe:
    print("\n⚠️ _is_safe_url rejected the URL. Fix _ALLOWED_DOMAINS in source.py")
    sys.exit(1)

# ✅ Test 2: Try downloading
print("\n" + "=" * 60)
print("TEST 2: UpdateDownloader")
print("=" * 60)

from core.updater.downloader import UpdateDownloader

downloader = UpdateDownloader(
    url=URL,
    expected_size=0,
    dest_dir=os.path.expanduser("~\\Desktop"),
)

def on_progress(percent, downloaded, total):
    print(f"  Progress: {percent}%  ({downloaded}/{total} bytes)")

def on_done(path):
    print(f"\n✅ DOWNLOAD OK: {path}")
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   File size: {size} bytes")
    app.quit()

def on_failed(reason):
    print(f"\n❌ DOWNLOAD FAILED: {reason}")
    app.quit()

downloader.progress.connect(on_progress)
downloader.finished_ok.connect(on_done)
downloader.failed.connect(on_failed)
downloader.start()

# Safety timeout: 2 minutes
QTimer.singleShot(120_000, lambda: (print("\n⏰ TIMEOUT"), app.quit()))

sys.exit(app.exec_())