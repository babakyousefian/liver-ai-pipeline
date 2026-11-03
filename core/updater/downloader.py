# core/updater/downloader.py
# -*- coding: utf-8 -*-
"""
UpdateDownloader: QThread that streams the file to disk with progress.
Supports resume via HTTP Range.
"""

import os
import tempfile
import requests
from PyQt5.QtCore import QThread, pyqtSignal

from .source import _is_safe_url, _USER_AGENT, _TIMEOUT


class UpdateDownloader(QThread):
    """
    Signals:
        progress(int percent, int downloaded, int total)
        finished_ok(str file_path)
        failed(str reason)
    """
    progress = pyqtSignal(int, int, int)   # percent, downloaded, total
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    CHUNK_SIZE = 65536  # 64 KB

    def __init__(self, url: str, dest_dir: str = None,
                 expected_size: int = 0, parent=None):
        super().__init__(parent)
        self.url = url
        self.dest_dir = dest_dir or tempfile.gettempdir()
        self.expected_size = expected_size
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if not _is_safe_url(self.url):
            self.failed.emit("unsafe_url")
            return

        filename = self._guess_filename()
        if not filename:
            self.failed.emit("bad_filename")
            return

        dest_path = os.path.join(self.dest_dir, filename)
        # ✅ If exists from previous attempt, resume
        existing = os.path.getsize(dest_path) if os.path.isfile(dest_path) else 0

        headers = {"User-Agent": _USER_AGENT}
        if existing > 0 and self.expected_size > 0:
            headers["Range"] = f"bytes={existing}-"

        try:
            with requests.get(self.url, headers=headers, stream=True,
                              timeout=_TIMEOUT) as r:
                # ✅ Handle 416 (already complete) and 206 (partial)
                if r.status_code == 416:
                    # File already complete
                    if self.expected_size == 0 or existing == self.expected_size:
                        self.finished_ok.emit(dest_path)
                        return
                    # Otherwise, restart from scratch
                    existing = 0
                    os.remove(dest_path) if os.path.exists(dest_path) else None
                    self.failed.emit("restart_needed")
                    return

                if r.status_code not in (200, 206):
                    self.failed.emit(f"http_{r.status_code}")
                    return

                total = int(r.headers.get("Content-Length", 0)) + existing
                if self.expected_size > 0:
                    total = self.expected_size

                mode = "ab" if r.status_code == 206 else "wb"
                downloaded = existing
                last_emit = 0

                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=self.CHUNK_SIZE):
                        if self._cancelled:
                            self.failed.emit("cancelled")
                            return
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        # emit every ~200KB or every 1%
                        if downloaded - last_emit > 200_000:
                            percent = int(downloaded * 100 / total) if total > 0 else 0
                            self.progress.emit(percent, downloaded, total)
                            last_emit = downloaded

                percent = 100
                self.progress.emit(percent, downloaded, total)
                self.finished_ok.emit(dest_path)

        except requests.exceptions.RequestException as e:
            self.failed.emit(f"network: {type(e).__name__}")
        except Exception as e:
            self.failed.emit(f"unexpected: {e}")

    def _guess_filename(self) -> str:
        """Extract filename from URL, sanitize, fallback to 'update.bin'."""
        from urllib.parse import urlparse, unquote
        try:
            p = urlparse(self.url)
            name = os.path.basename(unquote(p.path))
            # sanitize
            bad = ("/", "\\", "..", "\x00")
            if name and not any(b in name for b in bad):
                return name
        except Exception:
            pass
        return "update.bin"