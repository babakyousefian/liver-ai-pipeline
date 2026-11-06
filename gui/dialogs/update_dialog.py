# gui/dialogs/update_dialog.py
# -*- coding: utf-8 -*-
"""
UpdateDialog: theme-aware, animated, RTL-aware dialog for updates.
"""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QTextBrowser, QScrollArea, QWidget,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

# ✅ These will be provided by your main GUI module
try:
    from GUI_update import (
        tr, THEMES, make_tooltip,
        animate_dialog_open, animate_dialog_close,
        show_message, ask_question,
    )
except ImportError:
    # fallback (should never happen in production)
    def tr(lang, key): return key
    THEMES = {"theme_dark_aqua": {}}
    def make_tooltip(**kw): return ""
    def animate_dialog_open(d): pass
    def animate_dialog_close(d, on_finished=None):
        if on_finished: on_finished()
    def show_message(*a, **k): pass
    def ask_question(*a, **k): return False

from core.updater import (
    UpdateInfo, UpdateDownloader, verify_checksum, verify_size,
    install_frozen, install_source, is_frozen, get_app_dir,
    format_version,
)


class UpdateDialog(QDialog):
    """
    Full-featured update dialog.

    Signals:
        download_started()
        download_finished(str file_path)
        install_started()
        install_finished(bool ok)
    """
    download_started = pyqtSignal()
    download_finished = pyqtSignal(str)
    install_started = pyqtSignal()
    install_finished = pyqtSignal(bool)

    def __init__(self, info: UpdateInfo, current_version: str,
                 theme: dict, lang: str = "English", parent=None,
                 auto_download: bool = False):
        super().__init__(parent)
        self.info = info
        self.current_version = current_version
        self.theme = theme
        self.lang = lang
        self._downloader = None
        self._downloaded_path = None
        self._installing = False

        self.setWindowTitle(tr(lang, "update_available"))
        self.setFixedSize(620, 620)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()
        QTimer.singleShot(0, lambda: animate_dialog_open(self))

        if auto_download:
            QTimer.singleShot(500, self._start_download)

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(14)

        # ============================================================
        # Header: icon + title + version
        # ============================================================
        header = QHBoxLayout()
        header.setSpacing(12)

        icon = QLabel("🚀")
        icon.setFixedSize(48, 48)
        icon.setAlignment(Qt.AlignCenter)
        icon.setObjectName("update_icon")
        header.addWidget(icon)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.title_label = QLabel(tr(self.lang, "update_available"))
        self.title_label.setObjectName("update_title")
        self.version_label = QLabel(
            f"v{format_version(self.current_version)}  →  "
            f"v{format_version(self.info.version)}"
        )
        self.version_label.setObjectName("update_version")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.version_label)
        header.addLayout(title_box, 1)

        layout.addLayout(header)

        # ============================================================
        # Meta row: size + date + critical badge
        # ============================================================
        meta = QHBoxLayout()
        meta.setSpacing(16)

        self.size_label = QLabel(f"📦 {self.info.human_size}")
        self.size_label.setObjectName("update_meta")
        meta.addWidget(self.size_label)

        if self.info.release_date:
            date_str = self.info.release_date[:10]
            self.date_label = QLabel(f"📅 {date_str}")
            self.date_label.setObjectName("update_meta")
            meta.addWidget(self.date_label)

        if self.info.is_critical:
            badge = QLabel("⚠️ " + tr(self.lang, "critical_update"))
            badge.setObjectName("update_critical_badge")
            meta.addWidget(badge)

        meta.addStretch()
        layout.addLayout(meta)

        # ============================================================
        # Changelog (scrollable)
        # ============================================================
        changelog_title = QLabel(tr(self.lang, "whats_new"))
        changelog_title.setObjectName("update_section")
        layout.addWidget(changelog_title)

        self.changelog = QTextBrowser()
        self.changelog.setObjectName("update_changelog")
        self.changelog.setOpenExternalLinks(True)
        self.changelog.setMinimumHeight(200)
        self.changelog.setHtml(self._changelog_to_html(self.info.changelog))
        layout.addWidget(self.changelog, 1)

        # ============================================================
        # Progress bar (hidden by default)
        # ============================================================
        self.progress_frame = QFrame()
        self.progress_frame.setObjectName("update_progress_frame")
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 6, 0, 6)
        progress_layout.setSpacing(4)

        self.progress_label = QLabel(tr(self.lang, "downloading"))
        self.progress_label.setObjectName("update_progress_label")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setObjectName("update_progress")

        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        self.progress_frame.setVisible(False)
        layout.addWidget(self.progress_frame)

        # ============================================================
        # Buttons
        # ============================================================
        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self.skip_btn = QPushButton("⏭ " + tr(self.lang, "skip_version"))
        self.skip_btn.setObjectName("update_secondary_btn")
        self.skip_btn.clicked.connect(self._on_skip)
        buttons.addWidget(self.skip_btn)

        self.later_btn = QPushButton("⏰ " + tr(self.lang, "remind_later"))
        self.later_btn.setObjectName("update_secondary_btn")
        self.later_btn.clicked.connect(self._on_later)
        buttons.addWidget(self.later_btn)

        buttons.addStretch()

        self.download_btn = QPushButton("⬇️ " + tr(self.lang, "download_now"))
        self.download_btn.setObjectName("update_primary_btn")
        self.download_btn.clicked.connect(self._start_download)
        buttons.addWidget(self.download_btn)

        self.install_btn = QPushButton("✨ " + tr(self.lang, "install_now"))
        self.install_btn.setObjectName("update_primary_btn")
        self.install_btn.clicked.connect(self._start_install)
        self.install_btn.setVisible(False)
        buttons.addWidget(self.install_btn)

        layout.addLayout(buttons)

        # ✅ Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("update_close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        close_btn.setParent(self.container)
        close_btn.move(580, 10)

    # ============================================================
    # Theme
    # ============================================================
    def _apply_theme(self):
        t = self.theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t.get('glass_bg', '#0f172a')};
                border: 2px solid {t.get('glass_border', '#00e5ff')};
                border-radius: 20px;
            }}
            QLabel#update_icon {{
                font-size: 36px;
                background: transparent;
            }}
            QLabel#update_title {{
                font-size: 22px;
                font-weight: bold;
                color: {t.get('primary', '#00e5ff')};
                background: transparent;
            }}
            QLabel#update_version {{
                font-size: 13px;
                color: {t.get('text_dim', '#94a3b8')};
                background: transparent;
            }}
            QLabel#update_meta {{
                font-size: 12px;
                color: {t.get('text_dim', '#94a3b8')};
                background: transparent;
            }}
            QLabel#update_critical_badge {{
                font-size: 12px;
                font-weight: bold;
                color: {t.get('danger', '#f87171')};
                background: rgba(248, 113, 113, 0.15);
                border: 1px solid {t.get('danger', '#f87171')};
                border-radius: 6px;
                padding: 2px 8px;
            }}
            QLabel#update_section {{
                font-size: 13px;
                font-weight: bold;
                color: {t.get('text', '#f1f5f9')};
                background: transparent;
                margin-top: 4px;
            }}
            QTextBrowser#update_changelog {{
                background: {t.get('input_bg', 'rgba(30, 41, 59, 0.85)')};
                border: 1px solid {t.get('input_border', '#00e5ff')};
                border-radius: 10px;
                color: {t.get('text', '#f1f5f9')};
                padding: 10px;
                font-size: 12px;
            }}
            QFrame#update_progress_frame {{
                background: transparent;
            }}
            QLabel#update_progress_label {{
                font-size: 12px;
                color: {t.get('text', '#f1f5f9')};
                background: transparent;
            }}
            QProgressBar#update_progress {{
                background: {t.get('input_bg', 'rgba(30, 41, 59, 0.85)')};
                border: 1px solid {t.get('glass_border', '#00e5ff')};
                border-radius: 6px;
                text-align: center;
                color: {t.get('text', '#f1f5f9')};
                font-size: 11px;
                font-weight: bold;
                min-height: 18px;
            }}
            QProgressBar#update_progress::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t.get('primary', '#00e5ff')},
                    stop:1 {t.get('secondary', '#c084fc')}
                );
                border-radius: 5px;
            }}
            QPushButton#update_primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t.get('primary', '#00e5ff')},
                    stop:0.5 {t.get('accent', '#f472b6')},
                    stop:1 {t.get('secondary', '#c084fc')}
                );
                color: {t.get('button_text', '#0a0a14')};
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-width: 130px;
            }}
            QPushButton#update_primary_btn:hover {{
                background: {t.get('button_hover', '#67e8f9')};
            }}
            QPushButton#update_secondary_btn {{
                background: {t.get('input_bg', 'rgba(30, 41, 59, 0.85)')};
                color: {t.get('text', '#f1f5f9')};
                border: 1px solid {t.get('input_border', '#00e5ff')};
                border-radius: 10px;
                padding: 8px 16px;
                font-size: 12px;
            }}
            QPushButton#update_secondary_btn:hover {{
                background: {t.get('primary', '#00e5ff')};
                color: {t.get('button_text', '#0a0a14')};
            }}
            QPushButton#update_close_btn {{
                background: transparent;
                color: {t.get('text_dim', '#94a3b8')};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#update_close_btn:hover {{
                color: {t.get('danger', '#f87171')};
            }}
        """)

    # ============================================================
    # Changelog → HTML (very small markdown-ish)
    # ============================================================
    def _changelog_to_html(self, text: str) -> str:
        t = self.theme
        text_color = t.get("text", "#f1f5f9")
        primary = t.get("primary", "#00e5ff")
        if not text:
            return f"<p style='color:{text_color};'>{tr(self.lang, 'no_changelog')}</p>"
        # Basic: convert **bold**, `code`, # heading, - bullet
        import re
        html = text
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
        html = re.sub(r"`(.+?)`", rf"<code style='color:{primary};'>\1</code>", html)
        html = re.sub(r"^### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^- (.+)$", r"• \1", html, flags=re.MULTILINE)
        html = html.replace("\n", "<br>")
        return f"<div style='color:{text_color};font-size:12px;'>{html}</div>"

    # ============================================================
    # Download flow
    # ============================================================
    def _start_download(self):
        if self._downloader is not None and self._downloader.isRunning():
            return
        self.progress_frame.setVisible(True)
        self.download_btn.setVisible(False)
        self.skip_btn.setVisible(False)
        self.later_btn.setEnabled(False)
        self.progress_label.setText(tr(self.lang, "downloading"))
        self.progress_bar.setValue(0)

        self.download_started.emit()

        self._downloader = UpdateDownloader(
            url=self.info.download_url,
            expected_size=self.info.download_size,
            parent=self,
        )
        self._downloader.progress.connect(self._on_download_progress)
        self._downloader.finished_ok.connect(self._on_download_done)
        self._downloader.failed.connect(self._on_download_failed)
        self._downloader.start()

    def _on_download_progress(self, percent: int, downloaded: int, total: int):
        self.progress_bar.setValue(percent)
        if total > 0:
            mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self.progress_label.setText(
                f"{tr(self.lang, 'downloading')}  {mb:.1f} / {total_mb:.1f} MB"
            )

    def _on_download_done(self, path: str):
        self.progress_label.setText(tr(self.lang, "verifying"))
        self.progress_bar.setValue(100)

        # ✅ Verify size
        if not verify_size(path, self.info.download_size, tolerance=1024):
            self._on_download_failed("size_mismatch")
            return

        # ✅ Verify checksum
        if self.info.checksum_sha256:
            ok = verify_checksum(path, self.info.checksum_sha256)
            if not ok:
                self._on_download_failed("checksum_mismatch")
                return

        self._downloaded_path = path
        self.download_finished.emit(path)
        self.progress_frame.setVisible(False)
        self.download_btn.setVisible(False)
        self.install_btn.setVisible(True)
        self.later_btn.setEnabled(True)

        # ✅ Auto-install for critical updates
        if self.info.is_critical:
            QTimer.singleShot(500, self._start_install)

    def _on_download_failed(self, reason: str):
        self.progress_frame.setVisible(False)
        self.download_btn.setVisible(True)
        self.skip_btn.setVisible(True)
        self.later_btn.setEnabled(True)
        show_message(
            self, "critical",
            tr(self.lang, "error"),
            f"{tr(self.lang, 'download_failed')}\n{reason}",
        )

    # ============================================================
    # Install flow
    # ============================================================
    def _start_install(self):
        if not self._downloaded_path or self._installing:
            return
        self._installing = True
        self.install_started.emit()

        if is_frozen():
            # EXE mode: launch helper bat, then quit app
            ok = install_frozen(self._downloaded_path)
            if ok:
                show_message(
                    self, "information",
                    tr(self.lang, "update_ready"),
                    tr(self.lang, "app_will_restart"),
                )
                QTimer.singleShot(800, self._quit_app)
            else:
                self._installing = False
                show_message(
                    self, "critical",
                    tr(self.lang, "error"),
                    tr(self.lang, "install_failed"),
                )
                self.install_finished.emit(False)
        else:
            # Source mode: unzip into project root
            try:
                import zipfile
                if not zipfile.is_zipfile(self._downloaded_path):
                    show_message(
                        self, "warning",
                        tr(self.lang, "error"),
                        tr(self.lang, "source_update_requires_zip"),
                    )
                    self._installing = False
                    self.install_finished.emit(False)
                    return

                ok = install_source(
                    self._downloaded_path,
                    get_app_dir(),
                )
                if ok:
                    show_message(
                        self, "information",
                        tr(self.lang, "update_ready"),
                        tr(self.lang, "app_will_restart"),
                    )
                    QTimer.singleShot(800, self._quit_app)
                else:
                    self._installing = False
                    show_message(
                        self, "critical",
                        tr(self.lang, "error"),
                        tr(self.lang, "install_failed"),
                    )
                    self.install_finished.emit(False)
            except Exception as e:
                self._installing = False
                print(f"⚠️ install source failed: {e}")
                show_message(self, "critical", tr(self.lang, "error"), str(e))
                self.install_finished.emit(False)

    def _quit_app(self):
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.quit()
        except Exception:
            pass

    # ============================================================
    # Skip / Later / Close
    # ============================================================
    def _on_skip(self):
        # Emit a signal so MainWindow can save `skipped_version`
        self.done(2)  # custom result code for "skip"

    def _on_later(self):
        self.done(3)  # custom result code for "later"

    def closeEvent(self, event):
        # Cancel download if running
        if self._downloader is not None and self._downloader.isRunning():
            self._downloader.cancel()
            self._downloader.wait(2000)
        super().closeEvent(event)

# ============================================================
# ✅ Helper: One-click update flow (used by Settings dialog)
# ============================================================

def run_update_flow(parent, config, current_version: str,
                    theme: dict, lang: str = "English") -> bool:
    """
    Full one-click update flow using beautiful custom dialogs.
    """
    from PyQt5.QtWidgets import QApplication, QDialog
    from PyQt5.QtCore import QEventLoop, QTimer
    from core.updater import (
        UpdateChecker, verify_checksum,
        install_frozen, install_source,
        is_frozen, get_app_dir,
    )
    from GUI_update import tr
    from gui.dialogs.update_dialogs import (
        UpdateAvailableDialog,
        UpdateDownloadDialog,
        ConfirmInstallDialog,
        SimpleInfoDialog,
    )

    # ============================================================
    # STEP 1: Check for updates (silent)
    # ============================================================
    checker = UpdateChecker(
        config=config,
        current_version=current_version,
        force=True,
    )
    check_result = {"info": None, "error": None}
    check_loop = QEventLoop()

    def _on_available(info):
        check_result["info"] = info
        check_loop.quit()

    def _on_uptodate():
        check_result["info"] = None
        check_loop.quit()

    def _on_failed(reason):
        check_result["error"] = reason
        check_loop.quit()

    checker.update_available.connect(_on_available)
    checker.up_to_date.connect(_on_uptodate)
    checker.check_failed.connect(_on_failed)
    checker.start()

    timeout_timer = QTimer()
    timeout_timer.setSingleShot(True)
    timeout_timer.timeout.connect(lambda: (
        check_result.update({"error": "timeout"}),
        check_loop.quit()
    ))
    timeout_timer.start(30_000)

    check_loop.exec_()
    timeout_timer.stop()

    info = check_result["info"]

    # ============================================================
    # STEP 2a: No update → beautiful info dialog
    # ============================================================
    if info is None:
        if check_result["error"]:
            dlg = SimpleInfoDialog(
                title=tr(lang, "update_check_failed"),
                message=tr(lang, "check_internet") + f"\n\n({check_result['error']})",
                theme=theme, lang=lang, icon="⚠️", parent=parent,
            )
        else:
            dlg = SimpleInfoDialog(
                title=tr(lang, "up_to_date"),
                message=tr(lang, "you_have_latest"),
                theme=theme, lang=lang, icon="✅", parent=parent,
            )
        dlg.exec_()
        return False

    # ============================================================
    # STEP 2b: Update available → beautiful dialog
    # ============================================================
    dlg = UpdateAvailableDialog(
        info=info,
        current_version=current_version,
        theme=theme,
        lang=lang,
        parent=parent,
    )
    if dlg.exec_() != QDialog.Accepted:
        return False

    # ============================================================
    # STEP 3: Download with beautiful progress dialog
    # ============================================================
    dl_dialog = UpdateDownloadDialog(
        info=info, theme=theme, lang=lang, parent=parent,
    )

    download_path = {"path": None, "error": None}
    dl_loop = QEventLoop()

    def _on_ready(path):
        download_path["path"] = path
        dl_loop.quit()

    def _on_failed(reason):
        download_path["error"] = reason
        dl_loop.quit()

    def _on_canceled():
        download_path["error"] = "canceled_by_user"
        dl_loop.quit()

    dl_dialog.download_ready.connect(_on_ready)
    dl_dialog.download_failed.connect(_on_failed)
    dl_dialog.canceled.connect(_on_canceled)

    # Start download + show dialog
    QTimer.singleShot(100, dl_dialog.start_download)
    dl_dialog.exec_()  # blocks until dialog closes

    # Check if user canceled
    if download_path["error"] == "canceled_by_user":
        return False

    # Check if download failed
    if download_path["error"] or not download_path["path"]:
        err = download_path["error"] or "unknown"
        SimpleInfoDialog(
            title=tr(lang, "error"),
            message=f"{tr(lang, 'download_failed')}\n\nError: {err}",
            theme=theme, lang=lang, icon="❌", parent=parent,
        ).exec_()
        return False

    path = download_path["path"]

    # ============================================================
    # STEP 4: Verify checksum
    # ============================================================
    if info.checksum_sha256:
        if not verify_checksum(path, info.checksum_sha256):
            SimpleInfoDialog(
                title=tr(lang, "error"),
                message=tr(lang, "install_failed") + "\n\nChecksum mismatch",
                theme=theme, lang=lang, icon="❌", parent=parent,
            ).exec_()
            return False

    # ============================================================
    # STEP 5: Confirm install
    # ============================================================
    install_dlg = ConfirmInstallDialog(
        info=info, theme=theme, lang=lang, parent=parent,
    )
    if install_dlg.exec_() != QDialog.Accepted:
        return False

    # ============================================================
    # STEP 6: Install
    # ============================================================
    try:
        if is_frozen():
            ok = install_frozen(path, target_version=info.version)
        else:
            import zipfile
            if not zipfile.is_zipfile(path):
                SimpleInfoDialog(
                    title=tr(lang, "error"),
                    message=tr(lang, "source_update_requires_zip"),
                    theme=theme, lang=lang, icon="⚠️", parent=parent,
                ).exec_()
                return False
            ok = install_source(
                path,
                get_app_dir(),
                target_version=info.version,
            )

        if ok:
            SimpleInfoDialog(
                title=tr(lang, "update_ready"),
                message=tr(lang, "app_will_restart"),
                theme=theme, lang=lang, icon="✅", parent=parent,
            ).exec_()
            QTimer.singleShot(1000, QApplication.quit)
            return True
        else:
            SimpleInfoDialog(
                title=tr(lang, "error"),
                message=tr(lang, "install_failed"),
                theme=theme, lang=lang, icon="❌", parent=parent,
            ).exec_()
            return False
    except Exception as e:
        print(f"⚠️ Install failed: {e}")
        import traceback
        traceback.print_exc()
        SimpleInfoDialog(
            title=tr(lang, "error"),
            message=f"{tr(lang, 'install_failed')}\n\n{e}",
            theme=theme, lang=lang, icon="❌", parent=parent,
        ).exec_()
        return False
