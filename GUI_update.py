# GUI.py
# -*- coding: utf-8 -*-

"""
Liver AI Pipeline - GUI (ULTIMATE EDITION v2.2)

Beautiful, animated, user-friendly GUI for the Liver AI Pipeline.

Features:
- Animated splash screen (5-7 seconds)
- Login / Signup with PBKDF2 password hashing + rate limiting
- Account dropdown menu
- Settings (language + theme)
- Image classification with the KD Student model
- 10 languages
- 3 beautiful themes with multi-color gradients
- Calm 30-second animated gradient background (theme-aware palette)
- Smooth hover/click animations on all buttons
- Complete UI sound system (12 effects, single-play per click)
- Music player with mute/unmute

Author: Babak Yousefian
Version: 2.2.0
"""

import os
import sys
import json
import hashlib
import hmac
import secrets
import random
import math
import time
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

warnings.filterwarnings("ignore")

from core.updater import (UpdateChecker, UpdateConfig, CURRENT_VERSION, is_newer,)
from gui.dialogs.update_dialog import UpdateDialog
from gui.widgets.update_badge import UpdateBadge

# ✅ Password strength (optional)
try:
    from core.security import calculate_strength
except Exception:
    calculate_strength = None

# ✅ Health-check imports (برای rollback خودکار)
try:
    from core.updater.health import (
        check_update_pending,
        rollback_if_needed,
        confirm_update_success,
    )
except ImportError:
    def check_update_pending(): return {}
    def rollback_if_needed(_data): return False
    def confirm_update_success(): pass

# ✅ Preprocessing
try:
    from core.preprocessing.medical_preprocessor import (
        preprocess_medical_image,
        PreprocessResult,
    )
    HAS_PREPROCESSOR = True
    print("✅ Medical Preprocessor available")
except Exception as e:
    print(f"⚠️ Medical Preprocessor not available: {e}")
    HAS_PREPROCESSOR = False
    preprocess_medical_image = None
    PreprocessResult = None

# ✅ Audit log
try:
    from core.audit import audit_log
except Exception as e:
    print(f"⚠️ audit_log import failed: {e}")
    audit_log = None

from core.ui.smart_suggestions import (
    SuggestionStorage, Suggestion, SuggestionContext,
    SuggestionEngine, SmartSuggestionBar,
)

from core.ui.onboarding import (
    OnboardingStorage, OnboardingStep, OnboardingTour,
    TOUR_STEPS_GUEST, TOUR_STEPS_FULL,
)

from core.ui.empty_state import (
    make_empty_state_html, EmptyStateWidget,
    EMPTY_STATE_STYLES, EmptyStateStyle,
)

from core.ui.micro_interactions import (
    MotionSettings, shake_widget, add_focus_glow,
    show_success_check, animate_count_up, button_press_scale,
    smooth_scroll_to, bounce_in, color_pulse, page_transition_in,
    decorate_input, decorate_button_press,
    highlight_widget_error, highlight_widget_success,
)

from core.ui.confetti import (
    ConfettiOverlay,
    celebrate,
    get_pattern_for,
    CELEBRATION_PATTERNS,
)

# ✅ Voice Feedback
try:
    from core.ui.voice_feedback import VoiceManager
    HAS_VOICE = True
    print("✅ Voice Feedback available")
except Exception as e:
    print(f"⚠️ Voice Feedback not available: {e}")
    VoiceManager = None
    HAS_VOICE = False

# ✅ Image Guard — Out-of-Distribution Detection
try:
    from core.security.image_guard import (
        analyze_image as guard_analyze_image,
        is_confident_result as guard_is_confident_result,
        MIN_LIVER_SCORE,
        MIN_MODEL_CONFIDENCE,
    )
    HAS_IMAGE_GUARD = True
    print(f"✅ Image Guard available (MIN_LIVER={MIN_LIVER_SCORE}, MIN_CONF={MIN_MODEL_CONFIDENCE})")
except Exception as e:
    print(f"⚠️ Image Guard not available: {e}")
    HAS_IMAGE_GUARD = False
    guard_analyze_image = None
    guard_is_confident_result = None
    MIN_LIVER_SCORE = 0.55
    MIN_MODEL_CONFIDENCE = 0.55

from core.ui.font_manager import (
    FontSettings, FontManager,
    load_font_settings, save_font_settings,
    list_available_fonts, display_name_for,
    DEFAULT_FAMILY, DEFAULT_SIZE, DEFAULT_SCALE,
    MIN_SIZE, MAX_SIZE, MIN_SCALE, MAX_SCALE,
)

# ✅ Preprocessing (v1.3)
try:
    from core.preprocessing.medical_preprocessor import (
        preprocess_medical_image,
        PreprocessResult,
    )
    HAS_PREPROCESSOR = True
    print("✅ Medical Preprocessor available")
except Exception as e:
    print(f"⚠️ Medical Preprocessor not available: {e}")
    HAS_PREPROCESSOR = False
    preprocess_medical_image = None
    PreprocessResult = None

from core.updater import (UpdateChecker, UpdateConfig, CURRENT_VERSION, is_newer,)
from core.security.auto_lock import (
    AutoLockConfig, AutoLockManager,
    load_config as load_auto_lock_config,
    save_config as save_auto_lock_config,
)

# ================================================================
# ✅ PyQt5 Imports
# ================================================================

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QDialog, QMessageBox, QFileDialog, QFrame, QGraphicsOpacityEffect,
    QSizePolicy, QSpacerItem, QScrollArea, QCheckBox, QComboBox,
    QStackedWidget, QToolButton, QMenu, QGraphicsDropShadowEffect,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QShortcut,
    QToolTip, QGraphicsBlurEffect,  # ← اضافه شد
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize,
    QPoint, pyqtSignal, QParallelAnimationGroup, QSequentialAnimationGroup,
    QThread, QMimeData, QObject, QRectF, QRegExp,
)
from PyQt5.QtGui import (
    QPixmap, QIcon, QFont, QColor, QPalette, QPainter, QBrush,
    QLinearGradient, QRadialGradient, QPen, QFontDatabase, QTransform,
    QDragEnterEvent, QDropEvent, QImage, QWheelEvent, QMouseEvent,
    QKeySequence, QPainterPath, QCursor, QRegion, QPainterPath, QRegExpValidator,
)
from PyQt5.QtWidgets import QAction  # ✅ در PyQt5 اینجاست

# ✅ PDF export (optional — نیاز به pip install reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, Color, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, Flowable,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF
    HAS_REPORTLAB = True
    print("✅ reportlab available — PDF export enabled")
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️ reportlab not available — PDF export disabled")

# ✅ RTL support برای PDF (فارسی/عربی/عبری)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display as bidi_get_display
    HAS_RTL = True
    print("✅ RTL support available for PDF")
except ImportError:
    HAS_RTL = False
    arabic_reshaper = None
    bidi_get_display = None
    print("⚠️ RTL support not available (pip install arabic-reshaper python-bidi)")

# ================================================================
# ✅ Optional: Music + Sound
# ================================================================

import os as _os_pg
_os_pg.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

try:
    import pygame
    # ✅ pre_init باید قبل از init باشه — کیفیت بهتر افکت‌ها
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)  # ✅ کانال کافی برای موزیک + ۱۲ افکت
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False
    print("⚠️ Pygame not available — Music/Sound disabled")

# ================================================================
# ✅ Optional: Image Processing
# ================================================================

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV not available — Image processing disabled")

import traceback as _tb
_LOG_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "torch_debug.log")
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"✅ PyTorch OK\n")
        f.write(f"Version: {torch.__version__}\n")
        f.write(f"File: {torch.__file__}\n")
        f.write(f"CUDA: {torch.cuda.is_available()}\n")
except Exception as e:
    HAS_TORCH = False
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"❌ PyTorch FAILED\n")
        f.write(f"Error: {type(e).__name__}: {e}\n\n")
        f.write(_tb.format_exc())


# ================================================================
# ✅ PATHS
# ================================================================

if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
    BUNDLE_ROOT = getattr(sys, '_MEIPASS', PROJECT_ROOT)
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_ROOT = PROJECT_ROOT

FINAL_MODEL_PATH = os.path.join(
    BUNDLE_ROOT, "finalReports", "selfSupervisedLearning", "final_models",
    "final_model.pth",
)

USERS_DB_PATH = os.path.join(PROJECT_ROOT, "users_db.json")

ASSETS_DIR = os.path.join(BUNDLE_ROOT, "assets")
ICON_PATH = os.path.join(ASSETS_DIR, "favicon.ico")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
MUSIC_PATH = os.path.join(
    BUNDLE_ROOT, "packages", "facilities", "assets", "musics", "music3.mp3"
)
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")
# ✅ پسوندهای مجاز برای Drag & Drop و فایل‌دیالوگ
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".dcm", ".tif", ".tiff"}
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")   # ← اضافه شد
# ✅ مسیر تاریخچه (per-user)
HISTORY_DIR = os.path.join(PROJECT_ROOT, "history")


# ================================================================
# ✅ TRANSLATIONS (10 Languages - COMPLETE)
# ================================================================

TRANSLATIONS = {
    "English": {
        "app_title": "Liver AI-babak",
        "welcome": "Welcome",
        "login": "Login",
        "signup": "Sign Up",
        "logout": "Logout",
        "exit": "Exit",
        "settings": "Settings",
        "account_info": "Account Info",
        "username": "Username",
        "password": "Password",
        "repeat_password": "Repeat Password",
        "first_name": "First Name",
        "last_name": "Last Name",
        "email": "Email Address",
        "remember_me": "Remember Me",
        "login_btn": "Login",
        "signup_btn": "Sign Up",
        "account": "Account",
        "change_language": "Change Language",
        "change_theme": "Change Theme",
        "insert_sono": "Insert Sono",
        "process": "Process",
        "mute": "Mute",
        "unmute": "Unmute",
        "visit_webpage": "Visit Webpage",
        "please_wait": "Please wait...",
        "processing": "Processing image...",
        "result": "Result",
        "confidence": "Confidence",
        "grid": "Grid",
        "insert_image_first": "Please insert an image first!",
        "no_image": "No image inserted",
        "invalid_credentials": "Invalid credentials",
        "user_exists": "Username already exists",
        "password_requirements": "Password requirements",
        "password_rule": (
            "Password must be:\n"
            "• 8-16 characters long\n"
            "• Contains at least 1 lowercase letter\n"
            "• Contains at least 1 uppercase letter\n"
            "• Contains at least 4 digits\n"
            "• Contains at least 4 special characters"
        ),
        "passwords_dont_match": "Passwords do not match!",
        "invalid_email": "Please enter a valid email address",
        "empty_fields": "Please fill in all fields",
        "signup_success": "Signup successful! You can now login.",
        "logout_confirm": "Are you sure you want to delete your account?",
        "exit_confirm": "Are you sure you want to exit?",
        "theme_dark_aqua": "Theme Dark Aqua-Purple",
        "theme_solarized_white": "Theme Solarized White",
        "theme_solarized_dark": "Theme Solarized Dark",
        "theme_dark_forest": "Theme Dark Forest Green",
        "select_language": "Select Language",
        "select_theme": "Select Theme",
        "apply": "Apply",
        "cancel": "Cancel",
        "save": "Save",
        "success": "Success",
        "error": "Error",
        "warning": "Warning",
        "settings_applied": "Settings applied!",
        "grid_0": "Grid 0 (NL) — Normal Liver",
        "grid_1": "Grid 1 (G1) — Grade 1",
        "grid_2": "Grid 2 (G2) — Grade 2",
        "grid_3": "Grid 3 (G3) — Grade 3",
        "too_many_attempts": "Too many failed attempts. Please wait 5 minutes.",
        "invalid_username": "Username must be 3-32 chars (letters, numbers, underscore only).",
        "welcome_user": "Welcome, {user}!",
        "inference_failed": "Inference failed",
        "drop_here": "Drop image here",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Unsupported file type. Please drop an image.",
        "image_loaded": "Image loaded successfully",
        "help_about_title": "Help & About",
        "help_about_text": (
            "Liver AI Pipeline v3.0\n"
            "Developed by Babak Yousefian\n\n"
            "Keyboard Shortcuts:\n"
            "  Ctrl+O      — Insert Image\n"
            "  Ctrl+Enter  — Process Image\n"
            "  Ctrl+L      — Login\n"
            "  Ctrl+,      — Settings\n"
            "  Esc         — Close current dialog\n"
            "  F1          — Help / About\n\n"
            "Tip: You can also drag & drop an image anywhere on the window."
        ),
        # ✅ Dashboard keys
        "dashboard_title": "Dashboard",
        "dash_total": "Total Images",
        "dash_avg_conf": "Avg. Confidence",
        "dash_distribution": "Distribution",
        "dash_last_5": "Last 5 Results",
        "dash_no_results": "No results yet",
        "dash_clear_confirm": "Clear all statistics?",
        "dash_cleared": "Statistics cleared",
        "dashboard_toggle": "Toggle Dashboard",
        # ✅ PDF keys
        "export_pdf": "Export PDF",
        "pdf_saved": "PDF saved successfully",
        "pdf_failed": "PDF generation failed",
        "pdf_subtitle": "Analysis Report",
        "pdf_patient": "Patient:",
        "pdf_date": "Date:",
        "pdf_model": "Model:",
        "pdf_diagnosis": "DIAGNOSIS",
        "pdf_confidence": "Confidence",
        "pdf_class_probs": "CLASS PROBABILITIES",
        "pdf_disclaimer": (
            "This report was developed by Babak Yousefian under the supervision of experienced physicians. "
            "It is a professional analysis tool and does not require referral to a physician."
        ),
        "history_title": "History",
        "history_toggle": "Toggle History",
        "history_empty": "No history yet",
        "history_cleared": "History cleared",
        "history_clear_confirm": "Clear all history?",
        # ============================================================
        # ✅ TOOLTIP KEYS
        # ============================================================
        "tooltip_title_login": "Login",
        "tooltip_desc_login": "Sign in to your account",
        "tooltip_tip_login": "Access your saved history and stats",
        "tooltip_title_signup": "Sign Up",
        "tooltip_desc_signup": "Create a new account",
        "tooltip_title_account": "Account",
        "tooltip_desc_account": "Click to open account menu",
        "tooltip_desc_account_user": "Logged in as {user}",
        "tooltip_title_dashboard": "Dashboard",
        "tooltip_desc_dashboard": "Show/hide stats panel — total, distribution, recent results",
        "tooltip_title_history": "History",
        "tooltip_desc_history": "Show/hide last 20 results with thumbnails",
        "tooltip_title_visit": "Visit Webpage",
        "tooltip_desc_visit": "Open the project's GitHub page in your browser",
        "tooltip_tip_visit": "Opens in external browser",
        "tooltip_title_insert": "Insert Image",
        "tooltip_desc_insert": "Load a sono image to analyze",
        "tooltip_tip_insert": "You can also drag & drop an image anywhere on the window",
        "tooltip_title_process": "Process",
        "tooltip_desc_process": "Run AI analysis on the loaded image",
        "tooltip_tip_process": "Requires an image to be loaded first",
        "tooltip_title_mute": "Mute Music",
        "tooltip_desc_mute": "Pause the background music",
        "tooltip_title_unmute": "Unmute Music",
        "tooltip_desc_unmute": "Resume the background music",
        "tooltip_title_export": "Export PDF",
        "tooltip_desc_export": "Save the current result as a PDF report",
        "tooltip_tip_export": "Includes image, diagnosis, and class probabilities",
        "tooltip_title_dash_clear": "Clear Stats",
        "tooltip_desc_dash_clear": "Reset all statistics for this account",
        "tooltip_tip_dash_clear": "This action cannot be undone",
        "tooltip_title_dash_close": "Close Dashboard",
        "tooltip_desc_dash_close": "Hide the stats panel",
        "tooltip_desc_dash_total": "Total number of images processed",
        "tooltip_desc_dash_avg": "Average confidence across all results",
        "tooltip_desc_dash_dist": "Distribution of results across grades",
        "tooltip_title_hist_clear": "Clear History",
        "tooltip_desc_hist_clear": "Delete all saved results and thumbnails",
        "tooltip_tip_hist_clear": "This will delete all image files too",
        "tooltip_title_hist_close": "Close History",
        "tooltip_desc_hist_close": "Hide the history panel",
        "tooltip_tip_thumb_click": "Click to reload this result",
        "tooltip_tip_recent": "Click to reload this result",
        "tooltip_title_zoom_in": "Zoom In",
        "tooltip_desc_zoom_in": "Increase image magnification",
        "tooltip_title_zoom_out": "Zoom Out",
        "tooltip_desc_zoom_out": "Decrease image magnification",
        "tooltip_title_zoom_reset": "Reset View",
        "tooltip_desc_zoom_reset": "Fit the image to the window",
        "tooltip_tip_zoom_reset": "You can also double-click the image",
        # ✅ Notification keys
        "notif_inference_ok": "Inference complete",
        # ✅ Follow system theme keys
        "follow_system_theme": "Follow System Theme",
        "follow_system_theme_desc": "Automatically switch between light and dark based on system settings",
        # ✅ Grad-CAM keys
        "gradcam_title": "Attention Heatmap",
        "gradcam_show": "Show Attention Map",
        "gradcam_toggle": "Toggle Heatmap",
        "gradcam_save": "Save Image",
        "gradcam_close": "Close",
        "gradcam_saved": "Image saved successfully",
        "gradcam_no_heatmap": "No attention map available. Please process an image first.",
        # English (نمونه — اینا رو به همه‌ی زبان‌ها اضافه کن)
        "update_available": "Update Available",
        "update_check_failed": "Update Check Failed",
        "check_for_updates": "Check for Updates",
        "check_for_updates_desc": "Check GitHub/website for the latest version",
        "up_to_date": "You're Up to Date",
        "you_have_latest": "You are running the latest version.",
        "check_internet": "Please check your internet connection and try again.",
        "downloading": "Downloading...",
        "verifying": "Verifying...",
        "download_now": "Download Now",
        "install_now": "Install & Restart",
        "skip_version": "Skip This Version",
        "remind_later": "Remind Me Later",
        "whats_new": "What's New",
        "no_changelog": "No changelog provided.",
        "critical_update": "Critical Update",
        "update_ready": "Ready to Install",
        "app_will_restart": "The app will close and restart to apply the update.",
        "install_failed": "Installation failed. Your old version is safe.",
        "download_failed": "Download failed.",
        "source_update_requires_zip": "Source update requires a .zip file.",
        "session_locked": "Session Locked",
        "locked_for_user": "Locked for {user}",
        "unlock": "Unlock",
        "session_unlocked": "Session unlocked",
        "auto_lock_notice": "Session will lock after 2 minutes of inactivity.",
        "audit_log_title": "Audit Log",
        "audit_export": "Export",
        "audit_exported": "Exported successfully",
        "audit_export_failed": "Export failed",
        "audit_search_placeholder": "Search actions or details...",
        "audit_col_time": "Time",
        "audit_col_level": "Level",
        "audit_col_category": "Category",
        "audit_col_action": "Action",
        "audit_col_user": "User",
        "audit_details_title": "Event Details",
        "audit_status": "Showing {count} of {total} events",
        # Level labels
        "strength_very_weak": "Very Weak",
        "strength_weak": "Weak",
        "strength_medium": "Medium",
        "strength_strong": "Strong",
        "strength_very_strong": "Very Strong",

        # Rules checklist
        "strength_rule_length": "8-16 characters",
        "strength_rule_lowercase": "Lowercase letter",
        "strength_rule_uppercase": "Uppercase letter",
        "strength_rule_digits": "Digits",
        "strength_rule_special": "Special characters",

        # Feedback
        "strength_add_chars": "Add {n} more characters",
        "strength_remove_chars": "Remove {n} characters",
        "strength_need_digits": "Add {n} more digit(s)",
        "strength_need_special": "Add {n} more special character(s)",
        "strength_common_pattern": "Avoid common words",
        "strength_common_year": "Avoid common years",
        "strength_has_sequence": "Avoid sequences (1234, abcd)",
        "strength_has_repeats": "Avoid repeats (aaaa, 1111)",

        # Match indicator
        "passwords_match": "Passwords match",
        # (passwords_dont_match already exists)
        "2fa_setup_title": "Enable Two-Factor Authentication",
        "2fa_setup_subtitle": "Scan the QR code with Google Authenticator, Authy, or Microsoft Authenticator.",
        "2fa_manual_secret": "Or enter this code manually:",
        "2fa_enter_code": "Enter the 6-digit code from your app:",
        "2fa_verify_btn": "Verify",
        "2fa_backup_codes_title": "Save these backup codes!",
        "2fa_backup_codes_hint": "Each code can be used once. Store them somewhere safe.",
        "2fa_done_btn": "Done",
        "2fa_secret_copied": "Secret copied to clipboard",
        "2fa_invalid_code": "Invalid code. Please try again.",
        "2fa_invalid_code_format": "Code must be 6 digits.",
        "2fa_verify_title": "Two-Factor Authentication",
        "2fa_verify_subtitle": "Enter the 6-digit code from your authenticator app.",
        "2fa_enter_code_hint": "Please enter a code.",
        "2fa_use_backup_code": "Use a backup code instead",
        "2fa_use_totp_code": "Use authenticator code instead",
        "2fa_enter_backup_code": "Enter one of your backup codes.",
        "2fa_section_title": "Security",
        "2fa_enabled_status": "Two-Factor Authentication is enabled",
        "2fa_disabled_status": "Two-Factor Authentication is disabled",
        "2fa_enable_btn": "Enable 2FA",
        "2fa_disable_btn": "Disable 2FA",
        "2fa_disable_title": "Disable 2FA",
        "2fa_disable_confirm": "Are you sure? Your account will be less secure.",
        "2fa_disabled_success": "Two-Factor Authentication disabled",
        "2fa_enabled_success": "Two-Factor Authentication enabled",
        "2fa_hint_title": "How to use this QR code",
        "2fa_hint_step1": "Install an authenticator app: Google Authenticator, Microsoft Authenticator, or Authy.",
        "2fa_hint_step2": "Open the app and tap the \"+\" button, then choose \"Scan QR code\".",
        "2fa_hint_step3": "Point your phone's camera at the QR code above (use the app's camera, not your phone's camera app).",
        "2fa_hint_step4": "The app will show a 6-digit code. Enter it in the box below.",
        "2fa_manual_secret_hint": "Can't scan? Copy the key below and add it manually in your app (\"Enter a setup key\").",
        "auto_lock_section_title": "Auto-Lock",
        "auto_lock_enable": "Lock after inactivity",
        "auto_lock_timeout_label": "Lock after:",
        "auto_lock_warning_title": "Auto-Lock Warning",
        "auto_lock_warning_body": "Your session will be locked in {seconds} seconds due to inactivity.",
        "lock_now": "Lock Now",
        "font_section_title": "Font",
        "font_family_label": "Font family",
        "font_size_label": "Base size",
        "font_scale_label": "Display scale",
        "font_scale_small": "Small",
        "font_scale_normal": "Normal",
        "font_scale_large": "Large",
        "font_scale_xlarge": "Extra Large",
        "font_scale_xxlarge": "Huge",
        "font_preview_text": "The quick brown fox jumps over the lazy dog — 1234567890",
        "font_reset": "Reset to Default",
        "motion_section_title": "Motion & Animations",
        "reduce_motion_label": "Reduce motion",
        "reduce_motion_hint": "Disable animations if they make you uncomfortable.",
        # ============================================================
        # ✅ EMPTY STATE KEYS
        # ============================================================
        "empty_history_empty_title": "No history yet",
        "empty_history_empty_subtitle": "Process your first image to see it here",
        "empty_history_empty_hint": "Each result is saved automatically",
        "empty_history_no_user_title": "Sign in to view history",
        "empty_history_no_user_subtitle": "Your saved results will appear here",
        "empty_history_no_user_hint": "History is private to each account",
        "empty_dash_no_results_title": "No results yet",
        "empty_dash_no_results_subtitle": "Your recent analyses will appear here",
        "empty_dash_no_results_hint": "Process an image to get started",
        "empty_dash_no_dist_title": "No distribution data",
        "empty_dash_no_dist_subtitle": "Stats will populate as you process images",
        "empty_dash_no_dist_hint": "Every analysis counts",
        "empty_image_empty_title": "No image inserted",
        "empty_image_empty_subtitle": "Drag & drop an image here, or click Insert",
        "empty_image_empty_hint": "Supports PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Drop to analyze",
        "empty_image_drop_hint_subtitle": "Release the image to load it",
        "empty_image_drop_hint_hint": "We'll auto-detect the format",
        "empty_processing_wait_title": "Processing...",
        "empty_processing_wait_subtitle": "AI is analyzing your image",
        "empty_processing_wait_hint": "This usually takes a few seconds",
        # ============================================================
        # ✅ ONBOARDING TOUR KEYS
        # ============================================================
        "tour_show_again": "Show Tour Again",
        "tour_completed_msg": "Tour completed! You're ready to go.",
        "tour_skipped_msg": "Tour skipped. You can replay it from Account menu.",
        "tour_btn_next": "Next",
        "tour_btn_prev": "Back",
        "tour_btn_skip": "Skip",
        "tour_btn_finish": "Finish ✓",
        # Guest tour (2 steps)
        "tour_guest_login_title": "Sign In",
        "tour_guest_login_subtitle": "Click here to sign in to your account and unlock the full experience.",
        "tour_guest_signup_title": "Create an Account",
        "tour_guest_signup_subtitle": "New here? Click here to create an account in seconds.",
        # Full tour
        "tour_insert_title": "Insert Image",
        "tour_insert_subtitle": "Load a sono image from disk, or drag & drop it anywhere on the window.",
        "tour_process_title": "Process Image",
        "tour_process_subtitle": "Run the AI analysis on the loaded image. Results appear in a few seconds.",
        "tour_export_title": "Export PDF",
        "tour_export_subtitle": "Save the current result as a professional PDF medical report.",
        "tour_gradcam_title": "Attention Heatmap",
        "tour_gradcam_subtitle": "Visualize where the AI is looking — a heatmap over the analyzed image.",
        "tour_dashboard_title": "Dashboard",
        "tour_dashboard_subtitle": "Toggle the stats panel: total images, average confidence, and grade distribution.",
        "tour_history_title": "History",
        "tour_history_subtitle": "Browse your last 20 analyzed images. Click any thumbnail to reload its result.",
        "tour_visit_title": "Visit Webpage",
        "tour_visit_subtitle": "Open the project's GitHub page in your browser for docs and updates.",
        "tour_mute_title": "Mute Music",
        "tour_mute_subtitle": "Pause or resume the background music at any time.",
        "tour_zoom_title": "Zoom Controls",
        "tour_zoom_subtitle": "Use the mouse wheel to zoom, drag to pan, and double-click to reset.",
        "tour_account_title": "Account Menu",
        "tour_account_subtitle": "Click here to access Settings, Account Info, Audit Log, and more.",
        # ✅ Smart Suggestions
        "suggestion_image_ready":      "Image loaded — ready to process",
        "suggestion_image_ready_sub":  "Click Process to run the AI analysis.",
        "suggestion_pdf_ready":        "Result ready — export as PDF?",
        "suggestion_pdf_ready_sub":    "Save a professional medical report in one click.",
        "suggestion_gradcam_hint":     "Curious what the AI saw?",
        "suggestion_gradcam_hint_sub": "See the attention heatmap over the image.",
        "suggestion_repeated_grid":    "You keep getting {grid}",
        "suggestion_repeated_grid_sub":"Open the dashboard to see the distribution.",
        "suggestion_history_hint":     "Your history is building up",
        "suggestion_history_hint_sub": "Browse your last 20 analyzed images.",
        "suggestion_dashboard_hint":   "Your dashboard has interesting stats",
        "suggestion_dashboard_hint_sub":"See totals, averages, and distribution at a glance.",
        "suggestion_settings_hint":    "Personalize the app",
        "suggestion_settings_hint_sub":"Change language, theme, font, and enable 2FA.",
        "suggestion_idle_help":        "Need a quick guide?",
        "suggestion_idle_help_sub":    "Replay the onboarding tour anytime.",
        "image_not_liver_warning": "This image does not look like a liver ultrasound.\n\nPlease insert a proper liver ultrasound image.",
        "result_not_confident": "The AI could not confidently identify this image as a liver ultrasound.\n\nPlease insert a proper liver ultrasound image and try again.",
        "not_liver_title": "Not a Liver Ultrasound",
        "not_liver_hint": "Please use a proper liver ultrasound (sonogram) image",
        # ✅ Voice Feedback keys
        "voice_welcome": "Welcome, {user}",
        "voice_goodbye": "Goodbye",
        "voice_logged_out": "Logged out successfully",
        "voice_image_loaded": "Image loaded successfully",
        "voice_image_rejected": "This is not a liver ultrasound image",
        "voice_inference_result": "{grid_name}, confidence {conf} percent",
        "voice_pdf_saved": "PDF report saved successfully",
        "voice_error": "An error occurred",
        "voice_settings_applied": "Settings applied",
        "voice_language_changed": "Language changed to {lang}",
        "voice_theme_changed": "Theme changed",
        "voice_2fa_enabled": "Two factor authentication enabled",
        "voice_2fa_disabled": "Two factor authentication disabled",
        "voice_locked": "Session locked",
        "voice_unlocked": "Session unlocked",
        "voice_processing": "Processing image",
        "voice_processing_done": "Processing complete",
        "voice_voice_enabled": "Voice feedback enabled",
        "voice_voice_disabled": "Voice feedback disabled",
    },
    "Persian": {
        "app_title": "هوش مصنوعی کبد، بابک",
        "welcome": "خوش آمدید",
        "login": "ورود",
        "signup": "ثبت نام",
        "logout": "خروج از حساب",
        "exit": "خروج",
        "settings": "تنظیمات",
        "account_info": "اطلاعات حساب",
        "username": "نام کاربری",
        "password": "رمز عبور",
        "repeat_password": "تکرار رمز عبور",
        "first_name": "نام",
        "last_name": "نام خانوادگی",
        "email": "ایمیل",
        "remember_me": "مرا به خاطر بسپار",
        "login_btn": "ورود",
        "signup_btn": "ثبت نام",
        "account": "حساب",
        "change_language": "تغییر زبان",
        "change_theme": "تغییر تم",
        "insert_sono": "درج تصویر",
        "process": "پردازش",
        "mute": "بی صدا",
        "unmute": "باصدا",
        "visit_webpage": "مشاهده وب‌سایت",
        "please_wait": "لطفاً صبر کنید...",
        "processing": "در حال پردازش تصویر...",
        "result": "نتیجه",
        "confidence": "اطمینان",
        "grid": "گرید",
        "insert_image_first": "لطفاً ابتدا یک تصویر وارد کنید!",
        "no_image": "تصویری وارد نشده",
        "invalid_credentials": "نام کاربری یا رمز اشتباه",
        "user_exists": "نام کاربری قبلاً وجود دارد",
        "password_requirements": "الزامات رمز عبور",
        "password_rule": (
            "رمز عبور باید:\n"
            "• ۸ تا ۱۶ کاراکتر باشد\n"
            "• حداقل ۱ حرف کوچک داشته باشد\n"
            "• حداقل ۱ حرف بزرگ داشته باشد\n"
            "• حداقل ۴ عدد داشته باشد\n"
            "• حداقل ۴ کاراکتر خاص داشته باشد"
        ),
        "passwords_dont_match": "رمزهای عبور مطابقت ندارند!",
        "invalid_email": "لطفاً یک ایمیل معتبر وارد کنید",
        "empty_fields": "لطفاً همه فیلدها را پر کنید",
        "signup_success": "ثبت نام موفق! حالا می‌توانید وارد شوید.",
        "logout_confirm": "آیا مطمئن هستید که می‌خواهید حساب خود را حذف کنید؟",
        "exit_confirm": "آیا مطمئن هستید که می‌خواهید خارج شوید؟",
        "theme_dark_aqua": "تم تاریک آبی-بنفش",
        "theme_solarized_white": "تم سفید سولاریزه",
        "theme_solarized_dark": "تم تیره سولاریزه",
        "theme_dark_forest": "تم جنگل سبز تیره",
        "select_language": "انتخاب زبان",
        "select_theme": "انتخاب تم",
        "apply": "اعمال",
        "cancel": "لغو",
        "save": "ذخیره",
        "success": "موفق",
        "error": "خطا",
        "warning": "هشدار",
        "settings_applied": "تنظیمات اعمال شد!",
        "grid_0": "گرید ۰ (NL) — کبد سالم",
        "grid_1": "گرید ۱ (G1) — درجه ۱",
        "grid_2": "گرید ۲ (G2) — درجه ۲",
        "grid_3": "گرید ۳ (G3) — درجه ۳",
        "too_many_attempts": "تلاش‌های ناموفق زیاد. لطفاً ۵ دقیقه صبر کنید.",
        "invalid_username": "نام کاربری باید ۳ تا ۳۲ کاراکتر باشد (حروف، اعداد، آندرلاین).",
        "welcome_user": "خوش آمدی، {user}!",
        "inference_failed": "پردازش ناموفق",
        "drop_here": "تصویر را اینجا رها کنید",
        "drop_subtitle": "PNG، JPG، JPEG، BMP، DCM",
        "drop_invalid": "فرمت فایل پشتیبانی نمی‌شود. لطفاً یک تصویر رها کنید.",
        "image_loaded": "تصویر با موفقیت بارگذاری شد",
        "help_about_title": "راهنما و درباره",
        "help_about_text": (
            "هوش مصنوعی کبد نسخه ۳.۰\n"
            "توسعه‌دهنده: بابک یوسفیان\n\n"
            "میانبرهای کیبورد:\n"
            "  Ctrl+O      — درج تصویر\n"
            "  Ctrl+Enter  — پردازش تصویر\n"
            "  Ctrl+L      — ورود\n"
            "  Ctrl+,      — تنظیمات\n"
            "  Esc         — بستن دیالوگ فعال\n"
            "  F1          — راهنما / درباره\n\n"
            "نکته: می‌توانید تصویر را در هر جای پنجره رها (Drag & Drop) کنید."
        ),
        # ✅ Dashboard keys
        "dashboard_title": "داشبورد",
        "dash_total": "کل تصاویر",
        "dash_avg_conf": "میانگین اطمینان",
        "dash_distribution": "توزیع",
        "dash_last_5": "۵ نتیجه‌ی آخر",
        "dash_no_results": "هنوز نتیجه‌ای نیست",
        "dash_clear_confirm": "همه‌ی آمار پاک شود؟",
        "dash_cleared": "آمار پاک شد",
        "dashboard_toggle": "باز/بستن داشبورد",
        # ✅ PDF keys
        "export_pdf": "خروجی PDF",
        "pdf_saved": "PDF با موفقیت ذخیره شد",
        "pdf_failed": "تولید PDF ناموفق بود",
        "pdf_subtitle": "گزارش تحلیل",
        "pdf_patient": "بیمار:",
        "pdf_date": "تاریخ:",
        "pdf_model": "مدل:",
        "pdf_diagnosis": "تشخیص",
        "pdf_confidence": "اطمینان",
        "pdf_class_probs": "احتمال کلاس‌ها",
        "pdf_disclaimer": (
            "این گزارش توسط بابک یوسفیان و تحت نظارت پزشکان مجرب تهیه شده است. "
            "این یک ابزار تحلیل حرفه‌ای است و نیازی به مراجعه به پزشک ندارد."
        ),
        "history_title": "تاریخچه",
        "history_toggle": "باز/بستن تاریخچه",
        "history_empty": "هنوز تاریخچه‌ای نیست",
        "history_cleared": "تاریخچه پاک شد",
        "history_clear_confirm": "همه‌ی تاریخچه پاک شود؟",
        # ============================================================
        # ✅ TOOLTIP KEYS
        # ============================================================
        "tooltip_title_login": "ورود",
        "tooltip_desc_login": "وارد حساب کاربری خود شوید",
        "tooltip_tip_login": "دسترسی به تاریخچه و آمار ذخیره‌شده",
        "tooltip_title_signup": "ثبت نام",
        "tooltip_desc_signup": "یک حساب کاربری جدید بسازید",
        "tooltip_title_account": "حساب کاربری",
        "tooltip_desc_account": "برای باز کردن منوی حساب کلیک کنید",
        "tooltip_desc_account_user": "وارد شده به عنوان {user}",
        "tooltip_title_dashboard": "داشبورد",
        "tooltip_desc_dashboard": "نمایش/مخفی کردن پنل آمار — کل، توزیع، نتایج اخیر",
        "tooltip_title_history": "تاریخچه",
        "tooltip_desc_history": "نمایش/مخفی کردن ۲۰ نتیجه‌ی آخر با تصاویر کوچک",
        "tooltip_title_visit": "مشاهده وب‌سایت",
        "tooltip_desc_visit": "باز کردن صفحه‌ی GitHub پروژه در مرورگر",
        "tooltip_tip_visit": "در مرورگر خارجی باز می‌شود",
        "tooltip_title_insert": "درج تصویر",
        "tooltip_desc_insert": "بارگذاری تصویر سونوگرافی برای تحلیل",
        "tooltip_tip_insert": "می‌توانید تصویر را در هر جای پنجره رها (Drag & Drop) کنید",
        "tooltip_title_process": "پردازش",
        "tooltip_desc_process": "اجرای تحلیل هوش مصنوعی روی تصویر بارگذاری‌شده",
        "tooltip_tip_process": "ابتدا باید یک تصویر بارگذاری شود",
        "tooltip_title_mute": "بی‌صدا کردن موسیقی",
        "tooltip_desc_mute": "توقف موسیقی پس‌زمینه",
        "tooltip_title_unmute": "باصدا کردن موسیقی",
        "tooltip_desc_unmute": "ادامه‌ی موسیقی پس‌زمینه",
        "tooltip_title_export": "خروجی PDF",
        "tooltip_desc_export": "ذخیره‌ی نتیجه‌ی فعلی به‌صورت گزارش PDF",
        "tooltip_tip_export": "شامل تصویر، تشخیص، و احتمالات کلاس‌ها",
        "tooltip_title_dash_clear": "پاک کردن آمار",
        "tooltip_desc_dash_clear": "بازنشانی همه‌ی آمار این حساب",
        "tooltip_tip_dash_clear": "این عمل قابل بازگشت نیست",
        "tooltip_title_dash_close": "بستن داشبورد",
        "tooltip_desc_dash_close": "مخفی کردن پنل آمار",
        "tooltip_desc_dash_total": "تعداد کل تصاویر پردازش‌شده",
        "tooltip_desc_dash_avg": "میانگین اطمینان روی همه‌ی نتایج",
        "tooltip_desc_dash_dist": "توزیع نتایج در بین درجات",
        "tooltip_title_hist_clear": "پاک کردن تاریخچه",
        "tooltip_desc_hist_clear": "حذف همه‌ی نتایج و تصاویر ذخیره‌شده",
        "tooltip_tip_hist_clear": "این کار همه‌ی فایل‌های تصویری را هم حذف می‌کند",
        "tooltip_title_hist_close": "بستن تاریخچه",
        "tooltip_desc_hist_close": "مخفی کردن پنل تاریخچه",
        "tooltip_tip_thumb_click": "برای بارگذاری مجدد این نتیجه کلیک کنید",
        "tooltip_tip_recent": "برای بارگذاری مجدد این نتیجه کلیک کنید",
        "tooltip_title_zoom_in": "بزرگ‌نمایی",
        "tooltip_desc_zoom_in": "افزایش بزرگ‌نمایی تصویر",
        "tooltip_title_zoom_out": "کوچک‌نمایی",
        "tooltip_desc_zoom_out": "کاهش بزرگ‌نمایی تصویر",
        "tooltip_title_zoom_reset": "بازنشانی نما",
        "tooltip_desc_zoom_reset": "جا دادن تصویر در پنجره",
        "tooltip_tip_zoom_reset": "می‌توانید روی تصویر دوبار کلیک کنید",
        # ✅ Notification keys
        "notif_inference_ok": "پردازش کامل شد",
        # ✅ Follow system theme keys
        "follow_system_theme": "پیروی از تم سیستم",
        "follow_system_theme_desc": "تغییر خودکار بین تم روشن و تیره بر اساس تنظیمات سیستم",
        # ✅ Grad-CAM keys
        "gradcam_title": "نقشه‌ی توجه",
        "gradcam_show": "نمایش نقشه‌ی توجه",
        "gradcam_toggle": "تغییر حالت",
        "gradcam_save": "ذخیره‌ی تصویر",
        "gradcam_close": "بستن",
        "gradcam_saved": "تصویر با موفقیت ذخیره شد",
        "gradcam_no_heatmap": "نقشه‌ی توجه موجود نیست. لطفاً ابتدا یک تصویر پردازش کنید.",
        "update_available": "به‌روزرسانی موجود است",
        "update_check_failed": "بررسی به‌روزرسانی ناموفق بود",
        "check_for_updates": "بررسی به‌روزرسانی",
        "check_for_updates_desc": "بررسی گیت‌هاب/وب‌سایت برای آخرین نسخه",
        "up_to_date": "شما به‌روز هستید",
        "you_have_latest": "شما آخرین نسخه را اجرا می‌کنید.",
        "check_internet": "لطفاً اتصال اینترنت خود را بررسی کنید و دوباره تلاش کنید.",
        "downloading": "در حال دانلود...",
        "verifying": "در حال بررسی...",
        "download_now": "اکنون دانلود کن",
        "install_now": "نصب و راه‌اندازی مجدد",
        "skip_version": "رد کردن این نسخه",
        "remind_later": "بعداً یادآوری کن",
        "whats_new": "چه چیز جدیدی هست",
        "no_changelog": "تغییرات ثبت نشده است.",
        "critical_update": "به‌روزرسانی حیاتی",
        "update_ready": "آماده‌ی نصب",
        "app_will_restart": "برنامه بسته و برای اعمال به‌روزرسانی دوباره باز می‌شود.",
        "install_failed": "نصب ناموفق بود. نسخه‌ی قبلی شما سالم است.",
        "download_failed": "دانلود ناموفق بود.",
        "source_update_requires_zip": "به‌روزرسانی منبع نیاز به فایل .zip دارد.",
        "session_locked": "نشست قفل شد",
        "locked_for_user": "قفل شده برای {user}",
        "unlock": "باز کردن قفل",
        "session_unlocked": "نشست باز شد",
        "auto_lock_notice": "نشست بعد از ۲ دقیقه بی‌تحرکی قفل می‌شود.",
        "audit_log_title": "گزارش فعالیت‌ها",
        "audit_export": "خروجی",
        "audit_exported": "با موفقیت صادر شد",
        "audit_export_failed": "خروجی ناموفق بود",
        "audit_search_placeholder": "جستجو در فعالیت‌ها یا جزئیات...",
        "audit_col_time": "زمان",
        "audit_col_level": "سطح",
        "audit_col_category": "دسته",
        "audit_col_action": "فعالیت",
        "audit_col_user": "کاربر",
        "audit_details_title": "جزئیات رویداد",
        "audit_status": "نمایش {count} از {total} رویداد",
        "strength_very_weak": "خیلی ضعیف",
        "strength_weak": "ضعیف",
        "strength_medium": "متوسط",
        "strength_strong": "قوی",
        "strength_very_strong": "خیلی قوی",

        "strength_rule_length": "۸ تا ۱۶ کاراکتر",
        "strength_rule_lowercase": "حرف کوچک",
        "strength_rule_uppercase": "حرف بزرگ",
        "strength_rule_digits": "عدد",
        "strength_rule_special": "کاراکتر خاص",

        "strength_add_chars": "{n} کاراکتر دیگر اضافه کنید",
        "strength_remove_chars": "{n} کاراکتر حذف کنید",
        "strength_need_digits": "{n} عدد دیگر اضافه کنید",
        "strength_need_special": "{n} کاراکتر خاص دیگر اضافه کنید",
        "strength_common_pattern": "از کلمات رایج اجتناب کنید",
        "strength_common_year": "از سال‌های رایج اجتناب کنید",
        "strength_has_sequence": "از دنباله‌ها (1234، abcd) اجتناب کنید",
        "strength_has_repeats": "از تکرارها (aaaa، 1111) اجتناب کنید",

        "passwords_match": "رمزهای عبور مطابقت دارند",
        "2fa_setup_title": "فعال‌سازی احراز هویت دو مرحله‌ای",
        "2fa_setup_subtitle": "کد QR را با Google Authenticator، Authy یا Microsoft Authenticator اسکن کنید.",
        "2fa_manual_secret": "یا این کد را دستی وارد کنید:",
        "2fa_enter_code": "کد ۶ رقمی برنامه را وارد کنید:",
        "2fa_verify_btn": "تأیید",
        "2fa_backup_codes_title": "این کدهای پشتیبان را ذخیره کنید!",
        "2fa_backup_codes_hint": "هر کد فقط یک‌بار قابل استفاده است. آن‌ها را جای امنی نگه دارید.",
        "2fa_done_btn": "تمام",
        "2fa_secret_copied": "کد در کلیپ‌بورد کپی شد",
        "2fa_invalid_code": "کد نامعتبر است. دوباره تلاش کنید.",
        "2fa_invalid_code_format": "کد باید ۶ رقم باشد.",
        "2fa_verify_title": "احراز هویت دو مرحله‌ای",
        "2fa_verify_subtitle": "کد ۶ رقمی برنامه Authenticator را وارد کنید.",
        "2fa_enter_code_hint": "لطفاً یک کد وارد کنید.",
        "2fa_use_backup_code": "استفاده از کد پشتیبان",
        "2fa_use_totp_code": "استفاده از کد Authenticator",
        "2fa_enter_backup_code": "یکی از کدهای پشتیبان را وارد کنید.",
        "2fa_section_title": "امنیت",
        "2fa_enabled_status": "احراز هویت دو مرحله‌ای فعال است",
        "2fa_disabled_status": "احراز هویت دو مرحله‌ای غیرفعال است",
        "2fa_enable_btn": "فعال‌سازی 2FA",
        "2fa_disable_btn": "غیرفعال‌سازی 2FA",
        "2fa_disable_title": "غیرفعال‌سازی 2FA",
        "2fa_disable_confirm": "آیا مطمئنید؟ حساب شما کمتر امن خواهد بود.",
        "2fa_disabled_success": "احراز هویت دو مرحله‌ای غیرفعال شد",
        "2fa_enabled_success": "احراز هویت دو مرحله‌ای فعال شد",
        "2fa_hint_title": "نحوه استفاده از این QR Code",
        "2fa_hint_step1": "یه اپلیکیشن Authenticator نصب کن: Google Authenticator، Microsoft Authenticator، یا Authy.",
        "2fa_hint_step2": "اپ رو باز کن، روی دکمه‌ی \"+\" بزن، بعد \"Scan QR code\" رو انتخاب کن.",
        "2fa_hint_step3": "دوربین گوشی رو بگیر سمت QR Code بالا (از دوربین خود اپ استفاده کن، نه دوربین پیش‌فرض گوشی).",
        "2fa_hint_step4": "اپ یه کد ۶ رقمی نشون می‌ده. اون رو توی کادر زیر وارد کن.",
        "2fa_manual_secret_hint": "نمی‌تونی اسکن کنی؟ کد زیر رو کپی کن و دستی توی اپ اضافه کن (\"Enter a setup key\").",
        "auto_lock_section_title": "قفل خودکار",
        "auto_lock_enable": "قفل کردن بعد از بی‌تحرکی",
        "auto_lock_timeout_label": "قفل کردن بعد از:",
        "auto_lock_warning_title": "هشدار قفل خودکار",
        "auto_lock_warning_body": "نشست شما به دلیل بی‌تحرکی تا {seconds} ثانیه‌ی دیگر قفل می‌شود.",
        "lock_now": "قفل کردن فوری",
        "font_section_title": "فونت",
        "font_family_label": "نوع فونت",
        "font_size_label": "اندازه‌ی پایه",
        "font_scale_label": "مقیاس نمایش",
        "font_scale_small": "کوچک",
        "font_scale_normal": "معمولی",
        "font_scale_large": "بزرگ",
        "font_scale_xlarge": "خیلی بزرگ",
        "font_scale_xxlarge": "عظیم",
        "font_preview_text": "این یک متن نمونه برای پیش‌نمایش فونت است — ۱۲۳۴۵۶۷۸۹۰",
        "font_reset": "بازگشت به پیش‌فرض",
        "font_applied": "فونت به‌روزرسانی شد",
        "motion_section_title": "حرکت و انیمیشن",
        "reduce_motion_label": "کاهش حرکت",
        "reduce_motion_hint": "اگه انیمیشن‌ها باعث ناراحتی می‌شن، خاموششون کن.",
        # ============================================================
        # ✅ EMPTY STATE KEYS
        # ============================================================
        "empty_history_empty_title": "هنوز تاریخچه‌ای نیست",
        "empty_history_empty_subtitle": "اولین تصویرت رو پردازش کن تا اینجا ببینی",
        "empty_history_empty_hint": "هر نتیجه به‌صورت خودکار ذخیره می‌شه",
        "empty_history_no_user_title": "برای دیدن تاریخچه وارد شو",
        "empty_history_no_user_subtitle": "نتایج ذخیره‌شده‌ت اینجا ظاهر می‌شن",
        "empty_history_no_user_hint": "تاریخچه برای هر حساب خصوصیه",
        "empty_dash_no_results_title": "هنوز نتیجه‌ای نیست",
        "empty_dash_no_results_subtitle": "تحلیل‌های اخیرت اینجا نمایش داده می‌شن",
        "empty_dash_no_results_hint": "برای شروع یه تصویر پردازش کن",
        "empty_dash_no_dist_title": "داده‌ی توزیعی نیست",
        "empty_dash_no_dist_subtitle": "با پردازش تصاویر، آمار پر می‌شه",
        "empty_dash_no_dist_hint": "هر تحلیل مهمه",
        "empty_image_empty_title": "تصویری وارد نشده",
        "empty_image_empty_subtitle": "تصویر رو اینجا رها کن، یا روی «درج تصویر» کلیک کن",
        "empty_image_empty_hint": "پشتیبانی از PNG، JPG، JPEG، BMP، DCM",
        "empty_image_drop_hint_title": "برای تحلیل رها کن",
        "empty_image_drop_hint_subtitle": "تصویر رو رها کن تا لود بشه",
        "empty_image_drop_hint_hint": "فرمت به‌صورت خودکار تشخیص داده می‌شه",
        "empty_processing_wait_title": "در حال پردازش...",
        "empty_processing_wait_subtitle": "هوش مصنوعی در حال تحلیل تصویره",
        "empty_processing_wait_hint": "معمولاً چند ثانیه طول می‌کشه",
        # ============================================================
        # ✅ ONBOARDING TOUR KEYS
        # ============================================================
        "tour_show_again": "نمایش مجدد تور",
        "tour_completed_msg": "تور کامل شد! آماده‌ای برای شروع.",
        "tour_skipped_msg": "تور رد شد. می‌تونی از منوی حساب دوباره ببینیش.",
        "tour_btn_next": "بعدی",
        "tour_btn_prev": "قبلی",
        "tour_btn_skip": "رد کردن",
        "tour_btn_finish": "پایان ✓",
        # Guest tour (2 steps)
        "tour_guest_login_title": "ورود به حساب",
        "tour_guest_login_subtitle": "برای ورود به حسابت و دسترسی به همه‌ی امکانات، اینجا کلیک کن.",
        "tour_guest_signup_title": "ساخت حساب جدید",
        "tour_guest_signup_subtitle": "تازه‌واردی؟ اینجا کلیک کن تا در چند ثانیه حساب بسازی.",
        # Full tour
        "tour_insert_title": "درج تصویر",
        "tour_insert_subtitle": "یه تصویر سونوگرافی از دیسک بارگذاری کن، یا هر جای پنجره رهاش کن.",
        "tour_process_title": "پردازش تصویر",
        "tour_process_subtitle": "تحلیل هوش مصنوعی روی تصویر بارگذاری‌شده رو اجرا کن. نتیجه در چند ثانیه ظاهر می‌شه.",
        "tour_export_title": "خروجی PDF",
        "tour_export_subtitle": "نتیجه‌ی فعلی رو به‌صورت یه گزارش PDF پزشکی حرفه‌ای ذخیره کن.",
        "tour_gradcam_title": "نقشه‌ی توجه",
        "tour_gradcam_subtitle": "ببین هوش مصنوعی کجای تصویر رو نگاه می‌کنه — یه نقشه‌ی حرارتی روی تصویر.",
        "tour_dashboard_title": "داشبورد",
        "tour_dashboard_subtitle": "پنل آمار رو باز/بسته کن: کل تصاویر، میانگین اطمینان، و توزیع درجات.",
        "tour_history_title": "تاریخچه",
        "tour_history_subtitle": "۲۰ تصویر آخر تحلیل‌شده رو مرور کن. روی هر thumbnail کلیک کن تا نتیجه‌ش رو دوباره ببینی.",
        "tour_visit_title": "مشاهده وب‌سایت",
        "tour_visit_subtitle": "صفحه‌ی GitHub پروژه رو در مرورگر باز کن برای مستندات و آپدیت‌ها.",
        "tour_mute_title": "بی‌صدا کردن موسیقی",
        "tour_mute_subtitle": "هر زمان خواستی موسیقی پس‌زمینه رو متوقف یا از سر بگیر.",
        "tour_zoom_title": "کنترل زوم",
        "tour_zoom_subtitle": "با چرخ ماوس زوم کن، با درگ جابه‌جا کن، و با دابل‌کلیک ریست کن.",
        "tour_account_title": "منوی حساب",
        "tour_account_subtitle": "اینجا کلیک کن تا به تنظیمات، اطلاعات حساب، گزارش فعالیت‌ها، و بیشتر دسترسی داشته باشی.",
        # ✅ Smart Suggestions
        "suggestion_image_ready":      "تصویر آماده‌ست، پردازش کن؟",
        "suggestion_image_ready_sub":  "روی پردازش بزن تا تحلیل AI اجرا بشه.",
        "suggestion_pdf_ready":        "خروجی PDF بگیر؟",
        "suggestion_pdf_ready_sub":    "گزارش پزشکی حرفه‌ای رو با یه کلیک ذخیره کن.",
        "suggestion_gradcam_hint":     "نقشه‌ی توجه رو ببین؟",
        "suggestion_gradcam_hint_sub": "ببین AI کجای تصویر رو نگاه کرده.",
        "suggestion_repeated_grid":    "همیشه {grid} می‌گیری",
        "suggestion_repeated_grid_sub":"داشبورد رو باز کن تا توزیع رو ببینی.",
        "suggestion_history_hint":     "تاریخچه‌ت رو ببین",
        "suggestion_history_hint_sub": "۲۰ تصویر آخر تحلیل‌شده‌ت رو مرور کن.",
        "suggestion_dashboard_hint":   "داشبورد آمار جالبی داره",
        "suggestion_dashboard_hint_sub":"مجموع، میانگین و توزیع رو یه نگاه ببین.",
        "suggestion_settings_hint":    "تنظیمات رو شخصی‌سازی کن",
        "suggestion_settings_hint_sub":"زبان، تم، فونت رو عوض کن و 2FA رو فعال کن.",
        "suggestion_idle_help":        "راهنمای سریع می‌خوای؟",
        "suggestion_idle_help_sub":    "هر وقت خواستی تور آنبوردینگ رو دوباره ببین.",
        "image_not_liver_warning": "این تصویر شبیه سونوگرافی کبد نیست.\n\nلطفاً یک تصویر سونوگرافی کبد وارد کنید.",
        "result_not_confident": "هوش مصنوعی نتوانست با اطمینان این تصویر را به عنوان سونوگرافی کبد شناسایی کند.\n\nلطفاً یک تصویر سونوگرافی کبد وارد کنید و دوباره تلاش کنید.",
        "not_liver_title": "سونوگرافی کبد نیست",
        "not_liver_hint": "لطفاً از یک تصویر سونوگرافی کبد استفاده کنید",
        "voice_welcome": "خوش آمدی، {user}",
        "voice_goodbye": "خداحافظ",
        "voice_logged_out": "با موفقیت خارج شدید",
        "voice_image_loaded": "تصویر با موفقیت بارگذاری شد",
        "voice_image_rejected": "این تصویر سونوگرافی کبد نیست",
        "voice_inference_result": "{grid_name}، اطمینان {conf} درصد",
        "voice_pdf_saved": "گزارش PDF با موفقیت ذخیره شد",
        "voice_error": "خطایی رخ داد",
        "voice_settings_applied": "تنظیمات اعمال شد",
        "voice_language_changed": "زبان به {lang} تغییر کرد",
        "voice_theme_changed": "تم تغییر کرد",
        "voice_2fa_enabled": "احراز هویت دو مرحله‌ای فعال شد",
        "voice_2fa_disabled": "احراز هویت دو مرحله‌ای غیرفعال شد",
        "voice_locked": "نشست قفل شد",
        "voice_unlocked": "نشست باز شد",
        "voice_processing": "در حال پردازش تصویر",
        "voice_processing_done": "پردازش کامل شد",
        "voice_voice_enabled": "صداگذاری فعال شد",
        "voice_voice_disabled": "صداگذاری غیرفعال شد",
    },
    "USA": {
        "app_title": "Liver AI-babak",
        "welcome": "Welcome",
        "login": "Login",
        "signup": "Sign Up",
        "logout": "Logout",
        "exit": "Exit",
        "settings": "Settings",
        "account_info": "Account Info",
        "username": "Username",
        "password": "Password",
        "repeat_password": "Repeat Password",
        "first_name": "First Name",
        "last_name": "Last Name",
        "email": "Email Address",
        "remember_me": "Remember Me",
        "login_btn": "Login",
        "signup_btn": "Sign Up",
        "account": "Account",
        "change_language": "Change Language",
        "change_theme": "Change Theme",
        "insert_sono": "Insert Sono",
        "process": "Process",
        "mute": "Mute",
        "unmute": "Unmute",
        "visit_webpage": "Visit Webpage",
        "please_wait": "Please wait...",
        "processing": "Processing image...",
        "result": "Result",
        "confidence": "Confidence",
        "grid": "Grid",
        "insert_image_first": "Please insert an image first!",
        "no_image": "No image inserted",
        "invalid_credentials": "Invalid credentials",
        "user_exists": "Username already exists",
        "password_requirements": "Password requirements",
        "password_rule": (
            "Password must be:\n"
            "• 8-16 characters long\n"
            "• Contains at least 1 lowercase letter\n"
            "• Contains at least 1 uppercase letter\n"
            "• Contains at least 4 digits\n"
            "• Contains at least 4 special characters"
        ),
        "passwords_dont_match": "Passwords do not match!",
        "invalid_email": "Please enter a valid email address",
        "empty_fields": "Please fill in all fields",
        "signup_success": "Signup successful! You can now login.",
        "logout_confirm": "Are you sure you want to delete your account?",
        "exit_confirm": "Are you sure you want to exit?",
        "theme_dark_aqua": "Theme Dark Aqua-Purple",
        "theme_solarized_white": "Theme Solarized White",
        "theme_solarized_dark": "Theme Solarized Dark",
        "theme_dark_forest": "Theme Dark Forest Green",
        "select_language": "Select Language",
        "select_theme": "Select Theme",
        "apply": "Apply",
        "cancel": "Cancel",
        "save": "Save",
        "success": "Success",
        "error": "Error",
        "warning": "Warning",
        "settings_applied": "Settings applied!",
        "grid_0": "Grid 0 (NL) — Normal Liver",
        "grid_1": "Grid 1 (G1) — Grade 1",
        "grid_2": "Grid 2 (G2) — Grade 2",
        "grid_3": "Grid 3 (G3) — Grade 3",
        "too_many_attempts": "Too many failed attempts. Please wait 5 minutes.",
        "invalid_username": "Username must be 3-32 chars (letters, numbers, underscore only).",
        "welcome_user": "Welcome, {user}!",
        "inference_failed": "Inference failed",
        "drop_here": "Drop image here",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Unsupported file type. Please drop an image.",
        "image_loaded": "Image loaded successfully",
        "help_about_title": "Help & About",
        "help_about_text": (
            "Liver AI Pipeline v3.0\n"
            "Developed by Babak Yousefian\n\n"
            "Keyboard Shortcuts:\n"
            "  Ctrl+O      — Insert Image\n"
            "  Ctrl+Enter  — Process Image\n"
            "  Ctrl+L      — Login\n"
            "  Ctrl+,      — Settings\n"
            "  Esc         — Close current dialog\n"
            "  F1          — Help / About\n\n"
            "Tip: You can also drag & drop an image anywhere on the window."
        ),
        "dashboard_title": "Dashboard",
        "dash_total": "Total Images",
        "dash_avg_conf": "Avg. Confidence",
        "dash_distribution": "Distribution",
        "dash_last_5": "Last 5 Results",
        "dash_no_results": "No results yet",
        "dash_clear_confirm": "Clear all statistics?",
        "dash_cleared": "Statistics cleared",
        "dashboard_toggle": "Toggle Dashboard",
        "export_pdf": "Export PDF",
        "pdf_saved": "PDF saved successfully",
        "pdf_failed": "PDF generation failed",
        "pdf_subtitle": "Analysis Report",
        "pdf_patient": "Patient:",
        "pdf_date": "Date:",
        "pdf_model": "Model:",
        "pdf_diagnosis": "DIAGNOSIS",
        "pdf_confidence": "Confidence",
        "pdf_class_probs": "CLASS PROBABILITIES",
        "pdf_disclaimer": (
            "This report was developed by Babak Yousefian under the supervision of experienced physicians. "
            "It is a professional analysis tool and does not require referral to a physician."
        ),
        "history_title": "History",
        "history_toggle": "Toggle History",
        "history_empty": "No history yet",
        "history_cleared": "History cleared",
        "history_clear_confirm": "Clear all history?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Login",
        "tooltip_desc_login": "Sign in to your account",
        "tooltip_tip_login": "Access your saved history and stats",
        "tooltip_title_signup": "Sign Up",
        "tooltip_desc_signup": "Create a new account",
        "tooltip_title_account": "Account",
        "tooltip_desc_account": "Click to open account menu",
        "tooltip_desc_account_user": "Logged in as {user}",
        "tooltip_title_dashboard": "Dashboard",
        "tooltip_desc_dashboard": "Show/hide stats panel — total, distribution, recent results",
        "tooltip_title_history": "History",
        "tooltip_desc_history": "Show/hide last 20 results with thumbnails",
        "tooltip_title_visit": "Visit Webpage",
        "tooltip_desc_visit": "Open the project's GitHub page in your browser",
        "tooltip_tip_visit": "Opens in external browser",
        "tooltip_title_insert": "Insert Image",
        "tooltip_desc_insert": "Load a sono image to analyze",
        "tooltip_tip_insert": "You can also drag & drop an image anywhere on the window",
        "tooltip_title_process": "Process",
        "tooltip_desc_process": "Run AI analysis on the loaded image",
        "tooltip_tip_process": "Requires an image to be loaded first",
        "tooltip_title_mute": "Mute Music",
        "tooltip_desc_mute": "Pause the background music",
        "tooltip_title_unmute": "Unmute Music",
        "tooltip_desc_unmute": "Resume the background music",
        "tooltip_title_export": "Export PDF",
        "tooltip_desc_export": "Save the current result as a PDF report",
        "tooltip_tip_export": "Includes image, diagnosis, and class probabilities",
        "tooltip_title_dash_clear": "Clear Stats",
        "tooltip_desc_dash_clear": "Reset all statistics for this account",
        "tooltip_tip_dash_clear": "This action cannot be undone",
        "tooltip_title_dash_close": "Close Dashboard",
        "tooltip_desc_dash_close": "Hide the stats panel",
        "tooltip_desc_dash_total": "Total number of images processed",
        "tooltip_desc_dash_avg": "Average confidence across all results",
        "tooltip_desc_dash_dist": "Distribution of results across grades",
        "tooltip_title_hist_clear": "Clear History",
        "tooltip_desc_hist_clear": "Delete all saved results and thumbnails",
        "tooltip_tip_hist_clear": "This will delete all image files too",
        "tooltip_title_hist_close": "Close History",
        "tooltip_desc_hist_close": "Hide the history panel",
        "tooltip_tip_thumb_click": "Click to reload this result",
        "tooltip_tip_recent": "Click to reload this result",
        "tooltip_title_zoom_in": "Zoom In",
        "tooltip_desc_zoom_in": "Increase image magnification",
        "tooltip_title_zoom_out": "Zoom Out",
        "tooltip_desc_zoom_out": "Decrease image magnification",
        "tooltip_title_zoom_reset": "Reset View",
        "tooltip_desc_zoom_reset": "Fit the image to the window",
        "tooltip_tip_zoom_reset": "You can also double-click the image",
        "notif_inference_ok": "Inference complete",
        "follow_system_theme": "Follow System Theme",
        "follow_system_theme_desc": "Automatically switch between light and dark based on system settings",
        "gradcam_title": "Attention Heatmap",
        "gradcam_show": "Show Attention Map",
        "gradcam_toggle": "Toggle Heatmap",
        "gradcam_save": "Save Image",
        "gradcam_close": "Close",
        "gradcam_saved": "Image saved successfully",
        "gradcam_no_heatmap": "No attention map available. Please process an image first.",
        "update_available": "Update Available",
        "update_check_failed": "Update Check Failed",
        "check_for_updates": "Check for Updates",
        "check_for_updates_desc": "Check GitHub/website for the latest version",
        "up_to_date": "You're Up to Date",
        "you_have_latest": "You are running the latest version.",
        "check_internet": "Please check your internet connection and try again.",
        "downloading": "Downloading...",
        "verifying": "Verifying...",
        "download_now": "Download Now",
        "install_now": "Install & Restart",
        "skip_version": "Skip This Version",
        "remind_later": "Remind Me Later",
        "whats_new": "What's New",
        "no_changelog": "No changelog provided.",
        "critical_update": "Critical Update",
        "update_ready": "Ready to Install",
        "app_will_restart": "The app will close and restart to apply the update.",
        "install_failed": "Installation failed. Your old version is safe.",
        "download_failed": "Download failed.",
        "source_update_requires_zip": "Source update requires a .zip file.",
        "session_locked": "Session Locked",
        "locked_for_user": "Locked for {user}",
        "unlock": "Unlock",
        "session_unlocked": "Session unlocked",
        "auto_lock_notice": "Session will lock after 2 minutes of inactivity.",
        "audit_log_title": "Audit Log",
        "audit_export": "Export",
        "audit_exported": "Exported successfully",
        "audit_export_failed": "Export failed",
        "audit_search_placeholder": "Search actions or details...",
        "audit_col_time": "Time",
        "audit_col_level": "Level",
        "audit_col_category": "Category",
        "audit_col_action": "Action",
        "audit_col_user": "User",
        "audit_details_title": "Event Details",
        "audit_status": "Showing {count} of {total} events",
        "strength_very_weak": "Very Weak",
        "strength_weak": "Weak",
        "strength_medium": "Medium",
        "strength_strong": "Strong",
        "strength_very_strong": "Very Strong",
        "strength_rule_length": "8-16 characters",
        "strength_rule_lowercase": "Lowercase letter",
        "strength_rule_uppercase": "Uppercase letter",
        "strength_rule_digits": "Digits",
        "strength_rule_special": "Special characters",
        "strength_add_chars": "Add {n} more characters",
        "strength_remove_chars": "Remove {n} characters",
        "strength_need_digits": "Add {n} more digit(s)",
        "strength_need_special": "Add {n} more special character(s)",
        "strength_common_pattern": "Avoid common words",
        "strength_common_year": "Avoid common years",
        "strength_has_sequence": "Avoid sequences (1234, abcd)",
        "strength_has_repeats": "Avoid repeats (aaaa, 1111)",
        "passwords_match": "Passwords match",
        "2fa_setup_title": "Enable Two-Factor Authentication",
        "2fa_setup_subtitle": "Scan the QR code with Google Authenticator, Authy, or Microsoft Authenticator.",
        "2fa_manual_secret": "Or enter this code manually:",
        "2fa_enter_code": "Enter the 6-digit code from your app:",
        "2fa_verify_btn": "Verify",
        "2fa_backup_codes_title": "Save these backup codes!",
        "2fa_backup_codes_hint": "Each code can be used once. Store them somewhere safe.",
        "2fa_done_btn": "Done",
        "2fa_secret_copied": "Secret copied to clipboard",
        "2fa_invalid_code": "Invalid code. Please try again.",
        "2fa_invalid_code_format": "Code must be 6 digits.",
        "2fa_verify_title": "Two-Factor Authentication",
        "2fa_verify_subtitle": "Enter the 6-digit code from your authenticator app.",
        "2fa_enter_code_hint": "Please enter a code.",
        "2fa_use_backup_code": "Use a backup code instead",
        "2fa_use_totp_code": "Use authenticator code instead",
        "2fa_enter_backup_code": "Enter one of your backup codes.",
        "2fa_section_title": "Security",
        "2fa_enabled_status": "Two-Factor Authentication is enabled",
        "2fa_disabled_status": "Two-Factor Authentication is disabled",
        "2fa_enable_btn": "Enable 2FA",
        "2fa_disable_btn": "Disable 2FA",
        "2fa_disable_title": "Disable 2FA",
        "2fa_disable_confirm": "Are you sure? Your account will be less secure.",
        "2fa_disabled_success": "Two-Factor Authentication disabled",
        "2fa_enabled_success": "Two-Factor Authentication enabled",
        "2fa_hint_title": "How to use this QR code",
        "2fa_hint_step1": "Install an authenticator app: Google Authenticator, Microsoft Authenticator, or Authy.",
        "2fa_hint_step2": "Open the app and tap the \"+\" button, then choose \"Scan QR code\".",
        "2fa_hint_step3": "Point your phone's camera at the QR code above (use the app's camera, not your phone's camera app).",
        "2fa_hint_step4": "The app will show a 6-digit code. Enter it in the box below.",
        "2fa_manual_secret_hint": "Can't scan? Copy the key below and add it manually in your app (\"Enter a setup key\").",
        "auto_lock_section_title": "Auto-Lock",
        "auto_lock_enable": "Lock after inactivity",
        "auto_lock_timeout_label": "Lock after:",
        "auto_lock_warning_title": "Auto-Lock Warning",
        "auto_lock_warning_body": "Your session will be locked in {seconds} seconds due to inactivity.",
        "lock_now": "Lock Now",
        "font_section_title": "Font",
        "font_family_label": "Font family",
        "font_size_label": "Base size",
        "font_scale_label": "Display scale",
        "font_scale_small": "Small",
        "font_scale_normal": "Normal",
        "font_scale_large": "Large",
        "font_scale_xlarge": "Extra Large",
        "font_scale_xxlarge": "Huge",
        "font_preview_text": "The quick brown fox jumps over the lazy dog — 1234567890",
        "font_reset": "Reset to Default",
        "font_applied": "Font updated",
        "motion_section_title": "Motion & Animations",
        "reduce_motion_label": "Reduce motion",
        "reduce_motion_hint": "Disable animations if they make you uncomfortable.",
        "empty_history_empty_title": "No history yet",
        "empty_history_empty_subtitle": "Process your first image to see it here",
        "empty_history_empty_hint": "Each result is saved automatically",
        "empty_history_no_user_title": "Sign in to view history",
        "empty_history_no_user_subtitle": "Your saved results will appear here",
        "empty_history_no_user_hint": "History is private to each account",
        "empty_dash_no_results_title": "No results yet",
        "empty_dash_no_results_subtitle": "Your recent analyses will appear here",
        "empty_dash_no_results_hint": "Process an image to get started",
        "empty_dash_no_dist_title": "No distribution data",
        "empty_dash_no_dist_subtitle": "Stats will populate as you process images",
        "empty_dash_no_dist_hint": "Every analysis counts",
        "empty_image_empty_title": "No image inserted",
        "empty_image_empty_subtitle": "Drag & drop an image here, or click Insert",
        "empty_image_empty_hint": "Supports PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Drop to analyze",
        "empty_image_drop_hint_subtitle": "Release the image to load it",
        "empty_image_drop_hint_hint": "We'll auto-detect the format",
        "empty_processing_wait_title": "Processing...",
        "empty_processing_wait_subtitle": "AI is analyzing your image",
        "empty_processing_wait_hint": "This usually takes a few seconds",
        "tour_show_again": "Show Tour Again",
        "tour_completed_msg": "Tour completed! You're ready to go.",
        "tour_skipped_msg": "Tour skipped. You can replay it from Account menu.",
        "tour_btn_next": "Next",
        "tour_btn_prev": "Back",
        "tour_btn_skip": "Skip",
        "tour_btn_finish": "Finish ✓",
        "tour_guest_login_title": "Sign In",
        "tour_guest_login_subtitle": "Click here to sign in to your account and unlock the full experience.",
        "tour_guest_signup_title": "Create an Account",
        "tour_guest_signup_subtitle": "New here? Click here to create an account in seconds.",
        "tour_insert_title": "Insert Image",
        "tour_insert_subtitle": "Load a sono image from disk, or drag & drop it anywhere on the window.",
        "tour_process_title": "Process Image",
        "tour_process_subtitle": "Run the AI analysis on the loaded image. Results appear in a few seconds.",
        "tour_export_title": "Export PDF",
        "tour_export_subtitle": "Save the current result as a professional PDF medical report.",
        "tour_gradcam_title": "Attention Heatmap",
        "tour_gradcam_subtitle": "Visualize where the AI is looking — a heatmap over the analyzed image.",
        "tour_dashboard_title": "Dashboard",
        "tour_dashboard_subtitle": "Toggle the stats panel: total images, average confidence, and grade distribution.",
        "tour_history_title": "History",
        "tour_history_subtitle": "Browse your last 20 analyzed images. Click any thumbnail to reload its result.",
        "tour_visit_title": "Visit Webpage",
        "tour_visit_subtitle": "Open the project's GitHub page in your browser for docs and updates.",
        "tour_mute_title": "Mute Music",
        "tour_mute_subtitle": "Pause or resume the background music at any time.",
        "tour_zoom_title": "Zoom Controls",
        "tour_zoom_subtitle": "Use the mouse wheel to zoom, drag to pan, and double-click to reset.",
        "tour_account_title": "Account Menu",
        "tour_account_subtitle": "Click here to access Settings, Account Info, Audit Log, and more.",
        "suggestion_image_ready":      "Image loaded — ready to process",
        "suggestion_image_ready_sub":  "Click Process to run the AI analysis.",
        "suggestion_pdf_ready":        "Result ready — export as PDF?",
        "suggestion_pdf_ready_sub":    "Save a professional medical report in one click.",
        "suggestion_gradcam_hint":     "Curious what the AI saw?",
        "suggestion_gradcam_hint_sub": "See the attention heatmap over the image.",
        "suggestion_repeated_grid":    "You keep getting {grid}",
        "suggestion_repeated_grid_sub":"Open the dashboard to see the distribution.",
        "suggestion_history_hint":     "Your history is building up",
        "suggestion_history_hint_sub": "Browse your last 20 analyzed images.",
        "suggestion_dashboard_hint":   "Your dashboard has interesting stats",
        "suggestion_dashboard_hint_sub":"See totals, averages, and distribution at a glance.",
        "suggestion_settings_hint":    "Personalize the app",
        "suggestion_settings_hint_sub":"Change language, theme, font, and enable 2FA.",
        "suggestion_idle_help":        "Need a quick guide?",
        "suggestion_idle_help_sub":    "Replay the onboarding tour anytime.",
        "image_not_liver_warning": "This image does not look like a liver ultrasound.\n\nPlease insert a proper liver ultrasound image.",
        "result_not_confident": "The AI could not confidently identify this image as a liver ultrasound.\n\nPlease insert a proper liver ultrasound image and try again.",
        "not_liver_title": "Not a Liver Ultrasound",
        "not_liver_hint": "Please use a proper liver ultrasound (sonogram) image",
        # ✅ Voice Feedback keys
        "voice_welcome": "Welcome, {user}",
        "voice_goodbye": "Goodbye",
        "voice_logged_out": "Logged out successfully",
        "voice_image_loaded": "Image loaded successfully",
        "voice_image_rejected": "This is not a liver ultrasound image",
        "voice_inference_result": "{grid_name}, confidence {conf} percent",
        "voice_pdf_saved": "PDF report saved successfully",
        "voice_error": "An error occurred",
        "voice_settings_applied": "Settings applied",
        "voice_language_changed": "Language changed to {lang}",
        "voice_theme_changed": "Theme changed",
        "voice_2fa_enabled": "Two factor authentication enabled",
        "voice_2fa_disabled": "Two factor authentication disabled",
        "voice_locked": "Session locked",
        "voice_unlocked": "Session unlocked",
        "voice_processing": "Processing image",
        "voice_processing_done": "Processing complete",
        "voice_voice_enabled": "Voice feedback enabled",
        "voice_voice_disabled": "Voice feedback disabled",
    },
    "France": {
        "app_title": "Pipeline IA du Foie",
        "welcome": "Bienvenue",
        "login": "Connexion",
        "signup": "Inscription",
        "logout": "Déconnexion",
        "exit": "Quitter",
        "settings": "Paramètres",
        "account_info": "Informations du compte",
        "username": "Nom d'utilisateur",
        "password": "Mot de passe",
        "repeat_password": "Répéter le mot de passe",
        "first_name": "Prénom",
        "last_name": "Nom",
        "email": "Adresse e-mail",
        "remember_me": "Se souvenir de moi",
        "login_btn": "Connexion",
        "signup_btn": "Inscription",
        "account": "Compte",
        "change_language": "Changer la langue",
        "change_theme": "Changer le thème",
        "insert_sono": "Insérer une écho",
        "process": "Traiter",
        "mute": "Muet",
        "unmute": "Activer le son",
        "visit_webpage": "Visiter le site",
        "please_wait": "Veuillez patienter...",
        "processing": "Traitement de l'image...",
        "result": "Résultat",
        "confidence": "Confiance",
        "grid": "Grille",
        "insert_image_first": "Veuillez d'abord insérer une image !",
        "no_image": "Aucune image insérée",
        "invalid_credentials": "Identifiants invalides",
        "user_exists": "Le nom d'utilisateur existe déjà",
        "password_requirements": "Exigences du mot de passe",
        "password_rule": (
            "Le mot de passe doit :\n"
            "• Contenir 8 à 16 caractères\n"
            "• Contenir au moins 1 lettre minuscule\n"
            "• Contenir au moins 1 lettre majuscule\n"
            "• Contenir au moins 4 chiffres\n"
            "• Contenir au moins 4 caractères spéciaux"
        ),
        "passwords_dont_match": "Les mots de passe ne correspondent pas !",
        "invalid_email": "Veuillez saisir une adresse e-mail valide",
        "empty_fields": "Veuillez remplir tous les champs",
        "signup_success": "Inscription réussie ! Vous pouvez maintenant vous connecter.",
        "logout_confirm": "Voulez-vous vraiment supprimer votre compte ?",
        "exit_confirm": "Voulez-vous vraiment quitter ?",
        "theme_dark_aqua": "Thème Aqua-Violet sombre",
        "theme_solarized_white": "Thème Solarized clair",
        "theme_solarized_dark": "Thème Solarized sombre",
        "theme_dark_forest": "Thème Forêt Verte Sombre",
        "select_language": "Choisir la langue",
        "select_theme": "Choisir le thème",
        "apply": "Appliquer",
        "cancel": "Annuler",
        "save": "Enregistrer",
        "success": "Succès",
        "error": "Erreur",
        "warning": "Avertissement",
        "settings_applied": "Paramètres appliqués !",
        "grid_0": "Grille 0 (NL) — Foie normal",
        "grid_1": "Grille 1 (G1) — Grade 1",
        "grid_2": "Grille 2 (G2) — Grade 2",
        "grid_3": "Grille 3 (G3) — Grade 3",
        "too_many_attempts": "Trop de tentatives échouées. Veuillez attendre 5 minutes.",
        "invalid_username": "Le nom d'utilisateur doit comporter 3 à 32 caractères (lettres, chiffres, underscore).",
        "welcome_user": "Bienvenue, {user} !",
        "inference_failed": "Échec de l'inférence",
        "drop_here": "Déposez l'image ici",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Type de fichier non pris en charge. Veuillez déposer une image.",
        "image_loaded": "Image chargée avec succès",
        "help_about_title": "Aide & À propos",
        "help_about_text": (
            "Pipeline IA du Foie v3.0\n"
            "Développé par Babak Yousefian\n\n"
            "Raccourcis clavier :\n"
            "  Ctrl+O      — Insérer une image\n"
            "  Ctrl+Enter  — Traiter l'image\n"
            "  Ctrl+L      — Connexion\n"
            "  Ctrl+,      — Paramètres\n"
            "  Esc         — Fermer la boîte de dialogue\n"
            "  F1          — Aide / À propos\n\n"
            "Astuce : Vous pouvez également glisser-déposer une image n'importe où sur la fenêtre."
        ),
        "dashboard_title": "Tableau de bord",
        "dash_total": "Total des images",
        "dash_avg_conf": "Confiance moy.",
        "dash_distribution": "Répartition",
        "dash_last_5": "5 derniers résultats",
        "dash_no_results": "Aucun résultat pour l'instant",
        "dash_clear_confirm": "Effacer toutes les statistiques ?",
        "dash_cleared": "Statistiques effacées",
        "dashboard_toggle": "Afficher/Masquer le tableau de bord",
        "export_pdf": "Exporter PDF",
        "pdf_saved": "PDF enregistré avec succès",
        "pdf_failed": "Échec de la génération PDF",
        "pdf_subtitle": "Rapport d'analyse",
        "pdf_patient": "Patient :",
        "pdf_date": "Date :",
        "pdf_model": "Modèle :",
        "pdf_diagnosis": "DIAGNOSTIC",
        "pdf_confidence": "Confiance",
        "pdf_class_probs": "PROBABILITÉS PAR CLASSE",
        "pdf_disclaimer": (
            "Ce rapport a été développé par Babak Yousefian sous la supervision de médecins expérimentés. "
            "Il s'agit d'un outil d'analyse professionnel et ne nécessite pas de consultation médicale."
        ),
        "history_title": "Historique",
        "history_toggle": "Afficher/Masquer l'historique",
        "history_empty": "Aucun historique pour l'instant",
        "history_cleared": "Historique effacé",
        "history_clear_confirm": "Effacer tout l'historique ?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Connexion",
        "tooltip_desc_login": "Connectez-vous à votre compte",
        "tooltip_tip_login": "Accédez à votre historique et vos statistiques",
        "tooltip_title_signup": "Inscription",
        "tooltip_desc_signup": "Créer un nouveau compte",
        "tooltip_title_account": "Compte",
        "tooltip_desc_account": "Cliquez pour ouvrir le menu du compte",
        "tooltip_desc_account_user": "Connecté en tant que {user}",
        "tooltip_title_dashboard": "Tableau de bord",
        "tooltip_desc_dashboard": "Afficher/masquer le panneau de stats — total, répartition, résultats récents",
        "tooltip_title_history": "Historique",
        "tooltip_desc_history": "Afficher/masquer les 20 derniers résultats avec vignettes",
        "tooltip_title_visit": "Visiter le site",
        "tooltip_desc_visit": "Ouvrir la page GitHub du projet dans votre navigateur",
        "tooltip_tip_visit": "S'ouvre dans le navigateur externe",
        "tooltip_title_insert": "Insérer une image",
        "tooltip_desc_insert": "Charger une image d'écho à analyser",
        "tooltip_tip_insert": "Vous pouvez aussi glisser-déposer une image n'importe où sur la fenêtre",
        "tooltip_title_process": "Traiter",
        "tooltip_desc_process": "Lancer l'analyse IA sur l'image chargée",
        "tooltip_tip_process": "Nécessite d'abord une image chargée",
        "tooltip_title_mute": "Couper la musique",
        "tooltip_desc_mute": "Mettre en pause la musique de fond",
        "tooltip_title_unmute": "Activer la musique",
        "tooltip_desc_unmute": "Reprendre la musique de fond",
        "tooltip_title_export": "Exporter PDF",
        "tooltip_desc_export": "Enregistrer le résultat actuel au format PDF",
        "tooltip_tip_export": "Inclut l'image, le diagnostic et les probabilités par classe",
        "tooltip_title_dash_clear": "Effacer les stats",
        "tooltip_desc_dash_clear": "Réinitialiser toutes les statistiques de ce compte",
        "tooltip_tip_dash_clear": "Cette action est irréversible",
        "tooltip_title_dash_close": "Fermer le tableau",
        "tooltip_desc_dash_close": "Masquer le panneau de stats",
        "tooltip_desc_dash_total": "Nombre total d'images traitées",
        "tooltip_desc_dash_avg": "Confiance moyenne sur tous les résultats",
        "tooltip_desc_dash_dist": "Répartition des résultats par grade",
        "tooltip_title_hist_clear": "Effacer l'historique",
        "tooltip_desc_hist_clear": "Supprimer tous les résultats et vignettes enregistrés",
        "tooltip_tip_hist_clear": "Cela supprimera aussi tous les fichiers image",
        "tooltip_title_hist_close": "Fermer l'historique",
        "tooltip_desc_hist_close": "Masquer le panneau d'historique",
        "tooltip_tip_thumb_click": "Cliquez pour recharger ce résultat",
        "tooltip_tip_recent": "Cliquez pour recharger ce résultat",
        "tooltip_title_zoom_in": "Zoom avant",
        "tooltip_desc_zoom_in": "Augmenter le grossissement de l'image",
        "tooltip_title_zoom_out": "Zoom arrière",
        "tooltip_desc_zoom_out": "Réduire le grossissement de l'image",
        "tooltip_title_zoom_reset": "Réinitialiser la vue",
        "tooltip_desc_zoom_reset": "Ajuster l'image à la fenêtre",
        "tooltip_tip_zoom_reset": "Vous pouvez aussi double-cliquer sur l'image",
        "notif_inference_ok": "Inférence terminée",
        "follow_system_theme": "Suivre le thème du système",
        "follow_system_theme_desc": "Bascule automatiquement entre clair et sombre selon les paramètres système",
        "gradcam_title": "Carte de chaleur d'attention",
        "gradcam_show": "Afficher la carte d'attention",
        "gradcam_toggle": "Basculer la carte",
        "gradcam_save": "Enregistrer l'image",
        "gradcam_close": "Fermer",
        "gradcam_saved": "Image enregistrée avec succès",
        "gradcam_no_heatmap": "Aucune carte d'attention disponible. Veuillez d'abord traiter une image.",
        "update_available": "Mise à jour disponible",
        "update_check_failed": "Échec de la vérification des mises à jour",
        "check_for_updates": "Vérifier les mises à jour",
        "check_for_updates_desc": "Vérifier GitHub/le site web pour la dernière version",
        "up_to_date": "Vous êtes à jour",
        "you_have_latest": "Vous utilisez la dernière version.",
        "check_internet": "Veuillez vérifier votre connexion Internet et réessayer.",
        "downloading": "Téléchargement...",
        "verifying": "Vérification...",
        "download_now": "Télécharger maintenant",
        "install_now": "Installer et redémarrer",
        "skip_version": "Ignorer cette version",
        "remind_later": "Me rappeler plus tard",
        "whats_new": "Quoi de neuf",
        "no_changelog": "Aucun journal des modifications fourni.",
        "critical_update": "Mise à jour critique",
        "update_ready": "Prêt à installer",
        "app_will_restart": "L'application va se fermer et redémarrer pour appliquer la mise à jour.",
        "install_failed": "L'installation a échoué. Votre ancienne version est intacte.",
        "download_failed": "Le téléchargement a échoué.",
        "source_update_requires_zip": "La mise à jour source nécessite un fichier .zip.",
        "session_locked": "Session Verrouillée",
        "locked_for_user": "Verrouillée pour {user}",
        "unlock": "Déverrouiller",
        "session_unlocked": "Session déverrouillée",
        "auto_lock_notice": "La session se verrouillera après 2 minutes d'inactivité.",
        "audit_log_title": "Journal d'audit",
        "audit_export": "Exporter",
        "audit_exported": "Exporté avec succès",
        "audit_export_failed": "Échec de l'exportation",
        "audit_search_placeholder": "Rechercher des actions ou des détails...",
        "audit_col_time": "Heure",
        "audit_col_level": "Niveau",
        "audit_col_category": "Catégorie",
        "audit_col_action": "Action",
        "audit_col_user": "Utilisateur",
        "audit_details_title": "Détails de l'événement",
        "audit_status": "Affichage de {count} sur {total} événements",
        "strength_very_weak": "Très faible",
        "strength_weak": "Faible",
        "strength_medium": "Moyen",
        "strength_strong": "Fort",
        "strength_very_strong": "Très fort",
        "strength_rule_length": "8-16 caractères",
        "strength_rule_lowercase": "Lettre minuscule",
        "strength_rule_uppercase": "Lettre majuscule",
        "strength_rule_digits": "Chiffres",
        "strength_rule_special": "Caractères spéciaux",
        "strength_add_chars": "Ajoutez {n} caractère(s) de plus",
        "strength_remove_chars": "Retirez {n} caractère(s)",
        "strength_need_digits": "Ajoutez {n} chiffre(s) de plus",
        "strength_need_special": "Ajoutez {n} caractère(s) spécial(aux) de plus",
        "strength_common_pattern": "Évitez les mots courants",
        "strength_common_year": "Évitez les années courantes",
        "strength_has_sequence": "Évitez les séquences (1234, abcd)",
        "strength_has_repeats": "Évitez les répétitions (aaaa, 1111)",
        "passwords_match": "Les mots de passe correspondent",
        "2fa_setup_title": "Activer l'authentification à deux facteurs",
        "2fa_setup_subtitle": "Scannez le code QR avec Google Authenticator, Authy ou Microsoft Authenticator.",
        "2fa_manual_secret": "Ou saisissez ce code manuellement :",
        "2fa_enter_code": "Saisissez le code à 6 chiffres de votre application :",
        "2fa_verify_btn": "Vérifier",
        "2fa_backup_codes_title": "Enregistrez ces codes de secours !",
        "2fa_backup_codes_hint": "Chaque code ne peut être utilisé qu'une seule fois. Conservez-les en lieu sûr.",
        "2fa_done_btn": "Terminé",
        "2fa_secret_copied": "Code secret copié dans le presse-papiers",
        "2fa_invalid_code": "Code invalide. Veuillez réessayer.",
        "2fa_invalid_code_format": "Le code doit comporter 6 chiffres.",
        "2fa_verify_title": "Authentification à deux facteurs",
        "2fa_verify_subtitle": "Saisissez le code à 6 chiffres de votre application d'authentification.",
        "2fa_enter_code_hint": "Veuillez saisir un code.",
        "2fa_use_backup_code": "Utiliser un code de secours",
        "2fa_use_totp_code": "Utiliser le code de l'application",
        "2fa_enter_backup_code": "Saisissez l'un de vos codes de secours.",
        "2fa_section_title": "Sécurité",
        "2fa_enabled_status": "L'authentification à deux facteurs est activée",
        "2fa_disabled_status": "L'authentification à deux facteurs est désactivée",
        "2fa_enable_btn": "Activer 2FA",
        "2fa_disable_btn": "Désactiver 2FA",
        "2fa_disable_title": "Désactiver 2FA",
        "2fa_disable_confirm": "Êtes-vous sûr ? Votre compte sera moins sécurisé.",
        "2fa_disabled_success": "Authentification à deux facteurs désactivée",
        "2fa_enabled_success": "Authentification à deux facteurs activée",
        "2fa_hint_title": "Comment utiliser ce QR code",
        "2fa_hint_step1": "Installez une application d'authentification : Google Authenticator, Microsoft Authenticator ou Authy.",
        "2fa_hint_step2": "Ouvrez l'application, appuyez sur le bouton « + », puis choisissez « Scanner le QR code ».",
        "2fa_hint_step3": "Pointez la caméra de votre téléphone vers le QR code ci-dessus (utilisez la caméra de l'application, pas celle de votre téléphone).",
        "2fa_hint_step4": "L'application affichera un code à 6 chiffres. Saisissez-le dans la case ci-dessous.",
        "2fa_manual_secret_hint": "Vous ne pouvez pas scanner ? Copiez la clé ci-dessous et ajoutez-la manuellement dans votre application (« Saisir une clé de configuration »).",
        "auto_lock_section_title": "Verrouillage automatique",
        "auto_lock_enable": "Verrouiller après inactivité",
        "auto_lock_timeout_label": "Verrouiller après :",
        "auto_lock_warning_title": "Avertissement de verrouillage",
        "auto_lock_warning_body": "Votre session sera verrouillée dans {seconds} secondes en raison d'inactivité.",
        "lock_now": "Verrouiller maintenant",
        "font_section_title": "Police",
        "font_family_label": "Famille de police",
        "font_size_label": "Taille de base",
        "font_scale_label": "Échelle d'affichage",
        "font_scale_small": "Petite",
        "font_scale_normal": "Normale",
        "font_scale_large": "Grande",
        "font_scale_xlarge": "Très grande",
        "font_scale_xxlarge": "Énorme",
        "font_preview_text": "Portez ce vieux whisky au juge blond qui fume — 1234567890",
        "font_reset": "Réinitialiser",
        "font_applied": "Police mise à jour",
        "motion_section_title": "Mouvement et animations",
        "reduce_motion_label": "Réduire les animations",
        "reduce_motion_hint": "Désactivez les animations si elles vous mettent mal à l'aise.",
        "empty_history_empty_title": "Aucun historique pour l'instant",
        "empty_history_empty_subtitle": "Traitez votre première image pour la voir ici",
        "empty_history_empty_hint": "Chaque résultat est enregistré automatiquement",
        "empty_history_no_user_title": "Connectez-vous pour voir l'historique",
        "empty_history_no_user_subtitle": "Vos résultats enregistrés apparaîtront ici",
        "empty_history_no_user_hint": "L'historique est privé pour chaque compte",
        "empty_dash_no_results_title": "Aucun résultat pour l'instant",
        "empty_dash_no_results_subtitle": "Vos analyses récentes apparaîtront ici",
        "empty_dash_no_results_hint": "Traitez une image pour commencer",
        "empty_dash_no_dist_title": "Aucune donnée de distribution",
        "empty_dash_no_dist_subtitle": "Les statistiques se rempliront au fur et à mesure",
        "empty_dash_no_dist_hint": "Chaque analyse compte",
        "empty_image_empty_title": "Aucune image insérée",
        "empty_image_empty_subtitle": "Glissez-déposez une image ici, ou cliquez sur Insérer",
        "empty_image_empty_hint": "Prend en charge PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Déposez pour analyser",
        "empty_image_drop_hint_subtitle": "Relâchez l'image pour la charger",
        "empty_image_drop_hint_hint": "Le format sera détecté automatiquement",
        "empty_processing_wait_title": "Traitement...",
        "empty_processing_wait_subtitle": "L'IA analyse votre image",
        "empty_processing_wait_hint": "Cela prend généralement quelques secondes",
        "tour_show_again": "Revoir la visite guidée",
        "tour_completed_msg": "Visite terminée ! Vous êtes prêt.",
        "tour_skipped_msg": "Visite ignorée. Vous pouvez la revoir depuis le menu Compte.",
        "tour_btn_next": "Suivant",
        "tour_btn_prev": "Retour",
        "tour_btn_skip": "Ignorer",
        "tour_btn_finish": "Terminer ✓",
        "tour_guest_login_title": "Connexion",
        "tour_guest_login_subtitle": "Cliquez ici pour vous connecter et débloquer toutes les fonctionnalités.",
        "tour_guest_signup_title": "Créer un compte",
        "tour_guest_signup_subtitle": "Nouveau ici ? Cliquez ici pour créer un compte en quelques secondes.",
        "tour_insert_title": "Insérer une image",
        "tour_insert_subtitle": "Chargez une image d'écho depuis le disque, ou glissez-déposez-la n'importe où sur la fenêtre.",
        "tour_process_title": "Traiter l'image",
        "tour_process_subtitle": "Lancez l'analyse IA sur l'image chargée. Les résultats apparaissent en quelques secondes.",
        "tour_export_title": "Exporter PDF",
        "tour_export_subtitle": "Enregistrez le résultat actuel sous forme de rapport médical PDF professionnel.",
        "tour_gradcam_title": "Carte de chaleur d'attention",
        "tour_gradcam_subtitle": "Visualisez où l'IA regarde — une carte de chaleur sur l'image analysée.",
        "tour_dashboard_title": "Tableau de bord",
        "tour_dashboard_subtitle": "Affichez/masquez le panneau de statistiques : total d'images, confiance moyenne, distribution des grades.",
        "tour_history_title": "Historique",
        "tour_history_subtitle": "Parcourez vos 20 dernières images analysées. Cliquez sur une vignette pour recharger son résultat.",
        "tour_visit_title": "Visiter le site",
        "tour_visit_subtitle": "Ouvrez la page GitHub du projet dans votre navigateur pour la documentation et les mises à jour.",
        "tour_mute_title": "Couper la musique",
        "tour_mute_subtitle": "Mettez en pause ou reprenez la musique de fond à tout moment.",
        "tour_zoom_title": "Contrôles de zoom",
        "tour_zoom_subtitle": "Utilisez la molette pour zoomer, faites glisser pour déplacer, et double-cliquez pour réinitialiser.",
        "tour_account_title": "Menu du compte",
        "tour_account_subtitle": "Cliquez ici pour accéder aux Paramètres, Infos du compte, Journal d'audit, et plus.",
        "suggestion_image_ready": "Image chargée — prêt à traiter ?",
        "suggestion_image_ready_sub": "Cliquez sur Traiter pour lancer l'analyse IA.",
        "suggestion_pdf_ready": "Résultat prêt — exporter en PDF ?",
        "suggestion_pdf_ready_sub": "Enregistrez un rapport médical professionnel en un clic.",
        "suggestion_gradcam_hint": "Curieux de voir ce que l'IA a regardé ?",
        "suggestion_gradcam_hint_sub": "Voyez la carte de chaleur d'attention sur l'image.",
        "suggestion_repeated_grid": "Vous obtenez toujours {grid}",
        "suggestion_repeated_grid_sub": "Ouvrez le tableau de bord pour voir la répartition.",
        "suggestion_history_hint": "Votre historique s'enrichit",
        "suggestion_history_hint_sub": "Parcourez vos 20 dernières images analysées.",
        "suggestion_dashboard_hint": "Votre tableau de bord a des stats intéressantes",
        "suggestion_dashboard_hint_sub": "Totaux, moyennes et répartition en un coup d'œil.",
        "suggestion_settings_hint": "Personnalisez l'application",
        "suggestion_settings_hint_sub": "Changez la langue, le thème, la police et activez la 2FA.",
        "suggestion_idle_help": "Besoin d'un guide rapide ?",
        "suggestion_idle_help_sub": "Rejouez la visite guidée à tout moment.",
        "image_not_liver_warning": "Cette image ne ressemble pas à une échographie du foie.\n\nVeuillez insérer une image d'échographie du foie appropriée.",
        "result_not_confident": "L'IA n'a pas pu identifier avec certitude cette image comme une échographie du foie.\n\nVeuillez insérer une image d'échographie du foie appropriée et réessayer.",
        "not_liver_title": "Pas une échographie du foie",
        "not_liver_hint": "Veuillez utiliser une image d'échographie (sonogramme) du foie appropriée",
        "voice_welcome": "Bienvenue, {user}",
        "voice_goodbye": "Au revoir",
        "voice_logged_out": "Déconnexion réussie",
        "voice_image_loaded": "Image chargée avec succès",
        "voice_image_rejected": "Ce n'est pas une échographie du foie",
        "voice_inference_result": "{grid_name}, confiance {conf} pour cent",
        "voice_pdf_saved": "Rapport PDF enregistré avec succès",
        "voice_error": "Une erreur est survenue",
        "voice_settings_applied": "Paramètres appliqués",
        "voice_language_changed": "Langue changée en {lang}",
        "voice_theme_changed": "Thème changé",
        "voice_2fa_enabled": "Authentification à deux facteurs activée",
        "voice_2fa_disabled": "Authentification à deux facteurs désactivée",
        "voice_locked": "Session verrouillée",
        "voice_unlocked": "Session déverrouillée",
        "voice_processing": "Traitement de l'image",
        "voice_processing_done": "Traitement terminé",
        "voice_voice_enabled": "Retour vocal activé",
        "voice_voice_disabled": "Retour vocal désactivé",
    },
    "Russian": {
        "app_title": "Конвейер ИИ печени",
        "welcome": "Добро пожаловать",
        "login": "Войти",
        "signup": "Регистрация",
        "logout": "Выйти",
        "exit": "Выход",
        "settings": "Настройки",
        "account_info": "Информация об аккаунте",
        "username": "Имя пользователя",
        "password": "Пароль",
        "repeat_password": "Повторите пароль",
        "first_name": "Имя",
        "last_name": "Фамилия",
        "email": "Электронная почта",
        "remember_me": "Запомнить меня",
        "login_btn": "Войти",
        "signup_btn": "Регистрация",
        "account": "Аккаунт",
        "change_language": "Изменить язык",
        "change_theme": "Изменить тему",
        "insert_sono": "Вставить сонограмму",
        "process": "Обработать",
        "mute": "Без звука",
        "unmute": "Со звуком",
        "visit_webpage": "Посетить сайт",
        "please_wait": "Пожалуйста, подождите...",
        "processing": "Обработка изображения...",
        "result": "Результат",
        "confidence": "Уверенность",
        "grid": "Сетка",
        "insert_image_first": "Сначала вставьте изображение!",
        "no_image": "Изображение не вставлено",
        "invalid_credentials": "Неверные данные",
        "user_exists": "Имя пользователя уже существует",
        "password_requirements": "Требования к паролю",
        "password_rule": (
            "Пароль должен:\n"
            "• Содержать 8-16 символов\n"
            "• Содержать минимум 1 строчную букву\n"
            "• Содержать минимум 1 заглавную букву\n"
            "• Содержать минимум 4 цифры\n"
            "• Содержать минимум 4 специальных символа"
        ),
        "passwords_dont_match": "Пароли не совпадают!",
        "invalid_email": "Введите корректный адрес электронной почты",
        "empty_fields": "Заполните все поля",
        "signup_success": "Регистрация успешна! Теперь вы можете войти.",
        "logout_confirm": "Вы уверены, что хотите удалить аккаунт?",
        "exit_confirm": "Вы уверены, что хотите выйти?",
        "theme_dark_aqua": "Тёмная тема Aqua-Purple",
        "theme_solarized_white": "Светлая тема Solarized",
        "theme_solarized_dark": "Тёмная тема Solarized",
        "theme_dark_forest": "Тёмная лесная зелёная тема",
        "select_language": "Выбрать язык",
        "select_theme": "Выбрать тему",
        "apply": "Применить",
        "cancel": "Отмена",
        "save": "Сохранить",
        "success": "Успех",
        "error": "Ошибка",
        "warning": "Предупреждение",
        "settings_applied": "Настройки применены!",
        "grid_0": "Сетка 0 (NL) — Нормальная печень",
        "grid_1": "Сетка 1 (G1) — Степень 1",
        "grid_2": "Сетка 2 (G2) — Степень 2",
        "grid_3": "Сетка 3 (G3) — Степень 3",
        "too_many_attempts": "Слишком много неудачных попыток. Подождите 5 минут.",
        "invalid_username": "Имя пользователя должно быть 3-32 символа (буквы, цифры, подчёркивание).",
        "welcome_user": "Добро пожаловать, {user}!",
        "inference_failed": "Ошибка вывода",
        "drop_here": "Перетащите изображение сюда",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Неподдерживаемый тип файла. Пожалуйста, перетащите изображение.",
        "image_loaded": "Изображение успешно загружено",
        "help_about_title": "Справка и О программе",
        "help_about_text": (
            "Конвейер ИИ печени v3.0\n"
            "Разработано: Бабак Юсефян\n\n"
            "Горячие клавиши:\n"
            "  Ctrl+O      — Вставить изображение\n"
            "  Ctrl+Enter  — Обработать изображение\n"
            "  Ctrl+L      — Войти\n"
            "  Ctrl+,      — Настройки\n"
            "  Esc         — Закрыть активный диалог\n"
            "  F1          — Справка / О программе\n\n"
            "Совет: Вы также можете перетащить изображение в любое место окна."
        ),
        "dashboard_title": "Панель управления",
        "dash_total": "Всего изображений",
        "dash_avg_conf": "Средняя уверенность",
        "dash_distribution": "Распределение",
        "dash_last_5": "Последние 5 результатов",
        "dash_no_results": "Пока нет результатов",
        "dash_clear_confirm": "Очистить всю статистику?",
        "dash_cleared": "Статистика очищена",
        "dashboard_toggle": "Показать/Скрыть панель",
        "export_pdf": "Экспорт PDF",
        "pdf_saved": "PDF успешно сохранён",
        "pdf_failed": "Не удалось создать PDF",
        "pdf_subtitle": "Отчёт об анализе",
        "pdf_patient": "Пациент:",
        "pdf_date": "Дата:",
        "pdf_model": "Модель:",
        "pdf_diagnosis": "ДИАГНОЗ",
        "pdf_confidence": "Уверенность",
        "pdf_class_probs": "ВЕРОЯТНОСТИ КЛАССОВ",
        "pdf_disclaimer": (
            "Этот отчёт разработан Бабаком Юсефяном под наблюдением опытных врачей. "
            "Это профессиональный инструмент анализа, не требующий направления к врачу."
        ),
        "history_title": "История",
        "history_toggle": "Показать/Скрыть историю",
        "history_empty": "Пока нет истории",
        "history_cleared": "История очищена",
        "history_clear_confirm": "Очистить всю историю?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Войти",
        "tooltip_desc_login": "Войдите в свой аккаунт",
        "tooltip_tip_login": "Доступ к сохранённой истории и статистике",
        "tooltip_title_signup": "Регистрация",
        "tooltip_desc_signup": "Создать новый аккаунт",
        "tooltip_title_account": "Аккаунт",
        "tooltip_desc_account": "Нажмите, чтобы открыть меню аккаунта",
        "tooltip_desc_account_user": "Вы вошли как {user}",
        "tooltip_title_dashboard": "Панель управления",
        "tooltip_desc_dashboard": "Показать/скрыть панель статистики — всего, распределение, последние результаты",
        "tooltip_title_history": "История",
        "tooltip_desc_history": "Показать/скрыть последние 20 результатов с эскизами",
        "tooltip_title_visit": "Посетить сайт",
        "tooltip_desc_visit": "Открыть страницу GitHub проекта в браузере",
        "tooltip_tip_visit": "Открывается во внешнем браузере",
        "tooltip_title_insert": "Вставить изображение",
        "tooltip_desc_insert": "Загрузить сонограмму для анализа",
        "tooltip_tip_insert": "Можно также перетащить изображение в любое место окна",
        "tooltip_title_process": "Обработать",
        "tooltip_desc_process": "Запустить ИИ-анализ загруженного изображения",
        "tooltip_tip_process": "Сначала нужно загрузить изображение",
        "tooltip_title_mute": "Выключить музыку",
        "tooltip_desc_mute": "Приостановить фоновую музыку",
        "tooltip_title_unmute": "Включить музыку",
        "tooltip_desc_unmute": "Возобновить фоновую музыку",
        "tooltip_title_export": "Экспорт PDF",
        "tooltip_desc_export": "Сохранить текущий результат как PDF-отчёт",
        "tooltip_tip_export": "Включает изображение, диагноз и вероятности классов",
        "tooltip_title_dash_clear": "Очистить статистику",
        "tooltip_desc_dash_clear": "Сбросить всю статистику этого аккаунта",
        "tooltip_tip_dash_clear": "Это действие нельзя отменить",
        "tooltip_title_dash_close": "Закрыть панель",
        "tooltip_desc_dash_close": "Скрыть панель статистики",
        "tooltip_desc_dash_total": "Общее количество обработанных изображений",
        "tooltip_desc_dash_avg": "Средняя уверенность по всем результатам",
        "tooltip_desc_dash_dist": "Распределение результатов по степеням",
        "tooltip_title_hist_clear": "Очистить историю",
        "tooltip_desc_hist_clear": "Удалить все сохранённые результаты и эскизы",
        "tooltip_tip_hist_clear": "Это также удалит все файлы изображений",
        "tooltip_title_hist_close": "Закрыть историю",
        "tooltip_desc_hist_close": "Скрыть панель истории",
        "tooltip_tip_thumb_click": "Нажмите, чтобы загрузить этот результат заново",
        "tooltip_tip_recent": "Нажмите, чтобы загрузить этот результат заново",
        "tooltip_title_zoom_in": "Увеличить",
        "tooltip_desc_zoom_in": "Увеличить масштаб изображения",
        "tooltip_title_zoom_out": "Уменьшить",
        "tooltip_desc_zoom_out": "Уменьшить масштаб изображения",
        "tooltip_title_zoom_reset": "Сбросить вид",
        "tooltip_desc_zoom_reset": "Вписать изображение в окно",
        "tooltip_tip_zoom_reset": "Можно также дважды щёлкнуть по изображению",
        "notif_inference_ok": "Вывод завершён",
        "follow_system_theme": "Следовать теме системы",
        "follow_system_theme_desc": "Автоматически переключаться между светлой и тёмной темой в зависимости от настроек системы",
        "gradcam_title": "Тепловая карта внимания",
        "gradcam_show": "Показать карту внимания",
        "gradcam_toggle": "Переключить карту",
        "gradcam_save": "Сохранить изображение",
        "gradcam_close": "Закрыть",
        "gradcam_saved": "Изображение успешно сохранено",
        "gradcam_no_heatmap": "Карта внимания недоступна. Сначала обработайте изображение.",
        "update_available": "Доступно обновление",
        "update_check_failed": "Не удалось проверить обновления",
        "check_for_updates": "Проверить обновления",
        "check_for_updates_desc": "Проверить GitHub/сайт на наличие последней версии",
        "up_to_date": "У вас последняя версия",
        "you_have_latest": "Вы используете последнюю версию.",
        "check_internet": "Пожалуйста, проверьте подключение к интернету и повторите попытку.",
        "downloading": "Загрузка...",
        "verifying": "Проверка...",
        "download_now": "Скачать сейчас",
        "install_now": "Установить и перезапустить",
        "skip_version": "Пропустить эту версию",
        "remind_later": "Напомнить позже",
        "whats_new": "Что нового",
        "no_changelog": "Список изменений не предоставлен.",
        "critical_update": "Критическое обновление",
        "update_ready": "Готово к установке",
        "app_will_restart": "Приложение закроется и перезапустится для применения обновления.",
        "install_failed": "Установка не удалась. Ваша старая версия в безопасности.",
        "download_failed": "Загрузка не удалась.",
        "source_update_requires_zip": "Обновление исходного кода требует файла .zip.",
        "session_locked": "Сессия заблокирована",
        "locked_for_user": "Заблокировано для {user}",
        "unlock": "Разблокировать",
        "session_unlocked": "Сессия разблокирована",
        "auto_lock_notice": "Сессия заблокируется через 2 минуты бездействия.",
        "audit_log_title": "Журнал аудита",
        "audit_export": "Экспорт",
        "audit_exported": "Успешно экспортировано",
        "audit_export_failed": "Ошибка экспорта",
        "audit_search_placeholder": "Поиск действий или деталей...",
        "audit_col_time": "Время",
        "audit_col_level": "Уровень",
        "audit_col_category": "Категория",
        "audit_col_action": "Действие",
        "audit_col_user": "Пользователь",
        "audit_details_title": "Детали события",
        "audit_status": "Показано {count} из {total} событий",
        "strength_very_weak": "Очень слабый",
        "strength_weak": "Слабый",
        "strength_medium": "Средний",
        "strength_strong": "Сильный",
        "strength_very_strong": "Очень сильный",
        "strength_rule_length": "8-16 символов",
        "strength_rule_lowercase": "Строчная буква",
        "strength_rule_uppercase": "Заглавная буква",
        "strength_rule_digits": "Цифры",
        "strength_rule_special": "Специальные символы",
        "strength_add_chars": "Добавьте ещё {n} символ(ов)",
        "strength_remove_chars": "Уберите {n} символ(ов)",
        "strength_need_digits": "Добавьте ещё {n} цифр(ы)",
        "strength_need_special": "Добавьте ещё {n} специальных символ(ов)",
        "strength_common_pattern": "Избегайте распространённых слов",
        "strength_common_year": "Избегайте распространённых годов",
        "strength_has_sequence": "Избегайте последовательностей (1234, abcd)",
        "strength_has_repeats": "Избегайте повторов (aaaa, 1111)",
        "passwords_match": "Пароли совпадают",
        "2fa_setup_title": "Включить двухфакторную аутентификацию",
        "2fa_setup_subtitle": "Отсканируйте QR-код с помощью Google Authenticator, Authy или Microsoft Authenticator.",
        "2fa_manual_secret": "Или введите этот код вручную:",
        "2fa_enter_code": "Введите 6-значный код из приложения:",
        "2fa_verify_btn": "Подтвердить",
        "2fa_backup_codes_title": "Сохраните эти резервные коды!",
        "2fa_backup_codes_hint": "Каждый код можно использовать только один раз. Храните их в надёжном месте.",
        "2fa_done_btn": "Готово",
        "2fa_secret_copied": "Секретный код скопирован в буфер обмена",
        "2fa_invalid_code": "Неверный код. Попробуйте снова.",
        "2fa_invalid_code_format": "Код должен состоять из 6 цифр.",
        "2fa_verify_title": "Двухфакторная аутентификация",
        "2fa_verify_subtitle": "Введите 6-значный код из приложения-аутентификатора.",
        "2fa_enter_code_hint": "Пожалуйста, введите код.",
        "2fa_use_backup_code": "Использовать резервный код",
        "2fa_use_totp_code": "Использовать код приложения",
        "2fa_enter_backup_code": "Введите один из ваших резервных кодов.",
        "2fa_section_title": "Безопасность",
        "2fa_enabled_status": "Двухфакторная аутентификация включена",
        "2fa_disabled_status": "Двухфакторная аутентификация отключена",
        "2fa_enable_btn": "Включить 2FA",
        "2fa_disable_btn": "Отключить 2FA",
        "2fa_disable_title": "Отключить 2FA",
        "2fa_disable_confirm": "Вы уверены? Ваш аккаунт станет менее защищённым.",
        "2fa_disabled_success": "Двухфакторная аутентификация отключена",
        "2fa_enabled_success": "Двухфакторная аутентификация включена",
        "2fa_hint_title": "Как использовать этот QR-код",
        "2fa_hint_step1": "Установите приложение-аутентификатор: Google Authenticator, Microsoft Authenticator или Authy.",
        "2fa_hint_step2": "Откройте приложение, нажмите кнопку «+» и выберите «Сканировать QR-код».",
        "2fa_hint_step3": "Наведите камеру телефона на QR-код выше (используйте камеру приложения, а не камеру телефона).",
        "2fa_hint_step4": "Приложение покажет 6-значный код. Введите его в поле ниже.",
        "2fa_manual_secret_hint": "Не можете отсканировать? Скопируйте ключ ниже и добавьте его вручную в приложении («Ввести ключ настройки»).",
        "auto_lock_section_title": "Автоблокировка",
        "auto_lock_enable": "Блокировать после бездействия",
        "auto_lock_timeout_label": "Блокировать через:",
        "auto_lock_warning_title": "Предупреждение об автоблокировке",
        "auto_lock_warning_body": "Ваша сессия будет заблокирована через {seconds} секунд из-за бездействия.",
        "lock_now": "Заблокировать сейчас",
        "font_section_title": "Шрифт",
        "font_family_label": "Семейство шрифтов",
        "font_size_label": "Базовый размер",
        "font_scale_label": "Масштаб отображения",
        "font_scale_small": "Маленький",
        "font_scale_normal": "Обычный",
        "font_scale_large": "Большой",
        "font_scale_xlarge": "Очень большой",
        "font_scale_xxlarge": "Огромный",
        "font_preview_text": "Съешь же ещё этих мягких французских булок — 1234567890",
        "font_reset": "Сбросить по умолчанию",
        "font_applied": "Шрифт обновлён",
        "motion_section_title": "Движение и анимации",
        "reduce_motion_label": "Уменьшить движение",
        "reduce_motion_hint": "Отключите анимации, если они вызывают у вас дискомфорт.",
        "empty_history_empty_title": "Пока нет истории",
        "empty_history_empty_subtitle": "Обработайте первое изображение, чтобы увидеть его здесь",
        "empty_history_empty_hint": "Каждый результат сохраняется автоматически",
        "empty_history_no_user_title": "Войдите, чтобы увидеть историю",
        "empty_history_no_user_subtitle": "Ваши сохранённые результаты появятся здесь",
        "empty_history_no_user_hint": "История приватна для каждого аккаунта",
        "empty_dash_no_results_title": "Пока нет результатов",
        "empty_dash_no_results_subtitle": "Ваши недавние анализы появятся здесь",
        "empty_dash_no_results_hint": "Обработайте изображение, чтобы начать",
        "empty_dash_no_dist_title": "Нет данных о распределении",
        "empty_dash_no_dist_subtitle": "Статистика будет заполняться по мере обработки",
        "empty_dash_no_dist_hint": "Каждый анализ важен",
        "empty_image_empty_title": "Изображение не вставлено",
        "empty_image_empty_subtitle": "Перетащите изображение сюда или нажмите «Вставить»",
        "empty_image_empty_hint": "Поддерживаются PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Отпустите для анализа",
        "empty_image_drop_hint_subtitle": "Отпустите изображение, чтобы загрузить",
        "empty_image_drop_hint_hint": "Формат определится автоматически",
        "empty_processing_wait_title": "Обработка...",
        "empty_processing_wait_subtitle": "ИИ анализирует ваше изображение",
        "empty_processing_wait_hint": "Обычно это занимает несколько секунд",
        "tour_show_again": "Показать тур снова",
        "tour_completed_msg": "Тур завершён! Всё готово.",
        "tour_skipped_msg": "Тур пропущен. Вы можете повторить его из меню аккаунта.",
        "tour_btn_next": "Далее",
        "tour_btn_prev": "Назад",
        "tour_btn_skip": "Пропустить",
        "tour_btn_finish": "Готово ✓",
        "tour_guest_login_title": "Вход",
        "tour_guest_login_subtitle": "Нажмите здесь, чтобы войти в аккаунт и открыть все возможности.",
        "tour_guest_signup_title": "Создать аккаунт",
        "tour_guest_signup_subtitle": "Впервые здесь? Нажмите здесь, чтобы создать аккаунт за секунды.",
        "tour_insert_title": "Вставить изображение",
        "tour_insert_subtitle": "Загрузите сонограмму с диска или перетащите её в любое место окна.",
        "tour_process_title": "Обработать изображение",
        "tour_process_subtitle": "Запустите ИИ-анализ загруженного изображения. Результаты появятся через несколько секунд.",
        "tour_export_title": "Экспорт PDF",
        "tour_export_subtitle": "Сохраните текущий результат в виде профессионального медицинского PDF-отчёта.",
        "tour_gradcam_title": "Тепловая карта внимания",
        "tour_gradcam_subtitle": "Посмотрите, куда смотрит ИИ — тепловая карта поверх анализируемого изображения.",
        "tour_dashboard_title": "Панель управления",
        "tour_dashboard_subtitle": "Показать/скрыть панель статистики: всего изображений, средняя уверенность, распределение по степеням.",
        "tour_history_title": "История",
        "tour_history_subtitle": "Просмотрите последние 20 проанализированных изображений. Нажмите на миниатюру, чтобы загрузить результат заново.",
        "tour_visit_title": "Посетить сайт",
        "tour_visit_subtitle": "Откройте страницу GitHub проекта в браузере для документации и обновлений.",
        "tour_mute_title": "Выключить музыку",
        "tour_mute_subtitle": "Приостановите или возобновите фоновую музыку в любой момент.",
        "tour_zoom_title": "Управление масштабом",
        "tour_zoom_subtitle": "Колесо мыши — масштаб, перетаскивание — панорама, двойной клик — сброс.",
        "tour_account_title": "Меню аккаунта",
        "tour_account_subtitle": "Нажмите здесь, чтобы открыть Настройки, Информацию об аккаунте, Журнал аудита и другое.",
        "suggestion_image_ready": "Изображение загружено — готово к обработке",
        "suggestion_image_ready_sub": "Cliquez sur Traiter pour lancer l'analyse IA.",
        "suggestion_pdf_ready": "Résultat prêt — exporter en PDF ?",
        "suggestion_pdf_ready_sub": "Enregistrez un rapport médical professionnel en un clic.",
        "suggestion_gradcam_hint": "Curieux de voir ce que l'IA a regardé ?",
        "suggestion_gradcam_hint_sub": "Voyez la carte de chaleur d'attention sur l'image.",
        "suggestion_repeated_grid": "Vous obtenez toujours {grid}",
        "suggestion_repeated_grid_sub": "Ouvrez le tableau de bord pour voir la répartition.",
        "suggestion_history_hint": "Votre historique s'enrichit",
        "suggestion_history_hint_sub": "Parcourez vos 20 dernières images analysées.",
        "suggestion_dashboard_hint": "Votre tableau de bord a des stats intéressantes",
        "suggestion_dashboard_hint_sub": "Totaux, moyennes et répartition en un coup d'œil.",
        "suggestion_settings_hint": "Personnalisez l'application",
        "suggestion_settings_hint_sub": "Changez la langue, le thème, la police et activez la 2FA.",
        "suggestion_idle_help": "Besoin d'un guide rapide ?",
        "suggestion_idle_help_sub": "Rejouez la visite guidée à tout moment.",
        "image_not_liver_warning": "Это изображение не похоже на УЗИ печени.\n\nПожалуйста, вставьте подходящее изображение УЗИ печени.",
        "result_not_confident": "ИИ не смог уверенно определить это изображение как УЗИ печени.\n\nПожалуйста, вставьте подходящее изображение УЗИ печени и попробуйте снова.",
        "not_liver_title": "Не УЗИ печени",
        "not_liver_hint": "Пожалуйста, используйте подходящее изображение УЗИ (сонограмму) печени",
        "voice_welcome": "Добро пожаловать, {user}",
        "voice_goodbye": "До свидания",
        "voice_logged_out": "Выход выполнен успешно",
        "voice_image_loaded": "Изображение успешно загружено",
        "voice_image_rejected": "Это не УЗИ печени",
        "voice_inference_result": "{grid_name}, уверенность {conf} процентов",
        "voice_pdf_saved": "PDF-отчёт успешно сохранён",
        "voice_error": "Произошла ошибка",
        "voice_settings_applied": "Настройки применены",
        "voice_language_changed": "Язык изменён на {lang}",
        "voice_theme_changed": "Тема изменена",
        "voice_2fa_enabled": "Двухфакторная аутентификация включена",
        "voice_2fa_disabled": "Двухфакторная аутентификация отключена",
        "voice_locked": "Сессия заблокирована",
        "voice_unlocked": "Сессия разблокирована",
        "voice_processing": "Обработка изображения",
        "voice_processing_done": "Обработка завершена",
        "voice_voice_enabled": "Голосовая связь включена",
        "voice_voice_disabled": "Голосовая связь отключена",
    },
    "Greece": {
        "app_title": "Σωλήνας AI Ήπατος",
        "welcome": "Καλώς ήρθατε",
        "login": "Σύνδεση",
        "signup": "Εγγραφή",
        "logout": "Αποσύνδεση",
        "exit": "Έξοδος",
        "settings": "Ρυθμίσεις",
        "account_info": "Πληροφορίες λογαριασμού",
        "username": "Όνομα χρήστη",
        "password": "Κωδικός πρόσβασης",
        "repeat_password": "Επανάληψη κωδικού",
        "first_name": "Όνομα",
        "last_name": "Επώνυμο",
        "email": "Διεύθυνση email",
        "remember_me": "Να με θυμάσαι",
        "login_btn": "Σύνδεση",
        "signup_btn": "Εγγραφή",
        "account": "Λογαριασμός",
        "change_language": "Αλλαγή γλώσσας",
        "change_theme": "Αλλαγή θέματος",
        "insert_sono": "Εισαγωγή υπερήχου",
        "process": "Επεξεργασία",
        "mute": "Σίγαση",
        "unmute": "Κατάργηση σίγασης",
        "visit_webpage": "Επισκεφθείτε την ιστοσελίδα",
        "please_wait": "Παρακαλώ περιμένετε...",
        "processing": "Επεξεργασία εικόνας...",
        "result": "Αποτέλεσμα",
        "confidence": "Βεβαιότητα",
        "grid": "Πλέγμα",
        "insert_image_first": "Παρακαλώ εισάγετε πρώτα μια εικόνα!",
        "no_image": "Δεν έχει εισαχθεί εικόνα",
        "invalid_credentials": "Μη έγκυρα διαπιστευτήρια",
        "user_exists": "Το όνομα χρήστη υπάρχει ήδη",
        "password_requirements": "Απαιτήσεις κωδικού",
        "password_rule": (
            "Ο κωδικός πρέπει:\n"
            "• Να έχει 8-16 χαρακτήρες\n"
            "• Να περιέχει τουλάχιστον 1 πεζό γράμμα\n"
            "• Να περιέχει τουλάχιστον 1 κεφαλαίο γράμμα\n"
            "• Να περιέχει τουλάχιστον 4 ψηφία\n"
            "• Να περιέχει τουλάχιστον 4 ειδικούς χαρακτήρες"
        ),
        "passwords_dont_match": "Οι κωδικοί δεν ταιριάζουν!",
        "invalid_email": "Εισάγετε μια έγκυρη διεύθυνση email",
        "empty_fields": "Συμπληρώστε όλα τα πεδία",
        "signup_success": "Επιτυχής εγγραφή! Μπορείτε τώρα να συνδεθείτε.",
        "logout_confirm": "Είστε σίγουροι ότι θέλετε να διαγράψετε τον λογαριασμό σας;",
        "exit_confirm": "Είστε σίγουροι ότι θέλετε να βγείτε;",
        "theme_dark_aqua": "Σκούρο θέμα Aqua-Purple",
        "theme_solarized_white": "Φωτεινό θέμα Solarized",
        "theme_solarized_dark": "Σκούρο θέμα Solarized",
        "theme_dark_forest": "Σκούρο Πράσινο Δάσος",
        "select_language": "Επιλογή γλώσσας",
        "select_theme": "Επιλογή θέματος",
        "apply": "Εφαρμογή",
        "cancel": "Ακύρωση",
        "save": "Αποθήκευση",
        "success": "Επιτυχία",
        "error": "Σφάλμα",
        "warning": "Προειδοποίηση",
        "settings_applied": "Οι ρυθμίσεις εφαρμόστηκαν!",
        "grid_0": "Πλέγμα 0 (NL) — Φυσιολογικό ήπαρ",
        "grid_1": "Πλέγμα 1 (G1) — Βαθμός 1",
        "grid_2": "Πλέγμα 2 (G2) — Βαθμός 2",
        "grid_3": "Πλέγμα 3 (G3) — Βαθμός 3",
        "too_many_attempts": "Πολλές αποτυχημένες προσπάθειες. Περιμένετε 5 λεπτά.",
        "invalid_username": "Το όνομα χρήστη πρέπει να είναι 3-32 χαρακτήρες (γράμματα, αριθμοί, κάτω παύλα).",
        "welcome_user": "Καλώς ήρθες, {user}!",
        "inference_failed": "Αποτυχία εξαγωγής",
        "drop_here": "Αποθέστε την εικόνα εδώ",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Μη υποστηριζόμενος τύπος αρχείου. Παρακαλώ αποθέστε μια εικόνα.",
        "image_loaded": "Η εικόνα φορτώθηκε με επιτυχία",
        "help_about_title": "Βοήθεια & Σχετικά",
        "help_about_text": (
            "Σωλήνας AI Ήπατος v3.0\n"
            "Αναπτύχθηκε από τον Babak Yousefian\n\n"
            "Συντομεύσεις πληκτρολογίου:\n"
            "  Ctrl+O      — Εισαγωγή εικόνας\n"
            "  Ctrl+Enter  — Επεξεργασία εικόνας\n"
            "  Ctrl+L      — Σύνδεση\n"
            "  Ctrl+,      — Ρυθμίσεις\n"
            "  Esc         — Κλείσιμο ενεργού παραθύρου\n"
            "  F1          — Βοήθεια / Σχετικά\n\n"
            "Συμβουλή: Μπορείτε επίσης να σύρετε και να αφήσετε μια εικόνα οπουδήποτε στο παράθυρο."
        ),
        "dashboard_title": "Πίνακας ελέγχου",
        "dash_total": "Σύνολο εικόνων",
        "dash_avg_conf": "Μέση βεβαιότητα",
        "dash_distribution": "Κατανομή",
        "dash_last_5": "Τελευταία 5 αποτελέσματα",
        "dash_no_results": "Δεν υπάρχουν αποτελέσματα ακόμα",
        "dash_clear_confirm": "Να διαγραφούν όλα τα στατιστικά;",
        "dash_cleared": "Τα στατιστικά διαγράφηκαν",
        "dashboard_toggle": "Εμφάνιση/Απόκρυψη πίνακα",
        "export_pdf": "Εξαγωγή PDF",
        "pdf_saved": "Το PDF αποθηκεύτηκε με επιτυχία",
        "pdf_failed": "Αποτυχία δημιουργίας PDF",
        "pdf_subtitle": "Έκθεση ανάλυσης",
        "pdf_patient": "Ασθενής:",
        "pdf_date": "Ημερομηνία:",
        "pdf_model": "Μοντέλο:",
        "pdf_diagnosis": "ΔΙΑΓΝΩΣΗ",
        "pdf_confidence": "Βεβαιότητα",
        "pdf_class_probs": "ΠΙΘΑΝΟΤΗΤΕΣ ΚΛΑΣΕΩΝ",
        "pdf_disclaimer": (
            "Αυτή η έκθεση αναπτύχθηκε από τον Babak Yousefian υπό την επίβλεψη έμπειρων ιατρών. "
            "Είναι ένα επαγγελματικό εργαλείο ανάλυσης και δεν απαιτεί παραπομπή σε γιατρό."
        ),
        "history_title": "Ιστορικό",
        "history_toggle": "Εμφάνιση/Απόκρυψη ιστορικού",
        "history_empty": "Δεν υπάρχει ιστορικό ακόμα",
        "history_cleared": "Το ιστορικό διαγράφηκε",
        "history_clear_confirm": "Να διαγραφεί όλο το ιστορικό;",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Σύνδεση",
        "tooltip_desc_login": "Συνδεθείτε στον λογαριασμό σας",
        "tooltip_tip_login": "Πρόσβαση στο αποθηκευμένο ιστορικό και στατιστικά",
        "tooltip_title_signup": "Εγγραφή",
        "tooltip_desc_signup": "Δημιουργήστε νέο λογαριασμό",
        "tooltip_title_account": "Λογαριασμός",
        "tooltip_desc_account": "Κάντε κλικ για να ανοίξετε το μενού λογαριασμού",
        "tooltip_desc_account_user": "Συνδεδεμένος ως {user}",
        "tooltip_title_dashboard": "Πίνακας ελέγχου",
        "tooltip_desc_dashboard": "Εμφάνιση/απόκρυψη πίνακα στατιστικών — σύνολο, κατανομή, πρόσφατα αποτελέσματα",
        "tooltip_title_history": "Ιστορικό",
        "tooltip_desc_history": "Εμφάνιση/απόκρυψη των τελευταίων 20 αποτελεσμάτων με μικρογραφίες",
        "tooltip_title_visit": "Επισκεφθείτε την ιστοσελίδα",
        "tooltip_desc_visit": "Άνοιγμα της σελίδας GitHub του έργου στον browser",
        "tooltip_tip_visit": "Ανοίγει σε εξωτερικό browser",
        "tooltip_title_insert": "Εισαγωγή εικόνας",
        "tooltip_desc_insert": "Φορτώστε μια εικόνα υπερήχου για ανάλυση",
        "tooltip_tip_insert": "Μπορείτε επίσης να σύρετε και να αφήσετε μια εικόνα οπουδήποτε στο παράθυρο",
        "tooltip_title_process": "Επεξεργασία",
        "tooltip_desc_process": "Εκτέλεση ανάλυσης AI στην φορτωμένη εικόνα",
        "tooltip_tip_process": "Απαιτείται πρώτα η φόρτωση εικόνας",
        "tooltip_title_mute": "Σίγαση μουσικής",
        "tooltip_desc_mute": "Παύση της μουσικής υποβάθρου",
        "tooltip_title_unmute": "Κατάργηση σίγασης",
        "tooltip_desc_unmute": "Συνέχιση της μουσικής υποβάθρου",
        "tooltip_title_export": "Εξαγωγή PDF",
        "tooltip_desc_export": "Αποθήκευση του τρέχοντος αποτελέσματος ως αναφορά PDF",
        "tooltip_tip_export": "Περιλαμβάνει εικόνα, διάγνωση και πιθανότητες κλάσεων",
        "tooltip_title_dash_clear": "Εκκαθάριση στατιστικών",
        "tooltip_desc_dash_clear": "Επαναφορά όλων των στατιστικών αυτού του λογαριασμού",
        "tooltip_tip_dash_clear": "Αυτή η ενέργεια δεν αναιρείται",
        "tooltip_title_dash_close": "Κλείσιμο πίνακα",
        "tooltip_desc_dash_close": "Απόκρυψη του πίνακα στατιστικών",
        "tooltip_desc_dash_total": "Συνολικός αριθμός επεξεργασμένων εικόνων",
        "tooltip_desc_dash_avg": "Μέση βεβαιότητα σε όλα τα αποτελέσματα",
        "tooltip_desc_dash_dist": "Κατανομή αποτελεσμάτων ανά βαθμό",
        "tooltip_title_hist_clear": "Εκκαθάριση ιστορικού",
        "tooltip_desc_hist_clear": "Διαγραφή όλων των αποθηκευμένων αποτελεσμάτων και μικρογραφιών",
        "tooltip_tip_hist_clear": "Αυτό θα διαγράψει επίσης όλα τα αρχεία εικόνων",
        "tooltip_title_hist_close": "Κλείσιμο ιστορικού",
        "tooltip_desc_hist_close": "Απόκρυψη του πίνακα ιστορικού",
        "tooltip_tip_thumb_click": "Κάντε κλικ για επαναφόρτωση αυτού του αποτελέσματος",
        "tooltip_tip_recent": "Κάντε κλικ για επαναφόρτωση αυτού του αποτελέσματος",
        "tooltip_title_zoom_in": "Μεγέθυνση",
        "tooltip_desc_zoom_in": "Αύξηση μεγέθυνσης εικόνας",
        "tooltip_title_zoom_out": "Σμίκρυνση",
        "tooltip_desc_zoom_out": "Μείωση μεγέθυνσης εικόνας",
        "tooltip_title_zoom_reset": "Επαναφορά προβολής",
        "tooltip_desc_zoom_reset": "Προσαρμογή της εικόνας στο παράθυρο",
        "tooltip_tip_zoom_reset": "Μπορείτε επίσης να κάνετε διπλό κλικ στην εικόνα",
        "notif_inference_ok": "Η εξαγωγή ολοκληρώθηκε",
        "follow_system_theme": "Ακολούθηση θέματος συστήματος",
        "follow_system_theme_desc": "Αυτόματη εναλλαγή μεταξύ φωτεινού και σκούρου με βάση τις ρυθμίσεις συστήματος",
        "gradcam_title": "Χάρτης θερμότητας προσοχής",
        "gradcam_show": "Εμφάνιση χάρτη προσοχής",
        "gradcam_toggle": "Εναλλαγή χάρτη",
        "gradcam_save": "Αποθήκευση εικόνας",
        "gradcam_close": "Κλείσιμο",
        "gradcam_saved": "Η εικόνα αποθηκεύτηκε με επιτυχία",
        "gradcam_no_heatmap": "Δεν υπάρχει διαθέσιμος χάρτης προσοχής. Επεξεργαστείτε πρώτα μια εικόνα.",
        "update_available": "Διαθέσιμη ενημέρωση",
        "update_check_failed": "Ο έλεγχος ενημερώσεων απέτυχε",
        "check_for_updates": "Έλεγχος για ενημερώσεις",
        "check_for_updates_desc": "Έλεγχος GitHub/ιστοσελίδας για την τελευταία έκδοση",
        "up_to_date": "Είστε ενημερωμένοι",
        "you_have_latest": "Χρησιμοποιείτε την τελευταία έκδοση.",
        "check_internet": "Ελέγξτε τη σύνδεσή σας στο διαδίκτυο και δοκιμάστε ξανά.",
        "downloading": "Λήψη...",
        "verifying": "Επαλήθευση...",
        "download_now": "Λήψη τώρα",
        "install_now": "Εγκατάσταση & Επανεκκίνηση",
        "skip_version": "Παράλειψη αυτής της έκδοσης",
        "remind_later": "Υπενθύμιση αργότερα",
        "whats_new": "Τι νέο υπάρχει",
        "no_changelog": "Δεν παρέχεται αρχείο αλλαγών.",
        "critical_update": "Κρίσιμη ενημέρωση",
        "update_ready": "Έτοιμο για εγκατάσταση",
        "app_will_restart": "Η εφαρμογή θα κλείσει και θα επανεκκινήσει για να εφαρμοστεί η ενημέρωση.",
        "install_failed": "Η εγκατάσταση απέτυχε. Η παλιά σας έκδοση είναι ασφαλής.",
        "download_failed": "Η λήψη απέτυχε.",
        "source_update_requires_zip": "Η ενημέρωση πηγαίου κώδικα απαιτεί αρχείο .zip.",
        "session_locked": "Κλειδωμένη Συνεδρία",
        "locked_for_user": "Κλειδωμένη για {user}",
        "unlock": "Ξεκλείδωμα",
        "session_unlocked": "Η συνεδρία ξεκλειδώθηκε",
        "auto_lock_notice": "Η συνεδρία θα κλειδωθεί μετά από 2 λεπτά αδράνειας.",
        "audit_log_title": "Αρχείο Ελέγχου",
        "audit_export": "Εξαγωγή",
        "audit_exported": "Εξήχθη με επιτυχία",
        "audit_export_failed": "Η εξαγωγή απέτυχε",
        "audit_search_placeholder": "Αναζήτηση ενεργειών ή λεπτομερειών...",
        "audit_col_time": "Ώρα",
        "audit_col_level": "Επίπεδο",
        "audit_col_category": "Κατηγορία",
        "audit_col_action": "Ενέργεια",
        "audit_col_user": "Χρήστης",
        "audit_details_title": "Λεπτομέρειες Συμβάντος",
        "audit_status": "Εμφάνιση {count} από {total} συμβάντων",
        "strength_very_weak": "Πολύ αδύναμος",
        "strength_weak": "Αδύναμος",
        "strength_medium": "Μέτριος",
        "strength_strong": "Ισχυρός",
        "strength_very_strong": "Πολύ ισχυρός",
        "strength_rule_length": "8-16 χαρακτήρες",
        "strength_rule_lowercase": "Πεζό γράμμα",
        "strength_rule_uppercase": "Κεφαλαίο γράμμα",
        "strength_rule_digits": "Ψηφία",
        "strength_rule_special": "Ειδικοί χαρακτήρες",
        "strength_add_chars": "Προσθέστε {n} ακόμη χαρακτήρα(-ες)",
        "strength_remove_chars": "Αφαιρέστε {n} χαρακτήρα(-ες)",
        "strength_need_digits": "Προσθέστε {n} ακόμη ψηφίο(-α)",
        "strength_need_special": "Προσθέστε {n} ακόμη ειδικό(-ούς) χαρακτήρα(-ες)",
        "strength_common_pattern": "Αποφύγετε κοινές λέξεις",
        "strength_common_year": "Αποφύγετε κοινές χρονιές",
        "strength_has_sequence": "Αποφύγετε ακολουθίες (1234, abcd)",
        "strength_has_repeats": "Αποφύγετε επαναλήψεις (aaaa, 1111)",
        "passwords_match": "Οι κωδικοί ταιριάζουν",
        "2fa_setup_title": "Ενεργοποίηση ελέγχου ταυτότητας δύο παραγόντων",
        "2fa_setup_subtitle": "Σαρώστε τον κωδικό QR με το Google Authenticator, το Authy ή το Microsoft Authenticator.",
        "2fa_manual_secret": "Ή εισαγάγετε αυτόν τον κωδικό χειροκίνητα:",
        "2fa_enter_code": "Εισαγάγετε τον 6ψήφιο κωδικό από την εφαρμογή σας:",
        "2fa_verify_btn": "Επαλήθευση",
        "2fa_backup_codes_title": "Αποθηκεύστε αυτούς τους εφεδρικούς κωδικούς!",
        "2fa_backup_codes_hint": "Κάθε κωδικός μπορεί να χρησιμοποιηθεί μόνο μία φορά. Φυλάξτε τους σε ασφαλές μέρος.",
        "2fa_done_btn": "Τέλος",
        "2fa_secret_copied": "Ο μυστικός κωδικός αντιγράφηκε στο πρόχειρο",
        "2fa_invalid_code": "Μη έγκυρος κωδικός. Δοκιμάστε ξανά.",
        "2fa_invalid_code_format": "Ο κωδικός πρέπει να έχει 6 ψηφία.",
        "2fa_verify_title": "Έλεγχος ταυτότητας δύο παραγόντων",
        "2fa_verify_subtitle": "Εισαγάγετε τον 6ψήφιο κωδικό από την εφαρμογή ελέγχου ταυτότητας.",
        "2fa_enter_code_hint": "Παρακαλώ εισαγάγετε έναν κωδικό.",
        "2fa_use_backup_code": "Χρήση εφεδρικού κωδικού",
        "2fa_use_totp_code": "Χρήση κωδικού εφαρμογής",
        "2fa_enter_backup_code": "Εισαγάγετε έναν από τους εφεδρικούς κωδικούς σας.",
        "2fa_section_title": "Ασφάλεια",
        "2fa_enabled_status": "Ο έλεγχος ταυτότητας δύο παραγόντων είναι ενεργός",
        "2fa_disabled_status": "Ο έλεγχος ταυτότητας δύο παραγόντων είναι ανενεργός",
        "2fa_enable_btn": "Ενεργοποίηση 2FA",
        "2fa_disable_btn": "Απενεργοποίηση 2FA",
        "2fa_disable_title": "Απενεργοποίηση 2FA",
        "2fa_disable_confirm": "Είστε σίγουροι; Ο λογαριασμός σας θα είναι λιγότερο ασφαλής.",
        "2fa_disabled_success": "Ο έλεγχος ταυτότητας δύο παραγόντων απενεργοποιήθηκε",
        "2fa_enabled_success": "Ο έλεγχος ταυτότητας δύο παραγόντων ενεργοποιήθηκε",
        "2fa_hint_title": "Πώς να χρησιμοποιήσετε αυτόν τον κωδικό QR",
        "2fa_hint_step1": "Εγκαταστήστε μια εφαρμογή ελέγχου ταυτότητας: Google Authenticator, Microsoft Authenticator ή Authy.",
        "2fa_hint_step2": "Ανοίξτε την εφαρμογή, πατήστε το κουμπί «+» και επιλέξτε «Σάρωση κωδικού QR».",
        "2fa_hint_step3": "Στρέψτε την κάμερα του τηλεφώνου σας στον κωδικό QR παραπάνω (χρησιμοποιήστε την κάμερα της εφαρμογής, όχι της κάμερας του τηλεφώνου).",
        "2fa_hint_step4": "Η εφαρμογή θα εμφανίσει έναν 6ψήφιο κωδικό. Εισαγάγετέ τον στο παρακάτω πλαίσιο.",
        "2fa_manual_secret_hint": "Δεν μπορείτε να σαρώσετε; Αντιγράψτε το κλειδί παρακάτω και προσθέστε το χειροκίνητα στην εφαρμογή («Εισαγωγή κλειδιού ρύθμισης»).",
        "auto_lock_section_title": "Αυτόματο Κλείδωμα",
        "auto_lock_enable": "Κλείδωμα μετά από αδράνεια",
        "auto_lock_timeout_label": "Κλείδωμα μετά από:",
        "auto_lock_warning_title": "Προειδοποίηση Αυτόματου Κλειδώματος",
        "auto_lock_warning_body": "Η συνεδρία σας θα κλειδωθεί σε {seconds} δευτερόλεπτα λόγω αδράνειας.",
        "lock_now": "Κλείδωμα τώρα",
        "font_section_title": "Γραμματοσειρά",
        "font_family_label": "Οικογένεια γραμματοσειράς",
        "font_size_label": "Βασικό μέγεθος",
        "font_scale_label": "Κλίμακα εμφάνισης",
        "font_scale_small": "Μικρή",
        "font_scale_normal": "Κανονική",
        "font_scale_large": "Μεγάλη",
        "font_scale_xlarge": "Πολύ μεγάλη",
        "font_scale_xxlarge": "Τεράστια",
        "font_preview_text": "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία — 1234567890",
        "font_reset": "Επαναφορά προεπιλογής",
        "font_applied": "Η γραμματοσειρά ενημερώθηκε",
        "motion_section_title": "Κίνηση και Κινούμενα Σχέδια",
        "reduce_motion_label": "Μείωση κίνησης",
        "reduce_motion_hint": "Απενεργοποιήστε τα κινούμενα σχέδια αν σας προκαλούν δυσφορία.",
        "empty_history_empty_title": "Δεν υπάρχει ιστορικό ακόμα",
        "empty_history_empty_subtitle": "Επεξεργαστείτε την πρώτη σας εικόνα για να τη δείτε εδώ",
        "empty_history_empty_hint": "Κάθε αποτέλεσμα αποθηκεύεται αυτόματα",
        "empty_history_no_user_title": "Συνδεθείτε για να δείτε το ιστορικό",
        "empty_history_no_user_subtitle": "Τα αποθηκευμένα αποτελέσματά σας θα εμφανιστούν εδώ",
        "empty_history_no_user_hint": "Το ιστορικό είναι ιδιωτικό για κάθε λογαριασμό",
        "empty_dash_no_results_title": "Δεν υπάρχουν αποτελέσματα ακόμα",
        "empty_dash_no_results_subtitle": "Οι πρόσφατες αναλύσεις σας θα εμφανιστούν εδώ",
        "empty_dash_no_results_hint": "Επεξεργαστείτε μια εικόνα για να ξεκινήσετε",
        "empty_dash_no_dist_title": "Δεν υπάρχουν δεδομένα κατανομής",
        "empty_dash_no_dist_subtitle": "Τα στατιστικά θα συμπληρώνονται καθώς επεξεργάζεστε εικόνες",
        "empty_dash_no_dist_hint": "Κάθε ανάλυση μετράει",
        "empty_image_empty_title": "Δεν έχει εισαχθεί εικόνα",
        "empty_image_empty_subtitle": "Σύρετε και αφήστε μια εικόνα εδώ ή πατήστε Εισαγωγή",
        "empty_image_empty_hint": "Υποστηρίζει PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Αφήστε για ανάλυση",
        "empty_image_drop_hint_subtitle": "Αφήστε την εικόνα για να φορτωθεί",
        "empty_image_drop_hint_hint": "Η μορφή θα ανιχνευθεί αυτόματα",
        "empty_processing_wait_title": "Επεξεργασία...",
        "empty_processing_wait_subtitle": "Το AI αναλύει την εικόνα σας",
        "empty_processing_wait_hint": "Αυτό συνήθως διαρκεί λίγα δευτερόλεπτα",
        "tour_show_again": "Εμφάνιση ξανά της περιήγησης",
        "tour_completed_msg": "Η περιήγηση ολοκληρώθηκε! Είστε έτοιμοι.",
        "tour_skipped_msg": "Η περιήγηση παραλείφθηκε. Μπορείτε να την επαναλάβετε από το μενού Λογαριασμού.",
        "tour_btn_next": "Επόμενο",
        "tour_btn_prev": "Πίσω",
        "tour_btn_skip": "Παράλειψη",
        "tour_btn_finish": "Τέλος ✓",
        "tour_guest_login_title": "Σύνδεση",
        "tour_guest_login_subtitle": "Κάντε κλικ εδώ για να συνδεθείτε και να ξεκλειδώσετε όλες τις λειτουργίες.",
        "tour_guest_signup_title": "Δημιουργία λογαριασμού",
        "tour_guest_signup_subtitle": "Νέος εδώ; Κάντε κλικ εδώ για να δημιουργήσετε λογαριασμό σε δευτερόλεπτα.",
        "tour_insert_title": "Εισαγωγή εικόνας",
        "tour_insert_subtitle": "Φορτώστε μια εικόνα υπερήχου από τον δίσκο ή σύρετε και αφήστε την οπουδήποτε στο παράθυρο.",
        "tour_process_title": "Επεξεργασία εικόνας",
        "tour_process_subtitle": "Εκτελέστε ανάλυση AI στη φορτωμένη εικόνα. Τα αποτελέσματα εμφανίζονται σε λίγα δευτερόλεπτα.",
        "tour_export_title": "Εξαγωγή PDF",
        "tour_export_subtitle": "Αποθηκεύστε το τρέχον αποτέλεσμα ως επαγγελματική ιατρική αναφορά PDF.",
        "tour_gradcam_title": "Χάρτης θερμότητας προσοχής",
        "tour_gradcam_subtitle": "Δείτε πού κοιτάει το AI — έναν χάρτη θερμότητας πάνω στην αναλυόμενη εικόνα.",
        "tour_dashboard_title": "Πίνακας ελέγχου",
        "tour_dashboard_subtitle": "Εμφάνιση/απόκρυψη του πίνακα στατιστικών: σύνολο εικόνων, μέση βεβαιότητα, κατανομή βαθμών.",
        "tour_history_title": "Ιστορικό",
        "tour_history_subtitle": "Περιηγηθείτε στις τελευταίες 20 αναλυμένες εικόνες. Κάντε κλικ σε μια μικρογραφία για επαναφόρτωση.",
        "tour_visit_title": "Επισκεφθείτε την ιστοσελίδα",
        "tour_visit_subtitle": "Ανοίξτε τη σελίδα GitHub του έργου στον browser για τεκμηρίωση και ενημερώσεις.",
        "tour_mute_title": "Σίγαση μουσικής",
        "tour_mute_subtitle": "Παύση ή συνέχιση της μουσικής υποβάθρου ανά πάσα στιγμή.",
        "tour_zoom_title": "Έλεγχοι μεγέθυνσης",
        "tour_zoom_subtitle": "Χρησιμοποιήστε τον τροχό για μεγέθυνση, σύρετε για μετακίνηση, διπλό κλικ για επαναφορά.",
        "tour_account_title": "Μενού λογαριασμού",
        "tour_account_subtitle": "Κάντε κλικ εδώ για πρόσβαση σε Ρυθμίσεις, Πληροφορίες λογαριασμού, Αρχείο ελέγχου και άλλα.",
        "suggestion_image_ready": "Η εικόνα φορτώθηκε — έτοιμη για επεξεργασία",
        "suggestion_image_ready_sub": "Πατήστε Επεξεργασία για να ξεκινήσει η ανάλυση AI.",
        "suggestion_pdf_ready": "Το αποτέλεσμα είναι έτοιμο — εξαγωγή σε PDF;",
        "suggestion_pdf_ready_sub": "Αποθηκεύστε μια επαγγελματική ιατρική αναφορά με ένα κλικ.",
        "suggestion_gradcam_hint": "Περίεργος τι κοίταξε το AI;",
        "suggestion_gradcam_hint_sub": "Δείτε τον χάρτη θερμότητας προσοχής πάνω στην εικόνα.",
        "suggestion_repeated_grid": "Παίρνετε πάντα {grid}",
        "suggestion_repeated_grid_sub": "Ανοίξτε τον πίνακα για να δείτε την κατανομή.",
        "suggestion_history_hint": "Το ιστορικό σας μεγαλώνει",
        "suggestion_history_hint_sub": "Περιηγηθείτε στις τελευταίες 20 αναλυμένες εικόνες.",
        "suggestion_dashboard_hint": "Ο πίνακας σας έχει ενδιαφέροντα στατιστικά",
        "suggestion_dashboard_hint_sub": "Σύνολα, μέσοι όροι και κατανομή με μια ματιά.",
        "suggestion_settings_hint": "Προσαρμόστε την εφαρμογή",
        "suggestion_settings_hint_sub": "Αλλάξτε γλώσσα, θέμα, γραμματοσειρά και ενεργοποιήστε 2FA.",
        "suggestion_idle_help": "Χρειάζεστε έναν γρήγορο οδηγό;",
        "suggestion_idle_help_sub": "Επαναλάβετε την ξενάγηση οποιαδήποτε στιγμή.",
        "image_not_liver_warning": "Αυτή η εικόνα δεν μοιάζει με υπερηχογράφημα ήπατος.\n\nΠαρακαλώ εισάγετε μια κατάλληλη εικόνα υπερηχογραφήματος ήπατος.",
        "result_not_confident": "Το AI δεν μπόρεσε να αναγνωρίσει με βεβαιότητα αυτή την εικόνα ως υπερηχογράφημα ήπατος.\n\nΠαρακαλώ εισάγετε μια κατάλληλη εικόνα υπερηχογραφήματος ήπατος και δοκιμάστε ξανά.",
        "not_liver_title": "Δεν είναι υπερηχογράφημα ήπατος",
        "not_liver_hint": "Παρακαλώ χρησιμοποιήστε μια κατάλληλη εικόνα υπερηχογραφήματος (υπερήχου) ήπατος",
        "voice_welcome": "Καλώς ήρθες, {user}",
        "voice_goodbye": "Αντίο",
        "voice_logged_out": "Αποσυνδεθήκατε με επιτυχία",
        "voice_image_loaded": "Η εικόνα φορτώθηκε με επιτυχία",
        "voice_image_rejected": "Αυτό δεν είναι υπερηχογράφημα ήπατος",
        "voice_inference_result": "{grid_name}, βεβαιότητα {conf} τοις εκατό",
        "voice_pdf_saved": "Η αναφορά PDF αποθηκεύτηκε με επιτυχία",
        "voice_error": "Παρουσιάστηκε σφάλμα",
        "voice_settings_applied": "Οι ρυθμίσεις εφαρμόστηκαν",
        "voice_language_changed": "Η γλώσσα άλλαξε σε {lang}",
        "voice_theme_changed": "Το θέμα άλλαξε",
        "voice_2fa_enabled": "Ο έλεγχος ταυτότητας δύο παραγόντων ενεργοποιήθηκε",
        "voice_2fa_disabled": "Ο έλεγχος ταυτότητας δύο παραγόντων απενεργοποιήθηκε",
        "voice_locked": "Η συνεδρία κλειδώθηκε",
        "voice_unlocked": "Η συνεδρία ξεκλειδώθηκε",
        "voice_processing": "Επεξεργασία εικόνας",
        "voice_processing_done": "Η επεξεργασία ολοκληρώθηκε",
        "voice_voice_enabled": "Η φωνητική ανάδραση ενεργοποιήθηκε",
        "voice_voice_disabled": "Η φωνητική ανάδραση απενεργοποιήθηκε",
    },
    "Germany": {
        "app_title": "Leber KI-Pipeline",
        "welcome": "Willkommen",
        "login": "Anmelden",
        "signup": "Registrieren",
        "logout": "Abmelden",
        "exit": "Beenden",
        "settings": "Einstellungen",
        "account_info": "Kontoinformationen",
        "username": "Benutzername",
        "password": "Passwort",
        "repeat_password": "Passwort wiederholen",
        "first_name": "Vorname",
        "last_name": "Nachname",
        "email": "E-Mail-Adresse",
        "remember_me": "Angemeldet bleiben",
        "login_btn": "Anmelden",
        "signup_btn": "Registrieren",
        "account": "Konto",
        "change_language": "Sprache ändern",
        "change_theme": "Thema ändern",
        "insert_sono": "Sonogramm einfügen",
        "process": "Verarbeiten",
        "mute": "Stumm",
        "unmute": "Ton an",
        "visit_webpage": "Webseite besuchen",
        "please_wait": "Bitte warten...",
        "processing": "Bild wird verarbeitet...",
        "result": "Ergebnis",
        "confidence": "Konfidenz",
        "grid": "Raster",
        "insert_image_first": "Bitte zuerst ein Bild einfügen!",
        "no_image": "Kein Bild eingefügt",
        "invalid_credentials": "Ungültige Anmeldedaten",
        "user_exists": "Benutzername existiert bereits",
        "password_requirements": "Passwortanforderungen",
        "password_rule": (
            "Das Passwort muss:\n"
            "• 8-16 Zeichen lang sein\n"
            "• Mindestens 1 Kleinbuchstaben enthalten\n"
            "• Mindestens 1 Großbuchstaben enthalten\n"
            "• Mindestens 4 Ziffern enthalten\n"
            "• Mindestens 4 Sonderzeichen enthalten"
        ),
        "passwords_dont_match": "Passwörter stimmen nicht überein!",
        "invalid_email": "Bitte eine gültige E-Mail-Adresse eingeben",
        "empty_fields": "Bitte alle Felder ausfüllen",
        "signup_success": "Registrierung erfolgreich! Sie können sich jetzt anmelden.",
        "logout_confirm": "Möchten Sie Ihr Konto wirklich löschen?",
        "exit_confirm": "Möchten Sie wirklich beenden?",
        "theme_dark_aqua": "Dunkles Aqua-Purple-Thema",
        "theme_solarized_white": "Helles Solarized-Thema",
        "theme_solarized_dark": "Dunkles Solarized-Thema",
        "theme_dark_forest": "Dunkles Waldgrün-Thema",
        "select_language": "Sprache auswählen",
        "select_theme": "Thema auswählen",
        "apply": "Anwenden",
        "cancel": "Abbrechen",
        "save": "Speichern",
        "success": "Erfolg",
        "error": "Fehler",
        "warning": "Warnung",
        "settings_applied": "Einstellungen angewendet!",
        "grid_0": "Raster 0 (NL) — Normale Leber",
        "grid_1": "Raster 1 (G1) — Grad 1",
        "grid_2": "Raster 2 (G2) — Grad 2",
        "grid_3": "Raster 3 (G3) — Grad 3",
        "too_many_attempts": "Zu viele fehlgeschlagene Versuche. Bitte 5 Minuten warten.",
        "invalid_username": "Benutzername muss 3-32 Zeichen haben (Buchstaben, Zahlen, Unterstrich).",
        "welcome_user": "Willkommen, {user}!",
        "inference_failed": "Inferenz fehlgeschlagen",
        "drop_here": "Bild hier ablegen",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Nicht unterstützter Dateityp. Bitte ein Bild ablegen.",
        "image_loaded": "Bild erfolgreich geladen",
        "help_about_title": "Hilfe & Über",
        "help_about_text": (
            "Leber KI-Pipeline v3.0\n"
            "Entwickelt von Babak Yousefian\n\n"
            "Tastenkürzel:\n"
            "  Ctrl+O      — Bild einfügen\n"
            "  Ctrl+Enter  — Bild verarbeiten\n"
            "  Ctrl+L      — Anmelden\n"
            "  Ctrl+,      — Einstellungen\n"
            "  Esc         — Aktiven Dialog schließen\n"
            "  F1          — Hilfe / Über\n\n"
            "Tipp: Sie können ein Bild auch an eine beliebige Stelle im Fenster ziehen und ablegen."
        ),
        "dashboard_title": "Übersicht",
        "dash_total": "Bilder insgesamt",
        "dash_avg_conf": "Durchschn. Konfidenz",
        "dash_distribution": "Verteilung",
        "dash_last_5": "Letzte 5 Ergebnisse",
        "dash_no_results": "Noch keine Ergebnisse",
        "dash_clear_confirm": "Alle Statistiken löschen?",
        "dash_cleared": "Statistiken gelöscht",
        "dashboard_toggle": "Übersicht ein-/ausblenden",
        "export_pdf": "PDF exportieren",
        "pdf_saved": "PDF erfolgreich gespeichert",
        "pdf_failed": "PDF-Erstellung fehlgeschlagen",
        "pdf_subtitle": "Analysebericht",
        "pdf_patient": "Patient:",
        "pdf_date": "Datum:",
        "pdf_model": "Modell:",
        "pdf_diagnosis": "DIAGNOSE",
        "pdf_confidence": "Konfidenz",
        "pdf_class_probs": "KLASSENWAHRSCHEINLICHKEITEN",
        "pdf_disclaimer": (
            "Dieser Bericht wurde von Babak Yousefian unter Aufsicht erfahrener Ärzte erstellt. "
            "Es handelt sich um ein professionelles Analysewerkzeug, das keine ärztliche Überweisung erfordert."
        ),
        "history_title": "Verlauf",
        "history_toggle": "Verlauf ein-/ausblenden",
        "history_empty": "Noch kein Verlauf",
        "history_cleared": "Verlauf gelöscht",
        "history_clear_confirm": "Gesamten Verlauf löschen?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Anmelden",
        "tooltip_desc_login": "Melden Sie sich bei Ihrem Konto an",
        "tooltip_tip_login": "Zugriff auf gespeicherten Verlauf und Statistiken",
        "tooltip_title_signup": "Registrieren",
        "tooltip_desc_signup": "Ein neues Konto erstellen",
        "tooltip_title_account": "Konto",
        "tooltip_desc_account": "Klicken, um das Kontomenü zu öffnen",
        "tooltip_desc_account_user": "Angemeldet als {user}",
        "tooltip_title_dashboard": "Übersicht",
        "tooltip_desc_dashboard": "Statistik-Panel ein-/ausblenden — Gesamt, Verteilung, letzte Ergebnisse",
        "tooltip_title_history": "Verlauf",
        "tooltip_desc_history": "Letzte 20 Ergebnisse mit Miniaturansichten ein-/ausblenden",
        "tooltip_title_visit": "Webseite besuchen",
        "tooltip_desc_visit": "GitHub-Seite des Projekts im Browser öffnen",
        "tooltip_tip_visit": "Öffnet im externen Browser",
        "tooltip_title_insert": "Bild einfügen",
        "tooltip_desc_insert": "Ein Sono-Bild zum Analysieren laden",
        "tooltip_tip_insert": "Sie können ein Bild auch an eine beliebige Stelle im Fenster ziehen und ablegen",
        "tooltip_title_process": "Verarbeiten",
        "tooltip_desc_process": "KI-Analyse des geladenen Bildes starten",
        "tooltip_tip_process": "Erfordert zunächst ein geladenes Bild",
        "tooltip_title_mute": "Musik stumm",
        "tooltip_desc_mute": "Hintergrundmusik pausieren",
        "tooltip_title_unmute": "Musik an",
        "tooltip_desc_unmute": "Hintergrundmusik fortsetzen",
        "tooltip_title_export": "PDF exportieren",
        "tooltip_desc_export": "Aktuelles Ergebnis als PDF-Bericht speichern",
        "tooltip_tip_export": "Enthält Bild, Diagnose und Klassenwahrscheinlichkeiten",
        "tooltip_title_dash_clear": "Statistiken löschen",
        "tooltip_desc_dash_clear": "Alle Statistiken dieses Kontos zurücksetzen",
        "tooltip_tip_dash_clear": "Diese Aktion kann nicht rückgängig gemacht werden",
        "tooltip_title_dash_close": "Übersicht schließen",
        "tooltip_desc_dash_close": "Statistik-Panel ausblenden",
        "tooltip_desc_dash_total": "Gesamtzahl der verarbeiteten Bilder",
        "tooltip_desc_dash_avg": "Durchschnittliche Konfidenz über alle Ergebnisse",
        "tooltip_desc_dash_dist": "Verteilung der Ergebnisse nach Graden",
        "tooltip_title_hist_clear": "Verlauf löschen",
        "tooltip_desc_hist_clear": "Alle gespeicherten Ergebnisse und Miniaturansichten löschen",
        "tooltip_tip_hist_clear": "Dadurch werden auch alle Bilddateien gelöscht",
        "tooltip_title_hist_close": "Verlauf schließen",
        "tooltip_desc_hist_close": "Verlauf-Panel ausblenden",
        "tooltip_tip_thumb_click": "Klicken, um dieses Ergebnis neu zu laden",
        "tooltip_tip_recent": "Klicken, um dieses Ergebnis neu zu laden",
        "tooltip_title_zoom_in": "Vergrößern",
        "tooltip_desc_zoom_in": "Bildvergrößerung erhöhen",
        "tooltip_title_zoom_out": "Verkleinern",
        "tooltip_desc_zoom_out": "Bildvergrößerung verringern",
        "tooltip_title_zoom_reset": "Ansicht zurücksetzen",
        "tooltip_desc_zoom_reset": "Bild an das Fenster anpassen",
        "tooltip_tip_zoom_reset": "Sie können auch auf das Bild doppelklicken",
        "notif_inference_ok": "Inferenz abgeschlossen",
        "follow_system_theme": "Systemthema folgen",
        "follow_system_theme_desc": "Automatisch zwischen Hell und Dunkel wechseln basierend auf Systemeinstellungen",
        "gradcam_title": "Aufmerksamkeits-Heatmap",
        "gradcam_show": "Aufmerksamkeitskarte anzeigen",
        "gradcam_toggle": "Karte umschalten",
        "gradcam_save": "Bild speichern",
        "gradcam_close": "Schließen",
        "gradcam_saved": "Bild erfolgreich gespeichert",
        "gradcam_no_heatmap": "Keine Aufmerksamkeitskarte verfügbar. Bitte zuerst ein Bild verarbeiten.",
        "update_available": "Update verfügbar",
        "update_check_failed": "Update-Prüfung fehlgeschlagen",
        "check_for_updates": "Nach Updates suchen",
        "check_for_updates_desc": "GitHub/Website auf die neueste Version prüfen",
        "up_to_date": "Sie sind auf dem neuesten Stand",
        "you_have_latest": "Sie verwenden die neueste Version.",
        "check_internet": "Bitte überprüfen Sie Ihre Internetverbindung und versuchen Sie es erneut.",
        "downloading": "Wird heruntergeladen...",
        "verifying": "Wird überprüft...",
        "download_now": "Jetzt herunterladen",
        "install_now": "Installieren & Neu starten",
        "skip_version": "Diese Version überspringen",
        "remind_later": "Später erinnern",
        "whats_new": "Was ist neu",
        "no_changelog": "Kein Änderungsprotokoll bereitgestellt.",
        "critical_update": "Kritisches Update",
        "update_ready": "Bereit zur Installation",
        "app_will_restart": "Die App wird geschlossen und neu gestartet, um das Update anzuwenden.",
        "install_failed": "Installation fehlgeschlagen. Ihre alte Version ist sicher.",
        "download_failed": "Download fehlgeschlagen.",
        "source_update_requires_zip": "Quellcode-Update erfordert eine .zip-Datei.",
        "session_locked": "Sitzung gesperrt",
        "locked_for_user": "Gesperrt für {user}",
        "unlock": "Entsperren",
        "session_unlocked": "Sitzung entsperrt",
        "auto_lock_notice": "Die Sitzung wird nach 2 Minuten Inaktivität gesperrt.",
        "audit_log_title": "Audit-Protokoll",
        "audit_export": "Exportieren",
        "audit_exported": "Erfolgreich exportiert",
        "audit_export_failed": "Export fehlgeschlagen",
        "audit_search_placeholder": "Aktionen oder Details durchsuchen...",
        "audit_col_time": "Zeit",
        "audit_col_level": "Stufe",
        "audit_col_category": "Kategorie",
        "audit_col_action": "Aktion",
        "audit_col_user": "Benutzer",
        "audit_details_title": "Ereignisdetails",
        "audit_status": "Zeige {count} von {total} Ereignissen",
        "strength_very_weak": "Sehr schwach",
        "strength_weak": "Schwach",
        "strength_medium": "Mittel",
        "strength_strong": "Stark",
        "strength_very_strong": "Sehr stark",
        "strength_rule_length": "8-16 Zeichen",
        "strength_rule_lowercase": "Kleinbuchstabe",
        "strength_rule_uppercase": "Großbuchstabe",
        "strength_rule_digits": "Ziffern",
        "strength_rule_special": "Sonderzeichen",
        "strength_add_chars": "Fügen Sie {n} weitere Zeichen hinzu",
        "strength_remove_chars": "Entfernen Sie {n} Zeichen",
        "strength_need_digits": "Fügen Sie {n} weitere Ziffer(n) hinzu",
        "strength_need_special": "Fügen Sie {n} weitere Sonderzeichen hinzu",
        "strength_common_pattern": "Vermeiden Sie häufige Wörter",
        "strength_common_year": "Vermeiden Sie häufige Jahreszahlen",
        "strength_has_sequence": "Vermeiden Sie Sequenzen (1234, abcd)",
        "strength_has_repeats": "Vermeiden Sie Wiederholungen (aaaa, 1111)",
        "passwords_match": "Passwörter stimmen überein",
        "2fa_setup_title": "Zwei-Faktor-Authentifizierung aktivieren",
        "2fa_setup_subtitle": "Scannen Sie den QR-Code mit Google Authenticator, Authy oder Microsoft Authenticator.",
        "2fa_manual_secret": "Oder geben Sie diesen Code manuell ein:",
        "2fa_enter_code": "Geben Sie den 6-stelligen Code aus Ihrer App ein:",
        "2fa_verify_btn": "Bestätigen",
        "2fa_backup_codes_title": "Speichern Sie diese Backup-Codes!",
        "2fa_backup_codes_hint": "Jeder Code kann nur einmal verwendet werden. Bewahren Sie sie an einem sicheren Ort auf.",
        "2fa_done_btn": "Fertig",
        "2fa_secret_copied": "Geheimcode in die Zwischenablage kopiert",
        "2fa_invalid_code": "Ungültiger Code. Bitte versuchen Sie es erneut.",
        "2fa_invalid_code_format": "Der Code muss 6 Ziffern haben.",
        "2fa_verify_title": "Zwei-Faktor-Authentifizierung",
        "2fa_verify_subtitle": "Geben Sie den 6-stelligen Code aus Ihrer Authentifizierungs-App ein.",
        "2fa_enter_code_hint": "Bitte geben Sie einen Code ein.",
        "2fa_use_backup_code": "Backup-Code verwenden",
        "2fa_use_totp_code": "App-Code verwenden",
        "2fa_enter_backup_code": "Geben Sie einen Ihrer Backup-Codes ein.",
        "2fa_section_title": "Sicherheit",
        "2fa_enabled_status": "Zwei-Faktor-Authentifizierung ist aktiviert",
        "2fa_disabled_status": "Zwei-Faktor-Authentifizierung ist deaktiviert",
        "2fa_enable_btn": "2FA aktivieren",
        "2fa_disable_btn": "2FA deaktivieren",
        "2fa_disable_title": "2FA deaktivieren",
        "2fa_disable_confirm": "Sind Sie sicher? Ihr Konto wird weniger sicher sein.",
        "2fa_disabled_success": "Zwei-Faktor-Authentifizierung deaktiviert",
        "2fa_enabled_success": "Zwei-Faktor-Authentifizierung aktiviert",
        "2fa_hint_title": "So verwenden Sie diesen QR-Code",
        "2fa_hint_step1": "Installieren Sie eine Authentifizierungs-App: Google Authenticator, Microsoft Authenticator oder Authy.",
        "2fa_hint_step2": "Öffnen Sie die App, tippen Sie auf die Schaltfläche „+“ und wählen Sie „QR-Code scannen“.",
        "2fa_hint_step3": "Richten Sie die Kamera Ihres Telefons auf den QR-Code oben (verwenden Sie die Kamera der App, nicht die Kamera-App Ihres Telefons).",
        "2fa_hint_step4": "Die App zeigt einen 6-stelligen Code an. Geben Sie ihn im Feld unten ein.",
        "2fa_manual_secret_hint": "Können Sie nicht scannen? Kopieren Sie den Schlüssel unten und fügen Sie ihn manuell in Ihrer App hinzu („Einrichtungsschlüssel eingeben“).",
        "auto_lock_section_title": "Automatische Sperre",
        "auto_lock_enable": "Nach Inaktivität sperren",
        "auto_lock_timeout_label": "Sperren nach:",
        "auto_lock_warning_title": "Warnung: Automatische Sperre",
        "auto_lock_warning_body": "Ihre Sitzung wird in {seconds} Sekunden aufgrund von Inaktivität gesperrt.",
        "lock_now": "Jetzt sperren",
        "font_section_title": "Schriftart",
        "font_family_label": "Schriftfamilie",
        "font_size_label": "Basisgröße",
        "font_scale_label": "Anzeigeskalierung",
        "font_scale_small": "Klein",
        "font_scale_normal": "Normal",
        "font_scale_large": "Groß",
        "font_scale_xlarge": "Sehr groß",
        "font_scale_xxlarge": "Riesig",
        "font_preview_text": "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern — 1234567890",
        "font_reset": "Auf Standard zurücksetzen",
        "font_applied": "Schriftart aktualisiert",
        "motion_section_title": "Bewegung & Animationen",
        "reduce_motion_label": "Bewegung reduzieren",
        "reduce_motion_hint": "Deaktivieren Sie Animationen, wenn sie Ihnen unangenehm sind.",
        "empty_history_empty_title": "Noch kein Verlauf",
        "empty_history_empty_subtitle": "Verarbeiten Sie Ihr erstes Bild, um es hier zu sehen",
        "empty_history_empty_hint": "Jedes Ergebnis wird automatisch gespeichert",
        "empty_history_no_user_title": "Melden Sie sich an, um den Verlauf zu sehen",
        "empty_history_no_user_subtitle": "Ihre gespeicherten Ergebnisse erscheinen hier",
        "empty_history_no_user_hint": "Der Verlauf ist für jedes Konto privat",
        "empty_dash_no_results_title": "Noch keine Ergebnisse",
        "empty_dash_no_results_subtitle": "Ihre letzten Analysen erscheinen hier",
        "empty_dash_no_results_hint": "Verarbeiten Sie ein Bild, um zu beginnen",
        "empty_dash_no_dist_title": "Keine Verteilungsdaten",
        "empty_dash_no_dist_subtitle": "Statistiken füllen sich, während Sie Bilder verarbeiten",
        "empty_dash_no_dist_hint": "Jede Analyse zählt",
        "empty_image_empty_title": "Kein Bild eingefügt",
        "empty_image_empty_subtitle": "Ziehen Sie ein Bild hierher oder klicken Sie auf Einfügen",
        "empty_image_empty_hint": "Unterstützt PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Zum Analysieren ablegen",
        "empty_image_drop_hint_subtitle": "Lassen Sie das Bild los, um es zu laden",
        "empty_image_drop_hint_hint": "Das Format wird automatisch erkannt",
        "empty_processing_wait_title": "Verarbeitung...",
        "empty_processing_wait_subtitle": "Die KI analysiert Ihr Bild",
        "empty_processing_wait_hint": "Dies dauert normalerweise ein paar Sekunden",
        "tour_show_again": "Tour erneut anzeigen",
        "tour_completed_msg": "Tour abgeschlossen! Sie sind bereit.",
        "tour_skipped_msg": "Tour übersprungen. Sie können sie im Kontomenü wiederholen.",
        "tour_btn_next": "Weiter",
        "tour_btn_prev": "Zurück",
        "tour_btn_skip": "Überspringen",
        "tour_btn_finish": "Fertig ✓",
        "tour_guest_login_title": "Anmelden",
        "tour_guest_login_subtitle": "Klicken Sie hier, um sich anzumelden und alle Funktionen freizuschalten.",
        "tour_guest_signup_title": "Konto erstellen",
        "tour_guest_signup_subtitle": "Neu hier? Klicken Sie hier, um in Sekunden ein Konto zu erstellen.",
        "tour_insert_title": "Bild einfügen",
        "tour_insert_subtitle": "Laden Sie ein Sono-Bild von der Festplatte oder ziehen Sie es an eine beliebige Stelle im Fenster.",
        "tour_process_title": "Bild verarbeiten",
        "tour_process_subtitle": "Starten Sie die KI-Analyse des geladenen Bildes. Ergebnisse erscheinen in wenigen Sekunden.",
        "tour_export_title": "PDF exportieren",
        "tour_export_subtitle": "Speichern Sie das aktuelle Ergebnis als professionellen medizinischen PDF-Bericht.",
        "tour_gradcam_title": "Aufmerksamkeits-Heatmap",
        "tour_gradcam_subtitle": "Sehen Sie, wohin die KI schaut — eine Heatmap über dem analysierten Bild.",
        "tour_dashboard_title": "Übersicht",
        "tour_dashboard_subtitle": "Statistik-Panel ein-/ausblenden: Gesamtbilder, Durchschnittskonfidenz, Gradeverteilung.",
        "tour_history_title": "Verlauf",
        "tour_history_subtitle": "Durchsuchen Sie Ihre letzten 20 analysierten Bilder. Klicken Sie auf eine Miniatur, um das Ergebnis neu zu laden.",
        "tour_visit_title": "Webseite besuchen",
        "tour_visit_subtitle": "Öffnen Sie die GitHub-Seite des Projekts im Browser für Dokumentation und Updates.",
        "tour_mute_title": "Musik stumm",
        "tour_mute_subtitle": "Pausieren oder setzen Sie die Hintergrundmusik jederzeit fort.",
        "tour_zoom_title": "Zoom-Steuerung",
        "tour_zoom_subtitle": "Mausrad zum Zoomen, Ziehen zum Verschieben, Doppelklick zum Zurücksetzen.",
        "tour_account_title": "Kontomenü",
        "tour_account_subtitle": "Klicken Sie hier für Einstellungen, Kontoinformationen, Audit-Protokoll und mehr.",
        "suggestion_image_ready": "Bild geladen — bereit zur Verarbeitung",
        "suggestion_image_ready_sub": "Klicken Sie auf Verarbeiten, um die KI-Analyse zu starten.",
        "suggestion_pdf_ready": "Ergebnis bereit — als PDF exportieren?",
        "suggestion_pdf_ready_sub": "Speichern Sie einen professionellen medizinischen Bericht mit einem Klick.",
        "suggestion_gradcam_hint": "Neugierig, was die KI gesehen hat?",
        "suggestion_gradcam_hint_sub": "Sehen Sie die Aufmerksamkeits-Heatmap über dem Bild.",
        "suggestion_repeated_grid": "Sie erhalten immer {grid}",
        "suggestion_repeated_grid_sub": "Öffnen Sie die Übersicht, um die Verteilung zu sehen.",
        "suggestion_history_hint": "Ihr Verlauf wächst",
        "suggestion_history_hint_sub": "Durchsuchen Sie Ihre letzten 20 analysierten Bilder.",
        "suggestion_dashboard_hint": "Ihre Übersicht hat interessante Statistiken",
        "suggestion_dashboard_hint_sub": "Gesamtwerte, Durchschnitte und Verteilung auf einen Blick.",
        "suggestion_settings_hint": "Personalisieren Sie die App",
        "suggestion_settings_hint_sub": "Ändern Sie Sprache, Thema, Schriftart und aktivieren Sie 2FA.",
        "suggestion_idle_help": "Brauchen Sie eine kurze Anleitung?",
        "suggestion_idle_help_sub": "Wiederholen Sie die Einführung jederzeit.",
        "image_not_liver_warning": "Dieses Bild sieht nicht wie ein Leber-Ultraschall aus.\n\nBitte fügen Sie ein korrektes Leber-Ultraschallbild ein.",
        "result_not_confident": "Die KI konnte dieses Bild nicht sicher als Leber-Ultraschall identifizieren.\n\nBitte fügen Sie ein korrektes Leber-Ultraschallbild ein und versuchen Sie es erneut.",
        "not_liver_title": "Kein Leber-Ultraschall",
        "not_liver_hint": "Bitte verwenden Sie ein korrektes Leber-Ultraschallbild (Sonogramm)",
        "voice_welcome": "Willkommen, {user}",
        "voice_goodbye": "Auf Wiedersehen",
        "voice_logged_out": "Erfolgreich abgemeldet",
        "voice_image_loaded": "Bild erfolgreich geladen",
        "voice_image_rejected": "Dies ist kein Leber-Ultraschall",
        "voice_inference_result": "{grid_name}, Konfidenz {conf} Prozent",
        "voice_pdf_saved": "PDF-Bericht erfolgreich gespeichert",
        "voice_error": "Ein Fehler ist aufgetreten",
        "voice_settings_applied": "Einstellungen angewendet",
        "voice_language_changed": "Sprache geändert auf {lang}",
        "voice_theme_changed": "Thema geändert",
        "voice_2fa_enabled": "Zwei-Faktor-Authentifizierung aktiviert",
        "voice_2fa_disabled": "Zwei-Faktor-Authentifizierung deaktiviert",
        "voice_locked": "Sitzung gesperrt",
        "voice_unlocked": "Sitzung entsperrt",
        "voice_processing": "Bild wird verarbeitet",
        "voice_processing_done": "Verarbeitung abgeschlossen",
        "voice_voice_enabled": "Sprachfeedback aktiviert",
        "voice_voice_disabled": "Sprachfeedback deaktiviert",
    },
    "Arabic": {
        "app_title": "خط أنابيب الذكاء الاصطناعي للكبد",
        "welcome": "مرحباً",
        "login": "تسجيل الدخول",
        "signup": "التسجيل",
        "logout": "تسجيل الخروج",
        "exit": "خروج",
        "settings": "الإعدادات",
        "account_info": "معلومات الحساب",
        "username": "اسم المستخدم",
        "password": "كلمة المرور",
        "repeat_password": "تكرار كلمة المرور",
        "first_name": "الاسم الأول",
        "last_name": "اسم العائلة",
        "email": "البريد الإلكتروني",
        "remember_me": "تذكرني",
        "login_btn": "تسجيل الدخول",
        "signup_btn": "التسجيل",
        "account": "الحساب",
        "change_language": "تغيير اللغة",
        "change_theme": "تغيير المظهر",
        "insert_sono": "إدراج صورة",
        "process": "معالجة",
        "mute": "كتم الصوت",
        "unmute": "إلغاء الكتم",
        "visit_webpage": "زيارة الموقع",
        "please_wait": "يرجى الانتظار...",
        "processing": "جارٍ معالجة الصورة...",
        "result": "النتيجة",
        "confidence": "الثقة",
        "grid": "الشبكة",
        "insert_image_first": "يرجى إدراج صورة أولاً!",
        "no_image": "لم يتم إدراج صورة",
        "invalid_credentials": "بيانات اعتماد غير صالحة",
        "user_exists": "اسم المستخدم موجود بالفعل",
        "password_requirements": "متطلبات كلمة المرور",
        "password_rule": (
            "يجب أن تكون كلمة المرور:\n"
            "• من 8 إلى 16 حرفاً\n"
            "• تحتوي على حرف صغير واحد على الأقل\n"
            "• تحتوي على حرف كبير واحد على الأقل\n"
            "• تحتوي على 4 أرقام على الأقل\n"
            "• تحتوي على 4 رموز خاصة على الأقل"
        ),
        "passwords_dont_match": "كلمتا المرور غير متطابقتين!",
        "invalid_email": "يرجى إدخال بريد إلكتروني صالح",
        "empty_fields": "يرجى ملء جميع الحقول",
        "signup_success": "تم التسجيل بنجاح! يمكنك الآن تسجيل الدخول.",
        "logout_confirm": "هل أنت متأكد من حذف حسابك؟",
        "exit_confirm": "هل أنت متأكد من الخروج؟",
        "theme_dark_aqua": "مظهر داكن أزرق-بنفسجي",
        "theme_solarized_white": "مظهر فاتح سولاريزد",
        "theme_solarized_dark": "مظهر داكن سولاريزد",
        "theme_dark_forest": "مظهر الغابة الخضراء الداكن",
        "select_language": "اختر اللغة",
        "select_theme": "اختر المظهر",
        "apply": "تطبيق",
        "cancel": "إلغاء",
        "save": "حفظ",
        "success": "نجاح",
        "error": "خطأ",
        "warning": "تحذير",
        "settings_applied": "تم تطبيق الإعدادات!",
        "grid_0": "الشبكة 0 (NL) — كبد طبيعي",
        "grid_1": "الشبكة 1 (G1) — الدرجة 1",
        "grid_2": "الشبكة 2 (G2) — الدرجة 2",
        "grid_3": "الشبكة 3 (G3) — الدرجة 3",
        "too_many_attempts": "محاولات فاشلة كثيرة. يرجى الانتظار 5 دقائق.",
        "invalid_username": "اسم المستخدم يجب أن يكون 3-32 حرفاً (حروف، أرقام، شرطة سفلية).",
        "welcome_user": "مرحباً، {user}!",
        "inference_failed": "فشل الاستدلال",
        "drop_here": "أفلت الصورة هنا",
        "drop_subtitle": "PNG، JPG، JPEG، BMP، DCM",
        "drop_invalid": "نوع الملف غير مدعوم. يرجى إفلات صورة.",
        "image_loaded": "تم تحميل الصورة بنجاح",
        "help_about_title": "المساعدة وحول",
        "help_about_text": (
            "خط أنابيب الذكاء الاصطناعي للكبد الإصدار 3.0\n"
            "تم التطوير بواسطة باباك يوسفيان\n\n"
            "اختصارات لوحة المفاتيح:\n"
            "  Ctrl+O      — إدراج صورة\n"
            "  Ctrl+Enter  — معالجة الصورة\n"
            "  Ctrl+L      — تسجيل الدخول\n"
            "  Ctrl+,      — الإعدادات\n"
            "  Esc         — إغلاق الحوار النشط\n"
            "  F1          — المساعدة / حول\n\n"
            "نصيحة: يمكنك أيضًا سحب وإفلات صورة في أي مكان في النافذة."
        ),
        "dashboard_title": "لوحة التحكم",
        "dash_total": "إجمالي الصور",
        "dash_avg_conf": "متوسط الثقة",
        "dash_distribution": "التوزيع",
        "dash_last_5": "آخر 5 نتائج",
        "dash_no_results": "لا توجد نتائج بعد",
        "dash_clear_confirm": "مسح جميع الإحصائيات؟",
        "dash_cleared": "تم مسح الإحصائيات",
        "dashboard_toggle": "إظهار/إخفاء لوحة التحكم",
        "export_pdf": "تصدير PDF",
        "pdf_saved": "تم حفظ PDF بنجاح",
        "pdf_failed": "فشل إنشاء PDF",
        "pdf_subtitle": "تقرير التحليل",
        "pdf_patient": "المريض:",
        "pdf_date": "التاريخ:",
        "pdf_model": "النموذج:",
        "pdf_diagnosis": "التشخيص",
        "pdf_confidence": "الثقة",
        "pdf_class_probs": "احتمالات الفئات",
        "pdf_disclaimer": (
            "تم تطوير هذا التقرير بواسطة باباك يوسفيان تحت إشراف أطباء ذوي خبرة. "
            "إنها أداة تحليل احترافية ولا تتطلب إحالة إلى طبيب."
        ),
        "history_title": "السجل",
        "history_toggle": "إظهار/إخفاء السجل",
        "history_empty": "لا يوجد سجل بعد",
        "history_cleared": "تم مسح السجل",
        "history_clear_confirm": "مسح كل السجل؟",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "تسجيل الدخول",
        "tooltip_desc_login": "سجل الدخول إلى حسابك",
        "tooltip_tip_login": "الوصول إلى السجل والإحصائيات المحفوظة",
        "tooltip_title_signup": "التسجيل",
        "tooltip_desc_signup": "إنشاء حساب جديد",
        "tooltip_title_account": "الحساب",
        "tooltip_desc_account": "انقر لفتح قائمة الحساب",
        "tooltip_desc_account_user": "مسجل الدخول باسم {user}",
        "tooltip_title_dashboard": "لوحة التحكم",
        "tooltip_desc_dashboard": "إظهار/إخفاء لوحة الإحصائيات — الإجمالي، التوزيع، النتائج الأخيرة",
        "tooltip_title_history": "السجل",
        "tooltip_desc_history": "إظهار/إخفاء آخر 20 نتيجة مع الصور المصغرة",
        "tooltip_title_visit": "زيارة الموقع",
        "tooltip_desc_visit": "فتح صفحة GitHub للمشروع في المتصفح",
        "tooltip_tip_visit": "يفتح في المتصفح الخارجي",
        "tooltip_title_insert": "إدراج صورة",
        "tooltip_desc_insert": "تحميل صورة سونو للتحليل",
        "tooltip_tip_insert": "يمكنك أيضًا سحب وإفلات صورة في أي مكان في النافذة",
        "tooltip_title_process": "معالجة",
        "tooltip_desc_process": "تشغيل تحليل الذكاء الاصطناعي على الصورة المحملة",
        "tooltip_tip_process": "يتطلب تحميل صورة أولاً",
        "tooltip_title_mute": "كتم الموسيقى",
        "tooltip_desc_mute": "إيقاف موسيقى الخلفية مؤقتًا",
        "tooltip_title_unmute": "إلغاء كتم الموسيقى",
        "tooltip_desc_unmute": "استئناف موسيقى الخلفية",
        "tooltip_title_export": "تصدير PDF",
        "tooltip_desc_export": "حفظ النتيجة الحالية كتقرير PDF",
        "tooltip_tip_export": "يشمل الصورة والتشخيص واحتمالات الفئات",
        "tooltip_title_dash_clear": "مسح الإحصائيات",
        "tooltip_desc_dash_clear": "إعادة تعيين جميع الإحصائيات لهذا الحساب",
        "tooltip_tip_dash_clear": "لا يمكن التراجع عن هذا الإجراء",
        "tooltip_title_dash_close": "إغلاق لوحة التحكم",
        "tooltip_desc_dash_close": "إخفاء لوحة الإحصائيات",
        "tooltip_desc_dash_total": "العدد الإجمالي للصور المعالجة",
        "tooltip_desc_dash_avg": "متوسط الثقة عبر جميع النتائج",
        "tooltip_desc_dash_dist": "توزيع النتائج حسب الدرجات",
        "tooltip_title_hist_clear": "مسح السجل",
        "tooltip_desc_hist_clear": "حذف جميع النتائج والصور المصغرة المحفوظة",
        "tooltip_tip_hist_clear": "سيؤدي هذا أيضًا إلى حذف جميع ملفات الصور",
        "tooltip_title_hist_close": "إغلاق السجل",
        "tooltip_desc_hist_close": "إخفاء لوحة السجل",
        "tooltip_tip_thumb_click": "انقر لإعادة تحميل هذه النتيجة",
        "tooltip_tip_recent": "انقر لإعادة تحميل هذه النتيجة",
        "tooltip_title_zoom_in": "تكبير",
        "tooltip_desc_zoom_in": "زيادة تكبير الصورة",
        "tooltip_title_zoom_out": "تصغير",
        "tooltip_desc_zoom_out": "تقليل تكبير الصورة",
        "tooltip_title_zoom_reset": "إعادة تعيين العرض",
        "tooltip_desc_zoom_reset": "ملاءمة الصورة مع النافذة",
        "tooltip_tip_zoom_reset": "يمكنك أيضًا النقر المزدوج على الصورة",
        "notif_inference_ok": "اكتمل الاستدلال",
        "follow_system_theme": "متابعة سمة النظام",
        "follow_system_theme_desc": "التبديل تلقائيًا بين الفاتح والداكن بناءً على إعدادات النظام",
        "gradcam_title": "خريطة الانتباه الحرارية",
        "gradcam_show": "إظهار خريطة الانتباه",
        "gradcam_toggle": "تبديل الخريطة",
        "gradcam_save": "حفظ الصورة",
        "gradcam_close": "إغلاق",
        "gradcam_saved": "تم حفظ الصورة بنجاح",
        "gradcam_no_heatmap": "لا توجد خريطة انتباه متاحة. يرجى معالجة صورة أولاً.",
        "update_available": "تحديث متاح",
        "update_check_failed": "فشل التحقق من التحديثات",
        "check_for_updates": "التحقق من التحديثات",
        "check_for_updates_desc": "التحقق من GitHub/الموقع الإلكتروني للحصول على أحدث إصدار",
        "up_to_date": "أنت محدث",
        "you_have_latest": "أنت تستخدم أحدث إصدار.",
        "check_internet": "يرجى التحقق من اتصالك بالإنترنت والمحاولة مرة أخرى.",
        "downloading": "جارٍ التنزيل...",
        "verifying": "جارٍ التحقق...",
        "download_now": "تنزيل الآن",
        "install_now": "تثبيت وإعادة التشغيل",
        "skip_version": "تخطي هذا الإصدار",
        "remind_later": "ذكرني لاحقاً",
        "whats_new": "ما الجديد",
        "no_changelog": "لم يتم توفير سجل التغييرات.",
        "critical_update": "تحديث حرج",
        "update_ready": "جاهز للتثبيت",
        "app_will_restart": "سيتم إغلاق التطبيق وإعادة تشغيله لتطبيق التحديث.",
        "install_failed": "فشل التثبيت. إصدارك القديم آمن.",
        "download_failed": "فشل التنزيل.",
        "source_update_requires_zip": "تحديث المصدر يتطلب ملف .zip.",
        "session_locked": "الجلسة مقفلة",
        "locked_for_user": "مقفلة لـ {user}",
        "unlock": "إلغاء القفل",
        "session_unlocked": "تم إلغاء قفل الجلسة",
        "auto_lock_notice": "سيتم قفل الجلسة بعد دقيقتين من عدم النشاط.",
        "audit_log_title": "سجل التدقيق",
        "audit_export": "تصدير",
        "audit_exported": "تم التصدير بنجاح",
        "audit_export_failed": "فشل التصدير",
        "audit_search_placeholder": "البحث في الإجراءات أو التفاصيل...",
        "audit_col_time": "الوقت",
        "audit_col_level": "المستوى",
        "audit_col_category": "الفئة",
        "audit_col_action": "الإجراء",
        "audit_col_user": "المستخدم",
        "audit_details_title": "تفاصيل الحدث",
        "audit_status": "عرض {count} من {total} حدثاً",
        "strength_very_weak": "ضعيف جداً",
        "strength_weak": "ضعيف",
        "strength_medium": "متوسط",
        "strength_strong": "قوي",
        "strength_very_strong": "قوي جداً",
        "strength_rule_length": "8-16 حرفاً",
        "strength_rule_lowercase": "حرف صغير",
        "strength_rule_uppercase": "حرف كبير",
        "strength_rule_digits": "أرقام",
        "strength_rule_special": "رموز خاصة",
        "strength_add_chars": "أضف {n} حرفاً آخر",
        "strength_remove_chars": "احذف {n} حرفاً",
        "strength_need_digits": "أضف {n} رقماً آخر",
        "strength_need_special": "أضف {n} رمزاً خاصاً آخر",
        "strength_common_pattern": "تجنب الكلمات الشائعة",
        "strength_common_year": "تجنب السنوات الشائعة",
        "strength_has_sequence": "تجنب التسلسلات (1234، abcd)",
        "strength_has_repeats": "تجنب التكرارات (aaaa، 1111)",
        "passwords_match": "كلمتا المرور متطابقتان",
        "2fa_setup_title": "تفعيل المصادقة الثنائية",
        "2fa_setup_subtitle": "امسح رمز QR باستخدام Google Authenticator أو Authy أو Microsoft Authenticator.",
        "2fa_manual_secret": "أو أدخل هذا الرمز يدويًا:",
        "2fa_enter_code": "أدخل الرمز المكوّن من 6 أرقام من تطبيقك:",
        "2fa_verify_btn": "تحقق",
        "2fa_backup_codes_title": "احفظ رموز النسخ الاحتياطي هذه!",
        "2fa_backup_codes_hint": "يمكن استخدام كل رمز مرة واحدة فقط. احفظها في مكان آمن.",
        "2fa_done_btn": "تم",
        "2fa_secret_copied": "تم نسخ الرمز السري إلى الحافظة",
        "2fa_invalid_code": "رمز غير صالح. يرجى المحاولة مرة أخرى.",
        "2fa_invalid_code_format": "يجب أن يتكون الرمز من 6 أرقام.",
        "2fa_verify_title": "المصادقة الثنائية",
        "2fa_verify_subtitle": "أدخل الرمز المكوّن من 6 أرقام من تطبيق المصادقة.",
        "2fa_enter_code_hint": "يرجى إدخال رمز.",
        "2fa_use_backup_code": "استخدام رمز نسخ احتياطي",
        "2fa_use_totp_code": "استخدام رمز التطبيق",
        "2fa_enter_backup_code": "أدخل أحد رموز النسخ الاحتياطي.",
        "2fa_section_title": "الأمان",
        "2fa_enabled_status": "المصادقة الثنائية مفعّلة",
        "2fa_disabled_status": "المصادقة الثنائية معطّلة",
        "2fa_enable_btn": "تفعيل 2FA",
        "2fa_disable_btn": "تعطيل 2FA",
        "2fa_disable_title": "تعطيل 2FA",
        "2fa_disable_confirm": "هل أنت متأكد؟ سيصبح حسابك أقل أمانًا.",
        "2fa_disabled_success": "تم تعطيل المصادقة الثنائية",
        "2fa_enabled_success": "تم تفعيل المصادقة الثنائية",
        "2fa_hint_title": "كيفية استخدام رمز QR هذا",
        "2fa_hint_step1": "قم بتثبيت تطبيق مصادقة: Google Authenticator أو Microsoft Authenticator أو Authy.",
        "2fa_hint_step2": "افتح التطبيق واضغط على زر \"+\" ثم اختر \"مسح رمز QR\".",
        "2fa_hint_step3": "وجّه كاميرا هاتفك نحو رمز QR أعلاه (استخدم كاميرا التطبيق، وليس كاميرا هاتفك).",
        "2fa_hint_step4": "سيعرض التطبيق رمزًا مكوّنًا من 6 أرقام. أدخله في المربع أدناه.",
        "2fa_manual_secret_hint": "لا يمكنك المسح؟ انسخ المفتاح أدناه وأضفه يدويًا في تطبيقك (\"إدخال مفتاح الإعداد\").",
        "auto_lock_section_title": "القفل التلقائي",
        "auto_lock_enable": "القفل بعد عدم النشاط",
        "auto_lock_timeout_label": "القفل بعد:",
        "auto_lock_warning_title": "تحذير القفل التلقائي",
        "auto_lock_warning_body": "سيتم قفل جلستك خلال {seconds} ثانية بسبب عدم النشاط.",
        "lock_now": "قفل الآن",
        "font_section_title": "الخط",
        "font_family_label": "عائلة الخط",
        "font_size_label": "الحجم الأساسي",
        "font_scale_label": "مقياس العرض",
        "font_scale_small": "صغير",
        "font_scale_normal": "عادي",
        "font_scale_large": "كبير",
        "font_scale_xlarge": "كبير جدًا",
        "font_scale_xxlarge": "ضخم",
        "font_preview_text": "نص تجريبي لمعاينة الخط — 1234567890",
        "font_reset": "إعادة إلى الافتراضي",
        "font_applied": "تم تحديث الخط",
        "motion_section_title": "الحركة والرسوم المتحركة",
        "reduce_motion_label": "تقليل الحركة",
        "reduce_motion_hint": "عطّل الرسوم المتحركة إذا كانت تسبب لك الانزعاج.",
        "empty_history_empty_title": "لا يوجد سجل بعد",
        "empty_history_empty_subtitle": "قم بمعالجة صورتك الأولى لتظهر هنا",
        "empty_history_empty_hint": "يتم حفظ كل نتيجة تلقائيًا",
        "empty_history_no_user_title": "سجّل الدخول لعرض السجل",
        "empty_history_no_user_subtitle": "ستظهر نتائجك المحفوظة هنا",
        "empty_history_no_user_hint": "السجل خاص بكل حساب",
        "empty_dash_no_results_title": "لا توجد نتائج بعد",
        "empty_dash_no_results_subtitle": "ستظهر تحليلاتك الأخيرة هنا",
        "empty_dash_no_results_hint": "قم بمعالجة صورة للبدء",
        "empty_dash_no_dist_title": "لا توجد بيانات توزيع",
        "empty_dash_no_dist_subtitle": "ستمتلئ الإحصائيات أثناء معالجة الصور",
        "empty_dash_no_dist_hint": "كل تحليل مهم",
        "empty_image_empty_title": "لم يتم إدراج صورة",
        "empty_image_empty_subtitle": "اسحب وأفلت صورة هنا، أو انقر على إدراج",
        "empty_image_empty_hint": "يدعم PNG، JPG، JPEG، BMP، DCM",
        "empty_image_drop_hint_title": "أفلت للتحليل",
        "empty_image_drop_hint_subtitle": "أفلت الصورة لتحميلها",
        "empty_image_drop_hint_hint": "سيتم اكتشاف التنسيق تلقائيًا",
        "empty_processing_wait_title": "جارٍ المعالجة...",
        "empty_processing_wait_subtitle": "الذكاء الاصطناعي يحلل صورتك",
        "empty_processing_wait_hint": "يستغرق هذا عادةً بضع ثوانٍ",
        "tour_show_again": "إظهار الجولة مرة أخرى",
        "tour_completed_msg": "اكتملت الجولة! أنت جاهز.",
        "tour_skipped_msg": "تم تخطي الجولة. يمكنك إعادتها من قائمة الحساب.",
        "tour_btn_next": "التالي",
        "tour_btn_prev": "السابق",
        "tour_btn_skip": "تخطي",
        "tour_btn_finish": "إنهاء ✓",
        "tour_guest_login_title": "تسجيل الدخول",
        "tour_guest_login_subtitle": "انقر هنا لتسجيل الدخول وفتح جميع الميزات.",
        "tour_guest_signup_title": "إنشاء حساب",
        "tour_guest_signup_subtitle": "جديد هنا؟ انقر هنا لإنشاء حساب في ثوانٍ.",
        "tour_insert_title": "إدراج صورة",
        "tour_insert_subtitle": "حمّل صورة سونو من القرص، أو اسحبها وأفلتها في أي مكان في النافذة.",
        "tour_process_title": "معالجة الصورة",
        "tour_process_subtitle": "شغّل تحليل الذكاء الاصطناعي على الصورة المحملة. تظهر النتائج في بضع ثوانٍ.",
        "tour_export_title": "تصدير PDF",
        "tour_export_subtitle": "احفظ النتيجة الحالية كتقرير طبي احترافي بصيغة PDF.",
        "tour_gradcam_title": "خريطة الانتباه الحرارية",
        "tour_gradcam_subtitle": "تصوّر أين ينظر الذكاء الاصطناعي — خريطة حرارية فوق الصورة المحللة.",
        "tour_dashboard_title": "لوحة التحكم",
        "tour_dashboard_subtitle": "إظهار/إخفاء لوحة الإحصائيات: إجمالي الصور، متوسط الثقة، توزيع الدرجات.",
        "tour_history_title": "السجل",
        "tour_history_subtitle": "تصفّح آخر 20 صورة محللة. انقر على أي صورة مصغرة لإعادة تحميل نتيجتها.",
        "tour_visit_title": "زيارة الموقع",
        "tour_visit_subtitle": "افتح صفحة GitHub للمشروع في متصفحك للوثائق والتحديثات.",
        "tour_mute_title": "كتم الموسيقى",
        "tour_mute_subtitle": "أوقف أو استأنف موسيقى الخلفية في أي وقت.",
        "tour_zoom_title": "عناصر التحكم في التكبير",
        "tour_zoom_subtitle": "استخدم عجلة الماوس للتكبير، اسحب للتحريك، وانقر نقرًا مزدوجًا لإعادة الضبط.",
        "tour_account_title": "قائمة الحساب",
        "tour_account_subtitle": "انقر هنا للوصول إلى الإعدادات، معلومات الحساب، سجل التدقيق، والمزيد.",
        "suggestion_image_ready":      "تم تحميل الصورة — جاهزة للمعالجة",
        "suggestion_image_ready_sub":  "انقر على «معالجة» لتشغيل تحليل الذكاء الاصطناعي.",
        "suggestion_pdf_ready":        "النتيجة جاهزة — تصدير كـ PDF؟",
        "suggestion_pdf_ready_sub":    "احفظ تقريرًا طبيًا احترافيًا بنقرة واحدة.",
        "suggestion_gradcam_hint":     "هل أنت فضولي لمعرفة ما نظر إليه الذكاء الاصطناعي؟",
        "suggestion_gradcam_hint_sub": "شاهد خريطة الانتباه الحرارية فوق الصورة.",
        "suggestion_repeated_grid":    "أنت تحصل دائمًا على {grid}",
        "suggestion_repeated_grid_sub":"افتح لوحة التحكم لرؤية التوزيع.",
        "suggestion_history_hint":     "سجلك يتراكم",
        "suggestion_history_hint_sub": "تصفّح آخر 20 صورة تم تحليلها.",
        "suggestion_dashboard_hint":   "لوحة التحكم الخاصة بك لديها إحصائيات مثيرة",
        "suggestion_dashboard_hint_sub":"الإجماليات والمتوسطات والتوزيع بنظرة واحدة.",
        "suggestion_settings_hint":    "خصّص التطبيق",
        "suggestion_settings_hint_sub":"غيّر اللغة والمظهر والخط وفعّل المصادقة الثنائية.",
        "suggestion_idle_help":        "تحتاج إلى دليل سريع؟",
        "suggestion_idle_help_sub":    "أعد تشغيل الجولة التعريفية في أي وقت.",
        "image_not_liver_warning": "هذه الصورة لا تبدو كصورة موجات فوق صوتية للكبد.\n\nيرجى إدراج صورة موجات فوق صوتية صحيحة للكبد.",
        "result_not_confident": "لم يتمكن الذكاء الاصطناعي من تحديد هذه الصورة بثقة كصورة موجات فوق صوتية للكبد.\n\nيرجى إدراج صورة موجات فوق صوتية صحيحة للكبد والمحاولة مرة أخرى.",
        "not_liver_title": "ليست صورة موجات فوق صوتية للكبد",
        "not_liver_hint": "يرجى استخدام صورة موجات فوق صوتية (سونار) صحيحة للكبد",
        "voice_inference_result": "{grid_name}، الثقة {conf} بالمائة",
        "voice_pdf_saved": "تم حفظ تقرير PDF بنجاح",
        "voice_error": "حدث خطأ",
        "voice_settings_applied": "تم تطبيق الإعدادات",
        "voice_language_changed": "تم تغيير اللغة إلى {lang}",
        "voice_theme_changed": "تم تغيير المظهر",
        "voice_2fa_enabled": "تم تفعيل المصادقة الثنائية",
        "voice_2fa_disabled": "تم تعطيل المصادقة الثنائية",
        "voice_locked": "تم قفل الجلسة",
        "voice_unlocked": "تم إلغاء قفل الجلسة",
        "voice_processing": "جارٍ معالجة الصورة",
        "voice_processing_done": "اكتملت المعالجة",
        "voice_voice_enabled": "تم تفعيل التغذية الصوتية",
        "voice_voice_disabled": "تم تعطيل التغذية الصوتية",
    },
    "Ukranian": {
        "app_title": "Конвеєр ШІ печінки",
        "welcome": "Ласкаво просимо",
        "login": "Увійти",
        "signup": "Реєстрація",
        "logout": "Вийти",
        "exit": "Вихід",
        "settings": "Налаштування",
        "account_info": "Інформація про акаунт",
        "username": "Ім'я користувача",
        "password": "Пароль",
        "repeat_password": "Повторіть пароль",
        "first_name": "Ім'я",
        "last_name": "Прізвище",
        "email": "Електронна пошта",
        "remember_me": "Запам'ятати мене",
        "login_btn": "Увійти",
        "signup_btn": "Реєстрація",
        "account": "Акаунт",
        "change_language": "Змінити мову",
        "change_theme": "Змінити тему",
        "insert_sono": "Вставити сонограму",
        "process": "Обробити",
        "mute": "Без звуку",
        "unmute": "Зі звуком",
        "visit_webpage": "Відвідати сайт",
        "please_wait": "Будь ласка, зачекайте...",
        "processing": "Обробка зображення...",
        "result": "Результат",
        "confidence": "Впевненість",
        "grid": "Сітка",
        "insert_image_first": "Спочатку вставте зображення!",
        "no_image": "Зображення не вставлено",
        "invalid_credentials": "Невірні дані",
        "user_exists": "Ім'я користувача вже існує",
        "password_requirements": "Вимоги до пароля",
        "password_rule": (
            "Пароль повинен:\n"
            "• Містити 8-16 символів\n"
            "• Містити щонайменше 1 малу літеру\n"
            "• Містити щонайменше 1 велику літеру\n"
            "• Містити щонайменше 4 цифри\n"
            "• Містити щонайменше 4 спеціальні символи"
        ),
        "passwords_dont_match": "Паролі не збігаються!",
        "invalid_email": "Введіть коректну електронну адресу",
        "empty_fields": "Заповніть усі поля",
        "signup_success": "Реєстрація успішна! Тепер ви можете увійти.",
        "logout_confirm": "Ви впевнені, що хочете видалити акаунт?",
        "exit_confirm": "Ви впевнені, що хочете вийти?",
        "theme_dark_aqua": "Темна тема Aqua-Purple",
        "theme_solarized_white": "Світла тема Solarized",
        "theme_solarized_dark": "Темна тема Solarized",
        "theme_dark_forest": "Темна лісова зелена тема",
        "select_language": "Обрати мову",
        "select_theme": "Обрати тему",
        "apply": "Застосувати",
        "cancel": "Скасувати",
        "save": "Зберегти",
        "success": "Успіх",
        "error": "Помилка",
        "warning": "Попередження",
        "settings_applied": "Налаштування застосовано!",
        "grid_0": "Сітка 0 (NL) — Нормальна печінка",
        "grid_1": "Сітка 1 (G1) — Ступінь 1",
        "grid_2": "Сітка 2 (G2) — Ступінь 2",
        "grid_3": "Сітка 3 (G3) — Ступінь 3",
        "too_many_attempts": "Забагато невдалих спроб. Зачекайте 5 хвилин.",
        "invalid_username": "Ім'я користувача має бути 3-32 символи (літери, цифри, підкреслення).",
        "welcome_user": "Ласкаво просимо, {user}!",
        "inference_failed": "Помилка виведення",
        "drop_here": "Перетягніть зображення сюди",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Непідтримуваний тип файлу. Будь ласка, перетягніть зображення.",
        "image_loaded": "Зображення успішно завантажено",
        "help_about_title": "Довідка та Про програму",
        "help_about_text": (
            "Конвеєр ШІ печінки v3.0\n"
            "Розроблено: Бабак Юсефян\n\n"
            "Гарячі клавіші:\n"
            "  Ctrl+O      — Вставити зображення\n"
            "  Ctrl+Enter  — Обробити зображення\n"
            "  Ctrl+L      — Увійти\n"
            "  Ctrl+,      — Налаштування\n"
            "  Esc         — Закрити активний діалог\n"
            "  F1          — Довідка / Про програму\n\n"
            "Порада: Ви також можете перетягнути зображення в будь-яке місце вікна."
        ),
        "dashboard_title": "Панель керування",
        "dash_total": "Всього зображень",
        "dash_avg_conf": "Середня впевненість",
        "dash_distribution": "Розподіл",
        "dash_last_5": "Останні 5 результатів",
        "dash_no_results": "Поки немає результатів",
        "dash_clear_confirm": "Очистити всю статистику?",
        "dash_cleared": "Статистику очищено",
        "dashboard_toggle": "Показати/Приховати панель",
        "export_pdf": "Експорт PDF",
        "pdf_saved": "PDF успішно збережено",
        "pdf_failed": "Не вдалося створити PDF",
        "pdf_subtitle": "Звіт про аналіз",
        "pdf_patient": "Пацієнт:",
        "pdf_date": "Дата:",
        "pdf_model": "Модель:",
        "pdf_diagnosis": "ДІАГНОЗ",
        "pdf_confidence": "Впевненість",
        "pdf_class_probs": "ЙМОВІРНОСТІ КЛАСІВ",
        "pdf_disclaimer": (
            "Цей звіт розроблено Бабаком Юсефяном під наглядом досвідчених лікарів. "
            "Це професійний інструмент аналізу, який не потребує направлення до лікаря."
        ),
        "history_title": "Історія",
        "history_toggle": "Показати/Приховати історію",
        "history_empty": "Поки немає історії",
        "history_cleared": "Історію очищено",
        "history_clear_confirm": "Очистити всю історію?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Увійти",
        "tooltip_desc_login": "Увійдіть у свій акаунт",
        "tooltip_tip_login": "Доступ до збереженої історії та статистики",
        "tooltip_title_signup": "Реєстрація",
        "tooltip_desc_signup": "Створити новий акаунт",
        "tooltip_title_account": "Акаунт",
        "tooltip_desc_account": "Натисніть, щоб відкрити меню акаунта",
        "tooltip_desc_account_user": "Ви увійшли як {user}",
        "tooltip_title_dashboard": "Панель керування",
        "tooltip_desc_dashboard": "Показати/приховати панель статистики — всього, розподіл, останні результати",
        "tooltip_title_history": "Історія",
        "tooltip_desc_history": "Показати/приховати останні 20 результатів з мініатюрами",
        "tooltip_title_visit": "Відвідати сайт",
        "tooltip_desc_visit": "Відкрити сторінку GitHub проєкту в браузері",
        "tooltip_tip_visit": "Відкривається у зовнішньому браузері",
        "tooltip_title_insert": "Вставити зображення",
        "tooltip_desc_insert": "Завантажити сонограму для аналізу",
        "tooltip_tip_insert": "Можна також перетягнути зображення в будь-яке місце вікна",
        "tooltip_title_process": "Обробити",
        "tooltip_desc_process": "Запустити ШІ-аналіз завантаженого зображення",
        "tooltip_tip_process": "Спочатку потрібно завантажити зображення",
        "tooltip_title_mute": "Вимкнути музику",
        "tooltip_desc_mute": "Призупинити фонову музику",
        "tooltip_title_unmute": "Увімкнути музику",
        "tooltip_desc_unmute": "Продовжити фонову музику",
        "tooltip_title_export": "Експорт PDF",
        "tooltip_desc_export": "Зберегти поточний результат як PDF-звіт",
        "tooltip_tip_export": "Включає зображення, діагноз і ймовірності класів",
        "tooltip_title_dash_clear": "Очистити статистику",
        "tooltip_desc_dash_clear": "Скинути всю статистику цього акаунта",
        "tooltip_tip_dash_clear": "Цю дію неможливо скасувати",
        "tooltip_title_dash_close": "Закрити панель",
        "tooltip_desc_dash_close": "Приховати панель статистики",
        "tooltip_desc_dash_total": "Загальна кількість оброблених зображень",
        "tooltip_desc_dash_avg": "Середня впевненість за всіма результатами",
        "tooltip_desc_dash_dist": "Розподіл результатів за ступенями",
        "tooltip_title_hist_clear": "Очистити історію",
        "tooltip_desc_hist_clear": "Видалити всі збережені результати та мініатюри",
        "tooltip_tip_hist_clear": "Це також видалить усі файли зображень",
        "tooltip_title_hist_close": "Закрити історію",
        "tooltip_desc_hist_close": "Приховати панель історії",
        "tooltip_tip_thumb_click": "Натисніть, щоб повторно завантажити цей результат",
        "tooltip_tip_recent": "Натисніть, щоб повторно завантажити цей результат",
        "tooltip_title_zoom_in": "Збільшити",
        "tooltip_desc_zoom_in": "Збільшити масштаб зображення",
        "tooltip_title_zoom_out": "Зменшити",
        "tooltip_desc_zoom_out": "Зменшити масштаб зображення",
        "tooltip_title_zoom_reset": "Скинути вигляд",
        "tooltip_desc_zoom_reset": "Вписати зображення у вікно",
        "tooltip_tip_zoom_reset": "Можна також двічі клацнути по зображенню",
        "notif_inference_ok": "Виведення завершено",
        "follow_system_theme": "Слідувати темі системи",
        "follow_system_theme_desc": "Автоматично перемикатися між світлою та темною темою залежно від налаштувань системи",
        "gradcam_title": "Теплова карта уваги",
        "gradcam_show": "Показати карту уваги",
        "gradcam_toggle": "Перемкнути карту",
        "gradcam_save": "Зберегти зображення",
        "gradcam_close": "Закрити",
        "gradcam_saved": "Зображення успішно збережено",
        "gradcam_no_heatmap": "Карта уваги недоступна. Спочатку обробіть зображення.",
        "update_available": "Доступне оновлення",
        "update_check_failed": "Не вдалося перевірити оновлення",
        "check_for_updates": "Перевірити оновлення",
        "check_for_updates_desc": "Перевірити GitHub/сайт на наявність останньої версії",
        "up_to_date": "У вас остання версія",
        "you_have_latest": "Ви використовуєте останню версію.",
        "check_internet": "Будь ласка, перевірте підключення до Інтернету та спробуйте ще раз.",
        "downloading": "Завантаження...",
        "verifying": "Перевірка...",
        "download_now": "Завантажити зараз",
        "install_now": "Встановити та перезапустити",
        "skip_version": "Пропустити цю версію",
        "remind_later": "Нагадати пізніше",
        "whats_new": "Що нового",
        "no_changelog": "Список змін не надано.",
        "critical_update": "Критичне оновлення",
        "update_ready": "Готово до встановлення",
        "app_will_restart": "Програма закриється та перезапуститься для застосування оновлення.",
        "install_failed": "Встановлення не вдалося. Ваша стара версія в безпеці.",
        "download_failed": "Завантаження не вдалося.",
        "source_update_requires_zip": "Оновлення вихідного коду потребує файлу .zip.",
        "session_locked": "Сесія заблокована",
        "locked_for_user": "Заблоковано для {user}",
        "unlock": "Розблокувати",
        "session_unlocked": "Сесію розблоковано",
        "auto_lock_notice": "Сесія заблокується через 2 хвилини бездіяльності.",
        "audit_log_title": "Журнал аудиту",
        "audit_export": "Експорт",
        "audit_exported": "Успішно експортовано",
        "audit_export_failed": "Помилка експорту",
        "audit_search_placeholder": "Пошук дій або деталей...",
        "audit_col_time": "Час",
        "audit_col_level": "Рівень",
        "audit_col_category": "Категорія",
        "audit_col_action": "Дія",
        "audit_col_user": "Користувач",
        "audit_details_title": "Деталі події",
        "audit_status": "Показано {count} з {total} подій",
        "strength_very_weak": "Дуже слабкий",
        "strength_weak": "Слабкий",
        "strength_medium": "Середній",
        "strength_strong": "Сильний",
        "strength_very_strong": "Дуже сильний",
        "strength_rule_length": "8-16 символів",
        "strength_rule_lowercase": "Мала літера",
        "strength_rule_uppercase": "Велика літера",
        "strength_rule_digits": "Цифри",
        "strength_rule_special": "Спеціальні символи",
        "strength_add_chars": "Додайте ще {n} символ(ів)",
        "strength_remove_chars": "Приберіть {n} символ(ів)",
        "strength_need_digits": "Додайте ще {n} цифр(и)",
        "strength_need_special": "Додайте ще {n} спеціальних символ(ів)",
        "strength_common_pattern": "Уникайте поширених слів",
        "strength_common_year": "Уникайте поширених років",
        "strength_has_sequence": "Уникайте послідовностей (1234, abcd)",
        "strength_has_repeats": "Уникайте повторів (aaaa, 1111)",
        "passwords_match": "Паролі збігаються",
        "2fa_setup_title": "Увімкнути двофакторну автентифікацію",
        "2fa_setup_subtitle": "Відскануйте QR-код за допомогою Google Authenticator, Authy або Microsoft Authenticator.",
        "2fa_manual_secret": "Або введіть цей код вручну:",
        "2fa_enter_code": "Введіть 6-значний код із застосунку:",
        "2fa_verify_btn": "Підтвердити",
        "2fa_backup_codes_title": "Збережіть ці резервні коди!",
        "2fa_backup_codes_hint": "Кожен код можна використати лише один раз. Зберігайте їх у безпечному місці.",
        "2fa_done_btn": "Готово",
        "2fa_secret_copied": "Секретний код скопійовано до буфера обміну",
        "2fa_invalid_code": "Невірний код. Спробуйте ще раз.",
        "2fa_invalid_code_format": "Код має складатися з 6 цифр.",
        "2fa_verify_title": "Двофакторна автентифікація",
        "2fa_verify_subtitle": "Введіть 6-значний код із застосунку-автентифікатора.",
        "2fa_enter_code_hint": "Будь ласка, введіть код.",
        "2fa_use_backup_code": "Використати резервний код",
        "2fa_use_totp_code": "Використати код застосунку",
        "2fa_enter_backup_code": "Введіть один із ваших резервних кодів.",
        "2fa_section_title": "Безпека",
        "2fa_enabled_status": "Двофакторну автентифікацію увімкнено",
        "2fa_disabled_status": "Двофакторну автентифікацію вимкнено",
        "2fa_enable_btn": "Увімкнути 2FA",
        "2fa_disable_btn": "Вимкнути 2FA",
        "2fa_disable_title": "Вимкнути 2FA",
        "2fa_disable_confirm": "Ви впевнені? Ваш обліковий запис стане менш захищеним.",
        "2fa_disabled_success": "Двофакторну автентифікацію вимкнено",
        "2fa_enabled_success": "Двофакторну автентифікацію увімкнено",
        "2fa_hint_title": "Як використовувати цей QR-код",
        "2fa_hint_step1": "Встановіть застосунок-автентифікатор: Google Authenticator, Microsoft Authenticator або Authy.",
        "2fa_hint_step2": "Відкрийте застосунок, натисніть кнопку «+» і виберіть «Сканувати QR-код».",
        "2fa_hint_step3": "Наведіть камеру телефону на QR-код вище (використовуйте камеру застосунку, а не камеру телефону).",
        "2fa_hint_step4": "Застосунок покаже 6-значний код. Введіть його в полі нижче.",
        "2fa_manual_secret_hint": "Не можете відсканувати? Скопіюйте ключ нижче та додайте його вручну у застосунку («Ввести ключ налаштування»).",
        "auto_lock_section_title": "Автоблокування",
        "auto_lock_enable": "Блокувати після бездіяльності",
        "auto_lock_timeout_label": "Блокувати через:",
        "auto_lock_warning_title": "Попередження про автоблокування",
        "auto_lock_warning_body": "Вашу сесію буде заблоковано через {seconds} секунд через бездіяльність.",
        "lock_now": "Заблокувати зараз",
        "font_section_title": "Шрифт",
        "font_family_label": "Сімейство шрифтів",
        "font_size_label": "Базовий розмір",
        "font_scale_label": "Масштаб відображення",
        "font_scale_small": "Малий",
        "font_scale_normal": "Звичайний",
        "font_scale_large": "Великий",
        "font_scale_xlarge": "Дуже великий",
        "font_scale_xxlarge": "Величезний",
        "font_preview_text": "Чуєш їх, доцю, га? Кумедна ж ти, прощайся без ґольфів! — 1234567890",
        "font_reset": "Скинути до стандартного",
        "font_applied": "Шрифт оновлено",
        "motion_section_title": "Рух і анімації",
        "reduce_motion_label": "Зменшити рух",
        "reduce_motion_hint": "Вимкніть анімації, якщо вони викликають у вас дискомфорт.",
        "empty_history_empty_title": "Поки немає історії",
        "empty_history_empty_subtitle": "Обробіть перше зображення, щоб побачити його тут",
        "empty_history_empty_hint": "Кожен результат зберігається автоматично",
        "empty_history_no_user_title": "Увійдіть, щоб переглянути історію",
        "empty_history_no_user_subtitle": "Ваші збережені результати з'являться тут",
        "empty_history_no_user_hint": "Історія приватна для кожного акаунта",
        "empty_dash_no_results_title": "Поки немає результатів",
        "empty_dash_no_results_subtitle": "Ваші останні аналізи з'являться тут",
        "empty_dash_no_results_hint": "Обробіть зображення, щоб почати",
        "empty_dash_no_dist_title": "Немає даних розподілу",
        "empty_dash_no_dist_subtitle": "Статистика заповнюватиметься в міру обробки",
        "empty_dash_no_dist_hint": "Кожен аналіз важливий",
        "empty_image_empty_title": "Зображення не вставлено",
        "empty_image_empty_subtitle": "Перетягніть зображення сюди або натисніть «Вставити»",
        "empty_image_empty_hint": "Підтримуються PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Відпустіть для аналізу",
        "empty_image_drop_hint_subtitle": "Відпустіть зображення, щоб завантажити",
        "empty_image_drop_hint_hint": "Формат визначиться автоматично",
        "empty_processing_wait_title": "Обробка...",
        "empty_processing_wait_subtitle": "ШІ аналізує ваше зображення",
        "empty_processing_wait_hint": "Зазвичай це займає кілька секунд",
        "tour_show_again": "Показати тур знову",
        "tour_completed_msg": "Тур завершено! Ви готові.",
        "tour_skipped_msg": "Тур пропущено. Ви можете повторити його з меню акаунта.",
        "tour_btn_next": "Далі",
        "tour_btn_prev": "Назад",
        "tour_btn_skip": "Пропустити",
        "tour_btn_finish": "Готово ✓",
        "tour_guest_login_title": "Вхід",
        "tour_guest_login_subtitle": "Натисніть тут, щоб увійти в акаунт і відкрити всі можливості.",
        "tour_guest_signup_title": "Створити акаунт",
        "tour_guest_signup_subtitle": "Вперше тут? Натисніть тут, щоб створити акаунт за секунди.",
        "tour_insert_title": "Вставити зображення",
        "tour_insert_subtitle": "Завантажте сонограму з диска або перетягніть її будь-де у вікні.",
        "tour_process_title": "Обробити зображення",
        "tour_process_subtitle": "Запустіть ШІ-аналіз завантаженого зображення. Результати з'являться за кілька секунд.",
        "tour_export_title": "Експорт PDF",
        "tour_export_subtitle": "Збережіть поточний результат як професійний медичний PDF-звіт.",
        "tour_gradcam_title": "Теплова карта уваги",
        "tour_gradcam_subtitle": "Подивіться, куди дивиться ШІ — теплова карта поверх аналізованого зображення.",
        "tour_dashboard_title": "Панель керування",
        "tour_dashboard_subtitle": "Показати/приховати панель статистики: всього зображень, середня впевненість, розподіл за ступенями.",
        "tour_history_title": "Історія",
        "tour_history_subtitle": "Перегляньте останні 20 проаналізованих зображень. Натисніть на мініатюру, щоб завантажити результат знову.",
        "tour_visit_title": "Відвідати сайт",
        "tour_visit_subtitle": "Відкрийте сторінку GitHub проєкту в браузері для документації та оновлень.",
        "tour_mute_title": "Вимкнути музику",
        "tour_mute_subtitle": "Призупиніть або відновіть фонову музику будь-коли.",
        "tour_zoom_title": "Керування масштабом",
        "tour_zoom_subtitle": "Колесо миші — масштаб, перетягування — панорама, подвійний клік — скидання.",
        "tour_account_title": "Меню акаунта",
        "tour_account_subtitle": "Натисніть тут, щоб відкрити Налаштування, Інформацію про акаунт, Журнал аудиту та інше.",
        "suggestion_image_ready": "Зображення завантажено — готове до обробки",
        "suggestion_image_ready_sub": "Натисніть «Обробити», щоб запустити аналіз ШІ.",
        "suggestion_pdf_ready": "Результат готовий — експортувати в PDF?",
        "suggestion_pdf_ready_sub": "Збережіть професійний медичний звіт одним кліком.",
        "suggestion_gradcam_hint": "Цікаво, куди дивився ШІ?",
        "suggestion_gradcam_hint_sub": "Подивіться теплову карту уваги поверх зображення.",
        "suggestion_repeated_grid": "Ви постійно отримуєте {grid}",
        "suggestion_repeated_grid_sub": "Відкрийте панель, щоб побачити розподіл.",
        "suggestion_history_hint": "Ваша історія наповнюється",
        "suggestion_history_hint_sub": "Перегляньте останні 20 проаналізованих зображень.",
        "suggestion_dashboard_hint": "На вашій панелі є цікава статистика",
        "suggestion_dashboard_hint_sub": "Підсумки, середні значення та розподіл одним поглядом.",
        "suggestion_settings_hint": "Налаштуйте застосунок під себе",
        "suggestion_settings_hint_sub": "Змініть мову, тему, шрифт і увімкніть 2FA.",
        "suggestion_idle_help": "Потрібна швидка підказка?",
        "suggestion_idle_help_sub": "Повторіть навчальний тур у будь-який час.",
        "image_not_liver_warning": "Це зображення не схоже на УЗД печінки.\n\nБудь ласка, вставте відповідне зображення УЗД печінки.",
        "result_not_confident": "ШІ не зміг упевнено ідентифікувати це зображення як УЗД печінки.\n\nБудь ласка, вставте відповідне зображення УЗД печінки та спробуйте ще раз.",
        "not_liver_title": "Не УЗД печінки",
        "not_liver_hint": "Будь ласка, використовуйте відповідне зображення УЗД (сонограму) печінки",
        "voice_welcome": "Ласкаво просимо, {user}",
        "voice_goodbye": "До побачення",
        "voice_logged_out": "Вихід виконано успішно",
        "voice_image_loaded": "Зображення успішно завантажено",
        "voice_image_rejected": "Це не УЗД печінки",
        "voice_inference_result": "{grid_name}, впевненість {conf} відсотків",
        "voice_pdf_saved": "PDF-звіт успішно збережено",
        "voice_error": "Сталася помилка",
        "voice_settings_applied": "Налаштування застосовано",
        "voice_language_changed": "Мову змінено на {lang}",
        "voice_theme_changed": "Тему змінено",
        "voice_2fa_enabled": "Двофакторну автентифікацію увімкнено",
        "voice_2fa_disabled": "Двофакторну автентифікацію вимкнено",
        "voice_locked": "Сесію заблоковано",
        "voice_unlocked": "Сесію розблоковано",
        "voice_processing": "Обробка зображення",
        "voice_processing_done": "Обробку завершено",
        "voice_voice_enabled": "Голосовий зворотний зв'язок увімкнено",
        "voice_voice_disabled": "Голосовий зворотний зв'язок вимкнено",
    },
    "Italy": {
        "app_title": "Pipeline IA del Fegato",
        "welcome": "Benvenuto",
        "login": "Accedi",
        "signup": "Registrati",
        "logout": "Disconnetti",
        "exit": "Esci",
        "settings": "Impostazioni",
        "account_info": "Informazioni account",
        "username": "Nome utente",
        "password": "Password",
        "repeat_password": "Ripeti password",
        "first_name": "Nome",
        "last_name": "Cognome",
        "email": "Indirizzo email",
        "remember_me": "Ricordami",
        "login_btn": "Accedi",
        "signup_btn": "Registrati",
        "account": "Account",
        "change_language": "Cambia lingua",
        "change_theme": "Cambia tema",
        "insert_sono": "Inserisci ecografia",
        "process": "Elabora",
        "mute": "Muto",
        "unmute": "Attiva audio",
        "visit_webpage": "Visita il sito",
        "please_wait": "Attendere prego...",
        "processing": "Elaborazione immagine...",
        "result": "Risultato",
        "confidence": "Confidenza",
        "grid": "Griglia",
        "insert_image_first": "Inserisci prima un'immagine!",
        "no_image": "Nessuna immagine inserita",
        "invalid_credentials": "Credenziali non valide",
        "user_exists": "Nome utente già esistente",
        "password_requirements": "Requisiti password",
        "password_rule": (
            "La password deve:\n"
            "• Essere lunga 8-16 caratteri\n"
            "• Contenere almeno 1 lettera minuscola\n"
            "• Contenere almeno 1 lettera maiuscola\n"
            "• Contenere almeno 4 cifre\n"
            "• Contenere almeno 4 caratteri speciali"
        ),
        "passwords_dont_match": "Le password non corrispondono!",
        "invalid_email": "Inserisci un indirizzo email valido",
        "empty_fields": "Compila tutti i campi",
        "signup_success": "Registrazione riuscita! Ora puoi accedere.",
        "logout_confirm": "Sei sicuro di voler eliminare il tuo account?",
        "exit_confirm": "Sei sicuro di voler uscire?",
        "theme_dark_aqua": "Tema scuro Aqua-Purple",
        "theme_solarized_white": "Tema chiaro Solarized",
        "theme_solarized_dark": "Tema scuro Solarized",
        "theme_dark_forest": "Tema Foresta Verde Scuro",
        "select_language": "Seleziona lingua",
        "select_theme": "Seleziona tema",
        "apply": "Applica",
        "cancel": "Annulla",
        "save": "Salva",
        "success": "Successo",
        "error": "Errore",
        "warning": "Avviso",
        "settings_applied": "Impostazioni applicate!",
        "grid_0": "Griglia 0 (NL) — Fegato normale",
        "grid_1": "Griglia 1 (G1) — Grado 1",
        "grid_2": "Griglia 2 (G2) — Grado 2",
        "grid_3": "Griglia 3 (G3) — Grado 3",
        "too_many_attempts": "Troppi tentativi falliti. Attendere 5 minuti.",
        "invalid_username": "Il nome utente deve essere 3-32 caratteri (lettere, numeri, underscore).",
        "welcome_user": "Benvenuto, {user}!",
        "inference_failed": "Inferenza fallita",
        "drop_here": "Rilascia l'immagine qui",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Tipo di file non supportato. Rilascia un'immagine.",
        "image_loaded": "Immagine caricata con successo",
        "help_about_title": "Guida & Informazioni",
        "help_about_text": (
            "Pipeline IA del Fegato v3.0\n"
            "Sviluppato da Babak Yousefian\n\n"
            "Scorciatoie da tastiera:\n"
            "  Ctrl+O      — Inserisci immagine\n"
            "  Ctrl+Enter  — Elabora immagine\n"
            "  Ctrl+L      — Accedi\n"
            "  Ctrl+,      — Impostazioni\n"
            "  Esc         — Chiudi finestra attiva\n"
            "  F1          — Guida / Informazioni\n\n"
            "Suggerimento: puoi anche trascinare un'immagine in qualsiasi punto della finestra."
        ),
        "dashboard_title": "Pannello",
        "dash_total": "Immagini totali",
        "dash_avg_conf": "Confidenza media",
        "dash_distribution": "Distribuzione",
        "dash_last_5": "Ultimi 5 risultati",
        "dash_no_results": "Nessun risultato ancora",
        "dash_clear_confirm": "Cancellare tutte le statistiche?",
        "dash_cleared": "Statistiche cancellate",
        "dashboard_toggle": "Mostra/Nascondi pannello",
        "export_pdf": "Esporta PDF",
        "pdf_saved": "PDF salvato con successo",
        "pdf_failed": "Generazione PDF fallita",
        "pdf_subtitle": "Rapporto di analisi",
        "pdf_patient": "Paziente:",
        "pdf_date": "Data:",
        "pdf_model": "Modello:",
        "pdf_diagnosis": "DIAGNOSI",
        "pdf_confidence": "Confidenza",
        "pdf_class_probs": "PROBABILITÀ DELLE CLASSI",
        "pdf_disclaimer": (
            "Questo rapporto è stato sviluppato da Babak Yousefian sotto la supervisione di medici esperti. "
            "È uno strumento di analisi professionale e non richiede l'invio a un medico."
        ),
        "history_title": "Cronologia",
        "history_toggle": "Mostra/Nascondi cronologia",
        "history_empty": "Nessuna cronologia ancora",
        "history_cleared": "Cronologia cancellata",
        "history_clear_confirm": "Cancellare tutta la cronologia?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Accedi",
        "tooltip_desc_login": "Accedi al tuo account",
        "tooltip_tip_login": "Accedi alla cronologia e alle statistiche salvate",
        "tooltip_title_signup": "Registrati",
        "tooltip_desc_signup": "Crea un nuovo account",
        "tooltip_title_account": "Account",
        "tooltip_desc_account": "Clicca per aprire il menu dell'account",
        "tooltip_desc_account_user": "Connesso come {user}",
        "tooltip_title_dashboard": "Pannello",
        "tooltip_desc_dashboard": "Mostra/nascondi pannello statistiche — totale, distribuzione, risultati recenti",
        "tooltip_title_history": "Cronologia",
        "tooltip_desc_history": "Mostra/nascondi gli ultimi 20 risultati con miniature",
        "tooltip_title_visit": "Visita il sito",
        "tooltip_desc_visit": "Apri la pagina GitHub del progetto nel browser",
        "tooltip_tip_visit": "Si apre nel browser esterno",
        "tooltip_title_insert": "Inserisci immagine",
        "tooltip_desc_insert": "Carica un'immagine ecografica da analizzare",
        "tooltip_tip_insert": "Puoi anche trascinare un'immagine in qualsiasi punto della finestra",
        "tooltip_title_process": "Elabora",
        "tooltip_desc_process": "Avvia l'analisi AI sull'immagine caricata",
        "tooltip_tip_process": "Richiede prima un'immagine caricata",
        "tooltip_title_mute": "Muto",
        "tooltip_desc_mute": "Metti in pausa la musica di sottofondo",
        "tooltip_title_unmute": "Attiva audio",
        "tooltip_desc_unmute": "Riprendi la musica di sottofondo",
        "tooltip_title_export": "Esporta PDF",
        "tooltip_desc_export": "Salva il risultato corrente come report PDF",
        "tooltip_tip_export": "Include immagine, diagnosi e probabilità delle classi",
        "tooltip_title_dash_clear": "Cancella statistiche",
        "tooltip_desc_dash_clear": "Ripristina tutte le statistiche di questo account",
        "tooltip_tip_dash_clear": "Questa azione non può essere annullata",
        "tooltip_title_dash_close": "Chiudi pannello",
        "tooltip_desc_dash_close": "Nascondi il pannello delle statistiche",
        "tooltip_desc_dash_total": "Numero totale di immagini elaborate",
        "tooltip_desc_dash_avg": "Confidenza media su tutti i risultati",
        "tooltip_desc_dash_dist": "Distribuzione dei risultati per grado",
        "tooltip_title_hist_clear": "Cancella cronologia",
        "tooltip_desc_hist_clear": "Elimina tutti i risultati e le miniature salvati",
        "tooltip_tip_hist_clear": "Questo eliminerà anche tutti i file immagine",
        "tooltip_title_hist_close": "Chiudi cronologia",
        "tooltip_desc_hist_close": "Nascondi il pannello della cronologia",
        "tooltip_tip_thumb_click": "Clicca per ricaricare questo risultato",
        "tooltip_tip_recent": "Clicca per ricaricare questo risultato",
        "tooltip_title_zoom_in": "Zoom avanti",
        "tooltip_desc_zoom_in": "Aumenta l'ingrandimento dell'immagine",
        "tooltip_title_zoom_out": "Zoom indietro",
        "tooltip_desc_zoom_out": "Riduci l'ingrandimento dell'immagine",
        "tooltip_title_zoom_reset": "Reimposta vista",
        "tooltip_desc_zoom_reset": "Adatta l'immagine alla finestra",
        "tooltip_tip_zoom_reset": "Puoi anche fare doppio clic sull'immagine",
        "notif_inference_ok": "Inferenza completata",
        "follow_system_theme": "Segui il tema di sistema",
        "follow_system_theme_desc": "Passa automaticamente tra chiaro e scuro in base alle impostazioni di sistema",
        "gradcam_title": "Mappa di calore dell'attenzione",
        "gradcam_show": "Mostra mappa di attenzione",
        "gradcam_toggle": "Cambia mappa",
        "gradcam_save": "Salva immagine",
        "gradcam_close": "Chiudi",
        "gradcam_saved": "Immagine salvata con successo",
        "gradcam_no_heatmap": "Nessuna mappa di attenzione disponibile. Elabora prima un'immagine.",
        "update_available": "Aggiornamento disponibile",
        "update_check_failed": "Controllo aggiornamenti non riuscito",
        "check_for_updates": "Controlla aggiornamenti",
        "check_for_updates_desc": "Controlla GitHub/sito web per l'ultima versione",
        "up_to_date": "Sei aggiornato",
        "you_have_latest": "Stai utilizzando l'ultima versione.",
        "check_internet": "Controlla la connessione Internet e riprova.",
        "downloading": "Download in corso...",
        "verifying": "Verifica in corso...",
        "download_now": "Scarica ora",
        "install_now": "Installa e riavvia",
        "skip_version": "Salta questa versione",
        "remind_later": "Ricordamelo più tardi",
        "whats_new": "Novità",
        "no_changelog": "Nessun changelog fornito.",
        "critical_update": "Aggiornamento critico",
        "update_ready": "Pronto per l'installazione",
        "app_will_restart": "L'app verrà chiusa e riavviata per applicare l'aggiornamento.",
        "install_failed": "Installazione non riuscita. La tua vecchia versione è al sicuro.",
        "download_failed": "Download non riuscito.",
        "source_update_requires_zip": "L'aggiornamento sorgente richiede un file .zip.",
        "session_locked": "Sessione Bloccata",
        "locked_for_user": "Bloccata per {user}",
        "unlock": "Sblocca",
        "session_unlocked": "Sessione sbloccata",
        "auto_lock_notice": "La sessione si bloccherà dopo 2 minuti di inattività.",
        "audit_log_title": "Registro di Audit",
        "audit_export": "Esporta",
        "audit_exported": "Esportato con successo",
        "audit_export_failed": "Esportazione non riuscita",
        "audit_search_placeholder": "Cerca azioni o dettagli...",
        "audit_col_time": "Ora",
        "audit_col_level": "Livello",
        "audit_col_category": "Categoria",
        "audit_col_action": "Azione",
        "audit_col_user": "Utente",
        "audit_details_title": "Dettagli Evento",
        "audit_status": "Visualizzazione {count} di {total} eventi",
        "strength_very_weak": "Molto debole",
        "strength_weak": "Debole",
        "strength_medium": "Media",
        "strength_strong": "Forte",
        "strength_very_strong": "Molto forte",
        "strength_rule_length": "8-16 caratteri",
        "strength_rule_lowercase": "Lettera minuscola",
        "strength_rule_uppercase": "Lettera maiuscola",
        "strength_rule_digits": "Cifre",
        "strength_rule_special": "Caratteri speciali",
        "strength_add_chars": "Aggiungi {n} carattere/i in più",
        "strength_remove_chars": "Rimuovi {n} carattere/i",
        "strength_need_digits": "Aggiungi {n} cifra/e in più",
        "strength_need_special": "Aggiungi {n} carattere/i speciale/i in più",
        "strength_common_pattern": "Evita parole comuni",
        "strength_common_year": "Evita anni comuni",
        "strength_has_sequence": "Evita sequenze (1234, abcd)",
        "strength_has_repeats": "Evita ripetizioni (aaaa, 1111)",
        "passwords_match": "Le password corrispondono",
        "2fa_setup_title": "Abilita l'autenticazione a due fattori",
        "2fa_setup_subtitle": "Scansiona il codice QR con Google Authenticator, Authy o Microsoft Authenticator.",
        "2fa_manual_secret": "Oppure inserisci questo codice manualmente:",
        "2fa_enter_code": "Inserisci il codice a 6 cifre dalla tua app:",
        "2fa_verify_btn": "Verifica",
        "2fa_backup_codes_title": "Salva questi codici di backup!",
        "2fa_backup_codes_hint": "Ogni codice può essere usato una sola volta. Conservali in un luogo sicuro.",
        "2fa_done_btn": "Fatto",
        "2fa_secret_copied": "Codice segreto copiato negli appunti",
        "2fa_invalid_code": "Codice non valido. Riprova.",
        "2fa_invalid_code_format": "Il codice deve contenere 6 cifre.",
        "2fa_verify_title": "Autenticazione a due fattori",
        "2fa_verify_subtitle": "Inserisci il codice a 6 cifre dalla tua app di autenticazione.",
        "2fa_enter_code_hint": "Inserisci un codice.",
        "2fa_use_backup_code": "Usa un codice di backup",
        "2fa_use_totp_code": "Usa il codice dell'app",
        "2fa_enter_backup_code": "Inserisci uno dei tuoi codici di backup.",
        "2fa_section_title": "Sicurezza",
        "2fa_enabled_status": "L'autenticazione a due fattori è abilitata",
        "2fa_disabled_status": "L'autenticazione a due fattori è disabilitata",
        "2fa_enable_btn": "Abilita 2FA",
        "2fa_disable_btn": "Disabilita 2FA",
        "2fa_disable_title": "Disabilita 2FA",
        "2fa_disable_confirm": "Sei sicuro? Il tuo account sarà meno sicuro.",
        "2fa_disabled_success": "Autenticazione a due fattori disabilitata",
        "2fa_enabled_success": "Autenticazione a due fattori abilitata",
        "2fa_hint_title": "Come utilizzare questo codice QR",
        "2fa_hint_step1": "Installa un'app di autenticazione: Google Authenticator, Microsoft Authenticator o Authy.",
        "2fa_hint_step2": "Apri l'app, tocca il pulsante «+» e scegli «Scansiona codice QR».",
        "2fa_hint_step3": "Punta la fotocamera del telefono verso il codice QR sopra (usa la fotocamera dell'app, non quella del telefono).",
        "2fa_hint_step4": "L'app mostrerà un codice a 6 cifre. Inseriscilo nel campo sottostante.",
        "2fa_manual_secret_hint": "Non riesci a scansionare? Copia la chiave qui sotto e aggiungila manualmente nell'app («Inserisci chiave di configurazione»).",
        "auto_lock_section_title": "Blocco automatico",
        "auto_lock_enable": "Blocca dopo inattività",
        "auto_lock_timeout_label": "Blocca dopo:",
        "auto_lock_warning_title": "Avviso di blocco automatico",
        "auto_lock_warning_body": "La tua sessione verrà bloccata tra {seconds} secondi per inattività.",
        "lock_now": "Blocca ora",
        "font_section_title": "Carattere",
        "font_family_label": "Famiglia di caratteri",
        "font_size_label": "Dimensione base",
        "font_scale_label": "Scala di visualizzazione",
        "font_scale_small": "Piccolo",
        "font_scale_normal": "Normale",
        "font_scale_large": "Grande",
        "font_scale_xlarge": "Molto grande",
        "font_scale_xxlarge": "Enorme",
        "font_preview_text": "Ma la volpe, col suo balzo, raggiunse il piccolo cane — 1234567890",
        "font_reset": "Ripristina predefinito",
        "font_applied": "Carattere aggiornato",
        "motion_section_title": "Movimento e animazioni",
        "reduce_motion_label": "Riduci movimento",
        "reduce_motion_hint": "Disattiva le animazioni se ti mettono a disagio.",
        "empty_history_empty_title": "Nessuna cronologia ancora",
        "empty_history_empty_subtitle": "Elabora la tua prima immagine per vederla qui",
        "empty_history_empty_hint": "Ogni risultato viene salvato automaticamente",
        "empty_history_no_user_title": "Accedi per vedere la cronologia",
        "empty_history_no_user_subtitle": "I tuoi risultati salvati appariranno qui",
        "empty_history_no_user_hint": "La cronologia è privata per ogni account",
        "empty_dash_no_results_title": "Nessun risultato ancora",
        "empty_dash_no_results_subtitle": "Le tue analisi recenti appariranno qui",
        "empty_dash_no_results_hint": "Elabora un'immagine per iniziare",
        "empty_dash_no_dist_title": "Nessun dato di distribuzione",
        "empty_dash_no_dist_subtitle": "Le statistiche si riempiranno man mano",
        "empty_dash_no_dist_hint": "Ogni analisi conta",
        "empty_image_empty_title": "Nessuna immagine inserita",
        "empty_image_empty_subtitle": "Trascina un'immagine qui o clicca Inserisci",
        "empty_image_empty_hint": "Supporta PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Rilascia per analizzare",
        "empty_image_drop_hint_subtitle": "Rilascia l'immagine per caricarla",
        "empty_image_drop_hint_hint": "Il formato verrà rilevato automaticamente",
        "empty_processing_wait_title": "Elaborazione...",
        "empty_processing_wait_subtitle": "L'IA sta analizzando la tua immagine",
        "empty_processing_wait_hint": "Di solito richiede pochi secondi",
        "tour_show_again": "Mostra di nuovo il tour",
        "tour_completed_msg": "Tour completato! Sei pronto.",
        "tour_skipped_msg": "Tour saltato. Puoi ripeterlo dal menu Account.",
        "tour_btn_next": "Avanti",
        "tour_btn_prev": "Indietro",
        "tour_btn_skip": "Salta",
        "tour_btn_finish": "Fine ✓",
        "tour_guest_login_title": "Accedi",
        "tour_guest_login_subtitle": "Clicca qui per accedere e sbloccare tutte le funzionalità.",
        "tour_guest_signup_title": "Crea un account",
        "tour_guest_signup_subtitle": "Nuovo qui? Clicca qui per creare un account in pochi secondi.",
        "tour_insert_title": "Inserisci immagine",
        "tour_insert_subtitle": "Carica un'immagine ecografica dal disco, o trascinala in qualsiasi punto della finestra.",
        "tour_process_title": "Elabora immagine",
        "tour_process_subtitle": "Avvia l'analisi AI sull'immagine caricata. I risultati appaiono in pochi secondi.",
        "tour_export_title": "Esporta PDF",
        "tour_export_subtitle": "Salva il risultato corrente come report medico PDF professionale.",
        "tour_gradcam_title": "Mappa di calore dell'attenzione",
        "tour_gradcam_subtitle": "Visualizza dove guarda l'IA — una mappa di calore sopra l'immagine analizzata.",
        "tour_dashboard_title": "Pannello",
        "tour_dashboard_subtitle": "Mostra/nascondi il pannello statistiche: totale immagini, confidenza media, distribuzione dei gradi.",
        "tour_history_title": "Cronologia",
        "tour_history_subtitle": "Sfoglia le ultime 20 immagini analizzate. Clicca su una miniatura per ricaricare il risultato.",
        "tour_visit_title": "Visita il sito",
        "tour_visit_subtitle": "Apri la pagina GitHub del progetto nel browser per documentazione e aggiornamenti.",
        "tour_mute_title": "Muto",
        "tour_mute_subtitle": "Metti in pausa o riprendi la musica di sottofondo in qualsiasi momento.",
        "tour_zoom_title": "Controlli zoom",
        "tour_zoom_subtitle": "Usa la rotella per lo zoom, trascina per spostare, doppio clic per reimpostare.",
        "tour_account_title": "Menu account",
        "tour_account_subtitle": "Clicca qui per accedere a Impostazioni, Info account, Registro di audit e altro.",
        "suggestion_image_ready": "Immagine caricata — pronta per l'elaborazione",
        "suggestion_image_ready_sub": "Clicca su Elabora per avviare l'analisi IA.",
        "suggestion_pdf_ready": "Risultato pronto — esportare in PDF?",
        "suggestion_pdf_ready_sub": "Salva un rapporto medico professionale con un clic.",
        "suggestion_gradcam_hint": "Curioso di cosa ha visto l'IA?",
        "suggestion_gradcam_hint_sub": "Guarda la mappa di calore dell'attenzione sull'immagine.",
        "suggestion_repeated_grid": "Ottieni sempre {grid}",
        "suggestion_repeated_grid_sub": "Apri il pannello per vedere la distribuzione.",
        "suggestion_history_hint": "La tua cronologia si sta accumulando",
        "suggestion_history_hint_sub": "Sfoglia le tue ultime 20 immagini analizzate.",
        "suggestion_dashboard_hint": "Il tuo pannello ha statistiche interessanti",
        "suggestion_dashboard_hint_sub": "Totali, medie e distribuzione a colpo d'occhio.",
        "suggestion_settings_hint": "Personalizza l'app",
        "suggestion_settings_hint_sub": "Cambia lingua, tema, carattere e abilita la 2FA.",
        "suggestion_idle_help": "Ti serve una guida rapida?",
        "suggestion_idle_help_sub": "Ripeti il tour di benvenuto in qualsiasi momento.",
        "image_not_liver_warning": "Questa immagine non sembra un'ecografia del fegato.\n\nInserisci un'immagine ecografica del fegato appropriata.",
        "result_not_confident": "L'IA non è riuscita a identificare con certezza questa immagine come un'ecografia del fegato.\n\nInserisci un'immagine ecografica del fegato appropriata e riprova.",
        "not_liver_title": "Non è un'ecografia del fegato",
        "not_liver_hint": "Utilizza un'immagine ecografica (sonogramma) del fegato appropriata",
        "voice_welcome": "Benvenuto, {user}",
        "voice_goodbye": "Arrivederci",
        "voice_logged_out": "Disconnessione riuscita",
        "voice_image_loaded": "Immagine caricata con successo",
        "voice_image_rejected": "Questa non è un'ecografia del fegato",
        "voice_inference_result": "{grid_name}, confidenza {conf} per cento",
        "voice_pdf_saved": "Report PDF salvato con successo",
        "voice_error": "Si è verificato un errore",
        "voice_settings_applied": "Impostazioni applicate",
        "voice_language_changed": "Lingua cambiata in {lang}",
        "voice_theme_changed": "Tema cambiato",
        "voice_2fa_enabled": "Autenticazione a due fattori abilitata",
        "voice_2fa_disabled": "Autenticazione a due fattori disabilitata",
        "voice_locked": "Sessione bloccata",
        "voice_unlocked": "Sessione sbloccata",
        "voice_processing": "Elaborazione immagine",
        "voice_processing_done": "Elaborazione completata",
        "voice_voice_enabled": "Feedback vocale abilitato",
        "voice_voice_disabled": "Feedback vocale disabilitato",
    },
    "China": {
        "app_title": "肝脏AI-babak",
        "welcome": "欢迎",
        "login": "登录",
        "signup": "注册",
        "logout": "退出登录",
        "exit": "退出",
        "settings": "设置",
        "account_info": "账户信息",
        "username": "用户名",
        "password": "密码",
        "repeat_password": "重复密码",
        "first_name": "名字",
        "last_name": "姓氏",
        "email": "电子邮箱",
        "remember_me": "记住我",
        "login_btn": "登录",
        "signup_btn": "注册",
        "account": "账户",
        "change_language": "更改语言",
        "change_theme": "更改主题",
        "insert_sono": "插入超声图像",
        "process": "处理",
        "mute": "静音",
        "unmute": "取消静音",
        "visit_webpage": "访问网页",
        "please_wait": "请稍候...",
        "processing": "正在处理图像...",
        "result": "结果",
        "confidence": "置信度",
        "grid": "网格",
        "insert_image_first": "请先插入图像！",
        "no_image": "未插入图像",
        "invalid_credentials": "凭据无效",
        "user_exists": "用户名已存在",
        "password_requirements": "密码要求",
        "password_rule": (
            "密码必须：\n"
            "• 长度为 8-16 个字符\n"
            "• 至少包含 1 个小写字母\n"
            "• 至少包含 1 个大写字母\n"
            "• 至少包含 4 个数字\n"
            "• 至少包含 4 个特殊字符"
        ),
        "passwords_dont_match": "密码不匹配！",
        "invalid_email": "请输入有效的电子邮箱地址",
        "empty_fields": "请填写所有字段",
        "signup_success": "注册成功！您现在可以登录。",
        "logout_confirm": "您确定要删除账户吗？",
        "exit_confirm": "您确定要退出吗？",
        "theme_dark_aqua": "深色水蓝-紫色主题",
        "theme_solarized_white": "Solarized 浅色主题",
        "theme_solarized_dark": "Solarized 深色主题",
        "theme_dark_forest": "深林绿色主题",
        "select_language": "选择语言",
        "select_theme": "选择主题",
        "apply": "应用",
        "cancel": "取消",
        "save": "保存",
        "success": "成功",
        "error": "错误",
        "warning": "警告",
        "settings_applied": "设置已应用！",
        "grid_0": "网格 0 (NL) — 正常肝脏",
        "grid_1": "网格 1 (G1) — 1 级",
        "grid_2": "网格 2 (G2) — 2 级",
        "grid_3": "网格 3 (G3) — 3 级",
        "too_many_attempts": "失败次数过多。请等待 5 分钟。",
        "invalid_username": "用户名必须为 3-32 个字符（仅限字母、数字、下划线）。",
        "welcome_user": "欢迎，{user}！",
        "inference_failed": "推理失败",
        "drop_here": "将图像拖放到此处",
        "drop_subtitle": "PNG、JPG、JPEG、BMP、DCM",
        "drop_invalid": "不支持的文件类型。请拖放图像。",
        "image_loaded": "图像加载成功",
        "help_about_title": "帮助与关于",
        "help_about_text": (
            "肝脏AI管道 v3.0\n"
            "开发者：Babak Yousefian\n\n"
            "键盘快捷键：\n"
            "  Ctrl+O      — 插入图像\n"
            "  Ctrl+Enter  — 处理图像\n"
            "  Ctrl+L      — 登录\n"
            "  Ctrl+,      — 设置\n"
            "  Esc         — 关闭当前对话框\n"
            "  F1          — 帮助 / 关于\n\n"
            "提示：您也可以将图像拖放到窗口的任何位置。"
        ),
        "dashboard_title": "仪表板",
        "dash_total": "图像总数",
        "dash_avg_conf": "平均置信度",
        "dash_distribution": "分布",
        "dash_last_5": "最近 5 条结果",
        "dash_no_results": "暂无结果",
        "dash_clear_confirm": "清除所有统计数据？",
        "dash_cleared": "统计数据已清除",
        "dashboard_toggle": "显示/隐藏仪表板",
        "export_pdf": "导出 PDF",
        "pdf_saved": "PDF 已成功保存",
        "pdf_failed": "PDF 生成失败",
        "pdf_subtitle": "分析报告",
        "pdf_patient": "患者：",
        "pdf_date": "日期：",
        "pdf_model": "模型：",
        "pdf_diagnosis": "诊断",
        "pdf_confidence": "置信度",
        "pdf_class_probs": "类别概率",
        "pdf_disclaimer": (
            "本报告由 Babak Yousefian 在经验丰富的医生监督下开发。"
            "它是一个专业的分析工具，无需转诊给医生。"
        ),
        "history_title": "历史记录",
        "history_toggle": "显示/隐藏历史记录",
        "history_empty": "暂无历史记录",
        "history_cleared": "历史记录已清除",
        "history_clear_confirm": "清除所有历史记录？",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "登录",
        "tooltip_desc_login": "登录您的账户",
        "tooltip_tip_login": "访问已保存的历史记录和统计信息",
        "tooltip_title_signup": "注册",
        "tooltip_desc_signup": "创建新账户",
        "tooltip_title_account": "账户",
        "tooltip_desc_account": "点击打开账户菜单",
        "tooltip_desc_account_user": "已登录为 {user}",
        "tooltip_title_dashboard": "仪表板",
        "tooltip_desc_dashboard": "显示/隐藏统计面板 — 总计、分布、最近结果",
        "tooltip_title_history": "历史记录",
        "tooltip_desc_history": "显示/隐藏最近 20 条结果及缩略图",
        "tooltip_title_visit": "访问网页",
        "tooltip_desc_visit": "在浏览器中打开项目的 GitHub 页面",
        "tooltip_tip_visit": "在外部浏览器中打开",
        "tooltip_title_insert": "插入图像",
        "tooltip_desc_insert": "加载超声图像以进行分析",
        "tooltip_tip_insert": "您也可以将图像拖放到窗口的任何位置",
        "tooltip_title_process": "处理",
        "tooltip_desc_process": "对已加载的图像运行 AI 分析",
        "tooltip_tip_process": "需要先加载图像",
        "tooltip_title_mute": "静音音乐",
        "tooltip_desc_mute": "暂停背景音乐",
        "tooltip_title_unmute": "取消静音",
        "tooltip_desc_unmute": "恢复背景音乐",
        "tooltip_title_export": "导出 PDF",
        "tooltip_desc_export": "将当前结果保存为 PDF 报告",
        "tooltip_tip_export": "包含图像、诊断和类别概率",
        "tooltip_title_dash_clear": "清除统计",
        "tooltip_desc_dash_clear": "重置此账户的所有统计信息",
        "tooltip_tip_dash_clear": "此操作无法撤销",
        "tooltip_title_dash_close": "关闭仪表板",
        "tooltip_desc_dash_close": "隐藏统计面板",
        "tooltip_desc_dash_total": "已处理图像总数",
        "tooltip_desc_dash_avg": "所有结果的平均置信度",
        "tooltip_desc_dash_dist": "按等级划分的结果分布",
        "tooltip_title_hist_clear": "清除历史记录",
        "tooltip_desc_hist_clear": "删除所有保存的结果和缩略图",
        "tooltip_tip_hist_clear": "这也会删除所有图像文件",
        "tooltip_title_hist_close": "关闭历史记录",
        "tooltip_desc_hist_close": "隐藏历史记录面板",
        "tooltip_tip_thumb_click": "点击重新加载此结果",
        "tooltip_tip_recent": "点击重新加载此结果",
        "tooltip_title_zoom_in": "放大",
        "tooltip_desc_zoom_in": "增加图像放大倍数",
        "tooltip_title_zoom_out": "缩小",
        "tooltip_desc_zoom_out": "减小图像放大倍数",
        "tooltip_title_zoom_reset": "重置视图",
        "tooltip_desc_zoom_reset": "使图像适应窗口",
        "tooltip_tip_zoom_reset": "您也可以双击图像",
        "notif_inference_ok": "推理完成",
        "follow_system_theme": "跟随系统主题",
        "follow_system_theme_desc": "根据系统设置自动在浅色和深色之间切换",
        "gradcam_title": "注意力热力图",
        "gradcam_show": "显示注意力图",
        "gradcam_toggle": "切换热力图",
        "gradcam_save": "保存图像",
        "gradcam_close": "关闭",
        "gradcam_saved": "图像保存成功",
        "gradcam_no_heatmap": "无可用注意力图。请先处理图像。",
        "update_available": "有可用更新",
        "update_check_failed": "更新检查失败",
        "check_for_updates": "检查更新",
        "check_for_updates_desc": "从 GitHub/网站检查最新版本",
        "up_to_date": "您已是最新版本",
        "you_have_latest": "您正在运行最新版本。",
        "check_internet": "请检查您的网络连接并重试。",
        "downloading": "正在下载...",
        "verifying": "正在验证...",
        "download_now": "立即下载",
        "install_now": "安装并重启",
        "skip_version": "跳过此版本",
        "remind_later": "稍后提醒",
        "whats_new": "新特性",
        "no_changelog": "未提供更新日志。",
        "critical_update": "关键更新",
        "update_ready": "准备安装",
        "app_will_restart": "应用将关闭并重启以应用更新。",
        "install_failed": "安装失败。您的旧版本是安全的。",
        "download_failed": "下载失败。",
        "source_update_requires_zip": "源码更新需要 .zip 文件。",
        "session_locked": "会话已锁定",
        "locked_for_user": "已为 {user} 锁定",
        "unlock": "解锁",
        "session_unlocked": "会话已解锁",
        "auto_lock_notice": "会话将在 2 分钟无操作后锁定。",
        "audit_log_title": "审计日志",
        "audit_export": "导出",
        "audit_exported": "导出成功",
        "audit_export_failed": "导出失败",
        "audit_search_placeholder": "搜索操作或详情...",
        "audit_col_time": "时间",
        "audit_col_level": "级别",
        "audit_col_category": "类别",
        "audit_col_action": "操作",
        "audit_col_user": "用户",
        "audit_details_title": "事件详情",
        "audit_status": "显示 {count} / {total} 个事件",
        "strength_very_weak": "非常弱",
        "strength_weak": "弱",
        "strength_medium": "中等",
        "strength_strong": "强",
        "strength_very_strong": "非常强",
        "strength_rule_length": "8-16 个字符",
        "strength_rule_lowercase": "小写字母",
        "strength_rule_uppercase": "大写字母",
        "strength_rule_digits": "数字",
        "strength_rule_special": "特殊字符",
        "strength_add_chars": "再添加 {n} 个字符",
        "strength_remove_chars": "删除 {n} 个字符",
        "strength_need_digits": "再添加 {n} 个数字",
        "strength_need_special": "再添加 {n} 个特殊字符",
        "strength_common_pattern": "避免常见词汇",
        "strength_common_year": "避免常见年份",
        "strength_has_sequence": "避免序列（1234、abcd）",
        "strength_has_repeats": "避免重复（aaaa、1111）",
        "passwords_match": "密码匹配",
        "2fa_setup_title": "启用双因素认证",
        "2fa_setup_subtitle": "使用 Google Authenticator、Authy 或 Microsoft Authenticator 扫描二维码。",
        "2fa_manual_secret": "或手动输入此代码：",
        "2fa_enter_code": "输入您应用中的 6 位代码：",
        "2fa_verify_btn": "验证",
        "2fa_backup_codes_title": "请保存这些备用代码！",
        "2fa_backup_codes_hint": "每个代码仅可使用一次。请将它们存放在安全的地方。",
        "2fa_done_btn": "完成",
        "2fa_secret_copied": "密钥已复制到剪贴板",
        "2fa_invalid_code": "代码无效。请重试。",
        "2fa_invalid_code_format": "代码必须为 6 位数字。",
        "2fa_verify_title": "双因素认证",
        "2fa_verify_subtitle": "输入您验证器应用中的 6 位代码。",
        "2fa_enter_code_hint": "请输入代码。",
        "2fa_use_backup_code": "改用备用代码",
        "2fa_use_totp_code": "改用验证器代码",
        "2fa_enter_backup_code": "输入您的其中一个备用代码。",
        "2fa_section_title": "安全",
        "2fa_enabled_status": "双因素认证已启用",
        "2fa_disabled_status": "双因素认证已禁用",
        "2fa_enable_btn": "启用 2FA",
        "2fa_disable_btn": "禁用 2FA",
        "2fa_disable_title": "禁用 2FA",
        "2fa_disable_confirm": "您确定吗？您的账户将变得不那么安全。",
        "2fa_disabled_success": "双因素认证已禁用",
        "2fa_enabled_success": "双因素认证已启用",
        "2fa_hint_title": "如何使用此二维码",
        "2fa_hint_step1": "安装身份验证器应用：Google Authenticator、Microsoft Authenticator 或 Authy。",
        "2fa_hint_step2": "打开应用，点击「+」按钮，然后选择「扫描二维码」。",
        "2fa_hint_step3": "将手机摄像头对准上方的二维码（请使用应用内的摄像头，而不是手机自带的相机应用）。",
        "2fa_hint_step4": "应用会显示 6 位代码。在下方输入框中输入该代码。",
        "2fa_manual_secret_hint": "无法扫描？请复制下方密钥，并在应用中手动添加（「输入设置密钥」）。",
        "auto_lock_section_title": "自动锁定",
        "auto_lock_enable": "无操作后锁定",
        "auto_lock_timeout_label": "锁定时间:",
        "auto_lock_warning_title": "自动锁定警告",
        "auto_lock_warning_body": "由于无操作，您的会话将在 {seconds} 秒后锁定。",
        "lock_now": "立即锁定",
        "font_section_title": "字体",
        "font_family_label": "字体族",
        "font_size_label": "基础大小",
        "font_scale_label": "显示缩放",
        "font_scale_small": "小",
        "font_scale_normal": "正常",
        "font_scale_large": "大",
        "font_scale_xlarge": "特大",
        "font_scale_xxlarge": "巨大",
        "font_preview_text": "敏捷的棕色狐狸跳过懒狗 — 1234567890",
        "font_reset": "恢复默认",
        "font_applied": "字体已更新",
        "motion_section_title": "动作和动画",
        "reduce_motion_label": "减少动态效果",
        "reduce_motion_hint": "如果动画让您感到不适，请禁用它们。",
        "empty_history_empty_title": "暂无历史记录",
        "empty_history_empty_subtitle": "处理您的第一张图像即可在此查看",
        "empty_history_empty_hint": "每个结果都会自动保存",
        "empty_history_no_user_title": "登录以查看历史记录",
        "empty_history_no_user_subtitle": "您保存的结果将显示在此处",
        "empty_history_no_user_hint": "历史记录对每个账户私密",
        "empty_dash_no_results_title": "暂无结果",
        "empty_dash_no_results_subtitle": "您最近的分析将显示在此处",
        "empty_dash_no_results_hint": "处理图像即可开始",
        "empty_dash_no_dist_title": "暂无分布数据",
        "empty_dash_no_dist_subtitle": "处理图像后统计信息将逐渐填充",
        "empty_dash_no_dist_hint": "每次分析都很重要",
        "empty_image_empty_title": "未插入图像",
        "empty_image_empty_subtitle": "将图像拖放到此处，或点击「插入」",
        "empty_image_empty_hint": "支持 PNG、JPG、JPEG、BMP、DCM",
        "empty_image_drop_hint_title": "松开以分析",
        "empty_image_drop_hint_subtitle": "松开图像以加载",
        "empty_image_drop_hint_hint": "将自动识别格式",
        "empty_processing_wait_title": "正在处理...",
        "empty_processing_wait_subtitle": "AI 正在分析您的图像",
        "empty_processing_wait_hint": "通常需要几秒钟",
        "tour_show_again": "再次显示导览",
        "tour_completed_msg": "导览已完成！您可以开始了。",
        "tour_skipped_msg": "导览已跳过。您可以从账户菜单再次查看。",
        "tour_btn_next": "下一步",
        "tour_btn_prev": "上一步",
        "tour_btn_skip": "跳过",
        "tour_btn_finish": "完成 ✓",
        "tour_guest_login_title": "登录",
        "tour_guest_login_subtitle": "点击此处登录并解锁全部功能。",
        "tour_guest_signup_title": "创建账户",
        "tour_guest_signup_subtitle": "第一次来？点击此处几秒钟内创建账户。",
        "tour_insert_title": "插入图像",
        "tour_insert_subtitle": "从磁盘加载超声图像，或将图像拖放到窗口任意位置。",
        "tour_process_title": "处理图像",
        "tour_process_subtitle": "对已加载的图像运行 AI 分析。结果将在几秒内显示。",
        "tour_export_title": "导出 PDF",
        "tour_export_subtitle": "将当前结果保存为专业的 PDF 医疗报告。",
        "tour_gradcam_title": "注意力热力图",
        "tour_gradcam_subtitle": "可视化 AI 关注的位置 — 在分析图像上叠加的热力图。",
        "tour_dashboard_title": "仪表板",
        "tour_dashboard_subtitle": "显示/隐藏统计面板：图像总数、平均置信度、等级分布。",
        "tour_history_title": "历史记录",
        "tour_history_subtitle": "浏览最近 20 张已分析的图像。点击任意缩略图重新加载其结果。",
        "tour_visit_title": "访问网页",
        "tour_visit_subtitle": "在浏览器中打开项目的 GitHub 页面以查看文档和更新。",
        "tour_mute_title": "静音音乐",
        "tour_mute_subtitle": "随时暂停或恢复背景音乐。",
        "tour_zoom_title": "缩放控制",
        "tour_zoom_subtitle": "使用鼠标滚轮缩放，拖动平移，双击重置。",
        "tour_account_title": "账户菜单",
        "tour_account_subtitle": "点击此处访问设置、账户信息、审计日志等。",
        "suggestion_image_ready": "图像已加载 — 可以处理了",
        "suggestion_image_ready_sub": "点击「处理」以运行 AI 分析。",
        "suggestion_pdf_ready": "结果已就绪 — 导出为 PDF？",
        "suggestion_pdf_ready_sub": "一键保存专业医疗报告。",
        "suggestion_gradcam_hint": "想知道 AI 看到了什么吗？",
        "suggestion_gradcam_hint_sub": "查看图像上的注意力热力图。",
        "suggestion_repeated_grid": "你总是得到 {grid}",
        "suggestion_repeated_grid_sub": "打开仪表板查看分布。",
        "suggestion_history_hint": "你的历史记录正在累积",
        "suggestion_history_hint_sub": "浏览你最近分析的 20 张图像。",
        "suggestion_dashboard_hint": "你的仪表板有有趣的统计",
        "suggestion_dashboard_hint_sub": "一目了然地查看总数、平均值和分布。",
        "suggestion_settings_hint": "个性化应用",
        "suggestion_settings_hint_sub": "更改语言、主题、字体并启用 2FA。",
        "suggestion_idle_help": "需要快速指南吗？",
        "suggestion_idle_help_sub": "随时重新播放引导之旅。",
        "image_not_liver_warning": "此图像看起来不像肝脏超声图像。\n\n请插入正确的肝脏超声图像。",
        "result_not_confident": "AI 无法确信地将此图像识别为肝脏超声图像。\n\n请插入正确的肝脏超声图像并重试。",
        "not_liver_title": "不是肝脏超声图像",
        "not_liver_hint": "请使用正确的肝脏超声（声像图）图像",
        "voice_welcome": "欢迎，{user}",
        "voice_goodbye": "再见",
        "voice_logged_out": "已成功退出登录",
        "voice_image_loaded": "图像加载成功",
        "voice_image_rejected": "这不是肝脏超声图像",
        "voice_inference_result": "{grid_name}，置信度 {conf} 百分比",
        "voice_pdf_saved": "PDF 报告已成功保存",
        "voice_error": "发生错误",
        "voice_settings_applied": "设置已应用",
        "voice_language_changed": "语言已更改为 {lang}",
        "voice_theme_changed": "主题已更改",
        "voice_2fa_enabled": "双因素认证已启用",
        "voice_2fa_disabled": "双因素认证已禁用",
        "voice_locked": "会话已锁定",
        "voice_unlocked": "会话已解锁",
        "voice_processing": "正在处理图像",
        "voice_processing_done": "处理完成",
        "voice_voice_enabled": "语音反馈已启用",
        "voice_voice_disabled": "语音反馈已禁用",
    },
    "Japan": {
        "app_title": "肝臓AI-babak",
        "welcome": "ようこそ",
        "login": "ログイン",
        "signup": "新規登録",
        "logout": "ログアウト",
        "exit": "終了",
        "settings": "設定",
        "account_info": "アカウント情報",
        "username": "ユーザー名",
        "password": "パスワード",
        "repeat_password": "パスワードの確認",
        "first_name": "名",
        "last_name": "姓",
        "email": "メールアドレス",
        "remember_me": "ログイン状態を保持",
        "login_btn": "ログイン",
        "signup_btn": "新規登録",
        "account": "アカウント",
        "change_language": "言語を変更",
        "change_theme": "テーマを変更",
        "insert_sono": "超音波画像を挿入",
        "process": "処理",
        "mute": "ミュート",
        "unmute": "ミュート解除",
        "visit_webpage": "ウェブページを開く",
        "please_wait": "お待ちください...",
        "processing": "画像を処理中...",
        "result": "結果",
        "confidence": "信頼度",
        "grid": "グリッド",
        "insert_image_first": "最初に画像を挿入してください！",
        "no_image": "画像が挿入されていません",
        "invalid_credentials": "認証情報が無効です",
        "user_exists": "ユーザー名は既に存在します",
        "password_requirements": "パスワードの要件",
        "password_rule": (
            "パスワードは：\n"
            "• 8〜16文字であること\n"
            "• 小文字を1文字以上含むこと\n"
            "• 大文字を1文字以上含むこと\n"
            "• 数字を4桁以上含むこと\n"
            "• 特殊文字を4文字以上含むこと"
        ),
        "passwords_dont_match": "パスワードが一致しません！",
        "invalid_email": "有効なメールアドレスを入力してください",
        "empty_fields": "すべての項目を入力してください",
        "signup_success": "登録に成功しました！ログインできます。",
        "logout_confirm": "アカウントを削除してもよろしいですか？",
        "exit_confirm": "終了してもよろしいですか？",
        "theme_dark_aqua": "ダークアクアパープルテーマ",
        "theme_solarized_white": "ソラリゼッドホワイトテーマ",
        "theme_solarized_dark": "ソラリゼッドダークテーマ",
        "theme_dark_forest": "ダークフォレストグリーンテーマ",
        "select_language": "言語を選択",
        "select_theme": "テーマを選択",
        "apply": "適用",
        "cancel": "キャンセル",
        "save": "保存",
        "success": "成功",
        "error": "エラー",
        "warning": "警告",
        "settings_applied": "設定が適用されました！",
        "grid_0": "グリッド0 (NL) — 正常な肝臓",
        "grid_1": "グリッド1 (G1) — グレード1",
        "grid_2": "グリッド2 (G2) — グレード2",
        "grid_3": "グリッド3 (G3) — グレード3",
        "too_many_attempts": "失敗回数が多すぎます。5分お待ちください。",
        "invalid_username": "ユーザー名は3〜32文字（英数字とアンダースコアのみ）にしてください。",
        "welcome_user": "ようこそ、{user}さん！",
        "inference_failed": "推論に失敗しました",
        "drop_here": "ここに画像をドロップ",
        "drop_subtitle": "PNG、JPG、JPEG、BMP、DCM",
        "drop_invalid": "サポートされていないファイル形式です。画像をドロップしてください。",
        "image_loaded": "画像が正常に読み込まれました",
        "help_about_title": "ヘルプ＆情報",
        "help_about_text": (
            "肝臓AIパイプライン v3.0\n"
            "開発者：Babak Yousefian\n\n"
            "キーボードショートカット：\n"
            "  Ctrl+O      — 画像を挿入\n"
            "  Ctrl+Enter  — 画像を処理\n"
            "  Ctrl+L      — ログイン\n"
            "  Ctrl+,      — 設定\n"
            "  Esc         — 現在のダイアログを閉じる\n"
            "  F1          — ヘルプ / 情報\n\n"
            "ヒント：画像をウィンドウの任意の場所にドラッグ＆ドロップすることもできます。"
        ),
        "dashboard_title": "ダッシュボード",
        "dash_total": "合計画像数",
        "dash_avg_conf": "平均信頼度",
        "dash_distribution": "分布",
        "dash_last_5": "最新5件の結果",
        "dash_no_results": "まだ結果がありません",
        "dash_clear_confirm": "すべての統計を消去しますか？",
        "dash_cleared": "統計を消去しました",
        "dashboard_toggle": "ダッシュボードの表示/非表示",
        "export_pdf": "PDF をエクスポート",
        "pdf_saved": "PDF が正常に保存されました",
        "pdf_failed": "PDF の生成に失敗しました",
        "pdf_subtitle": "分析レポート",
        "pdf_patient": "患者：",
        "pdf_date": "日付：",
        "pdf_model": "モデル：",
        "pdf_diagnosis": "診断",
        "pdf_confidence": "信頼度",
        "pdf_class_probs": "クラス確率",
        "pdf_disclaimer": (
            "このレポートは、経験豊富な医師の監督のもと、Babak Yousefian によって開発されました。"
            "これは専門的な分析ツールであり、医師への紹介は必要ありません。"
        ),
        "history_title": "履歴",
        "history_toggle": "履歴の表示/非表示",
        "history_empty": "まだ履歴がありません",
        "history_cleared": "履歴を消去しました",
        "history_clear_confirm": "すべての履歴を消去しますか？",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "ログイン",
        "tooltip_desc_login": "アカウントにサインイン",
        "tooltip_tip_login": "保存された履歴と統計にアクセス",
        "tooltip_title_signup": "新規登録",
        "tooltip_desc_signup": "新しいアカウントを作成",
        "tooltip_title_account": "アカウント",
        "tooltip_desc_account": "クリックしてアカウントメニューを開く",
        "tooltip_desc_account_user": "{user} としてログイン中",
        "tooltip_title_dashboard": "ダッシュボード",
        "tooltip_desc_dashboard": "統計パネルの表示/非表示 — 合計、分布、最近の結果",
        "tooltip_title_history": "履歴",
        "tooltip_desc_history": "最近 20 件の結果とサムネイルの表示/非表示",
        "tooltip_title_visit": "ウェブページを開く",
        "tooltip_desc_visit": "ブラウザでプロジェクトの GitHub ページを開く",
        "tooltip_tip_visit": "外部ブラウザで開きます",
        "tooltip_title_insert": "画像を挿入",
        "tooltip_desc_insert": "分析する超音波画像を読み込む",
        "tooltip_tip_insert": "ウィンドウの任意の場所に画像をドラッグ＆ドロップすることもできます",
        "tooltip_title_process": "処理",
        "tooltip_desc_process": "読み込んだ画像に対して AI 分析を実行",
        "tooltip_tip_process": "まず画像を読み込む必要があります",
        "tooltip_title_mute": "ミュート",
        "tooltip_desc_mute": "バックグラウンドミュージックを一時停止",
        "tooltip_title_unmute": "ミュート解除",
        "tooltip_desc_unmute": "バックグラウンドミュージックを再開",
        "tooltip_title_export": "PDF をエクスポート",
        "tooltip_desc_export": "現在の結果を PDF レポートとして保存",
        "tooltip_tip_export": "画像、診断、クラス確率が含まれます",
        "tooltip_title_dash_clear": "統計を消去",
        "tooltip_desc_dash_clear": "このアカウントのすべての統計をリセット",
        "tooltip_tip_dash_clear": "この操作は元に戻せません",
        "tooltip_title_dash_close": "ダッシュボードを閉じる",
        "tooltip_desc_dash_close": "統計パネルを非表示",
        "tooltip_desc_dash_total": "処理された画像の総数",
        "tooltip_desc_dash_avg": "すべての結果の平均信頼度",
        "tooltip_desc_dash_dist": "グレード別の結果分布",
        "tooltip_title_hist_clear": "履歴を消去",
        "tooltip_desc_hist_clear": "保存されたすべての結果とサムネイルを削除",
        "tooltip_tip_hist_clear": "すべての画像ファイルも削除されます",
        "tooltip_title_hist_close": "履歴を閉じる",
        "tooltip_desc_hist_close": "履歴パネルを非表示",
        "tooltip_tip_thumb_click": "クリックしてこの結果を再読み込み",
        "tooltip_tip_recent": "クリックしてこの結果を再読み込み",
        "tooltip_title_zoom_in": "ズームイン",
        "tooltip_desc_zoom_in": "画像の倍率を上げる",
        "tooltip_title_zoom_out": "ズームアウト",
        "tooltip_desc_zoom_out": "画像の倍率を下げる",
        "tooltip_title_zoom_reset": "ビューをリセット",
        "tooltip_desc_zoom_reset": "画像をウィンドウに合わせる",
        "tooltip_tip_zoom_reset": "画像をダブルクリックすることもできます",
        "notif_inference_ok": "推論完了",
        "follow_system_theme": "システムテーマに従う",
        "follow_system_theme_desc": "システム設定に基づいてライトとダークを自動的に切り替えます",
        "gradcam_title": "アテンションヒートマップ",
        "gradcam_show": "アテンションマップを表示",
        "gradcam_toggle": "ヒートマップを切り替え",
        "gradcam_save": "画像を保存",
        "gradcam_close": "閉じる",
        "gradcam_saved": "画像が正常に保存されました",
        "gradcam_no_heatmap": "利用可能なアテンションマップがありません。先に画像を処理してください。",
        "update_available": "アップデートが利用可能です",
        "update_check_failed": "アップデートの確認に失敗しました",
        "check_for_updates": "アップデートを確認",
        "check_for_updates_desc": "GitHub/ウェブサイトで最新バージョンを確認",
        "up_to_date": "最新の状態です",
        "you_have_latest": "最新バージョンを実行しています。",
        "check_internet": "インターネット接続を確認してもう一度お試しください。",
        "downloading": "ダウンロード中...",
        "verifying": "検証中...",
        "download_now": "今すぐダウンロード",
        "install_now": "インストールして再起動",
        "skip_version": "このバージョンをスキップ",
        "remind_later": "後で通知",
        "whats_new": "新機能",
        "no_changelog": "変更履歴が提供されていません。",
        "critical_update": "重要な更新",
        "update_ready": "インストールの準備ができました",
        "app_will_restart": "アップデートを適用するため、アプリは閉じて再起動します。",
        "install_failed": "インストールに失敗しました。古いバージョンは安全です。",
        "download_failed": "ダウンロードに失敗しました。",
        "source_update_requires_zip": "ソースアップデートには .zip ファイルが必要です。",
        "session_locked": "セッションロック",
        "locked_for_user": "{user} のためにロック中",
        "unlock": "ロック解除",
        "session_unlocked": "セッションのロックを解除しました",
        "auto_lock_notice": "2分間操作がないとセッションがロックされます。",
        "audit_log_title": "監査ログ",
        "audit_export": "エクスポート",
        "audit_exported": "正常にエクスポートされました",
        "audit_export_failed": "エクスポートに失敗しました",
        "audit_search_placeholder": "アクションまたは詳細を検索...",
        "audit_col_time": "時刻",
        "audit_col_level": "レベル",
        "audit_col_category": "カテゴリ",
        "audit_col_action": "アクション",
        "audit_col_user": "ユーザー",
        "audit_details_title": "イベントの詳細",
        "audit_status": "{total} 件中 {count} 件を表示中",
        "strength_very_weak": "非常弱",
        "strength_weak": "弱",
        "strength_medium": "中等",
        "strength_strong": "强",
        "strength_very_strong": "非常强",
        "strength_rule_length": "8-16 个字符",
        "strength_rule_lowercase": "小写字母",
        "strength_rule_uppercase": "大写字母",
        "strength_rule_digits": "数字",
        "strength_rule_special": "特殊字符",
        "strength_add_chars": "再添加 {n} 个字符",
        "strength_remove_chars": "删除 {n} 个字符",
        "strength_need_digits": "再添加 {n} 个数字",
        "strength_need_special": "再添加 {n} 个特殊字符",
        "strength_common_pattern": "避免常见词汇",
        "strength_common_year": "避免常见年份",
        "strength_has_sequence": "避免序列（1234、abcd）",
        "strength_has_repeats": "避免重复（aaaa、1111）",
        "passwords_match": "密码匹配",
        "2fa_setup_title": "二要素認証を有効にする",
        "2fa_setup_subtitle": "Google Authenticator、Authy、または Microsoft Authenticator で QR コードをスキャンしてください。",
        "2fa_manual_secret": "または、このコードを手動で入力してください：",
        "2fa_enter_code": "アプリの 6 桁のコードを入力してください：",
        "2fa_verify_btn": "確認",
        "2fa_backup_codes_title": "これらのバックアップコードを保存してください！",
        "2fa_backup_codes_hint": "各コードは 1 回のみ使用できます。安全な場所に保管してください。",
        "2fa_done_btn": "完了",
        "2fa_secret_copied": "シークレットコードをクリップボードにコピーしました",
        "2fa_invalid_code": "無効なコードです。もう一度お試しください。",
        "2fa_invalid_code_format": "コードは 6 桁の数字である必要があります。",
        "2fa_verify_title": "二要素認証",
        "2fa_verify_subtitle": "認証アプリの 6 桁のコードを入力してください。",
        "2fa_enter_code_hint": "コードを入力してください。",
        "2fa_use_backup_code": "バックアップコードを使用",
        "2fa_use_totp_code": "認証アプリのコードを使用",
        "2fa_enter_backup_code": "バックアップコードの 1 つを入力してください。",
        "2fa_section_title": "セキュリティ",
        "2fa_enabled_status": "二要素認証が有効です",
        "2fa_disabled_status": "二要素認証が無効です",
        "2fa_enable_btn": "2FA を有効にする",
        "2fa_disable_btn": "2FA を無効にする",
        "2fa_disable_title": "2FA を無効にする",
        "2fa_disable_confirm": "よろしいですか？アカウントの安全性が低下します。",
        "2fa_disabled_success": "二要素認証を無効にしました",
        "2fa_enabled_success": "二要素認証を有効にしました",
        "2fa_hint_title": "この QR コードの使い方",
        "2fa_hint_step1": "認証アプリをインストールしてください：Google Authenticator、Microsoft Authenticator、または Authy。",
        "2fa_hint_step2": "アプリを開き、「+」ボタンをタップして「QR コードをスキャン」を選択します。",
        "2fa_hint_step3": "スマートフォンのカメラを上の QR コードに向けてください（アプリのカメラを使用し、スマートフォンのカメラアプリは使わないでください）。",
        "2fa_hint_step4": "アプリに 6 桁のコードが表示されます。下の入力欄に入力してください。",
        "2fa_manual_secret_hint": "スキャンできない場合は、下のキーをコピーしてアプリに手動で追加してください（「設定キーを入力」）。",
        "auto_lock_section_title": "自動ロック",
        "auto_lock_enable": "無操作後にロック",
        "auto_lock_timeout_label": "ロック時間:",
        "auto_lock_warning_title": "自動ロックの警告",
        "auto_lock_warning_body": "無操作のため、{seconds} 秒後にセッションがロックされます。",
        "lock_now": "今すぐロック",
        "font_section_title": "フォント",
        "font_family_label": "フォントファミリー",
        "font_size_label": "基本サイズ",
        "font_scale_label": "表示スケール",
        "font_scale_small": "小",
        "font_scale_normal": "標準",
        "font_scale_large": "大",
        "font_scale_xlarge": "特大",
        "font_scale_xxlarge": "巨大",
        "font_preview_text": "いろはにほへと ちりぬるを — 1234567890",
        "font_reset": "デフォルトに戻す",
        "font_applied": "フォントを更新しました",
        "motion_section_title": "モーションとアニメーション",
        "reduce_motion_label": "モーションを減らす",
        "reduce_motion_hint": "アニメーションが気になる場合は無効にしてください。",
        "empty_history_empty_title": "履歴はまだありません",
        "empty_history_empty_subtitle": "最初の画像を処理するとここに表示されます",
        "empty_history_empty_hint": "各結果は自動的に保存されます",
        "empty_history_no_user_title": "サインインして履歴を表示",
        "empty_history_no_user_subtitle": "保存された結果がここに表示されます",
        "empty_history_no_user_hint": "履歴はアカウントごとにプライベートです",
        "empty_dash_no_results_title": "結果はまだありません",
        "empty_dash_no_results_subtitle": "最近の分析がここに表示されます",
        "empty_dash_no_results_hint": "画像を処理して開始してください",
        "empty_dash_no_dist_title": "分布データがありません",
        "empty_dash_no_dist_subtitle": "画像を処理すると統計が表示されます",
        "empty_dash_no_dist_hint": "すべての分析が重要です",
        "empty_image_empty_title": "画像が挿入されていません",
        "empty_image_empty_subtitle": "ここに画像をドラッグ＆ドロップするか、挿入をクリック",
        "empty_image_empty_hint": "PNG、JPG、JPEG、BMP、DCM に対応",
        "empty_image_drop_hint_title": "ドロップして分析",
        "empty_image_drop_hint_subtitle": "画像を離して読み込み",
        "empty_image_drop_hint_hint": "形式は自動的に検出されます",
        "empty_processing_wait_title": "処理中...",
        "empty_processing_wait_subtitle": "AI が画像を分析しています",
        "empty_processing_wait_hint": "通常数秒かかります",
        "tour_show_again": "ツアーを再表示",
        "tour_completed_msg": "ツアーが完了しました！準備完了です。",
        "tour_skipped_msg": "ツアーをスキップしました。アカウントメニューから再表示できます。",
        "tour_btn_next": "次へ",
        "tour_btn_prev": "戻る",
        "tour_btn_skip": "スキップ",
        "tour_btn_finish": "完了 ✓",
        "tour_guest_login_title": "サインイン",
        "tour_guest_login_subtitle": "ここをクリックしてサインインし、すべての機能を利用できます。",
        "tour_guest_signup_title": "アカウント作成",
        "tour_guest_signup_subtitle": "初めてですか？ここをクリックして数秒でアカウントを作成できます。",
        "tour_insert_title": "画像を挿入",
        "tour_insert_subtitle": "ディスクから超音波画像を読み込むか、ウィンドウの任意の場所にドラッグ＆ドロップします。",
        "tour_process_title": "画像を処理",
        "tour_process_subtitle": "読み込んだ画像に対して AI 分析を実行します。結果は数秒で表示されます。",
        "tour_export_title": "PDF をエクスポート",
        "tour_export_subtitle": "現在の結果をプロフェッショナルな PDF 医療レポートとして保存します。",
        "tour_gradcam_title": "アテンションヒートマップ",
        "tour_gradcam_subtitle": "AI がどこを見ているかを可視化 — 分析画像に重ねたヒートマップ。",
        "tour_dashboard_title": "ダッシュボード",
        "tour_dashboard_subtitle": "統計パネルの表示/非表示：合計画像数、平均信頼度、グレード分布。",
        "tour_history_title": "履歴",
        "tour_history_subtitle": "最近分析した 20 枚の画像を閲覧できます。サムネイルをクリックして結果を再読み込みします。",
        "tour_visit_title": "ウェブページを開く",
        "tour_visit_subtitle": "ブラウザでプロジェクトの GitHub ページを開き、ドキュメントと更新を確認します。",
        "tour_mute_title": "ミュート",
        "tour_mute_subtitle": "バックグラウンドミュージックをいつでも一時停止・再開できます。",
        "tour_zoom_title": "ズームコントロール",
        "tour_zoom_subtitle": "マウスホイールでズーム、ドラッグでパン、ダブルクリックでリセット。",
        "tour_account_title": "アカウントメニュー",
        "tour_account_subtitle": "ここをクリックして設定、アカウント情報、監査ログなどにアクセスします。",
        "suggestion_image_ready": "画像を読み込みました — 処理できます",
        "suggestion_image_ready_sub":  "「処理」をクリックして AI 分析を実行してください。",
        "suggestion_pdf_ready":        "結果の準備ができました — PDF にエクスポートしますか？",
        "suggestion_pdf_ready_sub":    "プロフェッショナルな医療レポートをワンクリックで保存。",
        "suggestion_gradcam_hint":     "AI がどこを見たか気になりますか？",
        "suggestion_gradcam_hint_sub": "画像上のアテンションヒートマップをご覧ください。",
        "suggestion_repeated_grid":    "いつも {grid} になりますね",
        "suggestion_repeated_grid_sub":"ダッシュボードを開いて分布を確認してください。",
        "suggestion_history_hint":     "履歴がたまっています",
        "suggestion_history_hint_sub": "最近分析した 20 枚の画像を閲覧できます。",
        "suggestion_dashboard_hint":   "ダッシュボードに興味深い統計があります",
        "suggestion_dashboard_hint_sub":"合計、平均、分布をひと目で確認できます。",
        "suggestion_settings_hint":    "アプリをパーソナライズ",
        "suggestion_settings_hint_sub":"言語、テーマ、フォントを変更し、2FA を有効にできます。",
        "suggestion_idle_help":        "クイックガイドが必要ですか？",
        "suggestion_idle_help_sub":    "いつでもオンボーディングツアーを再再生できます。",
        "image_not_liver_warning": "この画像は肝臓の超音波画像ではないようです。\n\n適切な肝臓の超音波画像を挿入してください。",
        "result_not_confident": "AI はこの画像を肝臓の超音波画像として確信を持って識別できませんでした。\n\n適切な肝臓の超音波画像を挿入して、もう一度お試しください。",
        "not_liver_title": "肝臓の超音波画像ではありません",
        "not_liver_hint": "適切な肝臓の超音波（ソノグラム）画像を使用してください",
        "voice_welcome": "ようこそ、{user}さん",
        "voice_goodbye": "さようなら",
        "voice_logged_out": "正常にログアウトしました",
        "voice_image_loaded": "画像が正常に読み込まれました",
        "voice_image_rejected": "これは肝臓の超音波画像ではありません",
        "voice_inference_result": "{grid_name}、信頼度 {conf} パーセント",
        "voice_pdf_saved": "PDFレポートが正常に保存されました",
        "voice_error": "エラーが発生しました",
        "voice_settings_applied": "設定が適用されました",
        "voice_language_changed": "言語が{lang}に変更されました",
        "voice_theme_changed": "テーマが変更されました",
        "voice_2fa_enabled": "二要素認証が有効になりました",
        "voice_2fa_disabled": "二要素認証が無効になりました",
        "voice_locked": "セッションがロックされました",
        "voice_unlocked": "セッションのロックが解除されました",
        "voice_processing": "画像を処理中",
        "voice_processing_done": "処理が完了しました",
        "voice_voice_enabled": "音声フィードバックが有効になりました",
        "voice_voice_disabled": "音声フィードバックが無効になりました",
    },
    "SouthKorea": {
        "app_title": "간 AI-babak",
        "welcome": "환영합니다",
        "login": "로그인",
        "signup": "회원가입",
        "logout": "로그아웃",
        "exit": "종료",
        "settings": "설정",
        "account_info": "계정 정보",
        "username": "사용자 이름",
        "password": "비밀번호",
        "repeat_password": "비밀번호 확인",
        "first_name": "이름",
        "last_name": "성",
        "email": "이메일 주소",
        "remember_me": "로그인 상태 유지",
        "login_btn": "로그인",
        "signup_btn": "회원가입",
        "account": "계정",
        "change_language": "언어 변경",
        "change_theme": "테마 변경",
        "insert_sono": "초음파 이미지 삽입",
        "process": "처리",
        "mute": "음소거",
        "unmute": "음소거 해제",
        "visit_webpage": "웹페이지 방문",
        "please_wait": "잠시 기다려 주세요...",
        "processing": "이미지 처리 중...",
        "result": "결과",
        "confidence": "신뢰도",
        "grid": "그리드",
        "insert_image_first": "먼저 이미지를 삽입하세요!",
        "no_image": "이미지가 삽입되지 않았습니다",
        "invalid_credentials": "잘못된 인증 정보",
        "user_exists": "사용자 이름이 이미 존재합니다",
        "password_requirements": "비밀번호 요구 사항",
        "password_rule": (
            "비밀번호는 다음을 충족해야 합니다:\n"
            "• 8-16자 길이\n"
            "• 소문자 1개 이상 포함\n"
            "• 대문자 1개 이상 포함\n"
            "• 숫자 4개 이상 포함\n"
            "• 특수 문자 4개 이상 포함"
        ),
        "passwords_dont_match": "비밀번호가 일치하지 않습니다!",
        "invalid_email": "유효한 이메일 주소를 입력하세요",
        "empty_fields": "모든 필드를 입력하세요",
        "signup_success": "회원가입 성공! 이제 로그인할 수 있습니다.",
        "logout_confirm": "계정을 삭제하시겠습니까?",
        "exit_confirm": "종료하시겠습니까?",
        "theme_dark_aqua": "다크 아쿠아-퍼플 테마",
        "theme_solarized_white": "솔라라이즈드 화이트 테마",
        "theme_solarized_dark": "솔라라이즈드 다크 테마",
        "theme_dark_forest": "다크 포레스트 그린 테마",
        "select_language": "언어 선택",
        "select_theme": "테마 선택",
        "apply": "적용",
        "cancel": "취소",
        "save": "저장",
        "success": "성공",
        "error": "오류",
        "warning": "경고",
        "settings_applied": "설정이 적용되었습니다!",
        "grid_0": "그리드 0 (NL) — 정상 간",
        "grid_1": "그리드 1 (G1) — 등급 1",
        "grid_2": "그리드 2 (G2) — 등급 2",
        "grid_3": "그리드 3 (G3) — 등급 3",
        "too_many_attempts": "실패 횟수가 너무 많습니다. 5분 기다려 주세요.",
        "invalid_username": "사용자 이름은 3-32자(영문, 숫자, 밑줄만)여야 합니다.",
        "welcome_user": "환영합니다, {user}님!",
        "inference_failed": "추론 실패",
        "drop_here": "여기에 이미지를 드롭하세요",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "지원되지 않는 파일 형식입니다. 이미지를 드롭하세요.",
        "image_loaded": "이미지가 성공적으로 로드되었습니다",
        "help_about_title": "도움말 및 정보",
        "help_about_text": (
            "간 AI 파이프라인 v3.0\n"
            "개발자: Babak Yousefian\n\n"
            "키보드 단축키:\n"
            "  Ctrl+O      — 이미지 삽입\n"
            "  Ctrl+Enter  — 이미지 처리\n"
            "  Ctrl+L      — 로그인\n"
            "  Ctrl+,      — 설정\n"
            "  Esc         — 현재 대화상자 닫기\n"
            "  F1          — 도움말 / 정보\n\n"
            "팁: 창의 아무 곳에나 이미지를 드래그 앤 드롭할 수도 있습니다."
        ),
        "dashboard_title": "대시보드",
        "dash_total": "총 이미지",
        "dash_avg_conf": "평균 신뢰도",
        "dash_distribution": "분포",
        "dash_last_5": "최근 5개 결과",
        "dash_no_results": "아직 결과가 없습니다",
        "dash_clear_confirm": "모든 통계를 지우시겠습니까?",
        "dash_cleared": "통계가 지워졌습니다",
        "dashboard_toggle": "대시보드 표시/숨기기",
        "export_pdf": "PDF 내보내기",
        "pdf_saved": "PDF가 성공적으로 저장되었습니다",
        "pdf_failed": "PDF 생성 실패",
        "pdf_subtitle": "분석 보고서",
        "pdf_patient": "환자:",
        "pdf_date": "날짜:",
        "pdf_model": "모델:",
        "pdf_diagnosis": "진단",
        "pdf_confidence": "신뢰도",
        "pdf_class_probs": "클래스 확률",
        "pdf_disclaimer": (
            "이 보고서는 경험 많은 의사의 감독 하에 Babak Yousefian이 개발했습니다. "
            "전문 분석 도구이며 의사에게 의뢰할 필요가 없습니다."
        ),
        "history_title": "기록",
        "history_toggle": "기록 표시/숨기기",
        "history_empty": "아직 기록이 없습니다",
        "history_cleared": "기록이 지워졌습니다",
        "history_clear_confirm": "모든 기록을 지우시겠습니까?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "로그인",
        "tooltip_desc_login": "계정에 로그인",
        "tooltip_tip_login": "저장된 기록 및 통계에 액세스",
        "tooltip_title_signup": "회원가입",
        "tooltip_desc_signup": "새 계정 만들기",
        "tooltip_title_account": "계정",
        "tooltip_desc_account": "계정 메뉴를 열려면 클릭",
        "tooltip_desc_account_user": "{user}(으)로 로그인됨",
        "tooltip_title_dashboard": "대시보드",
        "tooltip_desc_dashboard": "통계 패널 표시/숨기기 — 총계, 분포, 최근 결과",
        "tooltip_title_history": "기록",
        "tooltip_desc_history": "최근 20개 결과와 썸네일 표시/숨기기",
        "tooltip_title_visit": "웹페이지 방문",
        "tooltip_desc_visit": "브라우저에서 프로젝트 GitHub 페이지 열기",
        "tooltip_tip_visit": "외부 브라우저에서 열립니다",
        "tooltip_title_insert": "이미지 삽입",
        "tooltip_desc_insert": "분석할 초음파 이미지 불러오기",
        "tooltip_tip_insert": "창의 아무 곳에나 이미지를 드래그 앤 드롭할 수도 있습니다",
        "tooltip_title_process": "처리",
        "tooltip_desc_process": "불러온 이미지에 AI 분석 실행",
        "tooltip_tip_process": "먼저 이미지를 불러와야 합니다",
        "tooltip_title_mute": "음소거",
        "tooltip_desc_mute": "배경 음악 일시정지",
        "tooltip_title_unmute": "음소거 해제",
        "tooltip_desc_unmute": "배경 음악 재개",
        "tooltip_title_export": "PDF 내보내기",
        "tooltip_desc_export": "현재 결과를 PDF 보고서로 저장",
        "tooltip_tip_export": "이미지, 진단 및 클래스 확률 포함",
        "tooltip_title_dash_clear": "통계 지우기",
        "tooltip_desc_dash_clear": "이 계정의 모든 통계 재설정",
        "tooltip_tip_dash_clear": "이 작업은 취소할 수 없습니다",
        "tooltip_title_dash_close": "대시보드 닫기",
        "tooltip_desc_dash_close": "통계 패널 숨기기",
        "tooltip_desc_dash_total": "처리된 총 이미지 수",
        "tooltip_desc_dash_avg": "모든 결과의 평균 신뢰도",
        "tooltip_desc_dash_dist": "등급별 결과 분포",
        "tooltip_title_hist_clear": "기록 지우기",
        "tooltip_desc_hist_clear": "저장된 모든 결과와 썸네일 삭제",
        "tooltip_tip_hist_clear": "모든 이미지 파일도 삭제됩니다",
        "tooltip_title_hist_close": "기록 닫기",
        "tooltip_desc_hist_close": "기록 패널 숨기기",
        "tooltip_tip_thumb_click": "이 결과를 다시 불러오려면 클릭",
        "tooltip_tip_recent": "이 결과를 다시 불러오려면 클릭",
        "tooltip_title_zoom_in": "확대",
        "tooltip_desc_zoom_in": "이미지 확대 비율 증가",
        "tooltip_title_zoom_out": "축소",
        "tooltip_desc_zoom_out": "이미지 확대 비율 감소",
        "tooltip_title_zoom_reset": "보기 재설정",
        "tooltip_desc_zoom_reset": "이미지를 창에 맞추기",
        "tooltip_tip_zoom_reset": "이미지를 두 번 클릭할 수도 있습니다",
        "notif_inference_ok": "추론 완료",
        "follow_system_theme": "시스템 테마 따르기",
        "follow_system_theme_desc": "시스템 설정에 따라 라이트와 다크를 자동으로 전환합니다",
        "gradcam_title": "어텐션 히트맵",
        "gradcam_show": "어텐션 맵 표시",
        "gradcam_toggle": "히트맵 전환",
        "gradcam_save": "이미지 저장",
        "gradcam_close": "닫기",
        "gradcam_saved": "이미지가 성공적으로 저장되었습니다",
        "gradcam_no_heatmap": "사용 가능한 어텐션 맵이 없습니다. 먼저 이미지를 처리하세요.",
        "update_available": "갱신이 있습니다",
        "update_check_failed": "갱신 검사에 실패하였습니다",
        "check_for_updates": "갱신 검사",
        "check_for_updates_desc": "GitHub/웹페지에서 최신 판본을 검사",
        "up_to_date": "최신 상태입니다",
        "you_have_latest": "최신 판본을 실행하고 있습니다.",
        "check_internet": "인터네트 련결을 확인하고 다시 시도하십시오.",
        "downloading": "내리적재 중...",
        "verifying": "검증 중...",
        "download_now": "지금 내리적재",
        "install_now": "설치 및 재기동",
        "skip_version": "이 판본 건너뛰기",
        "remind_later": "나중에 알림",
        "whats_new": "새 기능",
        "no_changelog": "변경 기록이 제공되지 않았습니다.",
        "critical_update": "중대한 갱신",
        "update_ready": "설치 준비 완료",
        "app_will_restart": "갱신을 적용하기 위해 프로그램이 종료되고 다시 시작됩니다.",
        "install_failed": "설치 실패. 기존 판본은 안전합니다.",
        "download_failed": "내리적재 실패.",
        "source_update_requires_zip": "원천 갱신에는 .zip 파일이 필요합니다.",
        "session_locked": "세션 잠김",
        "locked_for_user": "{user}님을 위해 잠김",
        "unlock": "잠금 해제",
        "session_unlocked": "세션 잠금 해제됨",
        "auto_lock_notice": "2분 동안 활동이 없으면 세션이 잠깁니다.",
        "audit_log_title": "감사 로그",
        "audit_export": "내보내기",
        "audit_exported": "내보내기 성공",
        "audit_export_failed": "내보내기 실패",
        "audit_search_placeholder": "작업 또는 세부 정보 검색...",
        "audit_col_time": "시간",
        "audit_col_level": "수준",
        "audit_col_category": "카테고리",
        "audit_col_action": "작업",
        "audit_col_user": "사용자",
        "audit_details_title": "이벤트 세부 정보",
        "audit_status": "{total}개 중 {count}개 이벤트 표시 중",
        "strength_very_weak": "매우 약함",
        "strength_weak": "약함",
        "strength_medium": "보통",
        "strength_strong": "강함",
        "strength_very_strong": "매우 강함",
        "strength_rule_length": "8-16자",
        "strength_rule_lowercase": "소문자",
        "strength_rule_uppercase": "대문자",
        "strength_rule_digits": "숫자",
        "strength_rule_special": "특수 문자",
        "strength_add_chars": "{n}자 더 추가하세요",
        "strength_remove_chars": "{n}자 제거하세요",
        "strength_need_digits": "숫자 {n}개 더 추가하세요",
        "strength_need_special": "특수 문자 {n}개 더 추가하세요",
        "strength_common_pattern": "일반적인 단어를 피하세요",
        "strength_common_year": "일반적인 연도를 피하세요",
        "strength_has_sequence": "연속된 문자열(1234, abcd)을 피하세요",
        "strength_has_repeats": "반복(aaaa, 1111)을 피하세요",
        "passwords_match": "비밀번호가 일치합니다",
        "2fa_setup_title": "2단계 인증 활성화",
        "2fa_setup_subtitle": "Google Authenticator, Authy 또는 Microsoft Authenticator로 QR 코드를 스캔하세요.",
        "2fa_manual_secret": "또는 이 코드를 수동으로 입력하세요:",
        "2fa_enter_code": "앱에서 6자리 코드를 입력하세요:",
        "2fa_verify_btn": "확인",
        "2fa_backup_codes_title": "이 백업 코드를 저장하세요!",
        "2fa_backup_codes_hint": "각 코드는 한 번만 사용할 수 있습니다. 안전한 곳에 보관하세요.",
        "2fa_done_btn": "완료",
        "2fa_secret_copied": "비밀 코드가 클립보드에 복사되었습니다",
        "2fa_invalid_code": "잘못된 코드입니다. 다시 시도하세요.",
        "2fa_invalid_code_format": "코드는 6자리 숫자여야 합니다.",
        "2fa_verify_title": "2단계 인증",
        "2fa_verify_subtitle": "인증 앱에서 6자리 코드를 입력하세요.",
        "2fa_enter_code_hint": "코드를 입력하세요.",
        "2fa_use_backup_code": "백업 코드 사용",
        "2fa_use_totp_code": "인증 앱 코드 사용",
        "2fa_enter_backup_code": "백업 코드 중 하나를 입력하세요.",
        "2fa_section_title": "보안",
        "2fa_enabled_status": "2단계 인증이 활성화되어 있습니다",
        "2fa_disabled_status": "2단계 인증이 비활성화되어 있습니다",
        "2fa_enable_btn": "2FA 활성화",
        "2fa_disable_btn": "2FA 비활성화",
        "2fa_disable_title": "2FA 비활성화",
        "2fa_disable_confirm": "확실합니까? 계정 보안이 약해집니다.",
        "2fa_disabled_success": "2단계 인증이 비활성화되었습니다",
        "2fa_enabled_success": "2단계 인증이 활성화되었습니다",
        "2fa_hint_title": "이 QR 코드 사용 방법",
        "2fa_hint_step1": "인증 앱을 설치하세요: Google Authenticator, Microsoft Authenticator 또는 Authy.",
        "2fa_hint_step2": "앱을 열고 \"+\" 버튼을 탭한 다음 \"QR 코드 스캔\"을 선택하세요.",
        "2fa_hint_step3": "휴대폰 카메라를 위의 QR 코드로 향하세요 (휴대폰 카메라 앱이 아닌 앱 내 카메라를 사용하세요).",
        "2fa_hint_step4": "앱에 6자리 코드가 표시됩니다. 아래 입력란에 입력하세요.",
        "2fa_manual_secret_hint": "스캔할 수 없나요? 아래 키를 복사하여 앱에 수동으로 추가하세요 (\"설정 키 입력\").",
        "auto_lock_section_title": "자동 잠금",
        "auto_lock_enable": "비활성 후 잠금",
        "auto_lock_timeout_label": "잠금 시간:",
        "auto_lock_warning_title": "자동 잠금 경고",
        "auto_lock_warning_body": "비활성으로 인해 {seconds}초 후 세션이 잠깁니다.",
        "lock_now": "지금 잠금",
        "font_section_title": "글꼴",
        "font_family_label": "글꼴 패밀리",
        "font_size_label": "기본 크기",
        "font_scale_label": "표시 비율",
        "font_scale_small": "작게",
        "font_scale_normal": "보통",
        "font_scale_large": "크게",
        "font_scale_xlarge": "매우 크게",
        "font_scale_xxlarge": "거대하게",
        "font_preview_text": "다람쥐 헌 쳇바퀴에 타고파 — 1234567890",
        "font_reset": "기본값으로 재설정",
        "font_applied": "글꼴이 업데이트되었습니다",
        "motion_section_title": "모션 및 애니메이션",
        "reduce_motion_label": "모션 줄이기",
        "reduce_motion_hint": "애니메이션이 불편하시면 비활성화하세요.",
        "empty_history_empty_title": "아직 기록이 없습니다",
        "empty_history_empty_subtitle": "첫 이미지를 처리하면 여기에 표시됩니다",
        "empty_history_empty_hint": "각 결과는 자동으로 저장됩니다",
        "empty_history_no_user_title": "기록을 보려면 로그인하세요",
        "empty_history_no_user_subtitle": "저장된 결과가 여기에 표시됩니다",
        "empty_history_no_user_hint": "기록은 계정마다 비공개입니다",
        "empty_dash_no_results_title": "아직 결과가 없습니다",
        "empty_dash_no_results_subtitle": "최근 분석이 여기에 표시됩니다",
        "empty_dash_no_results_hint": "이미지를 처리하여 시작하세요",
        "empty_dash_no_dist_title": "분포 데이터 없음",
        "empty_dash_no_dist_subtitle": "이미지를 처리하면 통계가 채워집니다",
        "empty_dash_no_dist_hint": "모든 분석이 중요합니다",
        "empty_image_empty_title": "이미지가 삽입되지 않았습니다",
        "empty_image_empty_subtitle": "여기에 이미지를 드래그 앤 드롭하거나 삽입을 클릭하세요",
        "empty_image_empty_hint": "PNG, JPG, JPEG, BMP, DCM 지원",
        "empty_image_drop_hint_title": "분석하려면 놓으세요",
        "empty_image_drop_hint_subtitle": "이미지를 놓아 로드",
        "empty_image_drop_hint_hint": "형식이 자동으로 감지됩니다",
        "empty_processing_wait_title": "처리 중...",
        "empty_processing_wait_subtitle": "AI가 이미지를 분석하고 있습니다",
        "empty_processing_wait_hint": "보통 몇 초 걸립니다",
        "tour_show_again": "투어 다시 보기",
        "tour_completed_msg": "투어 완료! 준비되었습니다.",
        "tour_skipped_msg": "투어를 건너뛰었습니다. 계정 메뉴에서 다시 볼 수 있습니다.",
        "tour_btn_next": "다음",
        "tour_btn_prev": "이전",
        "tour_btn_skip": "건너뛰기",
        "tour_btn_finish": "완료 ✓",
        "tour_guest_login_title": "로그인",
        "tour_guest_login_subtitle": "여기를 클릭하여 로그인하고 모든 기능을 사용하세요.",
        "tour_guest_signup_title": "계정 만들기",
        "tour_guest_signup_subtitle": "처음이신가요? 여기를 클릭하여 몇 초 만에 계정을 만드세요.",
        "tour_insert_title": "이미지 삽입",
        "tour_insert_subtitle": "디스크에서 초음파 이미지를 불러오거나 창의 아무 곳에나 드래그 앤 드롭하세요.",
        "tour_process_title": "이미지 처리",
        "tour_process_subtitle": "불러온 이미지에 AI 분석을 실행합니다. 몇 초 안에 결과가 나타납니다.",
        "tour_export_title": "PDF 내보내기",
        "tour_export_subtitle": "현재 결과를 전문적인 PDF 의료 보고서로 저장합니다.",
        "tour_gradcam_title": "어텐션 히트맵",
        "tour_gradcam_subtitle": "AI가 어디를 보는지 시각화 — 분석 이미지 위에 히트맵을 표시합니다.",
        "tour_dashboard_title": "대시보드",
        "tour_dashboard_subtitle": "통계 패널 표시/숨기기: 총 이미지, 평균 신뢰도, 등급 분포.",
        "tour_history_title": "기록",
        "tour_history_subtitle": "최근 분석한 20개 이미지를 찾아보세요. 썸네일을 클릭하여 결과를 다시 불러옵니다.",
        "tour_visit_title": "웹페이지 방문",
        "tour_visit_subtitle": "브라우저에서 프로젝트의 GitHub 페이지를 열어 문서와 업데이트를 확인하세요.",
        "tour_mute_title": "음소거",
        "tour_mute_subtitle": "배경 음악을 언제든지 일시정지하거나 다시 시작할 수 있습니다.",
        "tour_zoom_title": "줌 컨트롤",
        "tour_zoom_subtitle": "마우스 휠로 확대/축소, 드래그로 이동, 더블클릭으로 재설정합니다.",
        "tour_account_title": "계정 메뉴",
        "tour_account_subtitle": "여기를 클릭하여 설정, 계정 정보, 감사 로그 등에 액세스하세요.",
        "suggestion_image_ready":      "이미지가 로드되었습니다 — 처리할 준비 완료",
        "suggestion_image_ready_sub":  "「처리」를 클릭하여 AI 분석을 실행하세요.",
        "suggestion_pdf_ready":        "결과가 준비되었습니다 — PDF로 내보낼까요?",
        "suggestion_pdf_ready_sub":    "전문적인 의료 보고서를 한 번의 클릭으로 저장하세요.",
        "suggestion_gradcam_hint":     "AI가 어디를 보았는지 궁금하신가요?",
        "suggestion_gradcam_hint_sub": "이미지 위의 어텐션 히트맵을 확인하세요.",
        "suggestion_repeated_grid":    "항상 {grid}이 나옵니다",
        "suggestion_repeated_grid_sub":"대시보드를 열어 분포를 확인하세요.",
        "suggestion_history_hint":     "기록이 쌓이고 있습니다",
        "suggestion_history_hint_sub": "최근 분석한 20개 이미지를 둘러보세요.",
        "suggestion_dashboard_hint":   "대시보드에 흥미로운 통계가 있습니다",
        "suggestion_dashboard_hint_sub":"총계, 평균, 분포를 한눈에 확인하세요.",
        "suggestion_settings_hint":    "앱을 개인화하세요",
        "suggestion_settings_hint_sub":"언어, 테마, 글꼴을 변경하고 2FA를 활성화하세요.",
        "suggestion_idle_help":        "빠른 안내가 필요하신가요?",
        "suggestion_idle_help_sub":    "언제든지 온보딩 투어를 다시 재생하세요.",
        "image_not_liver_warning": "이 이미지는 간 초음파 이미지가 아닌 것 같습니다.\n\n올바른 간 초음파 이미지를 삽입하세요.",
        "result_not_confident": "AI가 이 이미지를 간 초음파 이미지로 확신 있게 식별하지 못했습니다.\n\n올바른 간 초음파 이미지를 삽입하고 다시 시도하세요.",
        "not_liver_title": "간 초음파 이미지가 아닙니다",
        "not_liver_hint": "올바른 간 초음파(소노그램) 이미지를 사용하세요",
        "voice_welcome": "환영합니다, {user}님",
        "voice_goodbye": "안녕히 가세요",
        "voice_logged_out": "성공적으로 로그아웃되었습니다",
        "voice_image_loaded": "이미지가 성공적으로 로드되었습니다",
        "voice_image_rejected": "이것은 간 초음파 이미지가 아닙니다",
        "voice_inference_result": "{grid_name}, 신뢰도 {conf} 퍼센트",
        "voice_pdf_saved": "PDF 보고서가 성공적으로 저장되었습니다",
        "voice_error": "오류가 발생했습니다",
        "voice_settings_applied": "설정이 적용되었습니다",
        "voice_language_changed": "언어가 {lang}(으)로 변경되었습니다",
        "voice_theme_changed": "테마가 변경되었습니다",
        "voice_2fa_enabled": "2단계 인증이 활성화되었습니다",
        "voice_2fa_disabled": "2단계 인증이 비활성화되었습니다",
        "voice_locked": "세션이 잠겼습니다",
        "voice_unlocked": "세션 잠금이 해제되었습니다",
        "voice_processing": "이미지 처리 중",
        "voice_processing_done": "처리가 완료되었습니다",
        "voice_voice_enabled": "음성 피드백이 활성화되었습니다",
        "voice_voice_disabled": "음성 피드백이 비활성화되었습니다",
    },
    "NorthKorea": {
        "app_title": "간 AI-babak",
        "welcome": "환영합니다",
        "login": "련결",
        "signup": "등록",
        "logout": "련결 해제",
        "exit": "종료",
        "settings": "설정",
        "account_info": "계정 정보",
        "username": "사용자 이름",
        "password": "암호",
        "repeat_password": "암호 확인",
        "first_name": "이름",
        "last_name": "성",
        "email": "전자우편 주소",
        "remember_me": "련결 상태 유지",
        "login_btn": "련결",
        "signup_btn": "등록",
        "account": "계정",
        "change_language": "언어 변경",
        "change_theme": "테마 변경",
        "insert_sono": "초음파 사진 넣기",
        "process": "처리",
        "mute": "소리 끄기",
        "unmute": "소리 켜기",
        "visit_webpage": "웹페지 방문",
        "please_wait": "잠시 기다려 주십시오...",
        "processing": "사진 처리 중...",
        "result": "결과",
        "confidence": "확신도",
        "grid": "그물",
        "insert_image_first": "먼저 사진을 넣으십시오!",
        "no_image": "사진이 들어가지 않았습니다",
        "invalid_credentials": "인증 정보가 옳지 않습니다",
        "user_exists": "사용자 이름이 이미 있습니다",
        "password_requirements": "암호 요구 사항",
        "password_rule": (
            "암호는 다음과 같아야 합니다:\n"
            "• 8-16자 길이\n"
            "• 소문자 1개 이상 포함\n"
            "• 대문자 1개 이상 포함\n"
            "• 숫자 4개 이상 포함\n"
            "• 특수 문자 4개 이상 포함"
        ),
        "passwords_dont_match": "암호가 일치하지 않습니다!",
        "invalid_email": "옳바른 전자우편 주소를 입력하십시오",
        "empty_fields": "모든 항목을 입력하십시오",
        "signup_success": "등록 성공! 이제 련결할 수 있습니다.",
        "logout_confirm": "계정을 삭제하시겠습니까?",
        "exit_confirm": "종료하시겠습니까?",
        "theme_dark_aqua": "어두운 아쿠아-보라 테마",
        "theme_solarized_white": "솔라리제드 흰색 테마",
        "theme_solarized_dark": "솔라리제드 어두운 테마",
        "theme_dark_forest": "어두운 숲 초록 테마",
        "select_language": "언어 선택",
        "select_theme": "테마 선택",
        "apply": "적용",
        "cancel": "취소",
        "save": "저장",
        "success": "성공",
        "error": "오류",
        "warning": "경고",
        "settings_applied": "설정이 적용되었습니다!",
        "grid_0": "그물 0 (NL) — 정상 간",
        "grid_1": "그물 1 (G1) — 등급 1",
        "grid_2": "그물 2 (G2) — 등급 2",
        "grid_3": "그물 3 (G3) — 등급 3",
        "too_many_attempts": "실패 횟수가 너무 많습니다. 5분 기다려 주십시오.",
        "invalid_username": "사용자 이름은 3-32자(영문, 숫자, 밑줄만)여야 합니다.",
        "welcome_user": "환영합니다, {user}님!",
        "inference_failed": "추론 실패",
        "drop_here": "여기에 사진을 놓으십시오",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "지원되지 않는 파일 형식입니다. 사진을 놓으십시오.",
        "image_loaded": "사진이 성공적으로 로드되었습니다",
        "help_about_title": "도움말 및 정보",
        "help_about_text": (
            "간 AI 흐름선 v3.0\n"
            "개발자: Babak Yousefian\n\n"
            "건반 단축키:\n"
            "  Ctrl+O      — 사진 넣기\n"
            "  Ctrl+Enter  — 사진 처리\n"
            "  Ctrl+L      — 련결\n"
            "  Ctrl+,      — 설정\n"
            "  Esc         — 현재 대화창 닫기\n"
            "  F1          — 도움말 / 정보\n\n"
            "비고: 창의 어느 곳에나 사진을 끌어놓을 수도 있습니다."
        ),
        "dashboard_title": "지시판",
        "dash_total": "사진 총수",
        "dash_avg_conf": "평균 확신도",
        "dash_distribution": "분포",
        "dash_last_5": "최근 5개 결과",
        "dash_no_results": "아직 결과가 없습니다",
        "dash_clear_confirm": "모든 통계를 지우시겠습니까?",
        "dash_cleared": "통계가 지워졌습니다",
        "dashboard_toggle": "대시보드 표시/숨기기",
        "export_pdf": "PDF 내보내기",
        "pdf_saved": "PDF가 성공적으로 저장되었습니다",
        "pdf_failed": "PDF 생성 실패",
        "pdf_subtitle": "분석 보고서",
        "pdf_patient": "환자:",
        "pdf_date": "날짜:",
        "pdf_model": "모델:",
        "pdf_diagnosis": "진단",
        "pdf_confidence": "신뢰도",
        "pdf_class_probs": "클래스 확률",
        "pdf_disclaimer": (
            "이 보고서는 경험 많은 의사의 감독 하에 Babak Yousefian이 개발했습니다. "
            "전문 분석 도구이며 의사에게 의뢰할 필요가 없습니다."
        ),
        "history_title": "력사기록",
        "history_toggle": "력사기록 보이기/숨기기",
        "history_empty": "아직 력사기록이 없습니다",
        "history_cleared": "력사기록이 지워졌습니다",
        "history_clear_confirm": "모든 력사기록을 지우시겠습니까?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "련결",
        "tooltip_desc_login": "계정에 련결",
        "tooltip_tip_login": "저장된 력사기록 및 통계에 접근",
        "tooltip_title_signup": "등록",
        "tooltip_desc_signup": "새 계정 만들기",
        "tooltip_title_account": "계정",
        "tooltip_desc_account": "계정 메뉴를 열려면 클릭",
        "tooltip_desc_account_user": "{user}(으)로 련결됨",
        "tooltip_title_dashboard": "지시판",
        "tooltip_desc_dashboard": "통계 판넬 보이기/숨기기 — 총계, 분포, 최근 결과",
        "tooltip_title_history": "력사기록",
        "tooltip_desc_history": "최근 20개 결과와 작은사진 보이기/숨기기",
        "tooltip_title_visit": "웹페지 방문",
        "tooltip_desc_visit": "브라우저에서 프로젝트 GitHub 페지 열기",
        "tooltip_tip_visit": "외부 브라우저에서 열립니다",
        "tooltip_title_insert": "사진 넣기",
        "tooltip_desc_insert": "분석할 초음파 사진 불러오기",
        "tooltip_tip_insert": "창의 어느 곳에나 사진을 끌어놓을 수도 있습니다",
        "tooltip_title_process": "처리",
        "tooltip_desc_process": "불러온 사진에 AI 분석 실행",
        "tooltip_tip_process": "먼저 사진을 불러와야 합니다",
        "tooltip_title_mute": "소리 끄기",
        "tooltip_desc_mute": "배경 음악 일시정지",
        "tooltip_title_unmute": "소리 켜기",
        "tooltip_desc_unmute": "배경 음악 재개",
        "tooltip_title_export": "PDF 내보내기",
        "tooltip_desc_export": "현재 결과를 PDF 보고서로 저장",
        "tooltip_tip_export": "사진, 진단 및 클래스 확률 포함",
        "tooltip_title_dash_clear": "통계 지우기",
        "tooltip_desc_dash_clear": "이 계정의 모든 통계 재설정",
        "tooltip_tip_dash_clear": "이 작업은 취소할 수 없습니다",
        "tooltip_title_dash_close": "지시판 닫기",
        "tooltip_desc_dash_close": "통계 판넬 숨기기",
        "tooltip_desc_dash_total": "처리된 총 사진수",
        "tooltip_desc_dash_avg": "모든 결과의 평균 확신도",
        "tooltip_desc_dash_dist": "등급별 결과 분포",
        "tooltip_title_hist_clear": "력사기록 지우기",
        "tooltip_desc_hist_clear": "저장된 모든 결과와 작은사진 삭제",
        "tooltip_tip_hist_clear": "모든 사진 파일도 삭제됩니다",
        "tooltip_title_hist_close": "력사기록 닫기",
        "tooltip_desc_hist_close": "력사기록 판넬 숨기기",
        "tooltip_tip_thumb_click": "이 결과를 다시 불러오려면 클릭",
        "tooltip_tip_recent": "이 결과를 다시 불러오려면 클릭",
        "tooltip_title_zoom_in": "확대",
        "tooltip_desc_zoom_in": "사진 확대 비률 증가",
        "tooltip_title_zoom_out": "축소",
        "tooltip_desc_zoom_out": "사진 확대 비률 감소",
        "tooltip_title_zoom_reset": "보기 재설정",
        "tooltip_desc_zoom_reset": "사진을 창에 맞추기",
        "tooltip_tip_zoom_reset": "사진을 두 번 클릭할 수도 있습니다",
        "notif_inference_ok": "추론 완료",
        "follow_system_theme": "체계 테마 따르기",
        "follow_system_theme_desc": "체계 설정에 따라 밝은 테마와 어두운 테마를 자동으로 전환합니다",
        "gradcam_title": "주의 렬지도",
        "gradcam_show": "주의 지도 보이기",
        "gradcam_toggle": "렬지도 전환",
        "gradcam_save": "사진 저장",
        "gradcam_close": "닫기",
        "gradcam_saved": "사진이 성공적으로 저장되었습니다",
        "gradcam_no_heatmap": "사용 가능한 주의 지도가 없습니다. 먼저 사진을 처리하십시오.",
        "update_available": "갱신이 있습니다",
        "update_check_failed": "갱신 검사에 실패하였습니다",
        "check_for_updates": "갱신 검사",
        "check_for_updates_desc": "GitHub/웹페지에서 최신 판본을 검사",
        "up_to_date": "최신 상태입니다",
        "you_have_latest": "최신 판본을 실행하고 있습니다.",
        "check_internet": "인터네트 련결을 확인하고 다시 시도하십시오.",
        "downloading": "내리적재 중...",
        "verifying": "검증 중...",
        "download_now": "지금 내리적재",
        "install_now": "설치 및 재기동",
        "skip_version": "이 판본 건너뛰기",
        "remind_later": "나중에 알림",
        "whats_new": "새 기능",
        "no_changelog": "변경 기록이 제공되지 않았습니다.",
        "critical_update": "중대한 갱신",
        "update_ready": "설치 준비 완료",
        "app_will_restart": "갱신을 적용하기 위해 프로그램이 종료되고 다시 시작됩니다.",
        "install_failed": "설치 실패. 기존 판본은 안전합니다.",
        "download_failed": "내리적재 실패.",
        "source_update_requires_zip": "원천 갱신에는 .zip 파일이 필요합니다.",
        "session_locked": "련결마감",
        "locked_for_user": "{user}를 위해 마감됨",
        "unlock": "마감 풀기",
        "session_unlocked": "련결마감이 풀렸습니다",
        "auto_lock_notice": "2분간 움직임이 없으면 련결이 마감됩니다.",
        "audit_log_title": "검열기록",
        "audit_export": "내보내기",
        "audit_exported": "내보내기 성공",
        "audit_export_failed": "내보내기 실패",
        "audit_search_placeholder": "행동 또는 세부사항 검색...",
        "audit_col_time": "시간",
        "audit_col_level": "수준",
        "audit_col_category": "분류",
        "audit_col_action": "행동",
        "audit_col_user": "사용자",
        "audit_details_title": "사건 세부사항",
        "audit_status": "{total}개 중 {count}개 사건 표시 중",
        "strength_very_weak": "아주 약함",
        "strength_weak": "약함",
        "strength_medium": "보통",
        "strength_strong": "강함",
        "strength_very_strong": "아주 강함",
        "strength_rule_length": "8-16자",
        "strength_rule_lowercase": "소문자",
        "strength_rule_uppercase": "대문자",
        "strength_rule_digits": "숫자",
        "strength_rule_special": "특수 문자",
        "strength_add_chars": "{n}자를 더 넣으십시오",
        "strength_remove_chars": "{n}자를 빼십시오",
        "strength_need_digits": "숫자 {n}개를 더 넣으십시오",
        "strength_need_special": "특수 문자 {n}개를 더 넣으십시오",
        "strength_common_pattern": "흔한 단어를 피하십시오",
        "strength_common_year": "흔한 년도를 피하십시오",
        "strength_has_sequence": "련속된 문자렬(1234, abcd)을 피하십시오",
        "strength_has_repeats": "반복(aaaa, 1111)을 피하십시오",
        "passwords_match": "암호가 일치합니다",
        "2fa_setup_title": "두단계 인증 활성화",
        "2fa_setup_subtitle": "Google Authenticator, Authy 또는 Microsoft Authenticator로 QR 코드를 스캔하십시오.",
        "2fa_manual_secret": "또는 이 코드를 수동으로 입력하십시오:",
        "2fa_enter_code": "응용프로그람에서 6자리 코드를 입력하십시오:",
        "2fa_verify_btn": "확인",
        "2fa_backup_codes_title": "이 예비 코드들을 저장하십시오!",
        "2fa_backup_codes_hint": "각 코드는 한번만 사용할수 있습니다. 안전한 곳에 보관하십시오.",
        "2fa_done_btn": "완료",
        "2fa_secret_copied": "비밀 코드가 클립보드에 복사되였습니다",
        "2fa_invalid_code": "잘못된 코드입니다. 다시 시도하십시오.",
        "2fa_invalid_code_format": "코드는 6자리수자여야 합니다.",
        "2fa_verify_title": "두단계 인증",
        "2fa_verify_subtitle": "인증 응용프로그람에서 6자리 코드를 입력하십시오.",
        "2fa_enter_code_hint": "코드를 입력하십시오.",
        "2fa_use_backup_code": "예비 코드 사용",
        "2fa_use_totp_code": "인증 응용프로그람 코드 사용",
        "2fa_enter_backup_code": "예비 코드중 하나를 입력하십시오.",
        "2fa_section_title": "보안",
        "2fa_enabled_status": "두단계 인증이 활성화되여 있습니다",
        "2fa_disabled_status": "두단계 인증이 비활성화되여 있습니다",
        "2fa_enable_btn": "2FA 활성화",
        "2fa_disable_btn": "2FA 비활성화",
        "2fa_disable_title": "2FA 비활성화",
        "2fa_disable_confirm": "확실합니까? 계정 보안이 약해집니다.",
        "2fa_disabled_success": "두단계 인증이 비활성화되였습니다",
        "2fa_enabled_success": "두단계 인증이 활성화되였습니다",
        "2fa_hint_title": "이 QR 코드 사용 방법",
        "2fa_hint_step1": "인증 앱을 설치하십시오: Google Authenticator, Microsoft Authenticator 또는 Authy.",
        "2fa_hint_step2": "앱을 열고 \"+\" 단추를 누른 다음 \"QR 코드 스캔\"을 선택하십시오.",
        "2fa_hint_step3": "손전화 카메라를 우의 QR 코드에 향하십시오 (손전화 카메라 앱이 아닌 앱안 카메라를 사용하십시오).",
        "2fa_hint_step4": "앱에 6자리 코드가 표시됩니다. 아래 입력란에 입력하십시오.",
        "2fa_manual_secret_hint": "스캔할수 없습니까? 아래 키를 복사하여 앱에 수동으로 추가하십시오 (\"설정 키 입력\").",
        "auto_lock_section_title": "자동 잠금",
        "auto_lock_enable": "비활성 후 잠금",
        "auto_lock_timeout_label": "잠금 시간:",
        "auto_lock_warning_title": "자동 잠금 경고",
        "auto_lock_warning_body": "비활성으로 인해 {seconds}초 후 세션이 잠깁니다.",
        "lock_now": "지금 잠금",
        "font_section_title": "글자체",
        "font_family_label": "글자체 계렬",
        "font_size_label": "기본 크기",
        "font_scale_label": "표시 비률",
        "font_scale_small": "작게",
        "font_scale_normal": "보통",
        "font_scale_large": "크게",
        "font_scale_xlarge": "아주 크게",
        "font_scale_xxlarge": "거대하게",
        "font_preview_text": "조선글자체를 미리보기하는 본문 — 1234567890",
        "font_reset": "기본값으로 되돌리기",
        "font_applied": "글자체가 갱신되였습니다",
        "motion_section_title": "움직임 및 동화상",
        "reduce_motion_label": "움직임 줄이기",
        "reduce_motion_hint": "동화상이 불편하시면 비활성화하십시오.",
        "empty_history_empty_title": "아직 력사기록이 없습니다",
        "empty_history_empty_subtitle": "첫 사진을 처리하면 여기에 표시됩니다",
        "empty_history_empty_hint": "각 결과는 자동으로 저장됩니다",
        "empty_history_no_user_title": "력사기록을 보려면 련결하십시오",
        "empty_history_no_user_subtitle": "저장된 결과가 여기에 표시됩니다",
        "empty_history_no_user_hint": "력사기록은 계정마다 비공개입니다",
        "empty_dash_no_results_title": "아직 결과가 없습니다",
        "empty_dash_no_results_subtitle": "최근 분석이 여기에 표시됩니다",
        "empty_dash_no_results_hint": "사진을 처리하여 시작하십시오",
        "empty_dash_no_dist_title": "분포 데이터 없음",
        "empty_dash_no_dist_subtitle": "사진을 처리하면 통계가 채워집니다",
        "empty_dash_no_dist_hint": "모든 분석이 중요합니다",
        "empty_image_empty_title": "사진이 들어가지 않았습니다",
        "empty_image_empty_subtitle": "여기에 사진을 끌어놓거나 넣기를 누르십시오",
        "empty_image_empty_hint": "PNG, JPG, JPEG, BMP, DCM 지원",
        "empty_image_drop_hint_title": "분석하려면 놓으십시오",
        "empty_image_drop_hint_subtitle": "사진을 놓아 로드",
        "empty_image_drop_hint_hint": "형식이 자동으로 감지됩니다",
        "empty_processing_wait_title": "처리 중...",
        "empty_processing_wait_subtitle": "AI가 사진을 분석하고 있습니다",
        "empty_processing_wait_hint": "보통 몇초 걸립니다",
        "tour_show_again": "탐방 다시 보기",
        "tour_completed_msg": "탐방 완료! 준비되였습니다.",
        "tour_skipped_msg": "탐방을 건너뛰였습니다. 계정메뉴에서 다시 볼수 있습니다.",
        "tour_btn_next": "다음",
        "tour_btn_prev": "이전",
        "tour_btn_skip": "건너뛰기",
        "tour_btn_finish": "완료 ✓",
        "tour_guest_login_title": "련결",
        "tour_guest_login_subtitle": "여기를 눌러 련결하고 모든 기능을 사용하십시오.",
        "tour_guest_signup_title": "계정 만들기",
        "tour_guest_signup_subtitle": "처음입니까? 여기를 눌러 몇초안에 계정을 만드십시오.",
        "tour_insert_title": "사진 넣기",
        "tour_insert_subtitle": "디스크에서 초음파 사진을 불러오거나 창의 어느 곳에나 끌어놓으십시오.",
        "tour_process_title": "사진 처리",
        "tour_process_subtitle": "불러온 사진에 AI 분석을 실행합니다. 몇초안에 결과가 나타납니다.",
        "tour_export_title": "PDF 내보내기",
        "tour_export_subtitle": "현재 결과를 전문적인 PDF 의료보고서로 저장합니다.",
        "tour_gradcam_title": "주의 렬지도",
        "tour_gradcam_subtitle": "AI가 어디를 보는지 시각화 — 분석 사진우에 렬지도를 표시합니다.",
        "tour_dashboard_title": "지시판",
        "tour_dashboard_subtitle": "통계판넬 보이기/숨기기: 총 사진수, 평균 확신도, 등급분포.",
        "tour_history_title": "력사기록",
        "tour_history_subtitle": "최근 분석한 20개 사진을 찾아보십시오. 작은사진을 눌러 결과를 다시 불러옵니다.",
        "tour_visit_title": "웹페지 방문",
        "tour_visit_subtitle": "브라우저에서 프로젝트의 GitHub 페지를 열어 문서와 갱신을 확인하십시오.",
        "tour_mute_title": "소리 끄기",
        "tour_mute_subtitle": "배경음악을 언제든지 일시정지하거나 다시 시작할수 있습니다.",
        "tour_zoom_title": "줌 제어",
        "tour_zoom_subtitle": "마우스굴림으로 확대/축소, 끌기로 이동, 두번클릭으로 재설정합니다.",
        "tour_account_title": "계정메뉴",
        "tour_account_subtitle": "여기를 눌러 설정, 계정정보, 검열기록 등에 접근하십시오.",
        "suggestion_image_ready": "사진을 불러왔습니다 — 처리할준비가 되였습니다",
        "suggestion_image_ready_sub": "《처리》를 눌러 AI 분석을 실행하십시오.",
        "suggestion_pdf_ready": "결과가 준비되였습니다 — PDF로 내보내시겠습니까?",
        "suggestion_pdf_ready_sub": "전문적인 의료보고서를 한번의 클릭으로 저장하십시오.",
        "suggestion_gradcam_hint": "AI가 어디를 보았는지 궁금하십니까?",
        "suggestion_gradcam_hint_sub": "사진우의 주의 렬지도를 보십시오.",
        "suggestion_repeated_grid": "항상 {grid}이 나옵니다",
        "suggestion_repeated_grid_sub": "지시판을 열어 분포를 보십시오.",
        "suggestion_history_hint": "력사기록이 쌓이고 있습니다",
        "suggestion_history_hint_sub": "최근 분석한 20개 사진을 찾아보십시오.",
        "suggestion_dashboard_hint": "지시판에 흥미로운 통계가 있습니다",
        "suggestion_dashboard_hint_sub": "총계, 평균, 분포를 한눈에 보십시오.",
        "suggestion_settings_hint": "프로그람을 개성화하십시오",
        "suggestion_settings_hint_sub": "언어, 테마, 글자체를 변경하고 2FA를 활성화하십시오.",
        "suggestion_idle_help": "빠른 안내가 필요하십니까?",
        "suggestion_idle_help_sub": "언제든지 안내tour를 다시 재생하십시오.",
        "image_not_liver_warning": "이 사진은 간 초음파 사진이 아닌것 같습니다.\n\n올바른 간 초음파 사진을 넣으십시오.",
        "result_not_confident": "AI가 이 사진을 간 초음파 사진으로 확신있게 식별하지 못했습니다.\n\n올바른 간 초음파 사진을 넣고 다시 시도하십시오.",
        "not_liver_title": "간 초음파 사진이 아닙니다",
        "not_liver_hint": "올바른 간 초음파(소노그람) 사진을 사용하십시오",
        "voice_welcome": "환영합니다, {user}동무",
        "voice_goodbye": "안녕히 계십시오",
        "voice_logged_out": "련결이 성공적으로 해제되였습니다",
        "voice_image_loaded": "사진이 성공적으로 로드되였습니다",
        "voice_image_rejected": "이것은 간 초음파 사진이 아닙니다",
        "voice_inference_result": "{grid_name}, 확신도 {conf} 프로",
        "voice_pdf_saved": "PDF 보고서가 성공적으로 저장되였습니다",
        "voice_error": "오유가 발생하였습니다",
        "voice_settings_applied": "설정이 적용되였습니다",
        "voice_language_changed": "언어가 {lang}(으)로 변경되였습니다",
        "voice_theme_changed": "테마가 변경되였습니다",
        "voice_2fa_enabled": "두단계 인증이 활성화되였습니다",
        "voice_2fa_disabled": "두단계 인증이 비활성화되였습니다",
        "voice_locked": "련결이 마감되였습니다",
        "voice_unlocked": "련결마감이 풀렸습니다",
        "voice_processing": "사진을 처리중입니다",
        "voice_processing_done": "처리가 완료되였습니다",
        "voice_voice_enabled": "음성안내가 활성화되였습니다",
        "voice_voice_disabled": "음성안내가 비활성화되였습니다",
    },
    "India": {
        "app_title": "लिवर AI-babak",
        "welcome": "स्वागत है",
        "login": "लॉगिन",
        "signup": "साइन अप",
        "logout": "लॉगआउट",
        "exit": "बाहर निकलें",
        "settings": "सेटिंग्स",
        "account_info": "खाता जानकारी",
        "username": "उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "repeat_password": "पासवर्ड दोहराएं",
        "first_name": "पहला नाम",
        "last_name": "अंतिम नाम",
        "email": "ईमेल पता",
        "remember_me": "मुझे याद रखें",
        "login_btn": "लॉगिन",
        "signup_btn": "साइन अप",
        "account": "खाता",
        "change_language": "भाषा बदलें",
        "change_theme": "थीम बदलें",
        "insert_sono": "सोनो छवि डालें",
        "process": "प्रक्रिया करें",
        "mute": "म्यूट करें",
        "unmute": "अनम्यूट करें",
        "visit_webpage": "वेबपेज देखें",
        "please_wait": "कृपया प्रतीक्षा करें...",
        "processing": "छवि संसाधित हो रही है...",
        "result": "परिणाम",
        "confidence": "विश्वास",
        "grid": "ग्रिड",
        "insert_image_first": "कृपया पहले एक छवि डालें!",
        "no_image": "कोई छवि नहीं डाली गई",
        "invalid_credentials": "अमान्य क्रेडेंशियल",
        "user_exists": "उपयोगकर्ता नाम पहले से मौजूद है",
        "password_requirements": "पासवर्ड आवश्यकताएं",
        "password_rule": (
            "पासवर्ड में होना चाहिए:\n"
            "• 8-16 वर्ण लंबा\n"
            "• कम से कम 1 छोटा अक्षर\n"
            "• कम से कम 1 बड़ा अक्षर\n"
            "• कम से कम 4 अंक\n"
            "• कम से कम 4 विशेष वर्ण"
        ),
        "passwords_dont_match": "पासवर्ड मेल नहीं खाते!",
        "invalid_email": "कृपया एक मान्य ईमेल पता दर्ज करें",
        "empty_fields": "कृपया सभी फ़ील्ड भरें",
        "signup_success": "साइन अप सफल! अब आप लॉगिन कर सकते हैं।",
        "logout_confirm": "क्या आप वाकई अपना खाता हटाना चाहते हैं?",
        "exit_confirm": "क्या आप वाकई बाहर निकलना चाहते हैं?",
        "theme_dark_aqua": "डार्क एक्वा-पर्पल थीम",
        "theme_solarized_white": "सोलराइज़्ड व्हाइट थीम",
        "theme_solarized_dark": "सोलराइज़्ड डार्क थीम",
        "theme_dark_forest": "डार्क फॉरेस्ट ग्रीन थीम",
        "select_language": "भाषा चुनें",
        "select_theme": "थीम चुनें",
        "apply": "लागू करें",
        "cancel": "रद्द करें",
        "save": "सहेजें",
        "success": "सफल",
        "error": "त्रुटि",
        "warning": "चेतावनी",
        "settings_applied": "सेटिंग्स लागू हो गईं!",
        "grid_0": "ग्रिड 0 (NL) — सामान्य लिवर",
        "grid_1": "ग्रिड 1 (G1) — ग्रेड 1",
        "grid_2": "ग्रिड 2 (G2) — ग्रेड 2",
        "grid_3": "ग्रिड 3 (G3) — ग्रेड 3",
        "too_many_attempts": "बहुत अधिक विफल प्रयास। कृपया 5 मिनट प्रतीक्षा करें।",
        "invalid_username": "उपयोगकर्ता नाम 3-32 वर्ण (अक्षर, संख्या, अंडरस्कोर) होना चाहिए।",
        "welcome_user": "स्वागत है, {user}!",
        "inference_failed": "अनुमान विफल",
        "drop_here": "छवि यहाँ छोड़ें",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "असमर्थित फ़ाइल प्रकार। कृपया एक छवि छोड़ें।",
        "image_loaded": "छवि सफलतापूर्वक लोड हुई",
        "help_about_title": "सहायता और परिचय",
        "help_about_text": (
            "लिवर AI पाइपलाइन v3.0\n"
            "विकसितकर्ता: Babak Yousefian\n\n"
            "कीबोर्ड शॉर्टकट:\n"
            "  Ctrl+O      — छवि डालें\n"
            "  Ctrl+Enter  — छवि संसाधित करें\n"
            "  Ctrl+L      — लॉगिन\n"
            "  Ctrl+,      — सेटिंग्स\n"
            "  Esc         — सक्रिय संवाद बंद करें\n"
            "  F1          — सहायता / परिचय\n\n"
            "सुझाव: आप विंडो में कहीं भी छवि खींचकर छोड़ सकते हैं।"
        ),
        "dashboard_title": "डैशबोर्ड",
        "dash_total": "कुल छवियाँ",
        "dash_avg_conf": "औसत विश्वास",
        "dash_distribution": "वितरण",
        "dash_last_5": "अंतिम 5 परिणाम",
        "dash_no_results": "अभी तक कोई परिणाम नहीं",
        "dash_clear_confirm": "सभी आँकड़े साफ़ करें?",
        "dash_cleared": "आँकड़े साफ़ हो गए",
        "dashboard_toggle": "डैशबोर्ड दिखाएं/छिपाएं",
        "export_pdf": "PDF निर्यात करें",
        "pdf_saved": "PDF सफलतापूर्वक सहेजा गया",
        "pdf_failed": "PDF निर्माण विफल",
        "pdf_subtitle": "विश्लेषण रिपोर्ट",
        "pdf_patient": "रोगी:",
        "pdf_date": "तिथि:",
        "pdf_model": "मॉडल:",
        "pdf_diagnosis": "निदान",
        "pdf_confidence": "विश्वास",
        "pdf_class_probs": "वर्ग संभावनाएँ",
        "pdf_disclaimer": (
            "यह रिपोर्ट अनुभवी चिकित्सकों की देखरेख में Babak Yousefian द्वारा विकसित की गई थी। "
            "यह एक पेशेवर विश्लेषण उपकरण है और इसके लिए चिकित्सक के पास रेफ़रल की आवश्यकता नहीं है।"
        ),
        "history_title": "इतिहास",
        "history_toggle": "इतिहास दिखाएं/छिपाएं",
        "history_empty": "अभी तक कोई इतिहास नहीं",
        "history_cleared": "इतिहास साफ़ हो गया",
        "history_clear_confirm": "सारा इतिहास साफ़ करें?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "लॉगिन",
        "tooltip_desc_login": "अपने खाते में साइन इन करें",
        "tooltip_tip_login": "सहेजे गए इतिहास और आँकड़ों तक पहुँचें",
        "tooltip_title_signup": "साइन अप",
        "tooltip_desc_signup": "नया खाता बनाएँ",
        "tooltip_title_account": "खाता",
        "tooltip_desc_account": "खाता मेनू खोलने के लिए क्लिक करें",
        "tooltip_desc_account_user": "{user} के रूप में लॉग इन",
        "tooltip_title_dashboard": "डैशबोर्ड",
        "tooltip_desc_dashboard": "आँकड़े पैनल दिखाएं/छिपाएं — कुल, वितरण, हाल के परिणाम",
        "tooltip_title_history": "इतिहास",
        "tooltip_desc_history": "पिछले 20 परिणाम थंबनेल के साथ दिखाएं/छिपाएं",
        "tooltip_title_visit": "वेबपेज देखें",
        "tooltip_desc_visit": "ब्राउज़र में प्रोजेक्ट का GitHub पेज खोलें",
        "tooltip_tip_visit": "बाहरी ब्राउज़र में खुलता है",
        "tooltip_title_insert": "छवि डालें",
        "tooltip_desc_insert": "विश्लेषण के लिए सोनो छवि लोड करें",
        "tooltip_tip_insert": "आप विंडो में कहीं भी छवि खींचकर छोड़ सकते हैं",
        "tooltip_title_process": "प्रक्रिया करें",
        "tooltip_desc_process": "लोड की गई छवि पर AI विश्लेषण चलाएं",
        "tooltip_tip_process": "पहले एक छवि लोड करनी होगी",
        "tooltip_title_mute": "संगीत म्यूट करें",
        "tooltip_desc_mute": "पृष्ठभूमि संगीत रोकें",
        "tooltip_title_unmute": "संगीत अनम्यूट करें",
        "tooltip_desc_unmute": "पृष्ठभूमि संगीत फिर से शुरू करें",
        "tooltip_title_export": "PDF निर्यात करें",
        "tooltip_desc_export": "वर्तमान परिणाम को PDF रिपोर्ट के रूप में सहेजें",
        "tooltip_tip_export": "छवि, निदान और वर्ग संभावनाएँ शामिल हैं",
        "tooltip_title_dash_clear": "आँकड़े साफ़ करें",
        "tooltip_desc_dash_clear": "इस खाते के सभी आँकड़े रीसेट करें",
        "tooltip_tip_dash_clear": "यह क्रिया पूर्ववत नहीं की जा सकती",
        "tooltip_title_dash_close": "डैशबोर्ड बंद करें",
        "tooltip_desc_dash_close": "आँकड़े पैनल छिपाएं",
        "tooltip_desc_dash_total": "संसाधित छवियों की कुल संख्या",
        "tooltip_desc_dash_avg": "सभी परिणामों पर औसत विश्वास",
        "tooltip_desc_dash_dist": "ग्रेड के अनुसार परिणामों का वितरण",
        "tooltip_title_hist_clear": "इतिहास साफ़ करें",
        "tooltip_desc_hist_clear": "सभी सहेजे गए परिणाम और थंबनेल हटाएं",
        "tooltip_tip_hist_clear": "यह सभी छवि फ़ाइलें भी हटा देगा",
        "tooltip_title_hist_close": "इतिहास बंद करें",
        "tooltip_desc_hist_close": "इतिहास पैनल छिपाएं",
        "tooltip_tip_thumb_click": "इस परिणाम को फिर से लोड करने के लिए क्लिक करें",
        "tooltip_tip_recent": "इस परिणाम को फिर से लोड करने के लिए क्लिक करें",
        "tooltip_title_zoom_in": "ज़ूम इन",
        "tooltip_desc_zoom_in": "छवि आवर्धन बढ़ाएं",
        "tooltip_title_zoom_out": "ज़ूम आउट",
        "tooltip_desc_zoom_out": "छवि आवर्धन घटाएं",
        "tooltip_title_zoom_reset": "दृश्य रीसेट करें",
        "tooltip_desc_zoom_reset": "छवि को विंडो में फिट करें",
        "tooltip_tip_zoom_reset": "आप छवि पर डबल-क्लिक भी कर सकते हैं",
        "notif_inference_ok": "अनुमान पूर्ण",
        "follow_system_theme": "सिस्टम थीम का पालन करें",
        "follow_system_theme_desc": "सिस्टम सेटिंग्स के आधार पर लाइट और डार्क के बीच स्वचालित रूप से स्विच करें",
        "gradcam_title": "अटेंशन हीटमैप",
        "gradcam_show": "अटेंशन मैप दिखाएं",
        "gradcam_toggle": "हीटमैप टॉगल करें",
        "gradcam_save": "छवि सहेजें",
        "gradcam_close": "बंद करें",
        "gradcam_saved": "छवि सफलतापूर्वक सहेजी गई",
        "gradcam_no_heatmap": "कोई अटेंशन मैप उपलब्ध नहीं। कृपया पहले एक छवि संसाधित करें।",
        "update_available": "अद्यतन उपलब्ध है",
        "update_check_failed": "अद्यतन जाँच विफल",
        "check_for_updates": "अद्यतन जाँचें",
        "check_for_updates_desc": "नवीनतम संस्करण के लिए GitHub/वेबसाइट जाँचें",
        "up_to_date": "आप अद्यतित हैं",
        "you_have_latest": "आप नवीनतम संस्करण चला रहे हैं।",
        "check_internet": "कृपया अपना इंटरनेट कनेक्शन जाँचें और पुनः प्रयास करें।",
        "downloading": "डाउनलोड हो रहा है...",
        "verifying": "सत्यापन हो रहा है...",
        "download_now": "अभी डाउनलोड करें",
        "install_now": "इंस्टॉल करें और पुनः आरंभ करें",
        "skip_version": "यह संस्करण छोड़ें",
        "remind_later": "बाद में याद दिलाएँ",
        "whats_new": "नया क्या है",
        "no_changelog": "कोई चेंजलॉग प्रदान नहीं किया गया।",
        "critical_update": "महत्वपूर्ण अद्यतन",
        "update_ready": "इंस्टॉल के लिए तैयार",
        "app_will_restart": "अद्यतन लागू करने के लिए ऐप बंद होकर पुनः आरंभ होगा।",
        "install_failed": "इंस्टॉलेशन विफल। आपका पुराना संस्करण सुरक्षित है।",
        "download_failed": "डाउनलोड विफल।",
        "source_update_requires_zip": "स्रोत अद्यतन के लिए .zip फ़ाइल आवश्यक है।",
        "session_locked": "सत्र लॉक",
        "locked_for_user": "{user} के लिए लॉक",
        "unlock": "अनलॉक करें",
        "session_unlocked": "सत्र अनलॉक हो गया",
        "auto_lock_notice": "2 मिनट निष्क्रियता के बाद सत्र लॉक हो जाएगा।",
        "audit_log_title": "ऑडिट लॉग",
        "audit_export": "निर्यात करें",
        "audit_exported": "सफलतापूर्वक निर्यात किया गया",
        "audit_export_failed": "निर्यात विफल",
        "audit_search_placeholder": "क्रियाएँ या विवरण खोजें...",
        "audit_col_time": "समय",
        "audit_col_level": "स्तर",
        "audit_col_category": "श्रेणी",
        "audit_col_action": "क्रिया",
        "audit_col_user": "उपयोगकर्ता",
        "audit_details_title": "इवेंट विवरण",
        "audit_status": "{total} में से {count} इवेंट दिखाए जा रहे हैं",
        "strength_very_weak": "बहुत कमज़ोर",
        "strength_weak": "कमज़ोर",
        "strength_medium": "मध्यम",
        "strength_strong": "मज़बूत",
        "strength_very_strong": "बहुत मज़बूत",
        "strength_rule_length": "8-16 वर्ण",
        "strength_rule_lowercase": "छोटा अक्षर",
        "strength_rule_uppercase": "बड़ा अक्षर",
        "strength_rule_digits": "अंक",
        "strength_rule_special": "विशेष वर्ण",
        "strength_add_chars": "{n} और वर्ण जोड़ें",
        "strength_remove_chars": "{n} वर्ण हटाएं",
        "strength_need_digits": "{n} और अंक जोड़ें",
        "strength_need_special": "{n} और विशेष वर्ण जोड़ें",
        "strength_common_pattern": "सामान्य शब्दों से बचें",
        "strength_common_year": "सामान्य वर्षों से बचें",
        "strength_has_sequence": "क्रम (1234, abcd) से बचें",
        "strength_has_repeats": "दोहराव (aaaa, 1111) से बचें",
        "passwords_match": "पासवर्ड मेल खाते हैं",
        "2fa_setup_title": "दो-कारक प्रमाणीकरण सक्षम करें",
        "2fa_setup_subtitle": "Google Authenticator, Authy या Microsoft Authenticator से QR कोड स्कैन करें।",
        "2fa_manual_secret": "या इस कोड को मैन्युअल रूप से दर्ज करें:",
        "2fa_enter_code": "अपने ऐप से 6 अंकों का कोड दर्ज करें:",
        "2fa_verify_btn": "सत्यापित करें",
        "2fa_backup_codes_title": "इन बैकअप कोड को सहेजें!",
        "2fa_backup_codes_hint": "प्रत्येक कोड केवल एक बार उपयोग किया जा सकता है। उन्हें सुरक्षित स्थान पर रखें।",
        "2fa_done_btn": "पूर्ण",
        "2fa_secret_copied": "गुप्त कोड क्लिपबोर्ड पर कॉपी किया गया",
        "2fa_invalid_code": "अमान्य कोड। कृपया पुनः प्रयास करें।",
        "2fa_invalid_code_format": "कोड 6 अंकों का होना चाहिए।",
        "2fa_verify_title": "दो-कारक प्रमाणीकरण",
        "2fa_verify_subtitle": "अपने प्रमाणक ऐप से 6 अंकों का कोड दर्ज करें।",
        "2fa_enter_code_hint": "कृपया एक कोड दर्ज करें।",
        "2fa_use_backup_code": "बैकअप कोड का उपयोग करें",
        "2fa_use_totp_code": "प्रमाणक कोड का उपयोग करें",
        "2fa_enter_backup_code": "अपने बैकअप कोड में से एक दर्ज करें।",
        "2fa_section_title": "सुरक्षा",
        "2fa_enabled_status": "दो-कारक प्रमाणीकरण सक्षम है",
        "2fa_disabled_status": "दो-कारक प्रमाणीकरण अक्षम है",
        "2fa_enable_btn": "2FA सक्षम करें",
        "2fa_disable_btn": "2FA अक्षम करें",
        "2fa_disable_title": "2FA अक्षम करें",
        "2fa_disable_confirm": "क्या आप सुनिश्चित हैं? आपका खाता कम सुरक्षित हो जाएगा।",
        "2fa_disabled_success": "दो-कारक प्रमाणीकरण अक्षम किया गया",
        "2fa_enabled_success": "दो-कारक प्रमाणीकरण सक्षम किया गया",
        "2fa_hint_title": "इस QR कोड का उपयोग कैसे करें",
        "2fa_hint_step1": "एक प्रमाणक ऐप इंस्टॉल करें: Google Authenticator, Microsoft Authenticator या Authy।",
        "2fa_hint_step2": "ऐप खोलें, \"+\" बटन पर टैप करें, फिर \"QR कोड स्कैन करें\" चुनें।",
        "2fa_hint_step3": "अपने फ़ोन का कैमरा ऊपर दिए गए QR कोड की ओर करें (ऐप के कैमरा का उपयोग करें, अपने फ़ोन के कैमरा ऐप का नहीं)।",
        "2fa_hint_step4": "ऐप 6 अंकों का कोड दिखाएगा। इसे नीचे दिए गए बॉक्स में दर्ज करें।",
        "2fa_manual_secret_hint": "स्कैन नहीं कर सकते? नीचे दी गई कुंजी को कॉपी करें और अपने ऐप में मैन्युअल रूप से जोड़ें (\"सेटअप कुंजी दर्ज करें\")।",
        "auto_lock_section_title": "स्वतः लॉक",
        "auto_lock_enable": "निष्क्रियता के बाद लॉक करें",
        "auto_lock_timeout_label": "लॉक समय:",
        "auto_lock_warning_title": "स्वतः लॉक चेतावनी",
        "auto_lock_warning_body": "निष्क्रियता के कारण आपका सत्र {seconds} सेकंड में लॉक हो जाएगा।",
        "lock_now": "अभी लॉक करें",
        "font_section_title": "फ़ॉन्ट",
        "font_family_label": "फ़ॉन्ट परिवार",
        "font_size_label": "आधार आकार",
        "font_scale_label": "प्रदर्शन स्केल",
        "font_scale_small": "छोटा",
        "font_scale_normal": "सामान्य",
        "font_scale_large": "बड़ा",
        "font_scale_xlarge": "बहुत बड़ा",
        "font_scale_xxlarge": "विशाल",
        "font_preview_text": "यह फ़ॉन्ट पूर्वावलोकन के लिए एक नमूना पाठ है — 1234567890",
        "font_reset": "डिफ़ॉल्ट पर रीसेट करें",
        "font_applied": "फ़ॉन्ट अपडेट किया गया",
        "motion_section_title": "मोशन और एनिमेशन",
        "reduce_motion_label": "मोशन कम करें",
        "reduce_motion_hint": "यदि एनिमेशन आपको असहज करते हैं तो उन्हें अक्षम करें।",
        "empty_history_empty_title": "अभी तक कोई इतिहास नहीं",
        "empty_history_empty_subtitle": "अपनी पहली छवि संसाधित करें और यहाँ देखें",
        "empty_history_empty_hint": "प्रत्येक परिणाम स्वतः सहेजा जाता है",
        "empty_history_no_user_title": "इतिहास देखने के लिए साइन इन करें",
        "empty_history_no_user_subtitle": "आपके सहेजे गए परिणाम यहाँ दिखाई देंगे",
        "empty_history_no_user_hint": "इतिहास प्रत्येक खाते के लिए निजी है",
        "empty_dash_no_results_title": "अभी तक कोई परिणाम नहीं",
        "empty_dash_no_results_subtitle": "आपके हाल के विश्लेषण यहाँ दिखाई देंगे",
        "empty_dash_no_results_hint": "शुरू करने के लिए एक छवि संसाधित करें",
        "empty_dash_no_dist_title": "कोई वितरण डेटा नहीं",
        "empty_dash_no_dist_subtitle": "छवियाँ संसाधित करते ही आँकड़े भरेंगे",
        "empty_dash_no_dist_hint": "हर विश्लेषण मायने रखता है",
        "empty_image_empty_title": "कोई छवि नहीं डाली गई",
        "empty_image_empty_subtitle": "छवि यहाँ खींचकर छोड़ें, या डालें पर क्लिक करें",
        "empty_image_empty_hint": "PNG, JPG, JPEG, BMP, DCM समर्थित",
        "empty_image_drop_hint_title": "विश्लेषण के लिए छोड़ें",
        "empty_image_drop_hint_subtitle": "लोड करने के लिए छवि छोड़ें",
        "empty_image_drop_hint_hint": "प्रारूप स्वतः पहचाना जाएगा",
        "empty_processing_wait_title": "प्रक्रिया हो रही है...",
        "empty_processing_wait_subtitle": "AI आपकी छवि का विश्लेषण कर रहा है",
        "empty_processing_wait_hint": "आमतौर पर कुछ सेकंड लगते हैं",
        "tour_show_again": "टूर फिर से दिखाएं",
        "tour_completed_msg": "टूर पूर्ण! आप तैयार हैं।",
        "tour_skipped_msg": "टूर छोड़ दिया गया। आप इसे खाता मेनू से फिर से देख सकते हैं।",
        "tour_btn_next": "अगला",
        "tour_btn_prev": "पिछला",
        "tour_btn_skip": "छोड़ें",
        "tour_btn_finish": "समाप्त ✓",
        "tour_guest_login_title": "साइन इन",
        "tour_guest_login_subtitle": "साइन इन करने और सभी सुविधाओं को अनलॉक करने के लिए यहाँ क्लिक करें।",
        "tour_guest_signup_title": "खाता बनाएँ",
        "tour_guest_signup_subtitle": "यहाँ नए हैं? कुछ सेकंड में खाता बनाने के लिए यहाँ क्लिक करें।",
        "tour_insert_title": "छवि डालें",
        "tour_insert_subtitle": "डिस्क से सोनो छवि लोड करें, या इसे विंडो में कहीं भी खींचकर छोड़ें।",
        "tour_process_title": "छवि संसाधित करें",
        "tour_process_subtitle": "लोड की गई छवि पर AI विश्लेषण चलाएं। परिणाम कुछ सेकंड में दिखाई देंगे।",
        "tour_export_title": "PDF निर्यात करें",
        "tour_export_subtitle": "वर्तमान परिणाम को पेशेवर PDF चिकित्सा रिपोर्ट के रूप में सहेजें।",
        "tour_gradcam_title": "अटेंशन हीटमैप",
        "tour_gradcam_subtitle": "देखें AI कहाँ देख रहा है — विश्लेषण की गई छवि पर हीटमैप।",
        "tour_dashboard_title": "डैशबोर्ड",
        "tour_dashboard_subtitle": "आँकड़े पैनल दिखाएं/छिपाएं: कुल छवियाँ, औसत विश्वास, ग्रेड वितरण।",
        "tour_history_title": "इतिहास",
        "tour_history_subtitle": "अपनी अंतिम 20 विश्लेषित छवियाँ ब्राउज़ करें। परिणाम फिर से लोड करने के लिए किसी थंबनेल पर क्लिक करें।",
        "tour_visit_title": "वेबपेज देखें",
        "tour_visit_subtitle": "दस्तावेज़ और अपडेट के लिए ब्राउज़र में प्रोजेक्ट का GitHub पेज खोलें।",
        "tour_mute_title": "संगीत म्यूट करें",
        "tour_mute_subtitle": "किसी भी समय पृष्ठभूमि संगीत रोकें या फिर से शुरू करें।",
        "tour_zoom_title": "ज़ूम नियंत्रण",
        "tour_zoom_subtitle": "ज़ूम के लिए माउस व्हील, पैन के लिए ड्रैग, रीसेट के लिए डबल-क्लिक करें।",
        "tour_account_title": "खाता मेनू",
        "tour_account_subtitle": "सेटिंग्स, खाता जानकारी, ऑडिट लॉग और अधिक तक पहुँचने के लिए यहाँ क्लिक करें।",
        "suggestion_image_ready": "छवि लोड हो गई — प्रक्रिया के लिए तैयार",
        "suggestion_image_ready_sub": "AI विश्लेषण चलाने के लिए प्रक्रिया पर क्लिक करें।",
        "suggestion_pdf_ready": "परिणाम तैयार है — PDF में निर्यात करें?",
        "suggestion_pdf_ready_sub": "एक क्लिक में पेशेवर चिकित्सा रिपोर्ट सहेजें।",
        "suggestion_gradcam_hint": "जानना चाहते हैं कि AI ने क्या देखा?",
        "suggestion_gradcam_hint_sub": "छवि पर अटेंशन हीटमैप देखें।",
        "suggestion_repeated_grid": "आपको हमेशा {grid} मिलता है",
        "suggestion_repeated_grid_sub": "वितरण देखने के लिए डैशबोर्ड खोलें।",
        "suggestion_history_hint": "आपका इतिहास बढ़ रहा है",
        "suggestion_history_hint_sub": "अपनी अंतिम 20 विश्लेषित छवियाँ ब्राउज़ करें।",
        "suggestion_dashboard_hint": "आपके डैशबोर्ड में रोचक आँकड़े हैं",
        "suggestion_dashboard_hint_sub": "कुल, औसत और वितरण एक नज़र में देखें।",
        "suggestion_settings_hint": "ऐप को व्यक्तिगत बनाएं",
        "suggestion_settings_hint_sub": "भाषा, थीम, फ़ॉन्ट बदलें और 2FA सक्षम करें।",
        "suggestion_idle_help": "त्वरित मार्गदर्शिका चाहिए?",
        "suggestion_idle_help_sub": "किसी भी समय ऑनबोर्डिंग टूर फिर से चलाएँ।",
        "image_not_liver_warning": "यह छवि लिवर अल्ट्रासाउंड जैसी नहीं दिखती।\n\nकृपया उचित लिवर अल्ट्रासाउंड छवि डालें।",
        "result_not_confident": "AI इस छवि को लिवर अल्ट्रासाउंड के रूप में आत्मविश्वास से पहचान नहीं सका।\n\nकृपया उचित लिवर अल्ट्रासाउंड छवि डालें और पुनः प्रयास करें।",
        "not_liver_title": "लिवर अल्ट्रासाउंड नहीं है",
        "not_liver_hint": "कृपया उचित लिवर अल्ट्रासाउंड (सोनोग्राम) छवि का उपयोग करें",
        "voice_welcome": "स्वागत है, {user}",
        "voice_goodbye": "अलविदा",
        "voice_logged_out": "सफलतापूर्वक लॉगआउट हो गया",
        "voice_image_loaded": "छवि सफलतापूर्वक लोड हो गई",
        "voice_image_rejected": "यह लिवर अल्ट्रासाउंड छवि नहीं है",
        "voice_inference_result": "{grid_name}, विश्वास {conf} प्रतिशत",
        "voice_pdf_saved": "PDF रिपोर्ट सफलतापूर्वक सहेजी गई",
        "voice_error": "एक त्रुटि हुई",
        "voice_settings_applied": "सेटिंग्स लागू हो गईं",
        "voice_language_changed": "भाषा {lang} में बदल गई",
        "voice_theme_changed": "थीम बदल गई",
        "voice_2fa_enabled": "दो-कारक प्रमाणीकरण सक्षम हो गया",
        "voice_2fa_disabled": "दो-कारक प्रमाणीकरण अक्षम हो गया",
        "voice_locked": "सत्र लॉक हो गया",
        "voice_unlocked": "सत्र अनलॉक हो गया",
        "voice_processing": "छवि संसाधित हो रही है",
        "voice_processing_done": "प्रसंस्करण पूरा हुआ",
        "voice_voice_enabled": "ध्वनि फ़ीडबैक सक्षम हो गया",
        "voice_voice_disabled": "ध्वनि फ़ीडबैक अक्षम हो गया",
    },
    "Portugal": {
        "app_title": "Fígado AI-babak",
        "welcome": "Bem-vindo",
        "login": "Entrar",
        "signup": "Cadastrar",
        "logout": "Sair",
        "exit": "Sair",
        "settings": "Configurações",
        "account_info": "Informações da conta",
        "username": "Nome de usuário",
        "password": "Senha",
        "repeat_password": "Repetir senha",
        "first_name": "Nome",
        "last_name": "Sobrenome",
        "email": "Endereço de e-mail",
        "remember_me": "Lembrar de mim",
        "login_btn": "Entrar",
        "signup_btn": "Cadastrar",
        "account": "Conta",
        "change_language": "Alterar idioma",
        "change_theme": "Alterar tema",
        "insert_sono": "Inserir Sono",
        "process": "Processar",
        "mute": "Silenciar",
        "unmute": "Ativar som",
        "visit_webpage": "Visitar página",
        "please_wait": "Aguarde...",
        "processing": "Processando imagem...",
        "result": "Resultado",
        "confidence": "Confiança",
        "grid": "Grade",
        "insert_image_first": "Por favor, insira uma imagem primeiro!",
        "no_image": "Nenhuma imagem inserida",
        "invalid_credentials": "Credenciais inválidas",
        "user_exists": "Nome de usuário já existe",
        "password_requirements": "Requisitos de senha",
        "password_rule": (
            "A senha deve:\n"
            "• Ter 8-16 caracteres\n"
            "• Conter pelo menos 1 letra minúscula\n"
            "• Conter pelo menos 1 letra maiúscula\n"
            "• Conter pelo menos 4 dígitos\n"
            "• Conter pelo menos 4 caracteres especiais"
        ),
        "passwords_dont_match": "As senhas não coincidem!",
        "invalid_email": "Por favor, insira um endereço de e-mail válido",
        "empty_fields": "Por favor, preencha todos os campos",
        "signup_success": "Cadastro bem-sucedido! Você já pode entrar.",
        "logout_confirm": "Tem certeza de que deseja excluir sua conta?",
        "exit_confirm": "Tem certeza de que deseja sair?",
        "theme_dark_aqua": "Tema Escuro Aqua-Roxo",
        "theme_solarized_white": "Tema Solarized Claro",
        "theme_solarized_dark": "Tema Solarized Escuro",
        "theme_dark_forest": "Tema Floresta Verde Escuro",
        "select_language": "Selecionar idioma",
        "select_theme": "Selecionar tema",
        "apply": "Aplicar",
        "cancel": "Cancelar",
        "save": "Salvar",
        "success": "Sucesso",
        "error": "Erro",
        "warning": "Aviso",
        "settings_applied": "Configurações aplicadas!",
        "grid_0": "Grade 0 (NL) — Fígado normal",
        "grid_1": "Grade 1 (G1) — Grau 1",
        "grid_2": "Grade 2 (G2) — Grau 2",
        "grid_3": "Grade 3 (G3) — Grau 3",
        "too_many_attempts": "Muitas tentativas falhas. Aguarde 5 minutos.",
        "invalid_username": "O nome de usuário deve ter 3-32 caracteres (letras, números, sublinhado).",
        "welcome_user": "Bem-vindo, {user}!",
        "inference_failed": "Falha na inferência",
        "drop_here": "Solte a imagem aqui",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Tipo de arquivo não suportado. Solte uma imagem.",
        "image_loaded": "Imagem carregada com sucesso",
        "help_about_title": "Ajuda & Sobre",
        "help_about_text": (
            "Pipeline IA do Fígado v3.0\n"
            "Desenvolvido por Babak Yousefian\n\n"
            "Atalhos de teclado:\n"
            "  Ctrl+O      — Inserir imagem\n"
            "  Ctrl+Enter  — Processar imagem\n"
            "  Ctrl+L      — Entrar\n"
            "  Ctrl+,      — Configurações\n"
            "  Esc         — Fechar diálogo ativo\n"
            "  F1          — Ajuda / Sobre\n\n"
            "Dica: Você também pode arrastar e soltar uma imagem em qualquer lugar da janela."
        ),
        "dashboard_title": "Painel",
        "dash_total": "Total de imagens",
        "dash_avg_conf": "Confiança média",
        "dash_distribution": "Distribuição",
        "dash_last_5": "Últimos 5 resultados",
        "dash_no_results": "Ainda sem resultados",
        "dash_clear_confirm": "Limpar todas as estatísticas?",
        "dash_cleared": "Estatísticas limpas",
        "dashboard_toggle": "Mostrar/Ocultar painel",
        "export_pdf": "Exportar PDF",
        "pdf_saved": "PDF salvo com sucesso",
        "pdf_failed": "Falha na geração do PDF",
        "pdf_subtitle": "Relatório de Análise",
        "pdf_patient": "Paciente:",
        "pdf_date": "Data:",
        "pdf_model": "Modelo:",
        "pdf_diagnosis": "DIAGNÓSTICO",
        "pdf_confidence": "Confiança",
        "pdf_class_probs": "PROBABILIDADES DAS CLASSES",
        "pdf_disclaimer": (
            "Este relatório foi desenvolvido por Babak Yousefian sob a supervisão de médicos experientes. "
            "É uma ferramenta de análise profissional e não requer encaminhamento a um médico."
        ),
        "history_title": "Histórico",
        "history_toggle": "Mostrar/Ocultar histórico",
        "history_empty": "Ainda sem histórico",
        "history_cleared": "Histórico limpo",
        "history_clear_confirm": "Limpar todo o histórico?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Entrar",
        "tooltip_desc_login": "Entre na sua conta",
        "tooltip_tip_login": "Acesse seu histórico e estatísticas salvos",
        "tooltip_title_signup": "Cadastrar",
        "tooltip_desc_signup": "Criar uma nova conta",
        "tooltip_title_account": "Conta",
        "tooltip_desc_account": "Clique para abrir o menu da conta",
        "tooltip_desc_account_user": "Conectado como {user}",
        "tooltip_title_dashboard": "Painel",
        "tooltip_desc_dashboard": "Mostrar/ocultar painel de estatísticas — total, distribuição, resultados recentes",
        "tooltip_title_history": "Histórico",
        "tooltip_desc_history": "Mostrar/ocultar os últimos 20 resultados com miniaturas",
        "tooltip_title_visit": "Visitar página",
        "tooltip_desc_visit": "Abrir a página do GitHub do projeto no navegador",
        "tooltip_tip_visit": "Abre no navegador externo",
        "tooltip_title_insert": "Inserir imagem",
        "tooltip_desc_insert": "Carregar uma imagem de sono para analisar",
        "tooltip_tip_insert": "Você também pode arrastar e soltar uma imagem em qualquer lugar da janela",
        "tooltip_title_process": "Processar",
        "tooltip_desc_process": "Executar análise de IA na imagem carregada",
        "tooltip_tip_process": "Requer uma imagem carregada primeiro",
        "tooltip_title_mute": "Silenciar música",
        "tooltip_desc_mute": "Pausar a música de fundo",
        "tooltip_title_unmute": "Ativar música",
        "tooltip_desc_unmute": "Retomar a música de fundo",
        "tooltip_title_export": "Exportar PDF",
        "tooltip_desc_export": "Salvar o resultado atual como relatório PDF",
        "tooltip_tip_export": "Inclui imagem, diagnóstico e probabilidades por classe",
        "tooltip_title_dash_clear": "Limpar estatísticas",
        "tooltip_desc_dash_clear": "Redefinir todas as estatísticas desta conta",
        "tooltip_tip_dash_clear": "Esta ação não pode ser desfeita",
        "tooltip_title_dash_close": "Fechar painel",
        "tooltip_desc_dash_close": "Ocultar o painel de estatísticas",
        "tooltip_desc_dash_total": "Número total de imagens processadas",
        "tooltip_desc_dash_avg": "Confiança média em todos os resultados",
        "tooltip_desc_dash_dist": "Distribuição dos resultados por grau",
        "tooltip_title_hist_clear": "Limpar histórico",
        "tooltip_desc_hist_clear": "Excluir todos os resultados e miniaturas salvos",
        "tooltip_tip_hist_clear": "Isso também excluirá todos os arquivos de imagem",
        "tooltip_title_hist_close": "Fechar histórico",
        "tooltip_desc_hist_close": "Ocultar o painel de histórico",
        "tooltip_tip_thumb_click": "Clique para recarregar este resultado",
        "tooltip_tip_recent": "Clique para recarregar este resultado",
        "tooltip_title_zoom_in": "Aumentar zoom",
        "tooltip_desc_zoom_in": "Aumentar ampliação da imagem",
        "tooltip_title_zoom_out": "Diminuir zoom",
        "tooltip_desc_zoom_out": "Reduzir ampliação da imagem",
        "tooltip_title_zoom_reset": "Redefinir visualização",
        "tooltip_desc_zoom_reset": "Ajustar imagem à janela",
        "tooltip_tip_zoom_reset": "Você também pode clicar duas vezes na imagem",
        "notif_inference_ok": "Inferência concluída",
        "follow_system_theme": "Seguir tema do sistema",
        "follow_system_theme_desc": "Alternar automaticamente entre claro e escuro com base nas configurações do sistema",
        "gradcam_title": "Mapa de calor de atenção",
        "gradcam_show": "Mostrar mapa de atenção",
        "gradcam_toggle": "Alternar mapa",
        "gradcam_save": "Salvar imagem",
        "gradcam_close": "Fechar",
        "gradcam_saved": "Imagem salva com sucesso",
        "gradcam_no_heatmap": "Nenhum mapa de atenção disponível. Por favor, processe uma imagem primeiro.",
        "update_available": "Atualização disponível",
        "update_check_failed": "Falha ao verificar atualizações",
        "check_for_updates": "Verificar atualizações",
        "check_for_updates_desc": "Verificar GitHub/website para a versão mais recente",
        "up_to_date": "Está atualizado",
        "you_have_latest": "Está a executar a versão mais recente.",
        "check_internet": "Verifique a sua ligação à Internet e tente novamente.",
        "downloading": "A transferir...",
        "verifying": "A verificar...",
        "download_now": "Transferir agora",
        "install_now": "Instalar e reiniciar",
        "skip_version": "Ignorar esta versão",
        "remind_later": "Lembrar mais tarde",
        "whats_new": "Novidades",
        "no_changelog": "Nenhum registo de alterações fornecido.",
        "critical_update": "Atualização crítica",
        "update_ready": "Pronto para instalar",
        "app_will_restart": "A aplicação será fechada e reiniciada para aplicar a atualização.",
        "install_failed": "Instalação falhou. A sua versão antiga está segura.",
        "download_failed": "Transferência falhou.",
        "source_update_requires_zip": "A atualização de código-fonte requer um ficheiro .zip.",
        "session_locked": "Sessão Bloqueada",
        "locked_for_user": "Bloqueada para {user}",
        "unlock": "Desbloquear",
        "session_unlocked": "Sessão desbloqueada",
        "auto_lock_notice": "A sessão será bloqueada após 2 minutos de inatividade.",
        "audit_log_title": "Registo de Auditoria",
        "audit_export": "Exportar",
        "audit_exported": "Exportado com sucesso",
        "audit_export_failed": "Falha na exportação",
        "audit_search_placeholder": "Pesquisar ações ou detalhes...",
        "audit_col_time": "Hora",
        "audit_col_level": "Nível",
        "audit_col_category": "Categoria",
        "audit_col_action": "Ação",
        "audit_col_user": "Utilizador",
        "audit_details_title": "Detalhes do Evento",
        "audit_status": "A mostrar {count} de {total} eventos",
        "strength_very_weak": "Muito fraca",
        "strength_weak": "Fraca",
        "strength_medium": "Média",
        "strength_strong": "Forte",
        "strength_very_strong": "Muito forte",
        "strength_rule_length": "8-16 caracteres",
        "strength_rule_lowercase": "Letra minúscula",
        "strength_rule_uppercase": "Letra maiúscula",
        "strength_rule_digits": "Dígitos",
        "strength_rule_special": "Caracteres especiais",
        "strength_add_chars": "Adicione mais {n} caractere(s)",
        "strength_remove_chars": "Remova {n} caractere(s)",
        "strength_need_digits": "Adicione mais {n} dígito(s)",
        "strength_need_special": "Adicione mais {n} caractere(s) especial(ais)",
        "strength_common_pattern": "Evite palavras comuns",
        "strength_common_year": "Evite anos comuns",
        "strength_has_sequence": "Evite sequências (1234, abcd)",
        "strength_has_repeats": "Evite repetições (aaaa, 1111)",
        "passwords_match": "As palavras-passe coincidem",
        "2fa_setup_title": "Ativar autenticação de dois fatores",
        "2fa_setup_subtitle": "Digitalize o código QR com Google Authenticator, Authy ou Microsoft Authenticator.",
        "2fa_manual_secret": "Ou introduza este código manualmente:",
        "2fa_enter_code": "Introduza o código de 6 dígitos da sua aplicação:",
        "2fa_verify_btn": "Verificar",
        "2fa_backup_codes_title": "Guarde estes códigos de recuperação!",
        "2fa_backup_codes_hint": "Cada código só pode ser usado uma vez. Guarde-os num local seguro.",
        "2fa_done_btn": "Concluído",
        "2fa_secret_copied": "Código secreto copiado para a área de transferência",
        "2fa_invalid_code": "Código inválido. Tente novamente.",
        "2fa_invalid_code_format": "O código deve ter 6 dígitos.",
        "2fa_verify_title": "Autenticação de dois fatores",
        "2fa_verify_subtitle": "Introduza o código de 6 dígitos da sua aplicação de autenticação.",
        "2fa_enter_code_hint": "Por favor, introduza um código.",
        "2fa_use_backup_code": "Usar um código de recuperação",
        "2fa_use_totp_code": "Usar código da aplicação",
        "2fa_enter_backup_code": "Introduza um dos seus códigos de recuperação.",
        "2fa_section_title": "Segurança",
        "2fa_enabled_status": "A autenticação de dois fatores está ativada",
        "2fa_disabled_status": "A autenticação de dois fatores está desativada",
        "2fa_enable_btn": "Ativar 2FA",
        "2fa_disable_btn": "Desativar 2FA",
        "2fa_disable_title": "Desativar 2FA",
        "2fa_disable_confirm": "Tem a certeza? A sua conta ficará menos segura.",
        "2fa_disabled_success": "Autenticação de dois fatores desativada",
        "2fa_enabled_success": "Autenticação de dois fatores ativada",
        "2fa_hint_title": "Como usar este código QR",
        "2fa_hint_step1": "Instale uma aplicação de autenticação: Google Authenticator, Microsoft Authenticator ou Authy.",
        "2fa_hint_step2": "Abra a aplicação, toque no botão «+» e escolha «Digitalizar código QR».",
        "2fa_hint_step3": "Aponte a câmara do seu telemóvel para o código QR acima (use a câmara da aplicação, não a aplicação de câmara do telemóvel).",
        "2fa_hint_step4": "A aplicação mostrará um código de 6 dígitos. Introduza-o na caixa abaixo.",
        "2fa_manual_secret_hint": "Não consegue digitalizar? Copie a chave abaixo e adicione-a manualmente na aplicação («Introduzir chave de configuração»).",
        "auto_lock_section_title": "Bloqueio automático",
        "auto_lock_enable": "Bloquear após inatividade",
        "auto_lock_timeout_label": "Bloquear após:",
        "auto_lock_warning_title": "Aviso de bloqueio automático",
        "auto_lock_warning_body": "A sua sessão será bloqueada em {seconds} segundos por inatividade.",
        "lock_now": "Bloquear agora",
        "font_section_title": "Tipo de letra",
        "font_family_label": "Família de letras",
        "font_size_label": "Tamanho base",
        "font_scale_label": "Escala de visualização",
        "font_scale_small": "Pequeno",
        "font_scale_normal": "Normal",
        "font_scale_large": "Grande",
        "font_scale_xlarge": "Muito grande",
        "font_scale_xxlarge": "Enorme",
        "font_preview_text": "Luís argüia à Júlia que «brações, fé, chá, óxido, pôr, zângão» eram palavras do português — 1234567890",
        "font_reset": "Repor predefinição",
        "font_applied": "Tipo de letra atualizado",
        "motion_section_title": "Movimento e animações",
        "reduce_motion_label": "Reduzir movimento",
        "reduce_motion_hint": "Desative as animações se o deixarem desconfortável.",
        "empty_history_empty_title": "Ainda sem histórico",
        "empty_history_empty_subtitle": "Processe a sua primeira imagem para a ver aqui",
        "empty_history_empty_hint": "Cada resultado é guardado automaticamente",
        "empty_history_no_user_title": "Inicie sessão para ver o histórico",
        "empty_history_no_user_subtitle": "Os seus resultados guardados aparecerão aqui",
        "empty_history_no_user_hint": "O histórico é privado para cada conta",
        "empty_dash_no_results_title": "Ainda sem resultados",
        "empty_dash_no_results_subtitle": "As suas análises recentes aparecerão aqui",
        "empty_dash_no_results_hint": "Processe uma imagem para começar",
        "empty_dash_no_dist_title": "Sem dados de distribuição",
        "empty_dash_no_dist_subtitle": "As estatísticas preencher-se-ão à medida",
        "empty_dash_no_dist_hint": "Cada análise conta",
        "empty_image_empty_title": "Nenhuma imagem inserida",
        "empty_image_empty_subtitle": "Arraste e solte uma imagem aqui, ou clique Inserir",
        "empty_image_empty_hint": "Suporta PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Solte para analisar",
        "empty_image_drop_hint_subtitle": "Liberte a imagem para a carregar",
        "empty_image_drop_hint_hint": "O formato será detetado automaticamente",
        "empty_processing_wait_title": "A processar...",
        "empty_processing_wait_subtitle": "A IA está a analisar a sua imagem",
        "empty_processing_wait_hint": "Normalmente demora alguns segundos",
        "tour_show_again": "Mostrar tour novamente",
        "tour_completed_msg": "Tour concluído! Está pronto.",
        "tour_skipped_msg": "Tour ignorado. Pode repeti-lo no menu Conta.",
        "tour_btn_next": "Seguinte",
        "tour_btn_prev": "Anterior",
        "tour_btn_skip": "Ignorar",
        "tour_btn_finish": "Concluir ✓",
        "tour_guest_login_title": "Entrar",
        "tour_guest_login_subtitle": "Clique aqui para entrar na sua conta e desbloquear todas as funcionalidades.",
        "tour_guest_signup_title": "Criar uma conta",
        "tour_guest_signup_subtitle": "Novo por aqui? Clique aqui para criar uma conta em segundos.",
        "tour_insert_title": "Inserir imagem",
        "tour_insert_subtitle": "Carregue uma imagem de eco do disco, ou arraste e solte-a em qualquer lugar da janela.",
        "tour_process_title": "Processar imagem",
        "tour_process_subtitle": "Execute a análise de IA na imagem carregada. Os resultados aparecem em poucos segundos.",
        "tour_export_title": "Exportar PDF",
        "tour_export_subtitle": "Guarde o resultado atual como um relatório médico PDF profissional.",
        "tour_gradcam_title": "Mapa de calor de atenção",
        "tour_gradcam_subtitle": "Veja onde a IA está a olhar — um mapa de calor sobre a imagem analisada.",
        "tour_dashboard_title": "Painel",
        "tour_dashboard_subtitle": "Mostrar/ocultar painel de estatísticas: total de imagens, confiança média, distribuição por grau.",
        "tour_history_title": "Histórico",
        "tour_history_subtitle": "Navegue pelas últimas 20 imagens analisadas. Clique numa miniatura para recarregar o resultado.",
        "tour_visit_title": "Visitar página",
        "tour_visit_subtitle": "Abra a página GitHub do projeto no navegador para documentação e atualizações.",
        "tour_mute_title": "Silenciar música",
        "tour_mute_subtitle": "Pause ou retome a música de fundo a qualquer momento.",
        "tour_zoom_title": "Controlos de zoom",
        "tour_zoom_subtitle": "Use a roda do rato para zoom, arraste para mover, e duplo-clique para reiniciar.",
        "tour_account_title": "Menu da conta",
        "tour_account_subtitle": "Clique aqui para aceder a Configurações, Informações da conta, Registo de auditoria e mais.",
        "suggestion_image_ready":      "Imagem carregada — pronta para processar",
        "suggestion_image_ready_sub":  "Clique em Processar para executar a análise de IA.",
        "suggestion_pdf_ready":        "Resultado pronto — exportar como PDF?",
        "suggestion_pdf_ready_sub":    "Guarde um relatório médico profissional com um clique.",
        "suggestion_gradcam_hint":     "Curioso para saber o que a IA viu?",
        "suggestion_gradcam_hint_sub": "Veja o mapa de calor de atenção sobre a imagem.",
        "suggestion_repeated_grid":    "Está sempre a obter {grid}",
        "suggestion_repeated_grid_sub":"Abra o painel para ver a distribuição.",
        "suggestion_history_hint":     "O seu histórico está a crescer",
        "suggestion_history_hint_sub": "Navegue pelas suas últimas 20 imagens analisadas.",
        "suggestion_dashboard_hint":   "O seu painel tem estatísticas interessantes",
        "suggestion_dashboard_hint_sub":"Totais, médias e distribuição num relance.",
        "suggestion_settings_hint":    "Personalize a aplicação",
        "suggestion_settings_hint_sub":"Altere idioma, tema, tipo de letra e ative 2FA.",
        "suggestion_idle_help":        "Precisa de um guia rápido?",
        "suggestion_idle_help_sub":    "Repita o tour de integração a qualquer momento.",
        "image_not_liver_warning": "Esta imagem não parece uma ecografia do fígado.\n\nPor favor, insira uma imagem de ecografia do fígado adequada.",
        "result_not_confident": "A IA não conseguiu identificar com confiança esta imagem como uma ecografia do fígado.\n\nPor favor, insira uma imagem de ecografia do fígado adequada e tente novamente.",
        "not_liver_title": "Não é uma ecografia do fígado",
        "not_liver_hint": "Por favor, use uma imagem de ecografia (sonograma) do fígado adequada",
        "voice_welcome": "Bem-vindo, {user}",
        "voice_goodbye": "Adeus",
        "voice_logged_out": "Sessão terminada com sucesso",
        "voice_image_loaded": "Imagem carregada com sucesso",
        "voice_image_rejected": "Isto não é uma ecografia do fígado",
        "voice_inference_result": "{grid_name}, confiança {conf} por cento",
        "voice_pdf_saved": "Relatório PDF guardado com sucesso",
        "voice_error": "Ocorreu um erro",
        "voice_settings_applied": "Definições aplicadas",
        "voice_language_changed": "Idioma alterado para {lang}",
        "voice_theme_changed": "Tema alterado",
        "voice_2fa_enabled": "Autenticação de dois fatores ativada",
        "voice_2fa_disabled": "Autenticação de dois fatores desativada",
        "voice_locked": "Sessão bloqueada",
        "voice_unlocked": "Sessão desbloqueada",
        "voice_processing": "A processar imagem",
        "voice_processing_done": "Processamento concluído",
        "voice_voice_enabled": "Feedback de voz ativado",
        "voice_voice_disabled": "Feedback de voz desativado",
    },
    "Brazil": {
        "app_title": "Fígado AI-babak",
        "welcome": "Bem-vindo",
        "login": "Entrar",
        "signup": "Cadastrar",
        "logout": "Sair",
        "exit": "Sair",
        "settings": "Configurações",
        "account_info": "Informações da conta",
        "username": "Nome de usuário",
        "password": "Senha",
        "repeat_password": "Repetir senha",
        "first_name": "Nome",
        "last_name": "Sobrenome",
        "email": "Endereço de e-mail",
        "remember_me": "Lembrar de mim",
        "login_btn": "Entrar",
        "signup_btn": "Cadastrar",
        "account": "Conta",
        "change_language": "Alterar idioma",
        "change_theme": "Alterar tema",
        "insert_sono": "Inserir Sono",
        "process": "Processar",
        "mute": "Silenciar",
        "unmute": "Ativar som",
        "visit_webpage": "Visitar página",
        "please_wait": "Aguarde...",
        "processing": "Processando imagem...",
        "result": "Resultado",
        "confidence": "Confiança",
        "grid": "Grade",
        "insert_image_first": "Por favor, insira uma imagem primeiro!",
        "no_image": "Nenhuma imagem inserida",
        "invalid_credentials": "Credenciais inválidas",
        "user_exists": "Nome de usuário já existe",
        "password_requirements": "Requisitos de senha",
        "password_rule": (
            "A senha deve:\n"
            "• Ter 8-16 caracteres\n"
            "• Conter pelo menos 1 letra minúscula\n"
            "• Conter pelo menos 1 letra maiúscula\n"
            "• Conter pelo menos 4 dígitos\n"
            "• Conter pelo menos 4 caracteres especiais"
        ),
        "passwords_dont_match": "As senhas não coincidem!",
        "invalid_email": "Por favor, insira um endereço de e-mail válido",
        "empty_fields": "Por favor, preencha todos os campos",
        "signup_success": "Cadastro realizado com sucesso! Agora você pode entrar.",
        "logout_confirm": "Tem certeza de que deseja excluir sua conta?",
        "exit_confirm": "Tem certeza de que deseja sair?",
        "theme_dark_aqua": "Tema Escuro Aqua-Roxo",
        "theme_solarized_white": "Tema Solarized Claro",
        "theme_solarized_dark": "Tema Solarized Escuro",
        "theme_dark_forest": "Tema Floresta Verde Escuro",
        "select_language": "Selecionar idioma",
        "select_theme": "Selecionar tema",
        "apply": "Aplicar",
        "cancel": "Cancelar",
        "save": "Salvar",
        "success": "Sucesso",
        "error": "Erro",
        "warning": "Aviso",
        "settings_applied": "Configurações aplicadas!",
        "grid_0": "Grade 0 (NL) — Fígado normal",
        "grid_1": "Grade 1 (G1) — Grau 1",
        "grid_2": "Grade 2 (G2) — Grau 2",
        "grid_3": "Grade 3 (G3) — Grau 3",
        "too_many_attempts": "Muitas tentativas falhas. Aguarde 5 minutos.",
        "invalid_username": "O nome de usuário deve ter 3-32 caracteres (letras, números, sublinhado).",
        "welcome_user": "Bem-vindo, {user}!",
        "inference_failed": "Falha na inferência",
        "drop_here": "Solte a imagem aqui",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Tipo de arquivo não suportado. Solte uma imagem.",
        "image_loaded": "Imagem carregada com sucesso",
        "help_about_title": "Ajuda & Sobre",
        "help_about_text": (
            "Pipeline IA do Fígado v3.0\n"
            "Desenvolvido por Babak Yousefian\n\n"
            "Atalhos de teclado:\n"
            "  Ctrl+O      — Inserir imagem\n"
            "  Ctrl+Enter  — Processar imagem\n"
            "  Ctrl+L      — Entrar\n"
            "  Ctrl+,      — Configurações\n"
            "  Esc         — Fechar diálogo ativo\n"
            "  F1          — Ajuda / Sobre\n\n"
            "Dica: Você também pode arrastar e soltar uma imagem em qualquer lugar da janela."
        ),
        "dashboard_title": "Painel",
        "dash_total": "Total de imagens",
        "dash_avg_conf": "Confiança média",
        "dash_distribution": "Distribuição",
        "dash_last_5": "Últimos 5 resultados",
        "dash_no_results": "Nenhum resultado ainda",
        "dash_clear_confirm": "Limpar todas as estatísticas?",
        "dash_cleared": "Estatísticas limpas",
        "dashboard_toggle": "Mostrar/Ocultar painel",
        "export_pdf": "Exportar PDF",
        "pdf_saved": "PDF salvo com sucesso",
        "pdf_failed": "Falha na geração do PDF",
        "pdf_subtitle": "Relatório de Análise",
        "pdf_patient": "Paciente:",
        "pdf_date": "Data:",
        "pdf_model": "Modelo:",
        "pdf_diagnosis": "DIAGNÓSTICO",
        "pdf_confidence": "Confiança",
        "pdf_class_probs": "PROBABILIDADES DAS CLASSES",
        "pdf_disclaimer": (
            "Este relatório foi desenvolvido por Babak Yousefian sob a supervisão de médicos experientes. "
            "É uma ferramenta de análise profissional e não requer encaminhamento a um médico."
        ),
        "history_title": "Histórico",
        "history_toggle": "Mostrar/Ocultar histórico",
        "history_empty": "Nenhum histórico ainda",
        "history_cleared": "Histórico limpo",
        "history_clear_confirm": "Limpar todo o histórico?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Entrar",
        "tooltip_desc_login": "Entre na sua conta",
        "tooltip_tip_login": "Acesse seu histórico e estatísticas salvos",
        "tooltip_title_signup": "Cadastrar",
        "tooltip_desc_signup": "Criar uma nova conta",
        "tooltip_title_account": "Conta",
        "tooltip_desc_account": "Clique para abrir o menu da conta",
        "tooltip_desc_account_user": "Conectado como {user}",
        "tooltip_title_dashboard": "Painel",
        "tooltip_desc_dashboard": "Mostrar/ocultar painel de estatísticas — total, distribuição, resultados recentes",
        "tooltip_title_history": "Histórico",
        "tooltip_desc_history": "Mostrar/ocultar os últimos 20 resultados com miniaturas",
        "tooltip_title_visit": "Visitar página",
        "tooltip_desc_visit": "Abrir a página do GitHub do projeto no navegador",
        "tooltip_tip_visit": "Abre no navegador externo",
        "tooltip_title_insert": "Inserir imagem",
        "tooltip_desc_insert": "Carregar uma imagem de sono para analisar",
        "tooltip_tip_insert": "Você também pode arrastar e soltar uma imagem em qualquer lugar da janela",
        "tooltip_title_process": "Processar",
        "tooltip_desc_process": "Executar análise de IA na imagem carregada",
        "tooltip_tip_process": "Requer uma imagem carregada primeiro",
        "tooltip_title_mute": "Silenciar música",
        "tooltip_desc_mute": "Pausar a música de fundo",
        "tooltip_title_unmute": "Ativar música",
        "tooltip_desc_unmute": "Retomar a música de fundo",
        "tooltip_title_export": "Exportar PDF",
        "tooltip_desc_export": "Salvar o resultado atual como relatório PDF",
        "tooltip_tip_export": "Inclui imagem, diagnóstico e probabilidades por classe",
        "tooltip_title_dash_clear": "Limpar estatísticas",
        "tooltip_desc_dash_clear": "Redefinir todas as estatísticas desta conta",
        "tooltip_tip_dash_clear": "Esta ação não pode ser desfeita",
        "tooltip_title_dash_close": "Fechar painel",
        "tooltip_desc_dash_close": "Ocultar o painel de estatísticas",
        "tooltip_desc_dash_total": "Número total de imagens processadas",
        "tooltip_desc_dash_avg": "Confiança média em todos os resultados",
        "tooltip_desc_dash_dist": "Distribuição dos resultados por grau",
        "tooltip_title_hist_clear": "Limpar histórico",
        "tooltip_desc_hist_clear": "Excluir todos os resultados e miniaturas salvos",
        "tooltip_tip_hist_clear": "Isso também excluirá todos os arquivos de imagem",
        "tooltip_title_hist_close": "Fechar histórico",
        "tooltip_desc_hist_close": "Ocultar o painel de histórico",
        "tooltip_tip_thumb_click": "Clique para recarregar este resultado",
        "tooltip_tip_recent": "Clique para recarregar este resultado",
        "tooltip_title_zoom_in": "Aumentar zoom",
        "tooltip_desc_zoom_in": "Aumentar ampliação da imagem",
        "tooltip_title_zoom_out": "Diminuir zoom",
        "tooltip_desc_zoom_out": "Reduzir ampliação da imagem",
        "tooltip_title_zoom_reset": "Redefinir visualização",
        "tooltip_desc_zoom_reset": "Ajustar imagem à janela",
        "tooltip_tip_zoom_reset": "Você também pode clicar duas vezes na imagem",
        "notif_inference_ok": "Inferência concluída",
        "follow_system_theme": "Seguir tema do sistema",
        "follow_system_theme_desc": "Alternar automaticamente entre claro e escuro com base nas configurações do sistema",
        "gradcam_title": "Mapa de calor de atenção",
        "gradcam_show": "Mostrar mapa de atenção",
        "gradcam_toggle": "Alternar mapa",
        "gradcam_save": "Salvar imagem",
        "gradcam_close": "Fechar",
        "gradcam_saved": "Imagem salva com sucesso",
        "gradcam_no_heatmap": "Nenhum mapa de atenção disponível. Por favor, processe uma imagem primeiro.",
        "update_available": "Atualização disponível",
        "update_check_failed": "Falha ao verificar atualizações",
        "check_for_updates": "Verificar atualizações",
        "check_for_updates_desc": "Verificar GitHub/site para a versão mais recente",
        "up_to_date": "Você está atualizado",
        "you_have_latest": "Você está executando a versão mais recente.",
        "check_internet": "Verifique sua conexão com a Internet e tente novamente.",
        "downloading": "Baixando...",
        "verifying": "Verificando...",
        "download_now": "Baixar agora",
        "install_now": "Instalar e reiniciar",
        "skip_version": "Pular esta versão",
        "remind_later": "Lembrar mais tarde",
        "whats_new": "Novidades",
        "no_changelog": "Nenhum changelog fornecido.",
        "critical_update": "Atualização crítica",
        "update_ready": "Pronto para instalar",
        "app_will_restart": "O aplicativo será fechado e reiniciado para aplicar a atualização.",
        "install_failed": "Falha na instalação. Sua versão antiga está segura.",
        "download_failed": "Falha no download.",
        "source_update_requires_zip": "A atualização de código-fonte requer um arquivo .zip.",
        "session_locked": "Sessão Bloqueada",
        "locked_for_user": "Bloqueada para {user}",
        "unlock": "Desbloquear",
        "session_unlocked": "Sessão desbloqueada",
        "auto_lock_notice": "A sessão será bloqueada após 2 minutos de inatividade.",
        "audit_log_title": "Registro de Auditoria",
        "audit_export": "Exportar",
        "audit_exported": "Exportado com sucesso",
        "audit_export_failed": "Falha na exportação",
        "audit_search_placeholder": "Pesquisar ações ou detalhes...",
        "audit_col_time": "Hora",
        "audit_col_level": "Nível",
        "audit_col_category": "Categoria",
        "audit_col_action": "Ação",
        "audit_col_user": "Usuário",
        "audit_details_title": "Detalhes do Evento",
        "audit_status": "Mostrando {count} de {total} eventos",
        "strength_very_weak": "Muito fraca",
        "strength_weak": "Fraca",
        "strength_medium": "Média",
        "strength_strong": "Forte",
        "strength_very_strong": "Muito forte",
        "strength_rule_length": "8-16 caracteres",
        "strength_rule_lowercase": "Letra minúscula",
        "strength_rule_uppercase": "Letra maiúscula",
        "strength_rule_digits": "Dígitos",
        "strength_rule_special": "Caracteres especiais",
        "strength_add_chars": "Adicione mais {n} caractere(s)",
        "strength_remove_chars": "Remova {n} caractere(s)",
        "strength_need_digits": "Adicione mais {n} dígito(s)",
        "strength_need_special": "Adicione mais {n} caractere(s) especial(is)",
        "strength_common_pattern": "Evite palavras comuns",
        "strength_common_year": "Evite anos comuns",
        "strength_has_sequence": "Evite sequências (1234, abcd)",
        "strength_has_repeats": "Evite repetições (aaaa, 1111)",
        "passwords_match": "As senhas coincidem",
        "2fa_setup_title": "Ativar autenticação de dois fatores",
        "2fa_setup_subtitle": "Escaneie o código QR com Google Authenticator, Authy ou Microsoft Authenticator.",
        "2fa_manual_secret": "Ou insira este código manualmente:",
        "2fa_enter_code": "Insira o código de 6 dígitos do seu aplicativo:",
        "2fa_verify_btn": "Verificar",
        "2fa_backup_codes_title": "Salve estes códigos de backup!",
        "2fa_backup_codes_hint": "Cada código só pode ser usado uma vez. Guarde-os em um local seguro.",
        "2fa_done_btn": "Concluído",
        "2fa_secret_copied": "Código secreto copiado para a área de transferência",
        "2fa_invalid_code": "Código inválido. Tente novamente.",
        "2fa_invalid_code_format": "O código deve ter 6 dígitos.",
        "2fa_verify_title": "Autenticação de dois fatores",
        "2fa_verify_subtitle": "Insira o código de 6 dígitos do seu aplicativo autenticador.",
        "2fa_enter_code_hint": "Por favor, insira um código.",
        "2fa_use_backup_code": "Usar um código de backup",
        "2fa_use_totp_code": "Usar código do aplicativo",
        "2fa_enter_backup_code": "Insira um dos seus códigos de backup.",
        "2fa_section_title": "Segurança",
        "2fa_enabled_status": "A autenticação de dois fatores está ativada",
        "2fa_disabled_status": "A autenticação de dois fatores está desativada",
        "2fa_enable_btn": "Ativar 2FA",
        "2fa_disable_btn": "Desativar 2FA",
        "2fa_disable_title": "Desativar 2FA",
        "2fa_disable_confirm": "Tem certeza? Sua conta ficará menos segura.",
        "2fa_disabled_success": "Autenticação de dois fatores desativada",
        "2fa_enabled_success": "Autenticação de dois fatores ativada",
        "2fa_hint_title": "Como usar este código QR",
        "2fa_hint_step1": "Instale um aplicativo autenticador: Google Authenticator, Microsoft Authenticator ou Authy.",
        "2fa_hint_step2": "Abra o aplicativo, toque no botão \"+\" e escolha \"Escanear código QR\".",
        "2fa_hint_step3": "Aponte a câmera do seu celular para o código QR acima (use a câmera do aplicativo, não o app de câmera do celular).",
        "2fa_hint_step4": "O aplicativo mostrará um código de 6 dígitos. Insira-o na caixa abaixo.",
        "2fa_manual_secret_hint": "Não consegue escanear? Copie a chave abaixo e adicione-a manualmente no aplicativo (\"Inserir chave de configuração\").",
        "auto_lock_section_title": "Bloqueio automático",
        "auto_lock_enable": "Bloquear após inatividade",
        "auto_lock_timeout_label": "Bloquear após:",
        "auto_lock_warning_title": "Aviso de bloqueio automático",
        "auto_lock_warning_body": "Sua sessão será bloqueada em {seconds} segundos por inatividade.",
        "lock_now": "Bloquear agora",
        "font_section_title": "Fonte",
        "font_family_label": "Família da fonte",
        "font_size_label": "Tamanho base",
        "font_scale_label": "Escala de exibição",
        "font_scale_small": "Pequena",
        "font_scale_normal": "Normal",
        "font_scale_large": "Grande",
        "font_scale_xlarge": "Muito grande",
        "font_scale_xxlarge": "Enorme",
        "font_preview_text": "Um pequeno jabuti xereta viu dez cegonhas felizes — 1234567890",
        "font_reset": "Redefinir para o padrão",
        "font_applied": "Fonte atualizada",
        "motion_section_title": "Movimento e animações",
        "reduce_motion_label": "Reduzir movimento",
        "reduce_motion_hint": "Desative as animações se elas te deixarem desconfortável.",
        "empty_history_empty_title": "Ainda sem histórico",
        "empty_history_empty_subtitle": "Processe sua primeira imagem para vê-la aqui",
        "empty_history_empty_hint": "Cada resultado é salvo automaticamente",
        "empty_history_no_user_title": "Entre para ver o histórico",
        "empty_history_no_user_subtitle": "Seus resultados salvos aparecerão aqui",
        "empty_history_no_user_hint": "O histórico é privado para cada conta",
        "empty_dash_no_results_title": "Ainda sem resultados",
        "empty_dash_no_results_subtitle": "Suas análises recentes aparecerão aqui",
        "empty_dash_no_results_hint": "Processe uma imagem para começar",
        "empty_dash_no_dist_title": "Sem dados de distribuição",
        "empty_dash_no_dist_subtitle": "As estatísticas serão preenchidas conforme você processa",
        "empty_dash_no_dist_hint": "Cada análise conta",
        "empty_image_empty_title": "Nenhuma imagem inserida",
        "empty_image_empty_subtitle": "Arraste e solte uma imagem aqui, ou clique Inserir",
        "empty_image_empty_hint": "Suporta PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Solte para analisar",
        "empty_image_drop_hint_subtitle": "Solte a imagem para carregá-la",
        "empty_image_drop_hint_hint": "O formato será detectado automaticamente",
        "empty_processing_wait_title": "Processando...",
        "empty_processing_wait_subtitle": "A IA está analisando sua imagem",
        "empty_processing_wait_hint": "Geralmente leva alguns segundos",
        "tour_show_again": "Mostrar tour novamente",
        "tour_completed_msg": "Tour concluído! Você está pronto.",
        "tour_skipped_msg": "Tour ignorado. Você pode repeti-lo no menu Conta.",
        "tour_btn_next": "Próximo",
        "tour_btn_prev": "Anterior",
        "tour_btn_skip": "Pular",
        "tour_btn_finish": "Concluir ✓",
        "tour_guest_login_title": "Entrar",
        "tour_guest_login_subtitle": "Clique aqui para entrar na sua conta e desbloquear todos os recursos.",
        "tour_guest_signup_title": "Criar uma conta",
        "tour_guest_signup_subtitle": "Novo por aqui? Clique aqui para criar uma conta em segundos.",
        "tour_insert_title": "Inserir imagem",
        "tour_insert_subtitle": "Carregue uma imagem de ultrassom do disco, ou arraste e solte-a em qualquer lugar da janela.",
        "tour_process_title": "Processar imagem",
        "tour_process_subtitle": "Execute a análise de IA na imagem carregada. Os resultados aparecem em poucos segundos.",
        "tour_export_title": "Exportar PDF",
        "tour_export_subtitle": "Salve o resultado atual como um relatório médico PDF profissional.",
        "tour_gradcam_title": "Mapa de calor de atenção",
        "tour_gradcam_subtitle": "Veja onde a IA está olhando — um mapa de calor sobre a imagem analisada.",
        "tour_dashboard_title": "Painel",
        "tour_dashboard_subtitle": "Mostrar/ocultar painel de estatísticas: total de imagens, confiança média, distribuição por grau.",
        "tour_history_title": "Histórico",
        "tour_history_subtitle": "Navegue pelas últimas 20 imagens analisadas. Clique em uma miniatura para recarregar o resultado.",
        "tour_visit_title": "Visitar página",
        "tour_visit_subtitle": "Abra a página GitHub do projeto no navegador para documentação e atualizações.",
        "tour_mute_title": "Silenciar música",
        "tour_mute_subtitle": "Pause ou retome a música de fundo a qualquer momento.",
        "tour_zoom_title": "Controles de zoom",
        "tour_zoom_subtitle": "Use a roda do mouse para zoom, arraste para mover, e clique duplo para redefinir.",
        "tour_account_title": "Menu da conta",
        "tour_account_subtitle": "Clique aqui para acessar Configurações, Informações da conta, Registro de auditoria e mais.",
        "suggestion_image_ready": "Imagem carregada — pronta para processar",
        "suggestion_image_ready_sub": "Clique em Processar para executar a análise de IA.",
        "suggestion_pdf_ready": "Resultado pronto — exportar como PDF?",
        "suggestion_pdf_ready_sub": "Salve um relatório médico profissional com um clique.",
        "suggestion_gradcam_hint": "Curioso para saber o que a IA viu?",
        "suggestion_gradcam_hint_sub": "Veja o mapa de calor de atenção sobre a imagem.",
        "suggestion_repeated_grid": "Você sempre obtém {grid}",
        "suggestion_repeated_grid_sub": "Abra o painel para ver a distribuição.",
        "suggestion_history_hint": "Seu histórico está crescendo",
        "suggestion_history_hint_sub": "Navegue pelas suas últimas 20 imagens analisadas.",
        "suggestion_dashboard_hint": "Seu painel tem estatísticas interessantes",
        "suggestion_dashboard_hint_sub": "Totais, médias e distribuição rapidamente.",
        "suggestion_settings_hint": "Personalize o aplicativo",
        "suggestion_settings_hint_sub": "Altere idioma, tema, fonte e ative 2FA.",
        "suggestion_idle_help": "Precisa de um guia rápido?",
        "suggestion_idle_help_sub": "Repita o tour de integração a qualquer momento.",
        "image_not_liver_warning": "Esta imagem não parece uma ultrassonografia do fígado.\n\nPor favor, insira uma imagem de ultrassonografia do fígado adequada.",
        "result_not_confident": "A IA não conseguiu identificar com confiança esta imagem como uma ultrassonografia do fígado.\n\nPor favor, insira uma imagem de ultrassonografia do fígado adequada e tente novamente.",
        "not_liver_title": "Não é uma ultrassonografia do fígado",
        "not_liver_hint": "Por favor, use uma imagem de ultrassonografia (sonograma) do fígado adequada",
        "voice_welcome": "Bem-vindo, {user}",
        "voice_goodbye": "Tchau",
        "voice_logged_out": "Logout realizado com sucesso",
        "voice_image_loaded": "Imagem carregada com sucesso",
        "voice_image_rejected": "Esta não é uma ultrassonografia do fígado",
        "voice_inference_result": "{grid_name}, confiança {conf} por cento",
        "voice_pdf_saved": "Relatório PDF salvo com sucesso",
        "voice_error": "Ocorreu um erro",
        "voice_settings_applied": "Configurações aplicadas",
        "voice_language_changed": "Idioma alterado para {lang}",
        "voice_theme_changed": "Tema alterado",
        "voice_2fa_enabled": "Autenticação de dois fatores ativada",
        "voice_2fa_disabled": "Autenticação de dois fatores desativada",
        "voice_locked": "Sessão bloqueada",
        "voice_unlocked": "Sessão desbloqueada",
        "voice_processing": "Processando imagem",
        "voice_processing_done": "Processamento concluído",
        "voice_voice_enabled": "Feedback de voz ativado",
        "voice_voice_disabled": "Feedback de voz desativado",
    },
    "Argentina": {
        "app_title": "Hígado AI-babak",
        "welcome": "Bienvenido",
        "login": "Iniciar sesión",
        "signup": "Registrarse",
        "logout": "Cerrar sesión",
        "exit": "Salir",
        "settings": "Configuración",
        "account_info": "Información de la cuenta",
        "username": "Nombre de usuario",
        "password": "Contraseña",
        "repeat_password": "Repetir contraseña",
        "first_name": "Nombre",
        "last_name": "Apellido",
        "email": "Correo electrónico",
        "remember_me": "Recordarme",
        "login_btn": "Iniciar sesión",
        "signup_btn": "Registrarse",
        "account": "Cuenta",
        "change_language": "Cambiar idioma",
        "change_theme": "Cambiar tema",
        "insert_sono": "Insertar Sono",
        "process": "Procesar",
        "mute": "Silenciar",
        "unmute": "Activar sonido",
        "visit_webpage": "Visitar página web",
        "please_wait": "Por favor espere...",
        "processing": "Procesando imagen...",
        "result": "Resultado",
        "confidence": "Confianza",
        "grid": "Cuadrícula",
        "insert_image_first": "¡Por favor, inserte una imagen primero!",
        "no_image": "Ninguna imagen insertada",
        "invalid_credentials": "Credenciales inválidas",
        "user_exists": "El nombre de usuario ya existe",
        "password_requirements": "Requisitos de contraseña",
        "password_rule": (
            "La contraseña debe:\n"
            "• Tener 8-16 caracteres\n"
            "• Contener al menos 1 letra minúscula\n"
            "• Contener al menos 1 letra mayúscula\n"
            "• Contener al menos 4 dígitos\n"
            "• Contener al menos 4 caracteres especiales"
        ),
        "passwords_dont_match": "¡Las contraseñas no coinciden!",
        "invalid_email": "Por favor, ingrese un correo electrónico válido",
        "empty_fields": "Por favor, complete todos los campos",
        "signup_success": "¡Registro exitoso! Ahora puede iniciar sesión.",
        "logout_confirm": "¿Está seguro de que desea eliminar su cuenta?",
        "exit_confirm": "¿Está seguro de que desea salir?",
        "theme_dark_aqua": "Tema Oscuro Aqua-Púrpura",
        "theme_solarized_white": "Tema Solarized Claro",
        "theme_solarized_dark": "Tema Solarized Oscuro",
        "theme_dark_forest": "Tema Bosque Verde Oscuro",
        "select_language": "Seleccionar idioma",
        "select_theme": "Seleccionar tema",
        "apply": "Aplicar",
        "cancel": "Cancelar",
        "save": "Guardar",
        "success": "Éxito",
        "error": "Error",
        "warning": "Advertencia",
        "settings_applied": "¡Configuración aplicada!",
        "grid_0": "Cuadrícula 0 (NL) — Hígado normal",
        "grid_1": "Cuadrícula 1 (G1) — Grado 1",
        "grid_2": "Cuadrícula 2 (G2) — Grado 2",
        "grid_3": "Cuadrícula 3 (G3) — Grado 3",
        "too_many_attempts": "Demasiados intentos fallidos. Espere 5 minutos.",
        "invalid_username": "El nombre de usuario debe tener 3-32 caracteres (letras, números, guion bajo).",
        "welcome_user": "¡Bienvenido, {user}!",
        "inference_failed": "Fallo en la inferencia",
        "drop_here": "Suelte la imagen aquí",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Tipo de archivo no compatible. Suelte una imagen.",
        "image_loaded": "Imagen cargada con éxito",
        "help_about_title": "Ayuda & Acerca de",
        "help_about_text": (
            "Pipeline IA de Hígado v3.0\n"
            "Desarrollado por Babak Yousefian\n\n"
            "Atajos de teclado:\n"
            "  Ctrl+O      — Insertar imagen\n"
            "  Ctrl+Enter  — Procesar imagen\n"
            "  Ctrl+L      — Iniciar sesión\n"
            "  Ctrl+,      — Configuración\n"
            "  Esc         — Cerrar diálogo activo\n"
            "  F1          — Ayuda / Acerca de\n\n"
            "Consejo: También puede arrastrar y soltar una imagen en cualquier lugar de la ventana."
        ),
        "dashboard_title": "Panel",
        "dash_total": "Total de imágenes",
        "dash_avg_conf": "Confianza promedio",
        "dash_distribution": "Distribución",
        "dash_last_5": "Últimos 5 resultados",
        "dash_no_results": "Aún sin resultados",
        "dash_clear_confirm": "¿Borrar todas las estadísticas?",
        "dash_cleared": "Estadísticas borradas",
        "dashboard_toggle": "Mostrar/Ocultar panel",
        "export_pdf": "Exportar PDF",
        "pdf_saved": "PDF guardado con éxito",
        "pdf_failed": "Error al generar PDF",
        "pdf_subtitle": "Informe de Análisis",
        "pdf_patient": "Paciente:",
        "pdf_date": "Fecha:",
        "pdf_model": "Modelo:",
        "pdf_diagnosis": "DIAGNÓSTICO",
        "pdf_confidence": "Confianza",
        "pdf_class_probs": "PROBABILIDADES DE CLASE",
        "pdf_disclaimer": (
            "Este informe fue desarrollado por Babak Yousefian bajo la supervisión de médicos experimentados. "
            "Es una herramienta de análisis profesional y no requiere derivación a un médico."
        ),
        "history_title": "Historial",
        "history_toggle": "Mostrar/Ocultar historial",
        "history_empty": "Aún sin historial",
        "history_cleared": "Historial borrado",
        "history_clear_confirm": "¿Borrar todo el historial?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Iniciar sesión",
        "tooltip_desc_login": "Inicie sesión en su cuenta",
        "tooltip_tip_login": "Acceda a su historial y estadísticas guardados",
        "tooltip_title_signup": "Registrarse",
        "tooltip_desc_signup": "Crear una cuenta nueva",
        "tooltip_title_account": "Cuenta",
        "tooltip_desc_account": "Haga clic para abrir el menú de la cuenta",
        "tooltip_desc_account_user": "Conectado como {user}",
        "tooltip_title_dashboard": "Panel",
        "tooltip_desc_dashboard": "Mostrar/ocultar panel de estadísticas — total, distribución, resultados recientes",
        "tooltip_title_history": "Historial",
        "tooltip_desc_history": "Mostrar/ocultar los últimos 20 resultados con miniaturas",
        "tooltip_title_visit": "Visitar página web",
        "tooltip_desc_visit": "Abrir la página de GitHub del proyecto en el navegador",
        "tooltip_tip_visit": "Se abre en el navegador externo",
        "tooltip_title_insert": "Insertar imagen",
        "tooltip_desc_insert": "Cargar una imagen de ecografía para analizar",
        "tooltip_tip_insert": "También puede arrastrar y soltar una imagen en cualquier lugar de la ventana",
        "tooltip_title_process": "Procesar",
        "tooltip_desc_process": "Ejecutar análisis de IA en la imagen cargada",
        "tooltip_tip_process": "Requiere cargar una imagen primero",
        "tooltip_title_mute": "Silenciar música",
        "tooltip_desc_mute": "Pausar la música de fondo",
        "tooltip_title_unmute": "Activar música",
        "tooltip_desc_unmute": "Reanudar la música de fondo",
        "tooltip_title_export": "Exportar PDF",
        "tooltip_desc_export": "Guardar el resultado actual como informe PDF",
        "tooltip_tip_export": "Incluye imagen, diagnóstico y probabilidades por clase",
        "tooltip_title_dash_clear": "Limpiar estadísticas",
        "tooltip_desc_dash_clear": "Restablecer todas las estadísticas de esta cuenta",
        "tooltip_tip_dash_clear": "Esta acción no se puede deshacer",
        "tooltip_title_dash_close": "Cerrar panel",
        "tooltip_desc_dash_close": "Ocultar el panel de estadísticas",
        "tooltip_desc_dash_total": "Número total de imágenes procesadas",
        "tooltip_desc_dash_avg": "Confianza promedio en todos los resultados",
        "tooltip_desc_dash_dist": "Distribución de resultados por grado",
        "tooltip_title_hist_clear": "Limpiar historial",
        "tooltip_desc_hist_clear": "Eliminar todos los resultados y miniaturas guardados",
        "tooltip_tip_hist_clear": "Esto también eliminará todos los archivos de imagen",
        "tooltip_title_hist_close": "Cerrar historial",
        "tooltip_desc_hist_close": "Ocultar el panel de historial",
        "tooltip_tip_thumb_click": "Haga clic para recargar este resultado",
        "tooltip_tip_recent": "Haga clic para recargar este resultado",
        "tooltip_title_zoom_in": "Acercar",
        "tooltip_desc_zoom_in": "Aumentar el zoom de la imagen",
        "tooltip_title_zoom_out": "Alejar",
        "tooltip_desc_zoom_out": "Reducir el zoom de la imagen",
        "tooltip_title_zoom_reset": "Restablecer vista",
        "tooltip_desc_zoom_reset": "Ajustar la imagen a la ventana",
        "tooltip_tip_zoom_reset": "También puede hacer doble clic en la imagen",
        "notif_inference_ok": "Inferencia completada",
        "follow_system_theme": "Seguir tema del sistema",
        "follow_system_theme_desc": "Cambiar automáticamente entre claro y oscuro según la configuración del sistema",
        "gradcam_title": "Mapa de calor de atención",
        "gradcam_show": "Mostrar mapa de atención",
        "gradcam_toggle": "Alternar mapa",
        "gradcam_save": "Guardar imagen",
        "gradcam_close": "Cerrar",
        "gradcam_saved": "Imagen guardada con éxito",
        "gradcam_no_heatmap": "No hay mapa de atención disponible. Por favor, procese una imagen primero.",
        "update_available": "Actualización disponible",
        "update_check_failed": "Falló la verificación de actualizaciones",
        "check_for_updates": "Buscar actualizaciones",
        "check_for_updates_desc": "Verificar GitHub/sitio web para la última versión",
        "up_to_date": "Estás actualizado",
        "you_have_latest": "Estás ejecutando la última versión.",
        "check_internet": "Por favor, revisá tu conexión a Internet y volvé a intentar.",
        "downloading": "Descargando...",
        "verifying": "Verificando...",
        "download_now": "Descargar ahora",
        "install_now": "Instalar y reiniciar",
        "skip_version": "Omitir esta versión",
        "remind_later": "Recordarme más tarde",
        "whats_new": "Novedades",
        "no_changelog": "No se proporcionó registro de cambios.",
        "critical_update": "Actualización crítica",
        "update_ready": "Listo para instalar",
        "app_will_restart": "La aplicación se cerrará y reiniciará para aplicar la actualización.",
        "install_failed": "Falló la instalación. Tu versión anterior está segura.",
        "download_failed": "Falló la descarga.",
        "source_update_requires_zip": "La actualización de código fuente requiere un archivo .zip.",
        "session_locked": "Sesión Bloqueada",
        "locked_for_user": "Bloqueada para {user}",
        "unlock": "Desbloquear",
        "session_unlocked": "Sesión desbloqueada",
        "auto_lock_notice": "La sesión se bloqueará después de 2 minutos de inactividad.",
        "audit_log_title": "Registro de Auditoría",
        "audit_export": "Exportar",
        "audit_exported": "Exportado con éxito",
        "audit_export_failed": "Falló la exportación",
        "audit_search_placeholder": "Buscar acciones o detalles...",
        "audit_col_time": "Hora",
        "audit_col_level": "Nivel",
        "audit_col_category": "Categoría",
        "audit_col_action": "Acción",
        "audit_col_user": "Usuario",
        "audit_details_title": "Detalles del Evento",
        "audit_status": "Mostrando {count} de {total} eventos",
        "strength_very_weak": "Muy débil",
        "strength_weak": "Débil",
        "strength_medium": "Media",
        "strength_strong": "Fuerte",
        "strength_very_strong": "Muy fuerte",
        "strength_rule_length": "8-16 caracteres",
        "strength_rule_lowercase": "Letra minúscula",
        "strength_rule_uppercase": "Letra mayúscula",
        "strength_rule_digits": "Dígitos",
        "strength_rule_special": "Caracteres especiales",
        "strength_add_chars": "Agregá {n} carácter(es) más",
        "strength_remove_chars": "Quitá {n} carácter(es)",
        "strength_need_digits": "Agregá {n} dígito(s) más",
        "strength_need_special": "Agregá {n} carácter(es) especial(es) más",
        "strength_common_pattern": "Evitá palabras comunes",
        "strength_common_year": "Evitá años comunes",
        "strength_has_sequence": "Evitá secuencias (1234, abcd)",
        "strength_has_repeats": "Evitá repeticiones (aaaa, 1111)",
        "passwords_match": "Las contraseñas coinciden",
        "2fa_setup_title": "Habilitar autenticación de dos factores",
        "2fa_setup_subtitle": "Escaneá el código QR con Google Authenticator, Authy o Microsoft Authenticator.",
        "2fa_manual_secret": "O ingresá este código manualmente:",
        "2fa_enter_code": "Ingresá el código de 6 dígitos de tu aplicación:",
        "2fa_verify_btn": "Verificar",
        "2fa_backup_codes_title": "¡Guardá estos códigos de respaldo!",
        "2fa_backup_codes_hint": "Cada código se puede usar solo una vez. Guardalos en un lugar seguro.",
        "2fa_done_btn": "Listo",
        "2fa_secret_copied": "Código secreto copiado al portapapeles",
        "2fa_invalid_code": "Código inválido. Intentá de nuevo.",
        "2fa_invalid_code_format": "El código debe tener 6 dígitos.",
        "2fa_verify_title": "Autenticación de dos factores",
        "2fa_verify_subtitle": "Ingresá el código de 6 dígitos de tu aplicación autenticadora.",
        "2fa_enter_code_hint": "Por favor, ingresá un código.",
        "2fa_use_backup_code": "Usar un código de respaldo",
        "2fa_use_totp_code": "Usar código de la aplicación",
        "2fa_enter_backup_code": "Ingresá uno de tus códigos de respaldo.",
        "2fa_section_title": "Seguridad",
        "2fa_enabled_status": "La autenticación de dos factores está habilitada",
        "2fa_disabled_status": "La autenticación de dos factores está deshabilitada",
        "2fa_enable_btn": "Habilitar 2FA",
        "2fa_disable_btn": "Deshabilitar 2FA",
        "2fa_disable_title": "Deshabilitar 2FA",
        "2fa_disable_confirm": "¿Estás seguro? Tu cuenta será menos segura.",
        "2fa_disabled_success": "Autenticación de dos factores deshabilitada",
        "2fa_enabled_success": "Autenticación de dos factores habilitada",
        "2fa_hint_title": "Cómo usar este código QR",
        "2fa_hint_step1": "Instalá una aplicación de autenticación: Google Authenticator, Microsoft Authenticator o Authy.",
        "2fa_hint_step2": "Abrí la aplicación, tocá el botón «+» y elegí «Escanear código QR».",
        "2fa_hint_step3": "Apuntá la cámara de tu celular al código QR de arriba (usá la cámara de la aplicación, no la app de cámara del celular).",
        "2fa_hint_step4": "La aplicación va a mostrar un código de 6 dígitos. Ingresalo en el cuadro de abajo.",
        "2fa_manual_secret_hint": "¿No podés escanear? Copiá la clave de abajo y agregala manualmente en tu aplicación («Ingresar clave de configuración»).",
        "auto_lock_section_title": "Bloqueo automático",
        "auto_lock_enable": "Bloquear después de inactividad",
        "auto_lock_timeout_label": "Bloquear después de:",
        "auto_lock_warning_title": "Aviso de bloqueo automático",
        "auto_lock_warning_body": "Tu sesión se bloqueará en {seconds} segundos por inactividad.",
        "lock_now": "Bloquear ahora",
        "font_section_title": "Tipografía",
        "font_family_label": "Familia tipográfica",
        "font_size_label": "Tamaño base",
        "font_scale_label": "Escala de visualización",
        "font_scale_small": "Chico",
        "font_scale_normal": "Normal",
        "font_scale_large": "Grande",
        "font_scale_xlarge": "Muy grande",
        "font_scale_xxlarge": "Enorme",
        "font_preview_text": "El veloz murciélago hindú comía feliz cardillo y kiwi — 1234567890",
        "font_reset": "Restablecer predeterminado",
        "font_applied": "Tipografía actualizada",
        "motion_section_title": "Movimiento y animaciones",
        "reduce_motion_label": "Reducir movimiento",
        "reduce_motion_hint": "Desactivá las animaciones si te resultan incómodas.",
        "empty_history_empty_title": "Aún sin historial",
        "empty_history_empty_subtitle": "Procesá tu primera imagen para verla acá",
        "empty_history_empty_hint": "Cada resultado se guarda automáticamente",
        "empty_history_no_user_title": "Iniciá sesión para ver el historial",
        "empty_history_no_user_subtitle": "Tus resultados guardados aparecerán acá",
        "empty_history_no_user_hint": "El historial es privado para cada cuenta",
        "empty_dash_no_results_title": "Aún sin resultados",
        "empty_dash_no_results_subtitle": "Tus análisis recientes aparecerán acá",
        "empty_dash_no_results_hint": "Procesá una imagen para empezar",
        "empty_dash_no_dist_title": "Sin datos de distribución",
        "empty_dash_no_dist_subtitle": "Las estadísticas se llenarán a medida",
        "empty_dash_no_dist_hint": "Cada análisis cuenta",
        "empty_image_empty_title": "Ninguna imagen insertada",
        "empty_image_empty_subtitle": "Arrastrá y soltá una imagen acá, o hacé clic en Insertar",
        "empty_image_empty_hint": "Soporta PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Soltá para analizar",
        "empty_image_drop_hint_subtitle": "Soltá la imagen para cargarla",
        "empty_image_drop_hint_hint": "El formato se detectará automáticamente",
        "empty_processing_wait_title": "Procesando...",
        "empty_processing_wait_subtitle": "La IA está analizando tu imagen",
        "empty_processing_wait_hint": "Usualmente toma unos segundos",
        "tour_show_again": "Mostrar tour de nuevo",
        "tour_completed_msg": "¡Tour completado! Estás listo.",
        "tour_skipped_msg": "Tour omitido. Podés repetirlo desde el menú Cuenta.",
        "tour_btn_next": "Siguiente",
        "tour_btn_prev": "Anterior",
        "tour_btn_skip": "Omitir",
        "tour_btn_finish": "Finalizar ✓",
        "tour_guest_login_title": "Iniciar sesión",
        "tour_guest_login_subtitle": "Hacé clic acá para iniciar sesión y desbloquear todas las funciones.",
        "tour_guest_signup_title": "Crear una cuenta",
        "tour_guest_signup_subtitle": "¿Nuevo por acá? Hacé clic acá para crear una cuenta en segundos.",
        "tour_insert_title": "Insertar imagen",
        "tour_insert_subtitle": "Cargá una imagen de ecografía desde el disco, o arrastrá y soltala en cualquier lugar de la ventana.",
        "tour_process_title": "Procesar imagen",
        "tour_process_subtitle": "Ejecutá el análisis de IA en la imagen cargada. Los resultados aparecen en pocos segundos.",
        "tour_export_title": "Exportar PDF",
        "tour_export_subtitle": "Guardá el resultado actual como un informe médico PDF profesional.",
        "tour_gradcam_title": "Mapa de calor de atención",
        "tour_gradcam_subtitle": "Visualizá dónde mira la IA — un mapa de calor sobre la imagen analizada.",
        "tour_dashboard_title": "Panel",
        "tour_dashboard_subtitle": "Mostrar/ocultar el panel de estadísticas: total de imágenes, confianza promedio, distribución por grado.",
        "tour_history_title": "Historial",
        "tour_history_subtitle": "Navegá por tus últimas 20 imágenes analizadas. Hacé clic en una miniatura para recargar el resultado.",
        "tour_visit_title": "Visitar página web",
        "tour_visit_subtitle": "Abrí la página GitHub del proyecto en tu navegador para documentación y actualizaciones.",
        "tour_mute_title": "Silenciar música",
        "tour_mute_subtitle": "Pausá o reanudá la música de fondo en cualquier momento.",
        "tour_zoom_title": "Controles de zoom",
        "tour_zoom_subtitle": "Usá la rueda del mouse para zoom, arrastrá para mover, y doble clic para reiniciar.",
        "tour_account_title": "Menú de la cuenta",
        "tour_account_subtitle": "Hacé clic acá para acceder a Configuración, Información de la cuenta, Registro de auditoría y más.",
        "suggestion_image_ready": "Imagen cargada — lista para procesar",
        "suggestion_image_ready_sub": "Hacé clic en Procesar para ejecutar el análisis de IA.",
        "suggestion_pdf_ready": "Resultado listo — ¿exportar como PDF?",
        "suggestion_pdf_ready_sub": "Guardá un informe médico profesional con un clic.",
        "suggestion_gradcam_hint": "¿Curioso por saber qué vio la IA?",
        "suggestion_gradcam_hint_sub": "Mirá el mapa de calor de atención sobre la imagen.",
        "suggestion_repeated_grid": "Siempre obtenés {grid}",
        "suggestion_repeated_grid_sub": "Abrí el panel para ver la distribución.",
        "suggestion_history_hint": "Tu historial está creciendo",
        "suggestion_history_hint_sub": "Navegá por tus últimas 20 imágenes analizadas.",
        "suggestion_dashboard_hint": "Tu panel tiene estadísticas interesantes",
        "suggestion_dashboard_hint_sub": "Totales, promedios y distribución de un vistazo.",
        "suggestion_settings_hint": "Personalizá la aplicación",
        "suggestion_settings_hint_sub": "Cambiá idioma, tema, tipografía y activá 2FA.",
        "suggestion_idle_help": "¿Necesitás una guía rápida?",
        "suggestion_idle_help_sub": "Repetí el tour de bienvenida cuando quieras.",
        "image_not_liver_warning": "Esta imagen no parece una ecografía del hígado.\n\nPor favor, insertá una imagen de ecografía del hígado adecuada.",
        "result_not_confident": "La IA no pudo identificar con confianza esta imagen como una ecografía del hígado.\n\nPor favor, insertá una imagen de ecografía del hígado adecuada e intentá de nuevo.",
        "not_liver_title": "No es una ecografía del hígado",
        "not_liver_hint": "Por favor, usá una imagen de ecografía (sonograma) del hígado adecuada",
        "voice_welcome": "Bienvenido, {user}",
        "voice_goodbye": "Chau",
        "voice_logged_out": "Sesión cerrada con éxito",
        "voice_image_loaded": "Imagen cargada con éxito",
        "voice_image_rejected": "Esta no es una ecografía del hígado",
        "voice_inference_result": "{grid_name}, confianza {conf} por ciento",
        "voice_pdf_saved": "Informe PDF guardado con éxito",
        "voice_error": "Ocurrió un error",
        "voice_settings_applied": "Configuración aplicada",
        "voice_language_changed": "Idioma cambiado a {lang}",
        "voice_theme_changed": "Tema cambiado",
        "voice_2fa_enabled": "Autenticación de dos factores habilitada",
        "voice_2fa_disabled": "Autenticación de dos factores deshabilitada",
        "voice_locked": "Sesión bloqueada",
        "voice_unlocked": "Sesión desbloqueada",
        "voice_processing": "Procesando imagen",
        "voice_processing_done": "Procesamiento completado",
        "voice_voice_enabled": "Respuesta de voz habilitada",
        "voice_voice_disabled": "Respuesta de voz deshabilitada",
    },
    "Morocco": {
        "app_title": "خط أنابيب الذكاء الاصطناعي للكبد",
        "welcome": "مرحباً",
        "login": "تسجيل الدخول",
        "signup": "التسجيل",
        "logout": "تسجيل الخروج",
        "exit": "خروج",
        "settings": "الإعدادات",
        "account_info": "معلومات الحساب",
        "username": "اسم المستخدم",
        "password": "كلمة المرور",
        "repeat_password": "تكرار كلمة المرور",
        "first_name": "الاسم الأول",
        "last_name": "اسم العائلة",
        "email": "البريد الإلكتروني",
        "remember_me": "تذكرني",
        "login_btn": "تسجيل الدخول",
        "signup_btn": "التسجيل",
        "account": "الحساب",
        "change_language": "تغيير اللغة",
        "change_theme": "تغيير المظهر",
        "insert_sono": "إدراج صورة",
        "process": "معالجة",
        "mute": "كتم الصوت",
        "unmute": "إلغاء الكتم",
        "visit_webpage": "زيارة الموقع",
        "please_wait": "يرجى الانتظار...",
        "processing": "جارٍ معالجة الصورة...",
        "result": "النتيجة",
        "confidence": "الثقة",
        "grid": "الشبكة",
        "insert_image_first": "يرجى إدراج صورة أولاً!",
        "no_image": "لم يتم إدراج صورة",
        "invalid_credentials": "بيانات اعتماد غير صالحة",
        "user_exists": "اسم المستخدم موجود بالفعل",
        "password_requirements": "متطلبات كلمة المرور",
        "password_rule": (
            "يجب أن تكون كلمة المرور:\n"
            "• من 8 إلى 16 حرفاً\n"
            "• تحتوي على حرف صغير واحد على الأقل\n"
            "• تحتوي على حرف كبير واحد على الأقل\n"
            "• تحتوي على 4 أرقام على الأقل\n"
            "• تحتوي على 4 رموز خاصة على الأقل"
        ),
        "passwords_dont_match": "كلمتا المرور غير متطابقتين!",
        "invalid_email": "يرجى إدخال بريد إلكتروني صالح",
        "empty_fields": "يرجى ملء جميع الحقول",
        "signup_success": "تم التسجيل بنجاح! يمكنك الآن تسجيل الدخول.",
        "logout_confirm": "هل أنت متأكد من حذف حسابك؟",
        "exit_confirm": "هل أنت متأكد من الخروج؟",
        "theme_dark_aqua": "مظهر داكن أزرق-بنفسجي",
        "theme_solarized_white": "مظهر فاتح سولاريزد",
        "theme_solarized_dark": "مظهر داكن سولاريزد",
        "theme_dark_forest": "مظهر الغابة الخضراء الداكن",
        "select_language": "اختر اللغة",
        "select_theme": "اختر المظهر",
        "apply": "تطبيق",
        "cancel": "إلغاء",
        "save": "حفظ",
        "success": "نجاح",
        "error": "خطأ",
        "warning": "تحذير",
        "settings_applied": "تم تطبيق الإعدادات!",
        "grid_0": "الشبكة 0 (NL) — كبد طبيعي",
        "grid_1": "الشبكة 1 (G1) — الدرجة 1",
        "grid_2": "الشبكة 2 (G2) — الدرجة 2",
        "grid_3": "الشبكة 3 (G3) — الدرجة 3",
        "too_many_attempts": "محاولات فاشلة كثيرة. يرجى الانتظار 5 دقائق.",
        "invalid_username": "اسم المستخدم يجب أن يكون 3-32 حرفاً (حروف، أرقام، شرطة سفلية).",
        "welcome_user": "مرحباً، {user}!",
        "inference_failed": "فشل الاستدلال",
        "drop_here": "أفلت الصورة هنا",
        "drop_subtitle": "PNG، JPG، JPEG، BMP، DCM",
        "drop_invalid": "نوع الملف غير مدعوم. يرجى إفلات صورة.",
        "image_loaded": "تم تحميل الصورة بنجاح",
        "help_about_title": "المساعدة وحول",
        "help_about_text": (
            "خط أنابيب الذكاء الاصطناعي للكبد الإصدار 3.0\n"
            "تم التطوير بواسطة باباك يوسفيان\n\n"
            "اختصارات لوحة المفاتيح:\n"
            "  Ctrl+O      — إدراج صورة\n"
            "  Ctrl+Enter  — معالجة الصورة\n"
            "  Ctrl+L      — تسجيل الدخول\n"
            "  Ctrl+,      — الإعدادات\n"
            "  Esc         — إغلاق الحوار النشط\n"
            "  F1          — المساعدة / حول\n\n"
            "نصيحة: يمكنك أيضًا سحب وإفلات صورة في أي مكان في النافذة."
        ),
        "dashboard_title": "لوحة التحكم",
        "dash_total": "إجمالي الصور",
        "dash_avg_conf": "متوسط الثقة",
        "dash_distribution": "التوزيع",
        "dash_last_5": "آخر 5 نتائج",
        "dash_no_results": "لا توجد نتائج بعد",
        "dash_clear_confirm": "مسح جميع الإحصائيات؟",
        "dash_cleared": "تم مسح الإحصائيات",
        "dashboard_toggle": "إظهار/إخفاء لوحة التحكم",
        "export_pdf": "تصدير PDF",
        "pdf_saved": "تم حفظ PDF بنجاح",
        "pdf_failed": "فشل إنشاء PDF",
        "pdf_subtitle": "تقرير التحليل",
        "pdf_patient": "المريض:",
        "pdf_date": "التاريخ:",
        "pdf_model": "النموذج:",
        "pdf_diagnosis": "التشخيص",
        "pdf_confidence": "الثقة",
        "pdf_class_probs": "احتمالات الفئات",
        "pdf_disclaimer": (
            "تم تطوير هذا التقرير بواسطة باباك يوسفيان تحت إشراف أطباء ذوي خبرة. "
            "إنها أداة تحليل احترافية ولا تتطلب إحالة إلى طبيب."
        ),
        "history_title": "السجل",
        "history_toggle": "إظهار/إخفاء السجل",
        "history_empty": "لا يوجد سجل بعد",
        "history_cleared": "تم مسح السجل",
        "history_clear_confirm": "مسح كل السجل؟",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "تسجيل الدخول",
        "tooltip_desc_login": "سجل الدخول إلى حسابك",
        "tooltip_tip_login": "الوصول إلى السجل والإحصائيات المحفوظة",
        "tooltip_title_signup": "التسجيل",
        "tooltip_desc_signup": "إنشاء حساب جديد",
        "tooltip_title_account": "الحساب",
        "tooltip_desc_account": "انقر لفتح قائمة الحساب",
        "tooltip_desc_account_user": "مسجل الدخول باسم {user}",
        "tooltip_title_dashboard": "لوحة التحكم",
        "tooltip_desc_dashboard": "إظهار/إخفاء لوحة الإحصائيات — الإجمالي، التوزيع، النتائج الأخيرة",
        "tooltip_title_history": "السجل",
        "tooltip_desc_history": "إظهار/إخفاء آخر 20 نتيجة مع الصور المصغرة",
        "tooltip_title_visit": "زيارة الموقع",
        "tooltip_desc_visit": "فتح صفحة GitHub للمشروع في المتصفح",
        "tooltip_tip_visit": "يفتح في المتصفح الخارجي",
        "tooltip_title_insert": "إدراج صورة",
        "tooltip_desc_insert": "تحميل صورة سونو للتحليل",
        "tooltip_tip_insert": "يمكنك أيضًا سحب وإفلات صورة في أي مكان في النافذة",
        "tooltip_title_process": "معالجة",
        "tooltip_desc_process": "تشغيل تحليل الذكاء الاصطناعي على الصورة المحملة",
        "tooltip_tip_process": "يتطلب تحميل صورة أولاً",
        "tooltip_title_mute": "كتم الموسيقى",
        "tooltip_desc_mute": "إيقاف موسيقى الخلفية مؤقتًا",
        "tooltip_title_unmute": "إلغاء كتم الموسيقى",
        "tooltip_desc_unmute": "استئناف موسيقى الخلفية",
        "tooltip_title_export": "تصدير PDF",
        "tooltip_desc_export": "حفظ النتيجة الحالية كتقرير PDF",
        "tooltip_tip_export": "يشمل الصورة والتشخيص واحتمالات الفئات",
        "tooltip_title_dash_clear": "مسح الإحصائيات",
        "tooltip_desc_dash_clear": "إعادة تعيين جميع الإحصائيات لهذا الحساب",
        "tooltip_tip_dash_clear": "لا يمكن التراجع عن هذا الإجراء",
        "tooltip_title_dash_close": "إغلاق لوحة التحكم",
        "tooltip_desc_dash_close": "إخفاء لوحة الإحصائيات",
        "tooltip_desc_dash_total": "العدد الإجمالي للصور المعالجة",
        "tooltip_desc_dash_avg": "متوسط الثقة عبر جميع النتائج",
        "tooltip_desc_dash_dist": "توزيع النتائج حسب الدرجات",
        "tooltip_title_hist_clear": "مسح السجل",
        "tooltip_desc_hist_clear": "حذف جميع النتائج والصور المصغرة المحفوظة",
        "tooltip_tip_hist_clear": "سيؤدي هذا أيضًا إلى حذف جميع ملفات الصور",
        "tooltip_title_hist_close": "إغلاق السجل",
        "tooltip_desc_hist_close": "إخفاء لوحة السجل",
        "tooltip_tip_thumb_click": "انقر لإعادة تحميل هذه النتيجة",
        "tooltip_tip_recent": "انقر لإعادة تحميل هذه النتيجة",
        "tooltip_title_zoom_in": "تكبير",
        "tooltip_desc_zoom_in": "زيادة تكبير الصورة",
        "tooltip_title_zoom_out": "تصغير",
        "tooltip_desc_zoom_out": "تقليل تكبير الصورة",
        "tooltip_title_zoom_reset": "إعادة تعيين العرض",
        "tooltip_desc_zoom_reset": "ملاءمة الصورة مع النافذة",
        "tooltip_tip_zoom_reset": "يمكنك أيضًا النقر المزدوج على الصورة",
        "notif_inference_ok": "اكتمل الاستدلال",
        "follow_system_theme": "متابعة سمة النظام",
        "follow_system_theme_desc": "التبديل تلقائيًا بين الفاتح والداكن بناءً على إعدادات النظام",
        "gradcam_title": "خريطة الانتباه الحرارية",
        "gradcam_show": "إظهار خريطة الانتباه",
        "gradcam_toggle": "تبديل الخريطة",
        "gradcam_save": "حفظ الصورة",
        "gradcam_close": "إغلاق",
        "gradcam_saved": "تم حفظ الصورة بنجاح",
        "gradcam_no_heatmap": "لا توجد خريطة انتباه متاحة. يرجى معالجة صورة أولاً.",
        "update_available": "تحديث متوفر",
        "update_check_failed": "فشل التحقق من التحديثات",
        "check_for_updates": "التحقق من التحديثات",
        "check_for_updates_desc": "التحقق من GitHub/الموقع للحصول على آخر إصدار",
        "up_to_date": "أنت محدَّث",
        "you_have_latest": "أنت تستعمل آخر إصدار.",
        "check_internet": "المرجو التحقق من اتصالك بالإنترنت والمحاولة مرة أخرى.",
        "downloading": "جارٍ التحميل...",
        "verifying": "جارٍ التحقق...",
        "download_now": "حمّل الآن",
        "install_now": "ثبّت وأعد التشغيل",
        "skip_version": "تخطَّ هذا الإصدار",
        "remind_later": "ذكّرني لاحقاً",
        "whats_new": "ما الجديد",
        "no_changelog": "لم يتم توفير سجل التغييرات.",
        "critical_update": "تحديث حرج",
        "update_ready": "جاهز للتثبيت",
        "app_will_restart": "سيتم إغلاق التطبيق وإعادة تشغيله لتطبيق التحديث.",
        "install_failed": "فشل التثبيت. إصدارك القديم آمن.",
        "download_failed": "فشل التحميل.",
        "source_update_requires_zip": "تحديث المصدر يتطلب ملف .zip.",
        "session_locked": "الجلسة مقفلة",
        "locked_for_user": "مقفلة لـ {user}",
        "unlock": "إلغاء القفل",
        "session_unlocked": "تم إلغاء قفل الجلسة",
        "auto_lock_notice": "سيتم قفل الجلسة بعد دقيقتين من عدم النشاط.",
        "audit_log_title": "سجل التدقيق",
        "audit_export": "تصدير",
        "audit_exported": "تم التصدير بنجاح",
        "audit_export_failed": "فشل التصدير",
        "audit_search_placeholder": "البحث في الإجراءات أو التفاصيل...",
        "audit_col_time": "الوقت",
        "audit_col_level": "المستوى",
        "audit_col_category": "الفئة",
        "audit_col_action": "الإجراء",
        "audit_col_user": "المستخدم",
        "audit_details_title": "تفاصيل الحدث",
        "audit_status": "عرض {count} من {total} حدثاً",
        "strength_very_weak": "ضعيف بزاف",
        "strength_weak": "ضعيف",
        "strength_medium": "متوسط",
        "strength_strong": "قوي",
        "strength_very_strong": "قوي بزاف",
        "strength_rule_length": "8-16 حرف",
        "strength_rule_lowercase": "حرف صغير",
        "strength_rule_uppercase": "حرف كبير",
        "strength_rule_digits": "أرقام",
        "strength_rule_special": "رموز خاصة",
        "strength_add_chars": "زيد {n} حرف آخر",
        "strength_remove_chars": "نقص {n} حرف",
        "strength_need_digits": "زيد {n} رقم آخر",
        "strength_need_special": "زيد {n} رمز خاص آخر",
        "strength_common_pattern": "تجنب الكلمات الشائعة",
        "strength_common_year": "تجنب السنوات الشائعة",
        "strength_has_sequence": "تجنب التسلسلات (1234، abcd)",
        "strength_has_repeats": "تجنب التكرارات (aaaa، 1111)",
        "passwords_match": "كلمتي السر متطابقتين",
        "2fa_setup_title": "تفعيل المصادقة الثنائية",
        "2fa_setup_subtitle": "امسح رمز QR باستخدام Google Authenticator أو Authy أو Microsoft Authenticator.",
        "2fa_manual_secret": "أو أدخل هذا الرمز يدويًا:",
        "2fa_enter_code": "دخّل الرمز المكوّن من 6 أرقام من التطبيق ديالك:",
        "2fa_verify_btn": "تحقق",
        "2fa_backup_codes_title": "حتافظ على رموز الاحتياط هادو!",
        "2fa_backup_codes_hint": "كل رمز يمكن استعمالو غير مرة وحدة. حطهم فبلاصة آمنة.",
        "2fa_done_btn": "تم",
        "2fa_secret_copied": "الرمز السري تنسخ للحافظة",
        "2fa_invalid_code": "رمز غير صالح. عاود المحاولة.",
        "2fa_invalid_code_format": "الرمز خاصو يكون 6 أرقام.",
        "2fa_verify_title": "المصادقة الثنائية",
        "2fa_verify_subtitle": "دخّل الرمز المكوّن من 6 أرقام من تطبيق المصادقة.",
        "2fa_enter_code_hint": "عافاك دخّل رمز.",
        "2fa_use_backup_code": "استعمل رمز احتياطي",
        "2fa_use_totp_code": "استعمل رمز التطبيق",
        "2fa_enter_backup_code": "دخّل واحد من رموز الاحتياط ديالك.",
        "2fa_section_title": "الأمان",
        "2fa_enabled_status": "المصادقة الثنائية مفعّلة",
        "2fa_disabled_status": "المصادقة الثنائية معطّلة",
        "2fa_enable_btn": "فعّل 2FA",
        "2fa_disable_btn": "عطّل 2FA",
        "2fa_disable_title": "عطّل 2FA",
        "2fa_disable_confirm": "واش متأكد؟ الحساب ديالك غادي يولي أقل أمان.",
        "2fa_disabled_success": "المصادقة الثنائية تعطّلت",
        "2fa_enabled_success": "المصادقة الثنائية تفعّلت",
        "2fa_hint_title": "كيفاش تستعمل هاد رمز QR",
        "2fa_hint_step1": "ركّب تطبيق مصادقة: Google Authenticator، Microsoft Authenticator، ولا Authy.",
        "2fa_hint_step2": "حل التطبيق، ضغط على زر \"+\"، ومن بعد اختار \"مسح رمز QR\".",
        "2fa_hint_step3": "وجّه كاميرا التيليفون ديالك لرمز QR اللي فوق (استعمل كاميرا التطبيق، ماشي كاميرا التيليفون).",
        "2fa_hint_step4": "التطبيق غادي يوري ليك رمز فيه 6 أرقام. دخّلو فالمربع اللي تحت.",
        "2fa_manual_secret_hint": "ماقدرتيش تسكاني؟ كوبي المفتاح اللي تحت وزيدو يدوياً فالتطبيق ديالك (\"إدخال مفتاح الإعداد\").",
        "auto_lock_section_title": "القفل التلقائي",
        "auto_lock_enable": "قفل من بعد عدم النشاط",
        "auto_lock_timeout_label": "قفل من بعد:",
        "auto_lock_warning_title": "تحذير القفل التلقائي",
        "auto_lock_warning_body": "الجلسة ديالك غادي تتقفل فـ {seconds} ثانية حيت مامكاينش نشاط.",
        "lock_now": "قفل دابا",
        "font_section_title": "الخط",
        "font_family_label": "نوع الخط",
        "font_size_label": "الحجم الأساسي",
        "font_scale_label": "مقياس العرض",
        "font_scale_small": "صغير",
        "font_scale_normal": "عادي",
        "font_scale_large": "كبير",
        "font_scale_xlarge": "كبير بزاف",
        "font_scale_xxlarge": "ضخم",
        "font_preview_text": "هاد النص نموذجي باش تشوف الخط — 1234567890",
        "font_reset": "رجع للإعدادات الأصلية",
        "font_applied": "الخط تبدل",
        "motion_section_title": "الحركة والأنيماسيون",
        "reduce_motion_label": "قلل الحركة",
        "reduce_motion_hint": "طفي الأنيماسيون إلا كان كيضايقك.",
        "empty_history_empty_title": "مازال ماكاينش سجل",
        "empty_history_empty_subtitle": "عالج أول صورة باش تشوفها هنا",
        "empty_history_empty_hint": "كل نتيجة كتسجل أوتوماتيكياً",
        "empty_history_no_user_title": "دخل باش تشوف السجل",
        "empty_history_no_user_subtitle": "النتايج المحفوظة ديالك غادي تبان هنا",
        "empty_history_no_user_hint": "السجل خاص بكل حساب",
        "empty_dash_no_results_title": "مازال ماكاينش نتايج",
        "empty_dash_no_results_subtitle": "التحليلات الأخيرة ديالك غادي تبان هنا",
        "empty_dash_no_results_hint": "عالج صورة باش تبدا",
        "empty_dash_no_dist_title": "ماكاينش بيانات التوزيع",
        "empty_dash_no_dist_subtitle": "الإحصائيات غادي تعمر معالجة الصور",
        "empty_dash_no_dist_hint": "كل تحليل مهم",
        "empty_image_empty_title": "ماكاينش صورة مدخلة",
        "empty_image_empty_subtitle": "سحب وحط صورة هنا، ولا كليكي على إدراج",
        "empty_image_empty_hint": "كيدعم PNG، JPG، JPEG، BMP، DCM",
        "empty_image_drop_hint_title": "حط باش نحلل",
        "empty_image_drop_hint_subtitle": "خلي الصورة باش تتحمل",
        "empty_image_drop_hint_hint": "الصيغة غادي تتعرف أوتوماتيكياً",
        "empty_processing_wait_title": "كيتم المعالجة...",
        "empty_processing_wait_subtitle": "الذكاء الاصطناعي كيحلل الصورة ديالك",
        "empty_processing_wait_hint": "عادة كياخد شي ثواني",
        "tour_show_again": "بين الجولة مرة أخرى",
        "tour_completed_msg": "الجولة كملات! راك واجد.",
        "tour_skipped_msg": "الجولة تخطات. تقدر تعاودها من قائمة الحساب.",
        "tour_btn_next": "التالي",
        "tour_btn_prev": "السابق",
        "tour_btn_skip": "تخطي",
        "tour_btn_finish": "سالينا ✓",
        "tour_guest_login_title": "دخول",
        "tour_guest_login_subtitle": "كليكي هنا باش تدخل لحسابك وتفتح جميع الميزات.",
        "tour_guest_signup_title": "دير حساب",
        "tour_guest_signup_subtitle": "جديد هنا؟ كليكي هنا باش تدير حساب فشي ثواني.",
        "tour_insert_title": "حط صورة",
        "tour_insert_subtitle": "حمّل صورة سونو من الديسك، ولا سحبها وحطها فأي بلاصة فالنافذة.",
        "tour_process_title": "عالج الصورة",
        "tour_process_subtitle": "شغّل تحليل الذكاء الاصطناعي على الصورة المحملة. النتايج كتبان فشي ثواني.",
        "tour_export_title": "صدّر PDF",
        "tour_export_subtitle": "سجل النتيجة الحالية كتقرير طبي PDF احترافي.",
        "tour_gradcam_title": "خريطة الانتباه الحرارية",
        "tour_gradcam_subtitle": "شوف فين كيشوف الذكاء الاصطناعي — خريطة حرارية فوق الصورة المحللة.",
        "tour_dashboard_title": "لوحة التحكم",
        "tour_dashboard_subtitle": "بين/خبي لوحة الإحصائيات: مجموع الصور، متوسط الثقة، توزيع الدرجات.",
        "tour_history_title": "السجل",
        "tour_history_subtitle": "تصفح آخر 20 صورة محللة. كليكي على أي صورة مصغرة باش تعاود تحمل نتيجتها.",
        "tour_visit_title": "زيارة الموقع",
        "tour_visit_subtitle": "حل صفحة GitHub ديال المشروع فالمتصفح باش تشوف الوثائق والتحديثات.",
        "tour_mute_title": "كتم الموسيقى",
        "tour_mute_subtitle": "وقّف ولا كمّل موسيقى الخلفية فأي وقت.",
        "tour_zoom_title": "التحكم فالتكبير",
        "tour_zoom_subtitle": "استعمل عجلة الماوس للتكبير، سحب للتحريك، ونقرة مزدوجة لإعادة الضبط.",
        "tour_account_title": "قائمة الحساب",
        "tour_account_subtitle": "كليكي هنا باش توصل للإعدادات، معلومات الحساب، سجل التدقيق، والمزيد.",
        "suggestion_image_ready": "الصورة تحمّلات — واجدة للمعالجة",
        "suggestion_image_ready_sub": "كليكي على «معالجة» باش يبدا تحليل الذكاء الاصطناعي.",
        "suggestion_pdf_ready": "النتيجة واجدة — تصدير ف PDF؟",
        "suggestion_pdf_ready_sub": "سجل تقرير طبي احترافي بنقرة وحدة.",
        "suggestion_gradcam_hint": "واش عندك الفضول تعرف فين شاف الذكاء الاصطناعي؟",
        "suggestion_gradcam_hint_sub": "شوف خريطة الانتباه الحرارية فوق الصورة.",
        "suggestion_repeated_grid": "دائماً كتاخد {grid}",
        "suggestion_repeated_grid_sub": "حل لوحة التحكم باش تشوف التوزيع.",
        "suggestion_history_hint": "السجل ديالك كيتعمر",
        "suggestion_history_hint_sub": "تصفح آخر 20 صورة محللة.",
        "suggestion_dashboard_hint": "لوحة التحكم ديالك فيها إحصائيات مثيرة",
        "suggestion_dashboard_hint_sub": "المجموع والمتوسط والتوزيع بنظرة وحدة.",
        "suggestion_settings_hint": "خصص التطبيق",
        "suggestion_settings_hint_sub": "بدل اللغة، المظهر، الخط وفعّل 2FA.",
        "suggestion_idle_help": "محتاج دليل سريع؟",
        "suggestion_idle_help_sub": "عاود الجولة التعريفية فأي وقت.",
        "image_not_liver_warning": "هاد الصورة ماكتبانش بحال صورة إيكوغرافي ديال الكبد.\n\nعافاك دخّل صورة إيكوغرافي ديال الكبد صحيحة.",
        "result_not_confident": "الذكاء الاصطناعي ماقدرش يتعرف على هاد الصورة بثقة كصورة إيكوغرافي ديال الكبد.\n\nعافاك دخّل صورة إيكوغرافي ديال الكبد صحيحة وعاود المحاولة.",
        "not_liver_title": "ماشي إيكوغرافي ديال الكبد",
        "not_liver_hint": "عافاك استعمل صورة إيكوغرافي (سونوغرام) ديال الكبد صحيحة",
        "voice_welcome": "مرحبا بيك، {user}",
        "voice_goodbye": "بسلامة",
        "voice_logged_out": "خرجتي بنجاح",
        "voice_image_loaded": "الصورة تحمّلات بنجاح",
        "voice_image_rejected": "هاد الصورة ماشي إيكوغرافي ديال الكبد",
        "voice_inference_result": "{grid_name}، الثقة {conf} بالمئة",
        "voice_pdf_saved": "تقرير PDF تسجل بنجاح",
        "voice_error": "وقع شي خطأ",
        "voice_settings_applied": "الإعدادات تطبقات",
        "voice_language_changed": "اللغة تبدلات لـ {lang}",
        "voice_theme_changed": "المظهر تبدل",
        "voice_2fa_enabled": "المصادقة الثنائية تفعّلت",
        "voice_2fa_disabled": "المصادقة الثنائية تعطّلت",
        "voice_locked": "الجلسة تقفلت",
        "voice_unlocked": "الجلسة تحلّت",
        "voice_processing": "كنعالجو الصورة",
        "voice_processing_done": "المعالجة كملات",
        "voice_voice_enabled": "الصوت تفعّل",
        "voice_voice_disabled": "الصوت تعطّل",
    },
    "Turkey": {
        "app_title": "Karaciğer AI-babak",
        "welcome": "Hoş geldiniz",
        "login": "Giriş Yap",
        "signup": "Kayıt Ol",
        "logout": "Çıkış Yap",
        "exit": "Çıkış",
        "settings": "Ayarlar",
        "account_info": "Hesap Bilgileri",
        "username": "Kullanıcı Adı",
        "password": "Şifre",
        "repeat_password": "Şifreyi Tekrarla",
        "first_name": "Ad",
        "last_name": "Soyad",
        "email": "E-posta Adresi",
        "remember_me": "Beni Hatırla",
        "login_btn": "Giriş Yap",
        "signup_btn": "Kayıt Ol",
        "account": "Hesap",
        "change_language": "Dili Değiştir",
        "change_theme": "Temayı Değiştir",
        "insert_sono": "Sono Ekle",
        "process": "İşle",
        "mute": "Sessize Al",
        "unmute": "Sesi Aç",
        "visit_webpage": "Web Sayfasını Ziyaret Et",
        "please_wait": "Lütfen bekleyin...",
        "processing": "Görüntü işleniyor...",
        "result": "Sonuç",
        "confidence": "Güven",
        "grid": "Izgara",
        "insert_image_first": "Lütfen önce bir görüntü ekleyin!",
        "no_image": "Görüntü eklenmedi",
        "invalid_credentials": "Geçersiz kimlik bilgileri",
        "user_exists": "Kullanıcı adı zaten mevcut",
        "password_requirements": "Şifre gereksinimleri",
        "password_rule": (
            "Şifre şunları içermelidir:\n"
            "• 8-16 karakter uzunluğunda\n"
            "• En az 1 küçük harf\n"
            "• En az 1 büyük harf\n"
            "• En az 4 rakam\n"
            "• En az 4 özel karakter"
        ),
        "passwords_dont_match": "Şifreler eşleşmiyor!",
        "invalid_email": "Lütfen geçerli bir e-posta adresi girin",
        "empty_fields": "Lütfen tüm alanları doldurun",
        "signup_success": "Kayıt başarılı! Artık giriş yapabilirsiniz.",
        "logout_confirm": "Hesabınızı silmek istediğinizden emin misiniz?",
        "exit_confirm": "Çıkmak istediğinizden emin misiniz?",
        "theme_dark_aqua": "Koyu Aqua-Mor Tema",
        "theme_solarized_white": "Solarized Açık Tema",
        "theme_solarized_dark": "Solarized Koyu Tema",
        "theme_dark_forest": "Koyu Orman Yeşili Tema",
        "select_language": "Dil Seç",
        "select_theme": "Tema Seç",
        "apply": "Uygula",
        "cancel": "İptal",
        "save": "Kaydet",
        "success": "Başarılı",
        "error": "Hata",
        "warning": "Uyarı",
        "settings_applied": "Ayarlar uygulandı!",
        "grid_0": "Izgara 0 (NL) — Normal Karaciğer",
        "grid_1": "Izgara 1 (G1) — Derece 1",
        "grid_2": "Izgara 2 (G2) — Derece 2",
        "grid_3": "Izgara 3 (G3) — Derece 3",
        "too_many_attempts": "Çok fazla başarısız deneme. Lütfen 5 dakika bekleyin.",
        "invalid_username": "Kullanıcı adı 3-32 karakter olmalıdır (harfler, rakamlar, alt çizgi).",
        "welcome_user": "Hoş geldiniz, {user}!",
        "inference_failed": "Çıkarım başarısız",
        "drop_here": "Görüntüyü buraya bırakın",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Desteklenmeyen dosya türü. Lütfen bir görüntü bırakın.",
        "image_loaded": "Görüntü başarıyla yüklendi",
        "help_about_title": "Yardım & Hakkında",
        "help_about_text": (
            "Karaciğer AI Boru Hattı v3.0\n"
            "Geliştiren: Babak Yousefian\n\n"
            "Klavye Kısayolları:\n"
            "  Ctrl+O      — Görüntü Ekle\n"
            "  Ctrl+Enter  — Görüntüyü İşle\n"
            "  Ctrl+L      — Giriş Yap\n"
            "  Ctrl+,      — Ayarlar\n"
            "  Esc         — Aktif pencereyi kapat\n"
            "  F1          — Yardım / Hakkında\n\n"
            "İpucu: Bir görüntüyü pencerenin herhangi bir yerine sürükleyip bırakabilirsiniz."
        ),
        "dashboard_title": "Panel",
        "dash_total": "Toplam Görüntü",
        "dash_avg_conf": "Ort. Güven",
        "dash_distribution": "Dağılım",
        "dash_last_5": "Son 5 Sonuç",
        "dash_no_results": "Henüz sonuç yok",
        "dash_clear_confirm": "Tüm istatistikler silinsin mi?",
        "dash_cleared": "İstatistikler silindi",
        "dashboard_toggle": "Paneli Göster/Gizle",
        "export_pdf": "PDF Dışa Aktar",
        "pdf_saved": "PDF başarıyla kaydedildi",
        "pdf_failed": "PDF oluşturma başarısız",
        "pdf_subtitle": "Analiz Raporu",
        "pdf_patient": "Hasta:",
        "pdf_date": "Tarih:",
        "pdf_model": "Model:",
        "pdf_diagnosis": "TEŞHİS",
        "pdf_confidence": "Güven",
        "pdf_class_probs": "SINIF OLASILIKLARI",
        "pdf_disclaimer": (
            "Bu rapor, deneyimli doktorların gözetimi altında Babak Yousefian tarafından geliştirilmiştir. "
            "Profesyonel bir analiz aracıdır ve bir doktora sevk gerektirmez."
        ),
        "history_title": "Geçmiş",
        "history_toggle": "Geçmişi Göster/Gizle",
        "history_empty": "Henüz geçmiş yok",
        "history_cleared": "Geçmiş silindi",
        "history_clear_confirm": "Tüm geçmiş silinsin mi?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Giriş Yap",
        "tooltip_desc_login": "Hesabınıza giriş yapın",
        "tooltip_tip_login": "Kayıtlı geçmişinize ve istatistiklerinize erişin",
        "tooltip_title_signup": "Kayıt Ol",
        "tooltip_desc_signup": "Yeni bir hesap oluşturun",
        "tooltip_title_account": "Hesap",
        "tooltip_desc_account": "Hesap menüsünü açmak için tıklayın",
        "tooltip_desc_account_user": "{user} olarak giriş yapıldı",
        "tooltip_title_dashboard": "Panel",
        "tooltip_desc_dashboard": "İstatistik panelini göster/gizle — toplam, dağılım, son sonuçlar",
        "tooltip_title_history": "Geçmiş",
        "tooltip_desc_history": "Son 20 sonucu küçük resimlerle göster/gizle",
        "tooltip_title_visit": "Web Sayfasını Ziyaret Et",
        "tooltip_desc_visit": "Projenin GitHub sayfasını tarayıcıda aç",
        "tooltip_tip_visit": "Harici tarayıcıda açılır",
        "tooltip_title_insert": "Görüntü Ekle",
        "tooltip_desc_insert": "Analiz etmek için bir sono görüntüsü yükleyin",
        "tooltip_tip_insert": "Ayrıca pencereye herhangi bir yere görüntü sürükleyip bırakabilirsiniz",
        "tooltip_title_process": "İşle",
        "tooltip_desc_process": "Yüklenen görüntü üzerinde AI analizi çalıştır",
        "tooltip_tip_process": "Önce bir görüntü yüklenmelidir",
        "tooltip_title_mute": "Müziği Sessize Al",
        "tooltip_desc_mute": "Arka plan müziğini duraklat",
        "tooltip_title_unmute": "Müziğin Sesini Aç",
        "tooltip_desc_unmute": "Arka plan müziğine devam et",
        "tooltip_title_export": "PDF Dışa Aktar",
        "tooltip_desc_export": "Mevcut sonucu PDF raporu olarak kaydet",
        "tooltip_tip_export": "Görüntü, teşhis ve sınıf olasılıklarını içerir",
        "tooltip_title_dash_clear": "İstatistikleri Temizle",
        "tooltip_desc_dash_clear": "Bu hesabın tüm istatistiklerini sıfırla",
        "tooltip_tip_dash_clear": "Bu işlem geri alınamaz",
        "tooltip_title_dash_close": "Paneli Kapat",
        "tooltip_desc_dash_close": "İstatistik panelini gizle",
        "tooltip_desc_dash_total": "İşlenen toplam görüntü sayısı",
        "tooltip_desc_dash_avg": "Tüm sonuçlar üzerindeki ortalama güven",
        "tooltip_desc_dash_dist": "Derecelere göre sonuç dağılımı",
        "tooltip_title_hist_clear": "Geçmişi Temizle",
        "tooltip_desc_hist_clear": "Kayıtlı tüm sonuçları ve küçük resimleri sil",
        "tooltip_tip_hist_clear": "Bu, tüm görüntü dosyalarını da silecek",
        "tooltip_title_hist_close": "Geçmişi Kapat",
        "tooltip_desc_hist_close": "Geçmiş panelini gizle",
        "tooltip_tip_thumb_click": "Bu sonucu yeniden yüklemek için tıklayın",
        "tooltip_tip_recent": "Bu sonucu yeniden yüklemek için tıklayın",
        "tooltip_title_zoom_in": "Yakınlaştır",
        "tooltip_desc_zoom_in": "Görüntü büyütmesini artır",
        "tooltip_title_zoom_out": "Uzaklaştır",
        "tooltip_desc_zoom_out": "Görüntü büyütmesini azalt",
        "tooltip_title_zoom_reset": "Görünümü Sıfırla",
        "tooltip_desc_zoom_reset": "Görüntüyü pencereye sığdır",
        "tooltip_tip_zoom_reset": "Görüntüye çift tıklayabilirsiniz",
        "notif_inference_ok": "Çıkarım tamamlandı",
        "follow_system_theme": "Sistem temasını takip et",
        "follow_system_theme_desc": "Sistem ayarlarına göre açık ve koyu arasında otomatik olarak geçiş yap",
        "gradcam_title": "Dikkat Isı Haritası",
        "gradcam_show": "Dikkat Haritasını Göster",
        "gradcam_toggle": "Haritayı Değiştir",
        "gradcam_save": "Görüntüyü Kaydet",
        "gradcam_close": "Kapat",
        "gradcam_saved": "Görüntü başarıyla kaydedildi",
        "gradcam_no_heatmap": "Kullanılabilir dikkat haritası yok. Lütfen önce bir görüntü işleyin.",
        "update_available": "Güncelleme mevcut",
        "update_check_failed": "Güncelleme denetimi başarısız",
        "check_for_updates": "Güncellemeleri denetle",
        "check_for_updates_desc": "En son sürüm için GitHub/web sitesini denetle",
        "up_to_date": "Güncelsiniz",
        "you_have_latest": "En son sürümü çalıştırıyorsunuz.",
        "check_internet": "Lütfen internet bağlantınızı kontrol edip tekrar deneyin.",
        "downloading": "İndiriliyor...",
        "verifying": "Doğrulanıyor...",
        "download_now": "Şimdi indir",
        "install_now": "Yükle ve yeniden başlat",
        "skip_version": "Bu sürümü atla",
        "remind_later": "Daha sonra hatırlat",
        "whats_new": "Yenilikler",
        "no_changelog": "Değişiklik günlüğü sağlanmadı.",
        "critical_update": "Kritik güncelleme",
        "update_ready": "Yüklemeye hazır",
        "app_will_restart": "Güncellemeyi uygulamak için uygulama kapanıp yeniden başlatılacak.",
        "install_failed": "Yükleme başarısız. Eski sürümünüz güvende.",
        "download_failed": "İndirme başarısız.",
        "source_update_requires_zip": "Kaynak güncellemesi bir .zip dosyası gerektirir.",
        "session_locked": "Oturum Kilitli",
        "locked_for_user": "{user} için kilitli",
        "unlock": "Kilidi Aç",
        "session_unlocked": "Oturum kilidi açıldı",
        "auto_lock_notice": "Oturum 2 dakika hareketsizlikten sonra kilitlenecek.",
        "audit_log_title": "Denetim Günlüğü",
        "audit_export": "Dışa Aktar",
        "audit_exported": "Başarıyla dışa aktarıldı",
        "audit_export_failed": "Dışa aktarma başarısız",
        "audit_search_placeholder": "Eylemleri veya ayrıntıları ara...",
        "audit_col_time": "Zaman",
        "audit_col_level": "Seviye",
        "audit_col_category": "Kategori",
        "audit_col_action": "Eylem",
        "audit_col_user": "Kullanıcı",
        "audit_details_title": "Olay Ayrıntıları",
        "audit_status": "{total} olaydan {count} tanesi gösteriliyor",
        "strength_very_weak": "Çok zayıf",
        "strength_weak": "Zayıf",
        "strength_medium": "Orta",
        "strength_strong": "Güçlü",
        "strength_very_strong": "Çok güçlü",
        "strength_rule_length": "8-16 karakter",
        "strength_rule_lowercase": "Küçük harf",
        "strength_rule_uppercase": "Büyük harf",
        "strength_rule_digits": "Rakamlar",
        "strength_rule_special": "Özel karakterler",
        "strength_add_chars": "{n} karakter daha ekleyin",
        "strength_remove_chars": "{n} karakter kaldırın",
        "strength_need_digits": "{n} rakam daha ekleyin",
        "strength_need_special": "{n} özel karakter daha ekleyin",
        "strength_common_pattern": "Yaygın kelimelerden kaçının",
        "strength_common_year": "Yaygın yıllardan kaçının",
        "strength_has_sequence": "Dizilerden (1234, abcd) kaçının",
        "strength_has_repeats": "Tekrarlardan (aaaa, 1111) kaçının",
        "passwords_match": "Şifreler eşleşiyor",
        "2fa_setup_title": "İki Faktörlü Kimlik Doğrulamayı Etkinleştir",
        "2fa_setup_subtitle": "QR kodunu Google Authenticator, Authy veya Microsoft Authenticator ile tarayın.",
        "2fa_manual_secret": "Veya bu kodu manuel olarak girin:",
        "2fa_enter_code": "Uygulamanızdaki 6 haneli kodu girin:",
        "2fa_verify_btn": "Doğrula",
        "2fa_backup_codes_title": "Bu yedek kodları kaydedin!",
        "2fa_backup_codes_hint": "Her kod yalnızca bir kez kullanılabilir. Güvenli bir yerde saklayın.",
        "2fa_done_btn": "Tamam",
        "2fa_secret_copied": "Gizli kod panoya kopyalandı",
        "2fa_invalid_code": "Geçersiz kod. Lütfen tekrar deneyin.",
        "2fa_invalid_code_format": "Kod 6 haneli olmalıdır.",
        "2fa_verify_title": "İki Faktörlü Kimlik Doğrulama",
        "2fa_verify_subtitle": "Kimlik doğrulayıcı uygulamanızdaki 6 haneli kodu girin.",
        "2fa_enter_code_hint": "Lütfen bir kod girin.",
        "2fa_use_backup_code": "Yedek kod kullan",
        "2fa_use_totp_code": "Uygulama kodunu kullan",
        "2fa_enter_backup_code": "Yedek kodlarınızdan birini girin.",
        "2fa_section_title": "Güvenlik",
        "2fa_enabled_status": "İki Faktörlü Kimlik Doğrulama etkin",
        "2fa_disabled_status": "İki Faktörlü Kimlik Doğrulama devre dışı",
        "2fa_enable_btn": "2FA'yı Etkinleştir",
        "2fa_disable_btn": "2FA'yı Devre Dışı Bırak",
        "2fa_disable_title": "2FA'yı Devre Dışı Bırak",
        "2fa_disable_confirm": "Emin misiniz? Hesabınız daha az güvenli olacak.",
        "2fa_disabled_success": "İki Faktörlü Kimlik Doğrulama devre dışı bırakıldı",
        "2fa_enabled_success": "İki Faktörlü Kimlik Doğrulama etkinleştirildi",
        "2fa_hint_title": "Bu QR kodunu nasıl kullanırım",
        "2fa_hint_step1": "Bir kimlik doğrulama uygulaması yükleyin: Google Authenticator, Microsoft Authenticator veya Authy.",
        "2fa_hint_step2": "Uygulamayı açın, \"+\" düğmesine dokunun ve \"QR kodunu tara\"yı seçin.",
        "2fa_hint_step3": "Telefonunuzun kamerasını yukarıdaki QR koduna doğrultun (telefonunuzun kamera uygulamasını değil, uygulamanın kamerasını kullanın).",
        "2fa_hint_step4": "Uygulama 6 haneli bir kod gösterecektir. Aşağıdaki kutuya girin.",
        "2fa_manual_secret_hint": "Tarayamıyor musunuz? Aşağıdaki anahtarı kopyalayın ve uygulamanıza manuel olarak ekleyin (\"Kurulum anahtarını gir\").",
        "auto_lock_section_title": "Otomatik Kilit",
        "auto_lock_enable": "Hareketsizlik sonrası kilitle",
        "auto_lock_timeout_label": "Kilitleme süresi:",
        "auto_lock_warning_title": "Otomatik Kilit Uyarısı",
        "auto_lock_warning_body": "Hareketsizlik nedeniyle oturumunuz {seconds} saniye içinde kilitlenecek.",
        "lock_now": "Şimdi Kilitle",
        "font_section_title": "Yazı Tipi",
        "font_family_label": "Yazı tipi ailesi",
        "font_size_label": "Temel boyut",
        "font_scale_label": "Görüntüleme ölçeği",
        "font_scale_small": "Küçük",
        "font_scale_normal": "Normal",
        "font_scale_large": "Büyük",
        "font_scale_xlarge": "Çok büyük",
        "font_scale_xxlarge": "Devasa",
        "font_preview_text": "Pijamalı hasta yağız şoföre çabucak güvendi — 1234567890",
        "font_reset": "Varsayılana sıfırla",
        "font_applied": "Yazı tipi güncellendi",
        "motion_section_title": "Hareket ve Animasyonlar",
        "reduce_motion_label": "Hareketi azalt",
        "reduce_motion_hint": "Animasyonlar sizi rahatsız ediyorsa devre dışı bırakın.",
        "empty_history_empty_title": "Henüz geçmiş yok",
        "empty_history_empty_subtitle": "Burada görmek için ilk görüntünüzü işleyin",
        "empty_history_empty_hint": "Her sonuç otomatik olarak kaydedilir",
        "empty_history_no_user_title": "Geçmişi görmek için giriş yapın",
        "empty_history_no_user_subtitle": "Kaydedilen sonuçlarınız burada görünecek",
        "empty_history_no_user_hint": "Geçmiş her hesap için özeldir",
        "empty_dash_no_results_title": "Henüz sonuç yok",
        "empty_dash_no_results_subtitle": "Son analizleriniz burada görünecek",
        "empty_dash_no_results_hint": "Başlamak için bir görüntü işleyin",
        "empty_dash_no_dist_title": "Dağılım verisi yok",
        "empty_dash_no_dist_subtitle": "Görüntüleri işledikçe istatistikler dolacak",
        "empty_dash_no_dist_hint": "Her analiz önemlidir",
        "empty_image_empty_title": "Görüntü eklenmedi",
        "empty_image_empty_subtitle": "Bir görüntüyü buraya sürükleyin veya Ekle'ye tıklayın",
        "empty_image_empty_hint": "PNG, JPG, JPEG, BMP, DCM destekler",
        "empty_image_drop_hint_title": "Analiz için bırakın",
        "empty_image_drop_hint_subtitle": "Yüklemek için görüntüyü bırakın",
        "empty_image_drop_hint_hint": "Format otomatik olarak algılanır",
        "empty_processing_wait_title": "İşleniyor...",
        "empty_processing_wait_subtitle": "AI görüntünüzü analiz ediyor",
        "empty_processing_wait_hint": "Genellikle birkaç saniye sürer",
        "tour_show_again": "Turu tekrar göster",
        "tour_completed_msg": "Tur tamamlandı! Hazırsınız.",
        "tour_skipped_msg": "Tur atlandı. Hesap menüsünden tekrar izleyebilirsiniz.",
        "tour_btn_next": "İleri",
        "tour_btn_prev": "Geri",
        "tour_btn_skip": "Atla",
        "tour_btn_finish": "Bitir ✓",
        "tour_guest_login_title": "Giriş Yap",
        "tour_guest_login_subtitle": "Hesabınıza giriş yapmak ve tüm özellikleri açmak için buraya tıklayın.",
        "tour_guest_signup_title": "Hesap Oluştur",
        "tour_guest_signup_subtitle": "Yeni misiniz? Saniyeler içinde hesap oluşturmak için buraya tıklayın.",
        "tour_insert_title": "Görüntü Ekle",
        "tour_insert_subtitle": "Diskten bir sono görüntüsü yükleyin veya pencereye herhangi bir yere sürükleyip bırakın.",
        "tour_process_title": "Görüntüyü İşle",
        "tour_process_subtitle": "Yüklenen görüntü üzerinde AI analizini çalıştırın. Sonuçlar birkaç saniye içinde görünür.",
        "tour_export_title": "PDF Dışa Aktar",
        "tour_export_subtitle": "Mevcut sonucu profesyonel bir PDF tıbbi raporu olarak kaydedin.",
        "tour_gradcam_title": "Dikkat Isı Haritası",
        "tour_gradcam_subtitle": "AI'nin nereye baktığını görün — analiz edilen görüntü üzerinde bir ısı haritası.",
        "tour_dashboard_title": "Panel",
        "tour_dashboard_subtitle": "İstatistik panelini aç/kapat: toplam görüntü, ortalama güven, derece dağılımı.",
        "tour_history_title": "Geçmiş",
        "tour_history_subtitle": "Son 20 analiz edilmiş görüntüye göz atın. Sonucu yeniden yüklemek için bir küçük resme tıklayın.",
        "tour_visit_title": "Web Sayfasını Ziyaret Et",
        "tour_visit_subtitle": "Belgeler ve güncellemeler için projenin GitHub sayfasını tarayıcınızda açın.",
        "tour_mute_title": "Müziği Sessize Al",
        "tour_mute_subtitle": "Arka plan müziğini istediğiniz zaman duraklatın veya devam ettirin.",
        "tour_zoom_title": "Yakınlaştırma Kontrolleri",
        "tour_zoom_subtitle": "Yakınlaştırmak için fare tekerleğini, kaydırmak için sürükleyin, sıfırlamak için çift tıklayın.",
        "tour_account_title": "Hesap Menüsü",
        "tour_account_subtitle": "Ayarlar, Hesap Bilgileri, Denetim Günlüğü ve daha fazlasına erişmek için buraya tıklayın.",
        "suggestion_image_ready": "Görüntü yüklendi — işlemeye hazır",
        "suggestion_image_ready_sub": "AI analizini başlatmak için İşle'ye tıklayın.",
        "suggestion_pdf_ready": "Sonuç hazır — PDF olarak dışa aktarılsın mı?",
        "suggestion_pdf_ready_sub": "Profesyonel bir tıbbi raporu tek tıkla kaydedin.",
        "suggestion_gradcam_hint": "AI'nin ne gördüğünü merak ediyor musunuz?",
        "suggestion_gradcam_hint_sub": "Görüntü üzerindeki dikkat ısı haritasına bakın.",
        "suggestion_repeated_grid": "Her zaman {grid} alıyorsunuz",
        "suggestion_repeated_grid_sub": "Dağılımı görmek için panoyu açın.",
        "suggestion_history_hint": "Geçmişiniz büyüyor",
        "suggestion_history_hint_sub": "Son 20 analiz edilmiş görüntünüze göz atın.",
        "suggestion_dashboard_hint": "Panoınızda ilginç istatistikler var",
        "suggestion_dashboard_hint_sub": "Toplamlar, ortalamalar ve dağılım bir bakışta.",
        "suggestion_settings_hint": "Uygulamayı kişiselleştirin",
        "suggestion_settings_hint_sub": "Dil, tema, yazı tipini değiştirin ve 2FA'yı etkinleştirin.",
        "suggestion_idle_help": "Hızlı bir kılavuza mı ihtiyacınız var?",
        "suggestion_idle_help_sub": "Tanıtım turunu istediğiniz zaman yeniden oynatın.",
        "image_not_liver_warning": "Bu görüntü karaciğer ultrasonu gibi görünmüyor.\n\nLütfen uygun bir karaciğer ultrason görüntüsü ekleyin.",
        "result_not_confident": "Yapay zeka bu görüntüyü karaciğer ultrasonu olarak güvenle tanımlayamadı.\n\nLütfen uygun bir karaciğer ultrason görüntüsü ekleyin ve tekrar deneyin.",
        "not_liver_title": "Karaciğer ultrasonu değil",
        "not_liver_hint": "Lütfen uygun bir karaciğer ultrason (sonogram) görüntüsü kullanın",
        "voice_welcome": "Hoş geldiniz, {user}",
        "voice_goodbye": "Hoşça kalın",
        "voice_logged_out": "Başarıyla çıkış yapıldı",
        "voice_image_loaded": "Görüntü başarıyla yüklendi",
        "voice_image_rejected": "Bu bir karaciğer ultrasonu değil",
        "voice_inference_result": "{grid_name}, güven yüzde {conf}",
        "voice_pdf_saved": "PDF raporu başarıyla kaydedildi",
        "voice_error": "Bir hata oluştu",
        "voice_settings_applied": "Ayarlar uygulandı",
        "voice_language_changed": "Dil {lang} olarak değiştirildi",
        "voice_theme_changed": "Tema değiştirildi",
        "voice_2fa_enabled": "İki faktörlü kimlik doğrulama etkinleştirildi",
        "voice_2fa_disabled": "İki faktörlü kimlik doğrulama devre dışı bırakıldı",
        "voice_locked": "Oturum kilitlendi",
        "voice_unlocked": "Oturum kilidi açıldı",
        "voice_processing": "Görüntü işleniyor",
        "voice_processing_done": "İşleme tamamlandı",
        "voice_voice_enabled": "Sesli geri bildirim etkinleştirildi",
        "voice_voice_disabled": "Sesli geri bildirim devre dışı bırakıldı",
    },
    "Switzerland": {
        "app_title": "Leber KI-babak",
        "welcome": "Willkommen",
        "login": "Anmelden",
        "signup": "Registrieren",
        "logout": "Abmelden",
        "exit": "Beenden",
        "settings": "Einstellungen",
        "account_info": "Kontoinformationen",
        "username": "Benutzername",
        "password": "Passwort",
        "repeat_password": "Passwort wiederholen",
        "first_name": "Vorname",
        "last_name": "Nachname",
        "email": "E-Mail-Adresse",
        "remember_me": "Angemeldet bleiben",
        "login_btn": "Anmelden",
        "signup_btn": "Registrieren",
        "account": "Konto",
        "change_language": "Sprache ändern",
        "change_theme": "Thema ändern",
        "insert_sono": "Sonogramm einfügen",
        "process": "Verarbeiten",
        "mute": "Stumm",
        "unmute": "Ton an",
        "visit_webpage": "Webseite besuchen",
        "please_wait": "Bitte warten...",
        "processing": "Bild wird verarbeitet...",
        "result": "Ergebnis",
        "confidence": "Konfidenz",
        "grid": "Raster",
        "insert_image_first": "Bitte zuerst ein Bild einfügen!",
        "no_image": "Kein Bild eingefügt",
        "invalid_credentials": "Ungültige Anmeldedaten",
        "user_exists": "Benutzername existiert bereits",
        "password_requirements": "Passwortanforderungen",
        "password_rule": (
            "Das Passwort muss:\n"
            "• 8-16 Zeichen lang sein\n"
            "• Mindestens 1 Kleinbuchstaben enthalten\n"
            "• Mindestens 1 Großbuchstaben enthalten\n"
            "• Mindestens 4 Ziffern enthalten\n"
            "• Mindestens 4 Sonderzeichen enthalten"
        ),
        "passwords_dont_match": "Passwörter stimmen nicht überein!",
        "invalid_email": "Bitte eine gültige E-Mail-Adresse eingeben",
        "empty_fields": "Bitte alle Felder ausfüllen",
        "signup_success": "Registrierung erfolgreich! Sie können sich jetzt anmelden.",
        "logout_confirm": "Möchten Sie Ihr Konto wirklich löschen?",
        "exit_confirm": "Möchten Sie wirklich beenden?",
        "theme_dark_aqua": "Dunkles Aqua-Purple-Thema",
        "theme_solarized_white": "Helles Solarized-Thema",
        "theme_solarized_dark": "Dunkles Solarized-Thema",
        "theme_dark_forest": "Dunkles Waldgrün-Thema",
        "select_language": "Sprache auswählen",
        "select_theme": "Thema auswählen",
        "apply": "Anwenden",
        "cancel": "Abbrechen",
        "save": "Speichern",
        "success": "Erfolg",
        "error": "Fehler",
        "warning": "Warnung",
        "settings_applied": "Einstellungen angewendet!",
        "grid_0": "Raster 0 (NL) — Normale Leber",
        "grid_1": "Raster 1 (G1) — Grad 1",
        "grid_2": "Raster 2 (G2) — Grad 2",
        "grid_3": "Raster 3 (G3) — Grad 3",
        "too_many_attempts": "Zu viele fehlgeschlagene Versuche. Bitte 5 Minuten warten.",
        "invalid_username": "Benutzername muss 3-32 Zeichen haben (Buchstaben, Zahlen, Unterstrich).",
        "welcome_user": "Willkommen, {user}!",
        "inference_failed": "Inferenz fehlgeschlagen",
        "drop_here": "Bild hier ablegen",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Nicht unterstützter Dateityp. Bitte ein Bild ablegen.",
        "image_loaded": "Bild erfolgreich geladen",
        "help_about_title": "Hilfe & Über",
        "help_about_text": (
            "Leber KI-Pipeline v3.0\n"
            "Entwickelt von Babak Yousefian\n\n"
            "Tastenkürzel:\n"
            "  Ctrl+O      — Bild einfügen\n"
            "  Ctrl+Enter  — Bild verarbeiten\n"
            "  Ctrl+L      — Anmelden\n"
            "  Ctrl+,      — Einstellungen\n"
            "  Esc         — Aktiven Dialog schließen\n"
            "  F1          — Hilfe / Über\n\n"
            "Tipp: Sie können ein Bild auch an eine beliebige Stelle im Fenster ziehen und ablegen."
        ),
        "dashboard_title": "Übersicht",
        "dash_total": "Bilder insgesamt",
        "dash_avg_conf": "Durchschn. Konfidenz",
        "dash_distribution": "Verteilung",
        "dash_last_5": "Letzte 5 Ergebnisse",
        "dash_no_results": "Noch keine Ergebnisse",
        "dash_clear_confirm": "Alle Statistiken löschen?",
        "dash_cleared": "Statistiken gelöscht",
        "dashboard_toggle": "Übersicht ein-/ausblenden",
        "export_pdf": "PDF exportieren",
        "pdf_saved": "PDF erfolgreich gespeichert",
        "pdf_failed": "PDF-Erstellung fehlgeschlagen",
        "pdf_subtitle": "Analysebericht",
        "pdf_patient": "Patient:",
        "pdf_date": "Datum:",
        "pdf_model": "Modell:",
        "pdf_diagnosis": "DIAGNOSE",
        "pdf_confidence": "Konfidenz",
        "pdf_class_probs": "KLASSENWAHRSCHEINLICHKEITEN",
        "pdf_disclaimer": (
            "Dieser Bericht wurde von Babak Yousefian unter Aufsicht erfahrener Ärzte erstellt. "
            "Es handelt sich um ein professionelles Analysewerkzeug, das keine ärztliche Überweisung erfordert."
        ),
        "history_title": "Verlauf",
        "history_toggle": "Verlauf ein-/ausblenden",
        "history_empty": "Noch kein Verlauf",
        "history_cleared": "Verlauf gelöscht",
        "history_clear_confirm": "Gesamten Verlauf löschen?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Anmelden",
        "tooltip_desc_login": "Melden Sie sich bei Ihrem Konto an",
        "tooltip_tip_login": "Zugriff auf gespeicherten Verlauf und Statistiken",
        "tooltip_title_signup": "Registrieren",
        "tooltip_desc_signup": "Ein neues Konto erstellen",
        "tooltip_title_account": "Konto",
        "tooltip_desc_account": "Klicken, um das Kontomenü zu öffnen",
        "tooltip_desc_account_user": "Angemeldet als {user}",
        "tooltip_title_dashboard": "Übersicht",
        "tooltip_desc_dashboard": "Statistik-Panel ein-/ausblenden — Gesamt, Verteilung, letzte Ergebnisse",
        "tooltip_title_history": "Verlauf",
        "tooltip_desc_history": "Letzte 20 Ergebnisse mit Miniaturansichten ein-/ausblenden",
        "tooltip_title_visit": "Webseite besuchen",
        "tooltip_desc_visit": "GitHub-Seite des Projekts im Browser öffnen",
        "tooltip_tip_visit": "Öffnet im externen Browser",
        "tooltip_title_insert": "Bild einfügen",
        "tooltip_desc_insert": "Ein Sono-Bild zum Analysieren laden",
        "tooltip_tip_insert": "Sie können ein Bild auch an eine beliebige Stelle im Fenster ziehen und ablegen",
        "tooltip_title_process": "Verarbeiten",
        "tooltip_desc_process": "KI-Analyse des geladenen Bildes starten",
        "tooltip_tip_process": "Erfordert zunächst ein geladenes Bild",
        "tooltip_title_mute": "Musik stumm",
        "tooltip_desc_mute": "Hintergrundmusik pausieren",
        "tooltip_title_unmute": "Musik an",
        "tooltip_desc_unmute": "Hintergrundmusik fortsetzen",
        "tooltip_title_export": "PDF exportieren",
        "tooltip_desc_export": "Aktuelles Ergebnis als PDF-Bericht speichern",
        "tooltip_tip_export": "Enthält Bild, Diagnose und Klassenwahrscheinlichkeiten",
        "tooltip_title_dash_clear": "Statistiken löschen",
        "tooltip_desc_dash_clear": "Alle Statistiken dieses Kontos zurücksetzen",
        "tooltip_tip_dash_clear": "Diese Aktion kann nicht rückgängig gemacht werden",
        "tooltip_title_dash_close": "Übersicht schließen",
        "tooltip_desc_dash_close": "Statistik-Panel ausblenden",
        "tooltip_desc_dash_total": "Gesamtzahl der verarbeiteten Bilder",
        "tooltip_desc_dash_avg": "Durchschnittliche Konfidenz über alle Ergebnisse",
        "tooltip_desc_dash_dist": "Verteilung der Ergebnisse nach Graden",
        "tooltip_title_hist_clear": "Verlauf löschen",
        "tooltip_desc_hist_clear": "Alle gespeicherten Ergebnisse und Miniaturansichten löschen",
        "tooltip_tip_hist_clear": "Dadurch werden auch alle Bilddateien gelöscht",
        "tooltip_title_hist_close": "Verlauf schließen",
        "tooltip_desc_hist_close": "Verlauf-Panel ausblenden",
        "tooltip_tip_thumb_click": "Klicken, um dieses Ergebnis neu zu laden",
        "tooltip_tip_recent": "Klicken, um dieses Ergebnis neu zu laden",
        "tooltip_title_zoom_in": "Vergrößern",
        "tooltip_desc_zoom_in": "Bildvergrößerung erhöhen",
        "tooltip_title_zoom_out": "Verkleinern",
        "tooltip_desc_zoom_out": "Bildvergrößerung verringern",
        "tooltip_title_zoom_reset": "Ansicht zurücksetzen",
        "tooltip_desc_zoom_reset": "Bild an das Fenster anpassen",
        "tooltip_tip_zoom_reset": "Sie können auch auf das Bild doppelklicken",
        "notif_inference_ok": "Inferenz abgeschlossen",
        "follow_system_theme": "Systemthema folgen",
        "follow_system_theme_desc": "Automatisch zwischen Hell und Dunkel wechseln basierend auf Systemeinstellungen",
        "gradcam_title": "Aufmerksamkeits-Heatmap",
        "gradcam_show": "Aufmerksamkeitskarte anzeigen",
        "gradcam_toggle": "Karte umschalten",
        "gradcam_save": "Bild speichern",
        "gradcam_close": "Schließen",
        "gradcam_saved": "Bild erfolgreich gespeichert",
        "gradcam_no_heatmap": "Keine Aufmerksamkeitskarte verfügbar. Bitte zuerst ein Bild verarbeiten.",
        "update_available": "Aktualisierung verfügbar",
        "update_check_failed": "Aktualisierungsprüfung fehlgeschlagen",
        "check_for_updates": "Nach Aktualisierungen suchen",
        "check_for_updates_desc": "GitHub/Website auf die neueste Version prüfen",
        "up_to_date": "Sie sind auf dem neuesten Stand",
        "you_have_latest": "Sie verwenden die neueste Version.",
        "check_internet": "Bitte überprüfen Sie Ihre Internetverbindung und versuchen Sie es erneut.",
        "downloading": "Wird heruntergeladen...",
        "verifying": "Wird überprüft...",
        "download_now": "Jetzt herunterladen",
        "install_now": "Installieren & neu starten",
        "skip_version": "Diese Version überspringen",
        "remind_later": "Später erinnern",
        "whats_new": "Was ist neu",
        "no_changelog": "Kein Änderungsprotokoll bereitgestellt.",
        "critical_update": "Kritisches Update",
        "update_ready": "Bereit zur Installation",
        "app_will_restart": "Die App wird geschlossen und neu gestartet, um das Update anzuwenden.",
        "install_failed": "Installation fehlgeschlagen. Ihre alte Version ist sicher.",
        "download_failed": "Download fehlgeschlagen.",
        "source_update_requires_zip": "Quellcode-Update erfordert eine .zip-Datei.",
        "session_locked": "Sitzung gesperrt",
        "locked_for_user": "Gesperrt für {user}",
        "unlock": "Entsperren",
        "session_unlocked": "Sitzung entsperrt",
        "auto_lock_notice": "Die Sitzung wird nach 2 Minuten Inaktivität gesperrt.",
        "audit_log_title": "Audit-Protokoll",
        "audit_export": "Exportieren",
        "audit_exported": "Erfolgreich exportiert",
        "audit_export_failed": "Export fehlgeschlagen",
        "audit_search_placeholder": "Aktionen oder Details durchsuchen...",
        "audit_col_time": "Zeit",
        "audit_col_level": "Stufe",
        "audit_col_category": "Kategorie",
        "audit_col_action": "Aktion",
        "audit_col_user": "Benutzer",
        "audit_details_title": "Ereignisdetails",
        "audit_status": "Zeige {count} von {total} Ereignissen",
        "strength_very_weak": "Sehr schwach",
        "strength_weak": "Schwach",
        "strength_medium": "Mittel",
        "strength_strong": "Stark",
        "strength_very_strong": "Sehr stark",
        "strength_rule_length": "8-16 Zeichen",
        "strength_rule_lowercase": "Kleinbuchstabe",
        "strength_rule_uppercase": "Grossbuchstabe",
        "strength_rule_digits": "Ziffern",
        "strength_rule_special": "Sonderzeichen",
        "strength_add_chars": "Fügen Sie {n} weitere Zeichen hinzu",
        "strength_remove_chars": "Entfernen Sie {n} Zeichen",
        "strength_need_digits": "Fügen Sie {n} weitere Ziffer(n) hinzu",
        "strength_need_special": "Fügen Sie {n} weitere Sonderzeichen hinzu",
        "strength_common_pattern": "Vermeiden Sie häufige Wörter",
        "strength_common_year": "Vermeiden Sie häufige Jahreszahlen",
        "strength_has_sequence": "Vermeiden Sie Sequenzen (1234, abcd)",
        "strength_has_repeats": "Vermeiden Sie Wiederholungen (aaaa, 1111)",
        "passwords_match": "Passwörter stimmen überein",
        "2fa_setup_title": "Zwei-Faktor-Authentifizierung aktivieren",
        "2fa_setup_subtitle": "Scannen Sie den QR-Code mit Google Authenticator, Authy oder Microsoft Authenticator.",
        "2fa_manual_secret": "Oder geben Sie diesen Code manuell ein:",
        "2fa_enter_code": "Geben Sie den 6-stelligen Code aus Ihrer App ein:",
        "2fa_verify_btn": "Bestätigen",
        "2fa_backup_codes_title": "Speichern Sie diese Backup-Codes!",
        "2fa_backup_codes_hint": "Jeder Code kann nur einmal verwendet werden. Bewahren Sie sie an einem sicheren Ort auf.",
        "2fa_done_btn": "Fertig",
        "2fa_secret_copied": "Geheimcode in die Zwischenablage kopiert",
        "2fa_invalid_code": "Ungültiger Code. Bitte versuchen Sie es erneut.",
        "2fa_invalid_code_format": "Der Code muss 6 Ziffern haben.",
        "2fa_verify_title": "Zwei-Faktor-Authentifizierung",
        "2fa_verify_subtitle": "Geben Sie den 6-stelligen Code aus Ihrer Authentifizierungs-App ein.",
        "2fa_enter_code_hint": "Bitte geben Sie einen Code ein.",
        "2fa_use_backup_code": "Backup-Code verwenden",
        "2fa_use_totp_code": "App-Code verwenden",
        "2fa_enter_backup_code": "Geben Sie einen Ihrer Backup-Codes ein.",
        "2fa_section_title": "Sicherheit",
        "2fa_enabled_status": "Zwei-Faktor-Authentifizierung ist aktiviert",
        "2fa_disabled_status": "Zwei-Faktor-Authentifizierung ist deaktiviert",
        "2fa_enable_btn": "2FA aktivieren",
        "2fa_disable_btn": "2FA deaktivieren",
        "2fa_disable_title": "2FA deaktivieren",
        "2fa_disable_confirm": "Sind Sie sicher? Ihr Konto wird weniger sicher sein.",
        "2fa_disabled_success": "Zwei-Faktor-Authentifizierung deaktiviert",
        "2fa_enabled_success": "Zwei-Faktor-Authentifizierung aktiviert",
        "2fa_hint_title": "So verwenden Sie diesen QR-Code",
        "2fa_hint_step1": "Installieren Sie eine Authentifizierungs-App: Google Authenticator, Microsoft Authenticator oder Authy.",
        "2fa_hint_step2": "Öffnen Sie die App, tippen Sie auf die Schaltfläche „+“ und wählen Sie „QR-Code scannen“.",
        "2fa_hint_step3": "Richten Sie die Kamera Ihres Telefons auf den QR-Code oben (verwenden Sie die Kamera der App, nicht die Kamera-App Ihres Telefons).",
        "2fa_hint_step4": "Die App zeigt einen 6-stelligen Code an. Geben Sie ihn im Feld unten ein.",
        "2fa_manual_secret_hint": "Können Sie nicht scannen? Kopieren Sie den Schlüssel unten und fügen Sie ihn manuell in Ihrer App hinzu („Einrichtungsschlüssel eingeben“).",
        "auto_lock_section_title": "Automatische Sperre",
        "auto_lock_enable": "Nach Inaktivität sperren",
        "auto_lock_timeout_label": "Sperren nach:",
        "auto_lock_warning_title": "Warnung: Automatische Sperre",
        "auto_lock_warning_body": "Ihre Sitzung wird in {seconds} Sekunden aufgrund von Inaktivität gesperrt.",
        "lock_now": "Jetzt sperren",
        "font_section_title": "Yazı Tipi",
        "font_family_label": "Yazı tipi ailesi",
        "font_size_label": "Temel boyut",
        "font_scale_label": "Görüntüleme ölçeği",
        "font_scale_small": "Küçük",
        "font_scale_normal": "Normal",
        "font_scale_large": "Büyük",
        "font_scale_xlarge": "Çok büyük",
        "font_scale_xxlarge": "Devasa",
        "font_preview_text": "Pijamalı hasta yağız şoföre çabucak güvendi — 1234567890",
        "font_reset": "Varsayılana sıfırla",
        "font_applied": "Yazı tipi güncellendi",
        "motion_section_title": "Bewegung & Animationen",
        "reduce_motion_label": "Bewegung reduzieren",
        "reduce_motion_hint": "Deaktivieren Sie Animationen, wenn sie Ihnen unangenehm sind.",
        "empty_history_empty_title": "Noch kein Verlauf",
        "empty_history_empty_subtitle": "Verarbeiten Sie Ihr erstes Bild, um es hier zu sehen",
        "empty_history_empty_hint": "Jedes Ergebnis wird automatisch gespeichert",
        "empty_history_no_user_title": "Melden Sie sich an, um den Verlauf zu sehen",
        "empty_history_no_user_subtitle": "Ihre gespeicherten Ergebnisse erscheinen hier",
        "empty_history_no_user_hint": "Der Verlauf ist für jedes Konto privat",
        "empty_dash_no_results_title": "Noch keine Ergebnisse",
        "empty_dash_no_results_subtitle": "Ihre letzten Analysen erscheinen hier",
        "empty_dash_no_results_hint": "Verarbeiten Sie ein Bild, um zu beginnen",
        "empty_dash_no_dist_title": "Keine Verteilungsdaten",
        "empty_dash_no_dist_subtitle": "Statistiken füllen sich beim Verarbeiten",
        "empty_dash_no_dist_hint": "Jede Analyse zählt",
        "empty_image_empty_title": "Kein Bild eingefügt",
        "empty_image_empty_subtitle": "Ziehen Sie ein Bild hierher oder klicken Sie auf Einfügen",
        "empty_image_empty_hint": "Unterstützt PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Zum Analysieren ablegen",
        "empty_image_drop_hint_subtitle": "Lassen Sie das Bild los, um es zu laden",
        "empty_image_drop_hint_hint": "Das Format wird automatisch erkannt",
        "empty_processing_wait_title": "Verarbeitung...",
        "empty_processing_wait_subtitle": "Die KI analysiert Ihr Bild",
        "empty_processing_wait_hint": "Dies dauert normalerweise ein paar Sekunden",
        "tour_show_again": "Tour erneut anzeigen",
        "tour_completed_msg": "Tour abgeschlossen! Sie sind bereit.",
        "tour_skipped_msg": "Tour übersprungen. Sie können sie im Kontomenü wiederholen.",
        "tour_btn_next": "Weiter",
        "tour_btn_prev": "Zurück",
        "tour_btn_skip": "Überspringen",
        "tour_btn_finish": "Fertig ✓",
        "tour_guest_login_title": "Anmelden",
        "tour_guest_login_subtitle": "Klicken Sie hier, um sich anzumelden und alle Funktionen freizuschalten.",
        "tour_guest_signup_title": "Konto erstellen",
        "tour_guest_signup_subtitle": "Neu hier? Klicken Sie hier, um in Sekunden ein Konto zu erstellen.",
        "tour_insert_title": "Bild einfügen",
        "tour_insert_subtitle": "Laden Sie ein Sono-Bild von der Festplatte oder ziehen Sie es an eine beliebige Stelle im Fenster.",
        "tour_process_title": "Bild verarbeiten",
        "tour_process_subtitle": "Starten Sie die KI-Analyse des geladenen Bildes. Ergebnisse erscheinen in wenigen Sekunden.",
        "tour_export_title": "PDF exportieren",
        "tour_export_subtitle": "Speichern Sie das aktuelle Ergebnis als professionellen medizinischen PDF-Bericht.",
        "tour_gradcam_title": "Aufmerksamkeits-Heatmap",
        "tour_gradcam_subtitle": "Sehen Sie, wohin die KI schaut — eine Heatmap über dem analysierten Bild.",
        "tour_dashboard_title": "Übersicht",
        "tour_dashboard_subtitle": "Statistik-Panel ein-/ausblenden: Gesamtbilder, Durchschnittskonfidenz, Gradeverteilung.",
        "tour_history_title": "Verlauf",
        "tour_history_subtitle": "Durchsuchen Sie Ihre letzten 20 analysierten Bilder. Klicken Sie auf eine Miniatur, um das Ergebnis neu zu laden.",
        "tour_visit_title": "Webseite besuchen",
        "tour_visit_subtitle": "Öffnen Sie die GitHub-Seite des Projekts im Browser für Dokumentation und Updates.",
        "tour_mute_title": "Musik stumm",
        "tour_mute_subtitle": "Pausieren oder setzen Sie die Hintergrundmusik jederzeit fort.",
        "tour_zoom_title": "Zoom-Steuerung",
        "tour_zoom_subtitle": "Mausrad zum Zoomen, Ziehen zum Verschieben, Doppelklick zum Zurücksetzen.",
        "tour_account_title": "Kontomenü",
        "tour_account_subtitle": "Klicken Sie hier für Einstellungen, Kontoinformationen, Audit-Protokoll und mehr.",
        "suggestion_image_ready": "Bild geladen — bereit zur Verarbeitung",
        "suggestion_image_ready_sub": "Klicken Sie auf Verarbeiten, um die KI-Analyse zu starten.",
        "suggestion_pdf_ready": "Ergebnis bereit — als PDF exportieren?",
        "suggestion_pdf_ready_sub": "Speichern Sie einen professionellen medizinischen Bericht mit einem Klick.",
        "suggestion_gradcam_hint": "Neugierig, was die KI gesehen hat?",
        "suggestion_gradcam_hint_sub": "Sehen Sie die Aufmerksamkeits-Heatmap über dem Bild.",
        "suggestion_repeated_grid": "Sie erhalten immer {grid}",
        "suggestion_repeated_grid_sub": "Öffnen Sie die Übersicht, um die Verteilung zu sehen.",
        "suggestion_history_hint": "Ihr Verlauf wächst",
        "suggestion_history_hint_sub": "Durchsuchen Sie Ihre letzten 20 analysierten Bilder.",
        "suggestion_dashboard_hint": "Ihre Übersicht hat interessante Statistiken",
        "suggestion_dashboard_hint_sub": "Gesamtwerte, Durchschnitte und Verteilung auf einen Blick.",
        "suggestion_settings_hint": "Personalisieren Sie die App",
        "suggestion_settings_hint_sub": "Ändern Sie Sprache, Thema, Schriftart und aktivieren Sie 2FA.",
        "suggestion_idle_help": "Brauchen Sie eine kurze Anleitung?",
        "suggestion_idle_help_sub": "Wiederholen Sie die Einführung jederzeit.",
        "image_not_liver_warning": "Dieses Bild sieht nicht wie ein Leber-Ultraschall aus.\n\nBitte fügen Sie ein korrektes Leber-Ultraschallbild ein.",
        "result_not_confident": "Die KI konnte dieses Bild nicht sicher als Leber-Ultraschall identifizieren.\n\nBitte fügen Sie ein korrektes Leber-Ultraschallbild ein und versuchen Sie es erneut.",
        "not_liver_title": "Kein Leber-Ultraschall",
        "not_liver_hint": "Bitte verwenden Sie ein korrektes Leber-Ultraschallbild (Sonogramm)",
        "voice_welcome": "Willkomme, {user}",
        "voice_goodbye": "Uf Wiederluege",
        "voice_logged_out": "Erfolgriich abgmäldet",
        "voice_image_loaded": "Bild erfolgriich glade",
        "voice_image_rejected": "Das isch kes Leber-Ultraschallbild",
        "voice_inference_result": "{grid_name}, Konfidenz {conf} Prozent",
        "voice_pdf_saved": "PDF-Bericht erfolgriich gspeicheret",
        "voice_error": "Es isch en Fehler ufträtte",
        "voice_settings_applied": "Iistellige aagwändet",
        "voice_language_changed": "Sprach gänderet uf {lang}",
        "voice_theme_changed": "Thema gänderet",
        "voice_2fa_enabled": "Zwei-Faktor-Authentifizierig aktiviert",
        "voice_2fa_disabled": "Zwei-Faktor-Authentifizierig deaktiviert",
        "voice_locked": "Sitzig gsperrt",
        "voice_unlocked": "Sitzig entsperrt",
        "voice_processing": "Bild wird verarbeitet",
        "voice_processing_done": "Verarbeitig abgschlosse",
        "voice_voice_enabled": "Sprachfeedback aktiviert",
        "voice_voice_disabled": "Sprachfeedback deaktiviert",
    },
    "Sweden": {
        "app_title": "Lever AI-babak",
        "welcome": "Välkommen",
        "login": "Logga in",
        "signup": "Registrera dig",
        "logout": "Logga ut",
        "exit": "Avsluta",
        "settings": "Inställningar",
        "account_info": "Kontoinformation",
        "username": "Användarnamn",
        "password": "Lösenord",
        "repeat_password": "Upprepa lösenord",
        "first_name": "Förnamn",
        "last_name": "Efternamn",
        "email": "E-postadress",
        "remember_me": "Kom ihåg mig",
        "login_btn": "Logga in",
        "signup_btn": "Registrera dig",
        "account": "Konto",
        "change_language": "Byt språk",
        "change_theme": "Byt tema",
        "insert_sono": "Infoga Sono",
        "process": "Bearbeta",
        "mute": "Tysta",
        "unmute": "Slå på ljud",
        "visit_webpage": "Besök webbplats",
        "please_wait": "Vänligen vänta...",
        "processing": "Bearbetar bild...",
        "result": "Resultat",
        "confidence": "Konfidens",
        "grid": "Rutnät",
        "insert_image_first": "Vänligen infoga en bild först!",
        "no_image": "Ingen bild infogad",
        "invalid_credentials": "Ogiltiga inloggningsuppgifter",
        "user_exists": "Användarnamnet finns redan",
        "password_requirements": "Lösenordskrav",
        "password_rule": (
            "Lösenordet måste:\n"
            "• Vara 8-16 tecken långt\n"
            "• Innehålla minst 1 liten bokstav\n"
            "• Innehålla minst 1 stor bokstav\n"
            "• Innehålla minst 4 siffror\n"
            "• Innehålla minst 4 specialtecken"
        ),
        "passwords_dont_match": "Lösenorden matchar inte!",
        "invalid_email": "Vänligen ange en giltig e-postadress",
        "empty_fields": "Vänligen fyll i alla fält",
        "signup_success": "Registrering lyckades! Du kan nu logga in.",
        "logout_confirm": "Är du säker på att du vill radera ditt konto?",
        "exit_confirm": "Är du säker på att du vill avsluta?",
        "theme_dark_aqua": "Mörkt Aqua-Lila-tema",
        "theme_solarized_white": "Ljust Solarized-tema",
        "theme_solarized_dark": "Mörkt Solarized-tema",
        "theme_dark_forest": "Mörkt Skogsgrönt Tema",
        "select_language": "Välj språk",
        "select_theme": "Välj tema",
        "apply": "Tillämpa",
        "cancel": "Avbryt",
        "save": "Spara",
        "success": "Framgång",
        "error": "Fel",
        "warning": "Varning",
        "settings_applied": "Inställningar tillämpade!",
        "grid_0": "Rutnät 0 (NL) — Normal lever",
        "grid_1": "Rutnät 1 (G1) — Grad 1",
        "grid_2": "Rutnät 2 (G2) — Grad 2",
        "grid_3": "Rutnät 3 (G3) — Grad 3",
        "too_many_attempts": "För många misslyckade försök. Vänligen vänta 5 minuter.",
        "invalid_username": "Användarnamnet måste vara 3-32 tecken (bokstäver, siffror, understreck).",
        "welcome_user": "Välkommen, {user}!",
        "inference_failed": "Slutledning misslyckades",
        "drop_here": "Släpp bilden här",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Filtypen stöds inte. Vänligen släpp en bild.",
        "image_loaded": "Bilden laddades framgångsrikt",
        "help_about_title": "Hjälp & Om",
        "help_about_text": (
            "Lever AI Pipeline v3.0\n"
            "Utvecklad av Babak Yousefian\n\n"
            "Tangentbordsgenvägar:\n"
            "  Ctrl+O      — Infoga bild\n"
            "  Ctrl+Enter  — Bearbeta bild\n"
            "  Ctrl+L      — Logga in\n"
            "  Ctrl+,      — Inställningar\n"
            "  Esc         — Stäng aktiv dialog\n"
            "  F1          — Hjälp / Om\n\n"
            "Tips: Du kan också dra och släppa en bild var som helst i fönstret."
        ),
        "dashboard_title": "Instrumentpanel",
        "dash_total": "Totalt antal bilder",
        "dash_avg_conf": "Genomsnittlig konfidens",
        "dash_distribution": "Fördelning",
        "dash_last_5": "Senaste 5 resultaten",
        "dash_no_results": "Inga resultat än",
        "dash_clear_confirm": "Rensa all statistik?",
        "dash_cleared": "Statistiken rensad",
        "dashboard_toggle": "Visa/Dölj panel",
        "export_pdf": "Exportera PDF",
        "pdf_saved": "PDF sparades framgångsrikt",
        "pdf_failed": "PDF-generering misslyckades",
        "pdf_subtitle": "Analysrapport",
        "pdf_patient": "Patient:",
        "pdf_date": "Datum:",
        "pdf_model": "Modell:",
        "pdf_diagnosis": "DIAGNOS",
        "pdf_confidence": "Konfidens",
        "pdf_class_probs": "KLASSSANNOLIKHETER",
        "pdf_disclaimer": (
            "Denna rapport utvecklades av Babak Yousefian under överinseende av erfarna läkare. "
            "Det är ett professionellt analysverktyg och kräver ingen remiss till läkare."
        ),
        "history_title": "Historik",
        "history_toggle": "Visa/Dölj historik",
        "history_empty": "Ingen historik än",
        "history_cleared": "Historiken rensad",
        "history_clear_confirm": "Rensa all historik?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Logga in",
        "tooltip_desc_login": "Logga in på ditt konto",
        "tooltip_tip_login": "Få åtkomst till din sparade historik och statistik",
        "tooltip_title_signup": "Registrera dig",
        "tooltip_desc_signup": "Skapa ett nytt konto",
        "tooltip_title_account": "Konto",
        "tooltip_desc_account": "Klicka för att öppna kontomenyn",
        "tooltip_desc_account_user": "Inloggad som {user}",
        "tooltip_title_dashboard": "Instrumentpanel",
        "tooltip_desc_dashboard": "Visa/dölj statistikpanel — totalt, fördelning, senaste resultat",
        "tooltip_title_history": "Historik",
        "tooltip_desc_history": "Visa/dölj de senaste 20 resultaten med miniatyrbilder",
        "tooltip_title_visit": "Besök webbplats",
        "tooltip_desc_visit": "Öppna projektets GitHub-sida i webbläsaren",
        "tooltip_tip_visit": "Öppnas i extern webbläsare",
        "tooltip_title_insert": "Infoga bild",
        "tooltip_desc_insert": "Ladda en sono-bild för analys",
        "tooltip_tip_insert": "Du kan också dra och släppa en bild var som helst i fönstret",
        "tooltip_title_process": "Bearbeta",
        "tooltip_desc_process": "Kör AI-analys på den laddade bilden",
        "tooltip_tip_process": "Kräver att en bild laddas först",
        "tooltip_title_mute": "Tysta musik",
        "tooltip_desc_mute": "Pausa bakgrundsmusiken",
        "tooltip_title_unmute": "Slå på musik",
        "tooltip_desc_unmute": "Återuppta bakgrundsmusiken",
        "tooltip_title_export": "Exportera PDF",
        "tooltip_desc_export": "Spara det aktuella resultatet som PDF-rapport",
        "tooltip_tip_export": "Inkluderar bild, diagnos och klassannolikheter",
        "tooltip_title_dash_clear": "Rensa statistik",
        "tooltip_desc_dash_clear": "Återställ all statistik för detta konto",
        "tooltip_tip_dash_clear": "Denna åtgärd kan inte ångras",
        "tooltip_title_dash_close": "Stäng panel",
        "tooltip_desc_dash_close": "Dölj statistikpanelen",
        "tooltip_desc_dash_total": "Totalt antal bearbetade bilder",
        "tooltip_desc_dash_avg": "Genomsnittlig konfidens över alla resultat",
        "tooltip_desc_dash_dist": "Fördelning av resultat efter grad",
        "tooltip_title_hist_clear": "Rensa historik",
        "tooltip_desc_hist_clear": "Ta bort alla sparade resultat och miniatyrbilder",
        "tooltip_tip_hist_clear": "Detta tar också bort alla bildfiler",
        "tooltip_title_hist_close": "Stäng historik",
        "tooltip_desc_hist_close": "Dölj historikpanelen",
        "tooltip_tip_thumb_click": "Klicka för att ladda om detta resultat",
        "tooltip_tip_recent": "Klicka för att ladda om detta resultat",
        "tooltip_title_zoom_in": "Zooma in",
        "tooltip_desc_zoom_in": "Öka bildförstoringen",
        "tooltip_title_zoom_out": "Zooma ut",
        "tooltip_desc_zoom_out": "Minska bildförstoringen",
        "tooltip_title_zoom_reset": "Återställ vy",
        "tooltip_desc_zoom_reset": "Anpassa bilden till fönstret",
        "tooltip_tip_zoom_reset": "Du kan också dubbelklicka på bilden",
        "notif_inference_ok": "Slutledning slutförd",
        "follow_system_theme": "Följ systemtema",
        "follow_system_theme_desc": "Växla automatiskt mellan ljust och mörkt baserat på systeminställningar",
        "gradcam_title": "Uppmärksamhetsvärmekarta",
        "gradcam_show": "Visa uppmärksamhetskarta",
        "gradcam_toggle": "Växla värmekarta",
        "gradcam_save": "Spara bild",
        "gradcam_close": "Stäng",
        "gradcam_saved": "Bilden sparades framgångsrikt",
        "gradcam_no_heatmap": "Ingen uppmärksamhetskarta tillgänglig. Bearbeta en bild först.",
        "update_available": "Uppdatering tillgänglig",
        "update_check_failed": "Uppdateringskontroll misslyckades",
        "check_for_updates": "Sök efter uppdateringar",
        "check_for_updates_desc": "Sök på GitHub/webbplatsen efter den senaste versionen",
        "up_to_date": "Du är uppdaterad",
        "you_have_latest": "Du kör den senaste versionen.",
        "check_internet": "Kontrollera din internetanslutning och försök igen.",
        "downloading": "Laddar ner...",
        "verifying": "Verifierar...",
        "download_now": "Ladda ner nu",
        "install_now": "Installera och starta om",
        "skip_version": "Hoppa över denna version",
        "remind_later": "Påminn mig senare",
        "whats_new": "Vad är nytt",
        "no_changelog": "Ingen ändringslogg tillhandahölls.",
        "critical_update": "Kritisk uppdatering",
        "update_ready": "Redo att installera",
        "app_will_restart": "Appen stängs och startas om för att tillämpa uppdateringen.",
        "install_failed": "Installationen misslyckades. Din gamla version är säker.",
        "download_failed": "Nedladdningen misslyckades.",
        "source_update_requires_zip": "Källuppdatering kräver en .zip-fil.",
        "session_locked": "Sessionen Låst",
        "locked_for_user": "Låst för {user}",
        "unlock": "Lås Upp",
        "session_unlocked": "Sessionen upplåst",
        "auto_lock_notice": "Sessionen låses efter 2 minuters inaktivitet.",
        "audit_log_title": "Granskningslogg",
        "audit_export": "Exportera",
        "audit_exported": "Exporterades framgångsrikt",
        "audit_export_failed": "Export misslyckades",
        "audit_search_placeholder": "Sök åtgärder eller detaljer...",
        "audit_col_time": "Tid",
        "audit_col_level": "Nivå",
        "audit_col_category": "Kategori",
        "audit_col_action": "Åtgärd",
        "audit_col_user": "Användare",
        "audit_details_title": "Händelsedetaljer",
        "audit_status": "Visar {count} av {total} händelser",
        "strength_very_weak": "Mycket svagt",
        "strength_weak": "Svagt",
        "strength_medium": "Medel",
        "strength_strong": "Starkt",
        "strength_very_strong": "Mycket starkt",
        "strength_rule_length": "8-16 tecken",
        "strength_rule_lowercase": "Liten bokstav",
        "strength_rule_uppercase": "Stor bokstav",
        "strength_rule_digits": "Siffror",
        "strength_rule_special": "Specialtecken",
        "strength_add_chars": "Lägg till {n} tecken till",
        "strength_remove_chars": "Ta bort {n} tecken",
        "strength_need_digits": "Lägg till {n} siffra(-or) till",
        "strength_need_special": "Lägg till {n} specialtecken till",
        "strength_common_pattern": "Undvik vanliga ord",
        "strength_common_year": "Undvik vanliga årtal",
        "strength_has_sequence": "Undvik sekvenser (1234, abcd)",
        "strength_has_repeats": "Undvik upprepningar (aaaa, 1111)",
        "passwords_match": "Lösenorden matchar",
        "2fa_setup_title": "Aktivera tvåfaktorsautentisering",
        "2fa_setup_subtitle": "Skanna QR-koden med Google Authenticator, Authy eller Microsoft Authenticator.",
        "2fa_manual_secret": "Eller ange den här koden manuellt:",
        "2fa_enter_code": "Ange den 6-siffriga koden från din app:",
        "2fa_verify_btn": "Verifiera",
        "2fa_backup_codes_title": "Spara dessa reservkoder!",
        "2fa_backup_codes_hint": "Varje kod kan endast användas en gång. Förvara dem på en säker plats.",
        "2fa_done_btn": "Klar",
        "2fa_secret_copied": "Hemlig kod kopierad till urklipp",
        "2fa_invalid_code": "Ogiltig kod. Försök igen.",
        "2fa_invalid_code_format": "Koden måste vara 6 siffror.",
        "2fa_verify_title": "Tvåfaktorsautentisering",
        "2fa_verify_subtitle": "Ange den 6-siffriga koden från din autentiseringsapp.",
        "2fa_enter_code_hint": "Ange en kod.",
        "2fa_use_backup_code": "Använd en reservkod",
        "2fa_use_totp_code": "Använd app-kod",
        "2fa_enter_backup_code": "Ange en av dina reservkoder.",
        "2fa_section_title": "Säkerhet",
        "2fa_enabled_status": "Tvåfaktorsautentisering är aktiverad",
        "2fa_disabled_status": "Tvåfaktorsautentisering är inaktiverad",
        "2fa_enable_btn": "Aktivera 2FA",
        "2fa_disable_btn": "Inaktivera 2FA",
        "2fa_disable_title": "Inaktivera 2FA",
        "2fa_disable_confirm": "Är du säker? Ditt konto blir mindre säkert.",
        "2fa_disabled_success": "Tvåfaktorsautentisering inaktiverad",
        "2fa_enabled_success": "Tvåfaktorsautentisering aktiverad",
        "2fa_hint_title": "Så här använder du den här QR-koden",
        "2fa_hint_step1": "Installera en autentiseringsapp: Google Authenticator, Microsoft Authenticator eller Authy.",
        "2fa_hint_step2": "Öppna appen, tryck på \"+\"-knappen och välj \"Skanna QR-kod\".",
        "2fa_hint_step3": "Rikta telefonens kamera mot QR-koden ovan (använd appens kamera, inte telefonens kameraapp).",
        "2fa_hint_step4": "Appen visar en 6-siffrig kod. Ange den i fältet nedan.",
        "2fa_manual_secret_hint": "Kan du inte skanna? Kopiera nyckeln nedan och lägg till den manuellt i din app (\"Ange inställningsnyckel\").",
        "auto_lock_section_title": "Automatisk låsning",
        "auto_lock_enable": "Lås efter inaktivitet",
        "auto_lock_timeout_label": "Lås efter:",
        "auto_lock_warning_title": "Varning för automatisk låsning",
        "auto_lock_warning_body": "Din session låses om {seconds} sekunder på grund av inaktivitet.",
        "lock_now": "Lås nu",
        "font_section_title": "Schriftart",
        "font_family_label": "Schriftfamilie",
        "font_size_label": "Basisgröße",
        "font_scale_label": "Anzeigeskalierung",
        "font_scale_small": "Klein",
        "font_scale_normal": "Normal",
        "font_scale_large": "Gross",
        "font_scale_xlarge": "Sehr gross",
        "font_scale_xxlarge": "Riesig",
        "font_preview_text": "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern — 1234567890",
        "font_reset": "Auf Standard zurücksetzen",
        "font_applied": "Schriftart aktualisiert",
        "motion_section_title": "Rörelse och animationer",
        "reduce_motion_label": "Minska rörelse",
        "reduce_motion_hint": "Inaktivera animationer om de gör dig obekväm.",
        "empty_history_empty_title": "Ingen historik än",
        "empty_history_empty_subtitle": "Bearbeta din första bild för att se den här",
        "empty_history_empty_hint": "Varje resultat sparas automatiskt",
        "empty_history_no_user_title": "Logga in för att se historiken",
        "empty_history_no_user_subtitle": "Dina sparade resultat visas här",
        "empty_history_no_user_hint": "Historiken är privat för varje konto",
        "empty_dash_no_results_title": "Inga resultat än",
        "empty_dash_no_results_subtitle": "Dina senaste analyser visas här",
        "empty_dash_no_results_hint": "Bearbeta en bild för att börja",
        "empty_dash_no_dist_title": "Ingen fördelningsdata",
        "empty_dash_no_dist_subtitle": "Statistik fylls på när du bearbetar bilder",
        "empty_dash_no_dist_hint": "Varje analys räknas",
        "empty_image_empty_title": "Ingen bild infogad",
        "empty_image_empty_subtitle": "Dra och släpp en bild här, eller klicka på Infoga",
        "empty_image_empty_hint": "Stöder PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Släpp för att analysera",
        "empty_image_drop_hint_subtitle": "Släpp bilden för att ladda den",
        "empty_image_drop_hint_hint": "Formatet identifieras automatiskt",
        "empty_processing_wait_title": "Bearbetar...",
        "empty_processing_wait_subtitle": "AI analyserar din bild",
        "empty_processing_wait_hint": "Det tar vanligtvis några sekunder",
        "tour_show_again": "Visa rundturen igen",
        "tour_completed_msg": "Rundturen klar! Du är redo.",
        "tour_skipped_msg": "Rundturen hoppades över. Du kan spela upp den från Kontomenyn.",
        "tour_btn_next": "Nästa",
        "tour_btn_prev": "Tillbaka",
        "tour_btn_skip": "Hoppa över",
        "tour_btn_finish": "Klar ✓",
        "tour_guest_login_title": "Logga in",
        "tour_guest_login_subtitle": "Klicka här för att logga in och låsa upp alla funktioner.",
        "tour_guest_signup_title": "Skapa ett konto",
        "tour_guest_signup_subtitle": "Ny här? Klicka här för att skapa ett konto på några sekunder.",
        "tour_insert_title": "Infoga bild",
        "tour_insert_subtitle": "Ladda en sono-bild från disken eller dra och släpp den var som helst i fönstret.",
        "tour_process_title": "Bearbeta bild",
        "tour_process_subtitle": "Kör AI-analysen på den laddade bilden. Resultat visas inom några sekunder.",
        "tour_export_title": "Exportera PDF",
        "tour_export_subtitle": "Spara det aktuella resultatet som en professionell medicinsk PDF-rapport.",
        "tour_gradcam_title": "Uppmärksamhetsvärmekarta",
        "tour_gradcam_subtitle": "Se var AI tittar — en värmekarta över den analyserade bilden.",
        "tour_dashboard_title": "Instrumentpanel",
        "tour_dashboard_subtitle": "Visa/dölj statistikpanelen: totalt antal bilder, genomsnittlig konfidens, gradfördelning.",
        "tour_history_title": "Historik",
        "tour_history_subtitle": "Bläddra bland dina senaste 20 analyserade bilder. Klicka på en miniatyr för att ladda om resultatet.",
        "tour_visit_title": "Besök webbplats",
        "tour_visit_subtitle": "Öppna projektets GitHub-sida i webbläsaren för dokumentation och uppdateringar.",
        "tour_mute_title": "Tysta musik",
        "tour_mute_subtitle": "Pausa eller återuppta bakgrundsmusiken när som helst.",
        "tour_zoom_title": "Zoomkontroller",
        "tour_zoom_subtitle": "Använd mushjulet för att zooma, dra för att panorera, och dubbelklicka för att återställa.",
        "tour_account_title": "Kontomeny",
        "tour_account_subtitle": "Klicka här för att komma åt Inställningar, Kontoinformation, Granskningslogg och mer.",
        "suggestion_image_ready": "Bilden laddad — redo att bearbetas",
        "suggestion_image_ready_sub": "Klicka på Bearbeta för att starta AI-analysen.",
        "suggestion_pdf_ready": "Resultatet klart — exportera som PDF?",
        "suggestion_pdf_ready_sub": "Spara en professionell medicinsk rapport med ett klick.",
        "suggestion_gradcam_hint": "Nyfiken på vad AI:n såg?",
        "suggestion_gradcam_hint_sub": "Se uppmärksamhetsvärmekartan över bilden.",
        "suggestion_repeated_grid": "Du får alltid {grid}",
        "suggestion_repeated_grid_sub": "Öppna panelen för att se fördelningen.",
        "suggestion_history_hint": "Din historik växer",
        "suggestion_history_hint_sub": "Bläddra bland dina senaste 20 analyserade bilder.",
        "suggestion_dashboard_hint": "Din panel har intressant statistik",
        "suggestion_dashboard_hint_sub": "Totaler, medelvärden och fördelning i ett ögonkast.",
        "suggestion_settings_hint": "Anpassa appen",
        "suggestion_settings_hint_sub": "Ändra språk, tema, typsnitt och aktivera 2FA.",
        "suggestion_idle_help": "Behöver du en snabbguide?",
        "suggestion_idle_help_sub": "Spela upp rundturen igen när som helst.",
        "image_not_liver_warning": "Den här bilden ser inte ut som ett leverultraljud.\n\nVänligen infoga en korrekt leverultraljudsbild.",
        "result_not_confident": "AI kunde inte med säkerhet identifiera den här bilden som ett leverultraljud.\n\nVänligen infoga en korrekt leverultraljudsbild och försök igen.",
        "not_liver_title": "Inte ett leverultraljud",
        "not_liver_hint": "Vänligen använd en korrekt leverultraljudsbild (sonogram)",
        "voice_welcome": "Välkommen, {user}",
        "voice_goodbye": "Hej då",
        "voice_logged_out": "Utloggningen lyckades",
        "voice_image_loaded": "Bilden laddades framgångsrikt",
        "voice_image_rejected": "Detta är inte ett leverultraljud",
        "voice_inference_result": "{grid_name}, konfidens {conf} procent",
        "voice_pdf_saved": "PDF-rapporten sparades framgångsrikt",
        "voice_error": "Ett fel uppstod",
        "voice_settings_applied": "Inställningarna tillämpades",
        "voice_language_changed": "Språket ändrades till {lang}",
        "voice_theme_changed": "Temat ändrades",
        "voice_2fa_enabled": "Tvåfaktorsautentisering aktiverad",
        "voice_2fa_disabled": "Tvåfaktorsautentisering inaktiverad",
        "voice_locked": "Sessionen låstes",
        "voice_unlocked": "Sessionen låstes upp",
        "voice_processing": "Bearbetar bild",
        "voice_processing_done": "Bearbetningen slutfördes",
        "voice_voice_enabled": "Röstfeedback aktiverad",
        "voice_voice_disabled": "Röstfeedback inaktiverad",
    },
    "Wales": {
        "app_title": "Afu AI-babak",
        "welcome": "Croeso",
        "login": "Mewngofnodi",
        "signup": "Cofrestru",
        "logout": "Allgofnodi",
        "exit": "Gadael",
        "settings": "Gosodiadau",
        "account_info": "Gwybodaeth Cyfrif",
        "username": "Enw Defnyddiwr",
        "password": "Cyfrinair",
        "repeat_password": "Ailadrodd Cyfrinair",
        "first_name": "Enw Cyntaf",
        "last_name": "Cyfenw",
        "email": "Cyfeiriad E-bost",
        "remember_me": "Cofio Fi",
        "login_btn": "Mewngofnodi",
        "signup_btn": "Cofrestru",
        "account": "Cyfrif",
        "change_language": "Newid Iaith",
        "change_theme": "Newid Thema",
        "insert_sono": "Mewnosod Sono",
        "process": "Prosesu",
        "mute": "Tawelu",
        "unmute": "Dad-dawelu",
        "visit_webpage": "Ymweld â'r Wefan",
        "please_wait": "Arhoswch os gwelwch yn dda...",
        "processing": "Wrthi'n prosesu delwedd...",
        "result": "Canlyniad",
        "confidence": "Hyder",
        "grid": "Grid",
        "insert_image_first": "Mewnosodwch ddelwedd yn gyntaf os gwelwch yn dda!",
        "no_image": "Dim delwedd wedi'i mewnosod",
        "invalid_credentials": "Manylion mewngofnodi annilys",
        "user_exists": "Mae'r enw defnyddiwr eisoes yn bodoli",
        "password_requirements": "Gofynion cyfrinair",
        "password_rule": (
            "Rhaid i'r cyfrinair fod:\n"
            "• 8-16 nod o hyd\n"
            "• Yn cynnwys o leiaf 1 llythyren fach\n"
            "• Yn cynnwys o leiaf 1 llythyren fawr\n"
            "• Yn cynnwys o leiaf 4 digid\n"
            "• Yn cynnwys o leiaf 4 nod arbennig"
        ),
        "passwords_dont_match": "Nid yw'r cyfrineiriau'n cyfateb!",
        "invalid_email": "Rhowch gyfeiriad e-bost dilys",
        "empty_fields": "Llenwch bob maes",
        "signup_success": "Cofrestru'n llwyddiannus! Gallwch nawr fewngofnodi.",
        "logout_confirm": "Ydych chi'n siŵr eich bod am ddileu eich cyfrif?",
        "exit_confirm": "Ydych chi'n siŵr eich bod am adael?",
        "theme_dark_aqua": "Thema Dywyll Aqua-Porffor",
        "theme_solarized_white": "Thema Solarized Golau",
        "theme_solarized_dark": "Thema Solarized Tywyll",
        "theme_dark_forest": "Thema Coedwig Werdd Tywyll",
        "select_language": "Dewis Iaith",
        "select_theme": "Dewis Thema",
        "apply": "Cymhwyso",
        "cancel": "Canslo",
        "save": "Cadw",
        "success": "Llwyddiant",
        "error": "Gwall",
        "warning": "Rhybudd",
        "settings_applied": "Gosodiadau wedi'u cymhwyso!",
        "grid_0": "Grid 0 (NL) — Afu Normal",
        "grid_1": "Grid 1 (G1) — Gradd 1",
        "grid_2": "Grid 2 (G2) — Gradd 2",
        "grid_3": "Grid 3 (G3) — Gradd 3",
        "too_many_attempts": "Gormod o ymdrechion aflwyddiannus. Arhoswch 5 munud.",
        "invalid_username": "Rhaid i'r enw defnyddiwr fod yn 3-32 nod (llythrennau, rhifau, tanlinell).",
        "welcome_user": "Croeso, {user}!",
        "inference_failed": "Methodd casgliad",
        "drop_here": "Gollwng delwedd yma",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Math o ffeil heb ei gefnogi. Gollwng delwedd.",
        "image_loaded": "Delwedd wedi'i llwytho'n llwyddiannus",
        "help_about_title": "Cymorth ac Ynglŷn â",
        "help_about_text": (
            "Afu AI Piblinell v3.0\n"
            "Wedi'i ddatblygu gan Babak Yousefian\n\n"
            "Llwybrau byr y bysellfwrdd:\n"
            "  Ctrl+O      — Mewnosod delwedd\n"
            "  Ctrl+Enter  — Prosesu delwedd\n"
            "  Ctrl+L      — Mewngofnodi\n"
            "  Ctrl+,      — Gosodiadau\n"
            "  Esc         — Cau'r dialog gweithredol\n"
            "  F1          — Cymorth / Ynglŷn â\n\n"
            "Awgrym: Gallwch hefyd lusgo a gollwng delwedd unrhyw le yn y ffenestr."
        ),
        "dashboard_title": "Dangosfwrdd",
        "dash_total": "Cyfanswm delweddau",
        "dash_avg_conf": "Hyder cyfartalog",
        "dash_distribution": "Dosbarthiad",
        "dash_last_5": "5 canlyniad diwethaf",
        "dash_no_results": "Dim canlyniadau eto",
        "dash_clear_confirm": "Clirio'r holl ystadegau?",
        "dash_cleared": "Ystadegau wedi'u clirio",
        "dashboard_toggle": "Dangos/Cuddio'r dangosfwrdd",
        "export_pdf": "Allforio PDF",
        "pdf_saved": "PDF wedi'i gadw'n llwyddiannus",
        "pdf_failed": "Methodd cynhyrchu PDF",
        "pdf_subtitle": "Adroddiad Dadansoddi",
        "pdf_patient": "Claf:",
        "pdf_date": "Dyddiad:",
        "pdf_model": "Model:",
        "pdf_diagnosis": "DIAGNOSIS",
        "pdf_confidence": "Hyder",
        "pdf_class_probs": "TEBYGOLRWYDD DOSBARTHIADAU",
        "pdf_disclaimer": (
            "Datblygwyd yr adroddiad hwn gan Babak Yousefian dan oruchwyliaeth meddygon profiadol. "
            "Mae'n offeryn dadansoddi proffesiynol ac nid oes angen atgyfeiriad at feddyg."
        ),
        "history_title": "Hanes",
        "history_toggle": "Dangos/Cuddio hanes",
        "history_empty": "Dim hanes eto",
        "history_cleared": "Hanes wedi'i glirio",
        "history_clear_confirm": "Clirio'r holl hanes?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Mewngofnodi",
        "tooltip_desc_login": "Mewngofnodwch i'ch cyfrif",
        "tooltip_tip_login": "Cael mynediad at eich hanes a'ch ystadegau sydd wedi'u cadw",
        "tooltip_title_signup": "Cofrestru",
        "tooltip_desc_signup": "Creu cyfrif newydd",
        "tooltip_title_account": "Cyfrif",
        "tooltip_desc_account": "Cliciwch i agor y ddewislen cyfrif",
        "tooltip_desc_account_user": "Wedi mewngofnodi fel {user}",
        "tooltip_title_dashboard": "Dangosfwrdd",
        "tooltip_desc_dashboard": "Dangos/cuddio panel ystadegau — cyfanswm, dosbarthiad, canlyniadau diweddar",
        "tooltip_title_history": "Hanes",
        "tooltip_desc_history": "Dangos/cuddio'r 20 canlyniad diwethaf gyda mân-luniau",
        "tooltip_title_visit": "Ymweld â'r Wefan",
        "tooltip_desc_visit": "Agor tudalen GitHub y prosiect yn eich porwr",
        "tooltip_tip_visit": "Yn agor mewn porwr allanol",
        "tooltip_title_insert": "Mewnosod delwedd",
        "tooltip_desc_insert": "Llwytho delwedd sono i'w dadansoddi",
        "tooltip_tip_insert": "Gallwch hefyd lusgo a gollwng delwedd unrhyw le yn y ffenestr",
        "tooltip_title_process": "Prosesu",
        "tooltip_desc_process": "Rhedeg dadansoddiad AI ar y ddelwedd a lwythwyd",
        "tooltip_tip_process": "Mae angen llwytho delwedd yn gyntaf",
        "tooltip_title_mute": "Tawelu cerddoriaeth",
        "tooltip_desc_mute": "Oedi cerddoriaeth cefndir",
        "tooltip_title_unmute": "Dad-dawelu cerddoriaeth",
        "tooltip_desc_unmute": "Ailddechrau cerddoriaeth cefndir",
        "tooltip_title_export": "Allforio PDF",
        "tooltip_desc_export": "Cadw'r canlyniad presennol fel adroddiad PDF",
        "tooltip_tip_export": "Yn cynnwys delwedd, diagnosis, a thebygolrwydd dosbarthiadau",
        "tooltip_title_dash_clear": "Clirio ystadegau",
        "tooltip_desc_dash_clear": "Ailosod holl ystadegau'r cyfrif hwn",
        "tooltip_tip_dash_clear": "Ni ellir dadwneud y weithred hon",
        "tooltip_title_dash_close": "Cau'r dangosfwrdd",
        "tooltip_desc_dash_close": "Cuddio'r panel ystadegau",
        "tooltip_desc_dash_total": "Cyfanswm nifer y delweddau a broseswyd",
        "tooltip_desc_dash_avg": "Hyder cyfartalog ar draws pob canlyniad",
        "tooltip_desc_dash_dist": "Dosbarthiad canlyniadau yn ôl gradd",
        "tooltip_title_hist_clear": "Clirio hanes",
        "tooltip_desc_hist_clear": "Dileu pob canlyniad a mân-lun sydd wedi'u cadw",
        "tooltip_tip_hist_clear": "Bydd hyn hefyd yn dileu pob ffeil delwedd",
        "tooltip_title_hist_close": "Cau hanes",
        "tooltip_desc_hist_close": "Cuddio panel hanes",
        "tooltip_tip_thumb_click": "Cliciwch i ail-lwytho'r canlyniad hwn",
        "tooltip_tip_recent": "Cliciwch i ail-lwytho'r canlyniad hwn",
        "tooltip_title_zoom_in": "Chwyddo i mewn",
        "tooltip_desc_zoom_in": "Cynyddu chwyddiant delwedd",
        "tooltip_title_zoom_out": "Chwyddo allan",
        "tooltip_desc_zoom_out": "Lleihau chwyddiant delwedd",
        "tooltip_title_zoom_reset": "Ailosod golwg",
        "tooltip_desc_zoom_reset": "Ffitio'r ddelwedd i'r ffenestr",
        "tooltip_tip_zoom_reset": "Gallwch hefyd glicio ddwywaith ar y ddelwedd",
        "notif_inference_ok": "Casgliad wedi'i gwblhau",
        "follow_system_theme": "Dilyn thema'r system",
        "follow_system_theme_desc": "Newid yn awtomatig rhwng golau a thywyll yn seiliedig ar osodiadau'r system",
        "gradcam_title": "Map Gwres Sylw",
        "gradcam_show": "Dangos Map Sylw",
        "gradcam_toggle": "Toglo Map Gwres",
        "gradcam_save": "Cadw Delwedd",
        "gradcam_close": "Cau",
        "gradcam_saved": "Delwedd wedi'i chadw'n llwyddiannus",
        "gradcam_no_heatmap": "Dim map sylw ar gael. Proseswch ddelwedd yn gyntaf.",
        "update_available": "Diweddariad ar gael",
        "update_check_failed": "Methodd gwirio diweddariadau",
        "check_for_updates": "Gwirio am ddiweddariadau",
        "check_for_updates_desc": "Gwirio GitHub/gwefan am y fersiwn diweddaraf",
        "up_to_date": "Rydych chi'n gyfredol",
        "you_have_latest": "Rydych chi'n rhedeg y fersiwn diweddaraf.",
        "check_internet": "Gwiriwch eich cysylltiad rhyngrwyd a rhoi cynnig arall arni.",
        "downloading": "Wrthi'n llwytho i lawr...",
        "verifying": "Wrthi'n gwirio...",
        "download_now": "Llwytho i lawr nawr",
        "install_now": "Gosod ac ailgychwyn",
        "skip_version": "Hepgor y fersiwn hon",
        "remind_later": "Atgoffa fi'n ddiweddarach",
        "whats_new": "Beth sy'n newydd",
        "no_changelog": "Ni ddarparwyd cofnod newidiadau.",
        "critical_update": "Diweddariad critigol",
        "update_ready": "Yn barod i'w osod",
        "app_will_restart": "Bydd yr ap yn cau ac yn ailgychwyn i gymhwyso'r diweddariad.",
        "install_failed": "Methodd y gosodiad. Mae eich hen fersiwn yn ddiogel.",
        "download_failed": "Methodd y llwytho i lawr.",
        "source_update_requires_zip": "Mae diweddariad ffynhonnell angen ffeil .zip.",
        "session_locked": "Sesiwn Wedi'i Gloi",
        "locked_for_user": "Wedi'i gloi ar gyfer {user}",
        "unlock": "Datgloi",
        "session_unlocked": "Sesiwn wedi'i datgloi",
        "auto_lock_notice": "Bydd y sesiwn yn cloi ar ôl 2 funud o anweithgarwch.",
        "audit_log_title": "Cofnod Archwilio",
        "audit_export": "Allforio",
        "audit_exported": "Allforiwyd yn llwyddiannus",
        "audit_export_failed": "Methodd yr allforio",
        "audit_search_placeholder": "Chwilio gweithredoedd neu fanylion...",
        "audit_col_time": "Amser",
        "audit_col_level": "Lefel",
        "audit_col_category": "Categori",
        "audit_col_action": "Gweithred",
        "audit_col_user": "Defnyddiwr",
        "audit_details_title": "Manylion Digwyddiad",
        "audit_status": "Yn dangos {count} o {total} digwyddiad",
        "strength_very_weak": "Gwan iawn",
        "strength_weak": "Gwan",
        "strength_medium": "Canolig",
        "strength_strong": "Cryf",
        "strength_very_strong": "Cryf iawn",
        "strength_rule_length": "8-16 nod",
        "strength_rule_lowercase": "Llythyren fach",
        "strength_rule_uppercase": "Llythyren fawr",
        "strength_rule_digits": "Digidau",
        "strength_rule_special": "Nodau arbennig",
        "strength_add_chars": "Ychwanegwch {n} nod arall",
        "strength_remove_chars": "Tynnwch {n} nod",
        "strength_need_digits": "Ychwanegwch {n} digid arall",
        "strength_need_special": "Ychwanegwch {n} nod arbennig arall",
        "strength_common_pattern": "Osgowch eiriau cyffredin",
        "strength_common_year": "Osgowch flynyddoedd cyffredin",
        "strength_has_sequence": "Osgowch ddilyniannau (1234, abcd)",
        "strength_has_repeats": "Osgowch ailadroddiadau (aaaa, 1111)",
        "passwords_match": "Mae'r cyfrineiriau'n cyfateb",
        "2fa_setup_title": "Galluogi Dilysu Dau Ffactor",
        "2fa_setup_subtitle": "Sganiwch y cod QR gyda Google Authenticator, Authy neu Microsoft Authenticator.",
        "2fa_manual_secret": "Neu nodwch y cod hwn â llaw:",
        "2fa_enter_code": "Nodwch y cod 6 digid o'ch ap:",
        "2fa_verify_btn": "Gwirio",
        "2fa_backup_codes_title": "Cadwch y codau wrth gefn hyn!",
        "2fa_backup_codes_hint": "Gellir defnyddio pob cod unwaith yn unig. Cadwch nhw mewn man diogel.",
        "2fa_done_btn": "Wedi gorffen",
        "2fa_secret_copied": "Cod cyfrinachol wedi'i gopïo i'r clipfwrdd",
        "2fa_invalid_code": "Cod annilys. Rhowch gynnig arall arni.",
        "2fa_invalid_code_format": "Rhaid i'r cod fod yn 6 digid.",
        "2fa_verify_title": "Dilysu Dau Ffactor",
        "2fa_verify_subtitle": "Nodwch y cod 6 digid o'ch ap dilysu.",
        "2fa_enter_code_hint": "Nodwch god os gwelwch yn dda.",
        "2fa_use_backup_code": "Defnyddio cod wrth gefn",
        "2fa_use_totp_code": "Defnyddio cod yr ap",
        "2fa_enter_backup_code": "Nodwch un o'ch codau wrth gefn.",
        "2fa_section_title": "Diogelwch",
        "2fa_enabled_status": "Mae Dilysu Dau Ffactor wedi'i alluogi",
        "2fa_disabled_status": "Mae Dilysu Dau Ffactor wedi'i analluogi",
        "2fa_enable_btn": "Galluogi 2FA",
        "2fa_disable_btn": "Analluogi 2FA",
        "2fa_disable_title": "Analluogi 2FA",
        "2fa_disable_confirm": "Ydych chi'n siŵr? Bydd eich cyfrif yn llai diogel.",
        "2fa_disabled_success": "Dilysu Dau Ffactor wedi'i analluogi",
        "2fa_enabled_success": "Dilysu Dau Ffactor wedi'i alluogi",
        "2fa_hint_title": "Sut i ddefnyddio'r cod QR hwn",
        "2fa_hint_step1": "Gosodwch ap dilysu: Google Authenticator, Microsoft Authenticator neu Authy.",
        "2fa_hint_step2": "Agorwch yr ap, tapiwch y botwm \"+\" ac yna dewiswch \"Sganio cod QR\".",
        "2fa_hint_step3": "Pwyntiwch gamera eich ffôn at y cod QR uchod (defnyddiwch gamera'r ap, nid ap camera eich ffôn).",
        "2fa_hint_step4": "Bydd yr ap yn dangos cod 6 digid. Nodwch ef yn y blwch isod.",
        "2fa_manual_secret_hint": "Methu sganio? Copïwch yr allwedd isod a'i hychwanegu â llaw yn eich ap (\"Nodi allwedd gosod\").",
        "auto_lock_section_title": "Cloi Awtomatig",
        "auto_lock_enable": "Cloi ar ôl anweithgarwch",
        "auto_lock_timeout_label": "Cloi ar ôl:",
        "auto_lock_warning_title": "Rhybudd Cloi Awtomatig",
        "auto_lock_warning_body": "Bydd eich sesiwn yn cael ei chloi mewn {seconds} eiliad oherwydd anweithgarwch.",
        "lock_now": "Cloi Nawr",
        "font_section_title": "Ffont",
        "font_family_label": "Teulu ffont",
        "font_size_label": "Maint sylfaenol",
        "font_scale_label": "Graddfa arddangos",
        "font_scale_small": "Bach",
        "font_scale_normal": "Arferol",
        "font_scale_large": "Mawr",
        "font_scale_xlarge": "Mawr iawn",
        "font_scale_xxlarge": "Enfawr",
        "font_preview_text": "Mae'r llwynog brown cyflym yn neidio dros y ci diog — 1234567890",
        "font_reset": "Ailosod i'r rhagosodedig",
        "font_applied": "Ffont wedi'i ddiweddaru",
        "motion_section_title": "Cynnig ac Animeiddiadau",
        "reduce_motion_label": "Lleihau cynnig",
        "reduce_motion_hint": "Analluogwch animeiddiadau os ydyn nhw'n gwneud ichi deimlo'n anghyfforddus.",
        "empty_history_empty_title": "Dim hanes eto",
        "empty_history_empty_subtitle": "Proseswch eich delwedd gyntaf i'w gweld yma",
        "empty_history_empty_hint": "Mae pob canlyniad yn cael ei gadw'n awtomatig",
        "empty_history_no_user_title": "Mewngofnodwch i weld hanes",
        "empty_history_no_user_subtitle": "Bydd eich canlyniadau wedi'u cadw yn ymddangos yma",
        "empty_history_no_user_hint": "Mae hanes yn breifat i bob cyfrif",
        "empty_dash_no_results_title": "Dim canlyniadau eto",
        "empty_dash_no_results_subtitle": "Bydd eich dadansoddiadau diweddar yn ymddangos yma",
        "empty_dash_no_results_hint": "Proseswch ddelwedd i ddechrau",
        "empty_dash_no_dist_title": "Dim data dosbarthiad",
        "empty_dash_no_dist_subtitle": "Bydd ystadegau'n llenwi wrth brosesu delweddau",
        "empty_dash_no_dist_hint": "Mae pob dadansoddiad yn cyfrif",
        "empty_image_empty_title": "Dim delwedd wedi'i mewnosod",
        "empty_image_empty_subtitle": "Llusgwch a gollwng delwedd yma, neu cliciwch Mewnosod",
        "empty_image_empty_hint": "Yn cefnogi PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Gollwng i ddadansoddi",
        "empty_image_drop_hint_subtitle": "Rhyddhewch y ddelwedd i'w llwytho",
        "empty_image_drop_hint_hint": "Bydd y fformat yn cael ei ganfod yn awtomatig",
        "empty_processing_wait_title": "Wrthi'n prosesu...",
        "empty_processing_wait_subtitle": "Mae AI yn dadansoddi eich delwedd",
        "empty_processing_wait_hint": "Mae hyn fel arfer yn cymryd ychydig eiliadau",
        "tour_show_again": "Dangos y daith eto",
        "tour_completed_msg": "Taith wedi'i chwblhau! Rydych chi'n barod.",
        "tour_skipped_msg": "Taith wedi'i hepgor. Gallwch ei hailchwarae o'r ddewislen Cyfrif.",
        "tour_btn_next": "Nesaf",
        "tour_btn_prev": "Yn ôl",
        "tour_btn_skip": "Hepgor",
        "tour_btn_finish": "Gorffen ✓",
        "tour_guest_login_title": "Mewngofnodi",
        "tour_guest_login_subtitle": "Cliciwch yma i fewngofnodi a datgloi'r holl nodweddion.",
        "tour_guest_signup_title": "Creu cyfrif",
        "tour_guest_signup_subtitle": "Yn newydd yma? Cliciwch yma i greu cyfrif mewn eiliadau.",
        "tour_insert_title": "Mewnosod delwedd",
        "tour_insert_subtitle": "Llwythwch ddelwedd sono o'r disg, neu lusgwch a gollwng unrhyw le yn y ffenestr.",
        "tour_process_title": "Prosesu delwedd",
        "tour_process_subtitle": "Rhedeg dadansoddiad AI ar y ddelwedd a lwythwyd. Mae canlyniadau'n ymddangos mewn ychydig eiliadau.",
        "tour_export_title": "Allforio PDF",
        "tour_export_subtitle": "Cadwch y canlyniad presennol fel adroddiad meddygol PDF proffesiynol.",
        "tour_gradcam_title": "Map gwres sylw",
        "tour_gradcam_subtitle": "Gwelwch ble mae'r AI yn edrych — map gwres dros y ddelwedd a ddadansoddwyd.",
        "tour_dashboard_title": "Dangosfwrdd",
        "tour_dashboard_subtitle": "Dangos/cuddio panel ystadegau: cyfanswm delweddau, hyder cyfartalog, dosbarthiad gradd.",
        "tour_history_title": "Hanes",
        "tour_history_subtitle": "Pori eich 20 delwedd a ddadansoddwyd diwethaf. Cliciwch ar unrhyw fân-lun i ail-lwytho ei ganlyniad.",
        "tour_visit_title": "Ymweld â'r wefan",
        "tour_visit_subtitle": "Agorwch dudalen GitHub y prosiect yn eich porwr ar gyfer dogfennaeth a diweddariadau.",
        "tour_mute_title": "Tawelu cerddoriaeth",
        "tour_mute_subtitle": "Oedi neu ailddechrau'r gerddoriaeth gefndir ar unrhyw adeg.",
        "tour_zoom_title": "Rheolyddion chwyddo",
        "tour_zoom_subtitle": "Defnyddiwch olwyn y llygoden i chwyddo, llusgwch i symud, a chliciwch ddwywaith i ailosod.",
        "tour_account_title": "Dewislen cyfrif",
        "tour_account_subtitle": "Cliciwch yma i gyrchu Gosodiadau, Gwybodaeth Cyfrif, Cofnod Archwilio, a mwy.",
        "suggestion_image_ready":      "Delwedd wedi'i llwytho — yn barod i'w phrosesu",
        "suggestion_image_ready_sub":  "Cliciwch Prosesu i redeg dadansoddiad AI.",
        "suggestion_pdf_ready":        "Canlyniad yn barod — allforio fel PDF?",
        "suggestion_pdf_ready_sub":    "Cadwch adroddiad meddygol proffesiynol gydag un clic.",
        "suggestion_gradcam_hint":     "Yn chwilfrydig beth welodd yr AI?",
        "suggestion_gradcam_hint_sub": "Gwelwch y map gwres sylw dros y ddelwedd.",
        "suggestion_repeated_grid":    "Rydych chi bob amser yn cael {grid}",
        "suggestion_repeated_grid_sub":"Agorwch y dangosfwrdd i weld y dosbarthiad.",
        "suggestion_history_hint":     "Mae eich hanes yn tyfu",
        "suggestion_history_hint_sub": "Pori eich 20 delwedd a ddadansoddwyd diwethaf.",
        "suggestion_dashboard_hint":   "Mae gan eich dangosfwrdd ystadegau diddorol",
        "suggestion_dashboard_hint_sub":"Cyfansymiau, cyfartaleddau a dosbarthiad ar unwaith.",
        "suggestion_settings_hint":    "Personolwch yr ap",
        "suggestion_settings_hint_sub":"Newidiwch iaith, thema, ffont a galluogi 2FA.",
        "suggestion_idle_help":        "Angen canllaw cyflym?",
        "suggestion_idle_help_sub":    "Ailchwarae'r daith gynefino unrhyw bryd.",
        "image_not_liver_warning": "Nid yw'r ddelwedd hon yn edrych fel uwchsain afu.\n\nRhowch ddelwedd uwchsain afu briodol.",
        "result_not_confident": "Ni allai'r AI adnabod y ddelwedd hon yn hyderus fel uwchsain afu.\n\nRhowch ddelwedd uwchsain afu briodol a rhowch gynnig arall arni.",
        "not_liver_title": "Nid yw'n uwchsain afu",
        "not_liver_hint": "Defnyddiwch ddelwedd uwchsain (sonogram) afu briodol",
        "voice_welcome": "Croeso, {user}",
        "voice_goodbye": "Hwyl fawr",
        "voice_logged_out": "Allgofnodwyd yn llwyddiannus",
        "voice_image_loaded": "Delwedd wedi'i llwytho'n llwyddiannus",
        "voice_image_rejected": "Nid yw hwn yn uwchsain afu",
        "voice_inference_result": "{grid_name}, hyder {conf} y cant",
        "voice_pdf_saved": "Adroddiad PDF wedi'i gadw'n llwyddiannus",
        "voice_error": "Digwyddodd gwall",
        "voice_settings_applied": "Gosodiadau wedi'u cymhwyso",
        "voice_language_changed": "Iaith wedi newid i {lang}",
        "voice_theme_changed": "Thema wedi newid",
        "voice_2fa_enabled": "Dilysu dau ffactor wedi'i alluogi",
        "voice_2fa_disabled": "Dilysu dau ffactor wedi'i analluogi",
        "voice_locked": "Sesiwn wedi'i chloi",
        "voice_unlocked": "Sesiwn wedi'i datgloi",
        "voice_processing": "Wrthi'n prosesu delwedd",
        "voice_processing_done": "Prosesu wedi'i gwblhau",
        "voice_voice_enabled": "Adborth llais wedi'i alluogi",
        "voice_voice_disabled": "Adborth llais wedi'i analluogi",
    },
    "Nederland": {
        "app_title": "Lever AI-babak",
        "welcome": "Welkom",
        "login": "Inloggen",
        "signup": "Registreren",
        "logout": "Uitloggen",
        "exit": "Afsluiten",
        "settings": "Instellingen",
        "account_info": "Accountinformatie",
        "username": "Gebruikersnaam",
        "password": "Wachtwoord",
        "repeat_password": "Herhaal wachtwoord",
        "first_name": "Voornaam",
        "last_name": "Achternaam",
        "email": "E-mailadres",
        "remember_me": "Onthoud mij",
        "login_btn": "Inloggen",
        "signup_btn": "Registreren",
        "account": "Account",
        "change_language": "Taal wijzigen",
        "change_theme": "Thema wijzigen",
        "insert_sono": "Sono invoegen",
        "process": "Verwerken",
        "mute": "Dempen",
        "unmute": "Dempen opheffen",
        "visit_webpage": "Bezoek website",
        "please_wait": "Even geduld alstublieft...",
        "processing": "Afbeelding verwerken...",
        "result": "Resultaat",
        "confidence": "Vertrouwen",
        "grid": "Raster",
        "insert_image_first": "Voeg eerst een afbeelding in!",
        "no_image": "Geen afbeelding ingevoegd",
        "invalid_credentials": "Ongeldige inloggegevens",
        "user_exists": "Gebruikersnaam bestaat al",
        "password_requirements": "Wachtwoordvereisten",
        "password_rule": (
            "Het wachtwoord moet:\n"
            "• 8-16 tekens lang zijn\n"
            "• Minimaal 1 kleine letter bevatten\n"
            "• Minimaal 1 hoofdletter bevatten\n"
            "• Minimaal 4 cijfers bevatten\n"
            "• Minimaal 4 speciale tekens bevatten"
        ),
        "passwords_dont_match": "Wachtwoorden komen niet overeen!",
        "invalid_email": "Voer een geldig e-mailadres in",
        "empty_fields": "Vul alle velden in",
        "signup_success": "Registratie geslaagd! U kunt nu inloggen.",
        "logout_confirm": "Weet u zeker dat u uw account wilt verwijderen?",
        "exit_confirm": "Weet u zeker dat u wilt afsluiten?",
        "theme_dark_aqua": "Donker Aqua-Paars thema",
        "theme_solarized_white": "Licht Solarized thema",
        "theme_solarized_dark": "Donker Solarized thema",
        "theme_dark_forest": "Donker Bosgroen Thema",
        "select_language": "Taal selecteren",
        "select_theme": "Thema selecteren",
        "apply": "Toepassen",
        "cancel": "Annuleren",
        "save": "Opslaan",
        "success": "Succes",
        "error": "Fout",
        "warning": "Waarschuwing",
        "settings_applied": "Instellingen toegepast!",
        "grid_0": "Raster 0 (NL) — Normale lever",
        "grid_1": "Raster 1 (G1) — Graad 1",
        "grid_2": "Raster 2 (G2) — Graad 2",
        "grid_3": "Raster 3 (G3) — Graad 3",
        "too_many_attempts": "Te veel mislukte pogingen. Wacht 5 minuten.",
        "invalid_username": "Gebruikersnaam moet 3-32 tekens zijn (letters, cijfers, underscore).",
        "welcome_user": "Welkom, {user}!",
        "inference_failed": "Inferentie mislukt",
        "drop_here": "Sleep afbeelding hierheen",
        "drop_subtitle": "PNG, JPG, JPEG, BMP, DCM",
        "drop_invalid": "Bestandstype niet ondersteund. Sleep een afbeelding.",
        "image_loaded": "Afbeelding succesvol geladen",
        "help_about_title": "Help & Over",
        "help_about_text": (
            "Lever AI-pijplijn v3.0\n"
            "Ontwikkeld door Babak Yousefian\n\n"
            "Sneltoetsen:\n"
            "  Ctrl+O      — Afbeelding invoegen\n"
            "  Ctrl+Enter  — Afbeelding verwerken\n"
            "  Ctrl+L      — Inloggen\n"
            "  Ctrl+,      — Instellingen\n"
            "  Esc         — Actief dialoogvenster sluiten\n"
            "  F1          — Help / Over\n\n"
            "Tip: U kunt ook een afbeelding slepen en neerzetten waar u maar wilt in het venster."
        ),
        "dashboard_title": "Dashboard",
        "dash_total": "Totaal afbeeldingen",
        "dash_avg_conf": "Gem. vertrouwen",
        "dash_distribution": "Verdeling",
        "dash_last_5": "Laatste 5 resultaten",
        "dash_no_results": "Nog geen resultaten",
        "dash_clear_confirm": "Alle statistieken wissen?",
        "dash_cleared": "Statistieken gewist",
        "dashboard_toggle": "Dashboard tonen/verbergen",
        "export_pdf": "PDF exporteren",
        "pdf_saved": "PDF succesvol opgeslagen",
        "pdf_failed": "PDF-generatie mislukt",
        "pdf_subtitle": "Analyserapport",
        "pdf_patient": "Patiënt:",
        "pdf_date": "Datum:",
        "pdf_model": "Model:",
        "pdf_diagnosis": "DIAGNOSE",
        "pdf_confidence": "Vertrouwen",
        "pdf_class_probs": "KLASSEKANSEN",
        "pdf_disclaimer": (
            "Dit rapport is ontwikkeld door Babak Yousefian onder toezicht van ervaren artsen. "
            "Het is een professioneel analysedoeleinde en vereist geen doorverwijzing naar een arts."
        ),
        "history_title": "Geschiedenis",
        "history_toggle": "Geschiedenis tonen/verbergen",
        "history_empty": "Nog geen geschiedenis",
        "history_cleared": "Geschiedenis gewist",
        "history_clear_confirm": "Alle geschiedenis wissen?",
        # ✅ TOOLTIP KEYS
        "tooltip_title_login": "Inloggen",
        "tooltip_desc_login": "Log in op uw account",
        "tooltip_tip_login": "Toegang tot uw opgeslagen geschiedenis en statistieken",
        "tooltip_title_signup": "Registreren",
        "tooltip_desc_signup": "Een nieuw account aanmaken",
        "tooltip_title_account": "Account",
        "tooltip_desc_account": "Klik om het accountmenu te openen",
        "tooltip_desc_account_user": "Ingelogd als {user}",
        "tooltip_title_dashboard": "Dashboard",
        "tooltip_desc_dashboard": "Statistiekenpaneel tonen/verbergen — totaal, verdeling, recente resultaten",
        "tooltip_title_history": "Geschiedenis",
        "tooltip_desc_history": "Laatste 20 resultaten met miniaturen tonen/verbergen",
        "tooltip_title_visit": "Bezoek website",
        "tooltip_desc_visit": "Open de GitHub-pagina van het project in uw browser",
        "tooltip_tip_visit": "Opent in externe browser",
        "tooltip_title_insert": "Afbeelding invoegen",
        "tooltip_desc_insert": "Laad een sono-afbeelding om te analyseren",
        "tooltip_tip_insert": "U kunt ook een afbeelding slepen en neerzetten waar u maar wilt in het venster",
        "tooltip_title_process": "Verwerken",
        "tooltip_desc_process": "AI-analyse uitvoeren op de geladen afbeelding",
        "tooltip_tip_process": "Vereist eerst een geladen afbeelding",
        "tooltip_title_mute": "Muziek dempen",
        "tooltip_desc_mute": "Achtergrondmuziek pauzeren",
        "tooltip_title_unmute": "Muziek aan",
        "tooltip_desc_unmute": "Achtergrondmuziek hervatten",
        "tooltip_title_export": "PDF exporteren",
        "tooltip_desc_export": "Sla het huidige resultaat op als PDF-rapport",
        "tooltip_tip_export": "Bevat afbeelding, diagnose en klassekansen",
        "tooltip_title_dash_clear": "Statistieken wissen",
        "tooltip_desc_dash_clear": "Alle statistieken van dit account resetten",
        "tooltip_tip_dash_clear": "Deze actie kan niet ongedaan worden gemaakt",
        "tooltip_title_dash_close": "Dashboard sluiten",
        "tooltip_desc_dash_close": "Statistiekenpaneel verbergen",
        "tooltip_desc_dash_total": "Totaal aantal verwerkte afbeeldingen",
        "tooltip_desc_dash_avg": "Gemiddeld vertrouwen over alle resultaten",
        "tooltip_desc_dash_dist": "Verdeling van resultaten per graad",
        "tooltip_title_hist_clear": "Geschiedenis wissen",
        "tooltip_desc_hist_clear": "Alle opgeslagen resultaten en miniaturen verwijderen",
        "tooltip_tip_hist_clear": "Dit verwijdert ook alle afbeeldingsbestanden",
        "tooltip_title_hist_close": "Geschiedenis sluiten",
        "tooltip_desc_hist_close": "Geschiedenispaneel verbergen",
        "tooltip_tip_thumb_click": "Klik om dit resultaat opnieuw te laden",
        "tooltip_tip_recent": "Klik om dit resultaat opnieuw te laden",
        "tooltip_title_zoom_in": "Inzoomen",
        "tooltip_desc_zoom_in": "Afbeeldingsvergroting verhogen",
        "tooltip_title_zoom_out": "Uitzoomen",
        "tooltip_desc_zoom_out": "Afbeeldingsvergroting verlagen",
        "tooltip_title_zoom_reset": "Weergave resetten",
        "tooltip_desc_zoom_reset": "Afbeelding aanpassen aan venster",
        "tooltip_tip_zoom_reset": "U kunt ook dubbelklikken op de afbeelding",
        "notif_inference_ok": "Inferentie voltooid",
        "follow_system_theme": "Systeemthema volgen",
        "follow_system_theme_desc": "Automatisch schakelen tussen licht en donker op basis van systeeminstellingen",
        "gradcam_title": "Aandachtswarmtekaart",
        "gradcam_show": "Aandachtskaart tonen",
        "gradcam_toggle": "Warmtekaart wisselen",
        "gradcam_save": "Afbeelding opslaan",
        "gradcam_close": "Sluiten",
        "gradcam_saved": "Afbeelding succesvol opgeslagen",
        "gradcam_no_heatmap": "Geen aandachtskaart beschikbaar. Verwerk eerst een afbeelding.",
        "update_available": "Update beschikbaar",
        "update_check_failed": "Update-controle mislukt",
        "check_for_updates": "Controleren op updates",
        "check_for_updates_desc": "Controleer GitHub/website op de nieuwste versie",
        "up_to_date": "U bent up-to-date",
        "you_have_latest": "U gebruikt de nieuwste versie.",
        "check_internet": "Controleer uw internetverbinding en probeer het opnieuw.",
        "downloading": "Downloaden...",
        "verifying": "Verifiëren...",
        "download_now": "Nu downloaden",
        "install_now": "Installeren en herstarten",
        "skip_version": "Deze versie overslaan",
        "remind_later": "Later herinneren",
        "whats_new": "Wat is er nieuw",
        "no_changelog": "Geen changelog opgegeven.",
        "critical_update": "Kritieke update",
        "update_ready": "Klaar om te installeren",
        "app_will_restart": "De app wordt gesloten en opnieuw gestart om de update toe te passen.",
        "install_failed": "Installatie mislukt. Uw oude versie is veilig.",
        "download_failed": "Download mislukt.",
        "source_update_requires_zip": "Bron-update vereist een .zip-bestand.",
        "session_locked": "Sessie Vergrendeld",
        "locked_for_user": "Vergrendeld voor {user}",
        "unlock": "Ontgrendelen",
        "session_unlocked": "Sessie ontgrendeld",
        "auto_lock_notice": "De sessie wordt vergrendeld na 2 minuten inactiviteit.",
        "audit_log_title": "Auditlogboek",
        "audit_export": "Exporteren",
        "audit_exported": "Succesvol geëxporteerd",
        "audit_export_failed": "Export mislukt",
        "audit_search_placeholder": "Zoek acties of details...",
        "audit_col_time": "Tijd",
        "audit_col_level": "Niveau",
        "audit_col_category": "Categorie",
        "audit_col_action": "Actie",
        "audit_col_user": "Gebruiker",
        "audit_details_title": "Gebeurtenisdetails",
        "audit_status": "{count} van {total} gebeurtenissen weergegeven",
        "strength_very_weak": "Zeer zwak",
        "strength_weak": "Zwak",
        "strength_medium": "Gemiddeld",
        "strength_strong": "Sterk",
        "strength_very_strong": "Zeer sterk",
        "strength_rule_length": "8-16 tekens",
        "strength_rule_lowercase": "Kleine letter",
        "strength_rule_uppercase": "Hoofdletter",
        "strength_rule_digits": "Cijfers",
        "strength_rule_special": "Speciale tekens",
        "strength_add_chars": "Voeg nog {n} teken(s) toe",
        "strength_remove_chars": "Verwijder {n} teken(s)",
        "strength_need_digits": "Voeg nog {n} cijfer(s) toe",
        "strength_need_special": "Voeg nog {n} speciaal/speciale teken(s) toe",
        "strength_common_pattern": "Vermijd veelvoorkomende woorden",
        "strength_common_year": "Vermijd veelvoorkomende jaartallen",
        "strength_has_sequence": "Vermijd reeksen (1234, abcd)",
        "strength_has_repeats": "Vermijd herhalingen (aaaa, 1111)",
        "passwords_match": "Wachtwoorden komen overeen",
        "2fa_setup_title": "Tweefactorauthenticatie inschakelen",
        "2fa_setup_subtitle": "Scan de QR-code met Google Authenticator, Authy of Microsoft Authenticator.",
        "2fa_manual_secret": "Of voer deze code handmatig in:",
        "2fa_enter_code": "Voer de 6-cijferige code uit uw app in:",
        "2fa_verify_btn": "Verifiëren",
        "2fa_backup_codes_title": "Bewaar deze back-upcodes!",
        "2fa_backup_codes_hint": "Elke code kan slechts één keer worden gebruikt. Bewaar ze op een veilige plek.",
        "2fa_done_btn": "Klaar",
        "2fa_secret_copied": "Geheime code gekopieerd naar klembord",
        "2fa_invalid_code": "Ongeldige code. Probeer het opnieuw.",
        "2fa_invalid_code_format": "De code moet 6 cijfers bevatten.",
        "2fa_verify_title": "Tweefactorauthenticatie",
        "2fa_verify_subtitle": "Voer de 6-cijferige code uit uw authenticator-app in.",
        "2fa_enter_code_hint": "Voer een code in.",
        "2fa_use_backup_code": "Een back-upcode gebruiken",
        "2fa_use_totp_code": "App-code gebruiken",
        "2fa_enter_backup_code": "Voer een van uw back-upcodes in.",
        "2fa_section_title": "Beveiliging",
        "2fa_enabled_status": "Tweefactorauthenticatie is ingeschakeld",
        "2fa_disabled_status": "Tweefactorauthenticatie is uitgeschakeld",
        "2fa_enable_btn": "2FA inschakelen",
        "2fa_disable_btn": "2FA uitschakelen",
        "2fa_disable_title": "2FA uitschakelen",
        "2fa_disable_confirm": "Weet u het zeker? Uw account wordt minder veilig.",
        "2fa_disabled_success": "Tweefactorauthenticatie uitgeschakeld",
        "2fa_enabled_success": "Tweefactorauthenticatie ingeschakeld",
        "2fa_hint_title": "Hoe u deze QR-code gebruikt",
        "2fa_hint_step1": "Installeer een authenticator-app: Google Authenticator, Microsoft Authenticator of Authy.",
        "2fa_hint_step2": "Open de app, tik op de \"+\"-knop en kies \"QR-code scannen\".",
        "2fa_hint_step3": "Richt de camera van uw telefoon op de QR-code hierboven (gebruik de camera van de app, niet de camera-app van uw telefoon).",
        "2fa_hint_step4": "De app toont een 6-cijferige code. Voer deze in het vak hieronder in.",
        "2fa_manual_secret_hint": "Kunt u niet scannen? Kopieer de sleutel hieronder en voeg deze handmatig toe in uw app (\"Instelsleutel invoeren\").",
        "auto_lock_section_title": "Automatische vergrendeling",
        "auto_lock_enable": "Vergrendelen na inactiviteit",
        "auto_lock_timeout_label": "Vergrendelen na:",
        "auto_lock_warning_title": "Waarschuwing automatische vergrendeling",
        "auto_lock_warning_body": "Uw sessie wordt over {seconds} seconden vergrendeld wegens inactiviteit.",
        "lock_now": "Nu vergrendelen",
        "font_section_title": "Lettertype",
        "font_family_label": "Lettertypefamilie",
        "font_size_label": "Basisgrootte",
        "font_scale_label": "Weergaveschaal",
        "font_scale_small": "Klein",
        "font_scale_normal": "Normaal",
        "font_scale_large": "Groot",
        "font_scale_xlarge": "Extra groot",
        "font_scale_xxlarge": "Enorm",
        "font_preview_text": "Pa's wijze lynx bezag vroom het fikse aquaduct — 1234567890",
        "font_reset": "Terug naar standaard",
        "font_applied": "Lettertype bijgewerkt",
        "motion_section_title": "Beweging en animaties",
        "reduce_motion_label": "Beweging verminderen",
        "reduce_motion_hint": "Schakel animaties uit als ze u ongemakkelijk maken.",
        "empty_history_empty_title": "Nog geen geschiedenis",
        "empty_history_empty_subtitle": "Verwerk uw eerste afbeelding om deze hier te zien",
        "empty_history_empty_hint": "Elk resultaat wordt automatisch opgeslagen",
        "empty_history_no_user_title": "Log in om geschiedenis te zien",
        "empty_history_no_user_subtitle": "Uw opgeslagen resultaten verschijnen hier",
        "empty_history_no_user_hint": "Geschiedenis is privé voor elk account",
        "empty_dash_no_results_title": "Nog geen resultaten",
        "empty_dash_no_results_subtitle": "Uw recente analyses verschijnen hier",
        "empty_dash_no_results_hint": "Verwerk een afbeelding om te beginnen",
        "empty_dash_no_dist_title": "Geen distributiegegevens",
        "empty_dash_no_dist_subtitle": "Statistieken vullen zich terwijl u afbeeldingen verwerkt",
        "empty_dash_no_dist_hint": "Elke analyse telt",
        "empty_image_empty_title": "Geen afbeelding ingevoegd",
        "empty_image_empty_subtitle": "Sleep een afbeelding hierheen, of klik op Invoegen",
        "empty_image_empty_hint": "Ondersteunt PNG, JPG, JPEG, BMP, DCM",
        "empty_image_drop_hint_title": "Sleep om te analyseren",
        "empty_image_drop_hint_subtitle": "Laat de afbeelding los om te laden",
        "empty_image_drop_hint_hint": "Het formaat wordt automatisch gedetecteerd",
        "empty_processing_wait_title": "Verwerken...",
        "empty_processing_wait_subtitle": "AI analyseert uw afbeelding",
        "empty_processing_wait_hint": "Dit duurt meestal een paar seconden",
        "tour_show_again": "Tour opnieuw tonen",
        "tour_completed_msg": "Tour voltooid! Je bent klaar.",
        "tour_skipped_msg": "Tour overgeslagen. Je kunt hem opnieuw bekijken via het Account-menu.",
        "tour_btn_next": "Volgende",
        "tour_btn_prev": "Terug",
        "tour_btn_skip": "Overslaan",
        "tour_btn_finish": "Voltooien ✓",
        "tour_guest_login_title": "Inloggen",
        "tour_guest_login_subtitle": "Klik hier om in te loggen en alle functies te ontgrendelen.",
        "tour_guest_signup_title": "Account aanmaken",
        "tour_guest_signup_subtitle": "Nieuw hier? Klik hier om in enkele seconden een account aan te maken.",
        "tour_insert_title": "Afbeelding invoegen",
        "tour_insert_subtitle": "Laad een sono-afbeelding van schijf, of sleep deze naar een willekeurige plek in het venster.",
        "tour_process_title": "Afbeelding verwerken",
        "tour_process_subtitle": "Voer de AI-analyse uit op de geladen afbeelding. Resultaten verschijnen binnen enkele seconden.",
        "tour_export_title": "PDF exporteren",
        "tour_export_subtitle": "Sla het huidige resultaat op als een professioneel medisch PDF-rapport.",
        "tour_gradcam_title": "Aandachtswarmtekaart",
        "tour_gradcam_subtitle": "Bekijk waar de AI naar kijkt — een warmtekaart over de geanalyseerde afbeelding.",
        "tour_dashboard_title": "Dashboard",
        "tour_dashboard_subtitle": "Statistiekenpaneel tonen/verbergen: totaal afbeeldingen, gemiddeld vertrouwen, graadverdeling.",
        "tour_history_title": "Geschiedenis",
        "tour_history_subtitle": "Blader door je laatste 20 geanalyseerde afbeeldingen. Klik op een miniatuur om het resultaat opnieuw te laden.",
        "tour_visit_title": "Bezoek website",
        "tour_visit_subtitle": "Open de GitHub-pagina van het project in je browser voor documentatie en updates.",
        "tour_mute_title": "Muziek dempen",
        "tour_mute_subtitle": "Pauzeer of hervat de achtergrondmuziek op elk moment.",
        "tour_zoom_title": "Zoom-besturing",
        "tour_zoom_subtitle": "Gebruik het muiswiel om te zoomen, sleep om te pannen, en dubbelklik om te resetten.",
        "tour_account_title": "Accountmenu",
        "tour_account_subtitle": "Klik hier voor Instellingen, Accountinformatie, Auditlogboek en meer.",
        "suggestion_image_ready": "Afbeelding geladen — klaar om te verwerken",
        "suggestion_image_ready_sub": "Klik op Verwerken om de AI-analyse uit te voeren.",
        "suggestion_pdf_ready": "Resultaat klaar — exporteren als PDF?",
        "suggestion_pdf_ready_sub": "Sla een professioneel medisch rapport op met één klik.",
        "suggestion_gradcam_hint": "Nieuwsgierig wat de AI zag?",
        "suggestion_gradcam_hint_sub": "Bekijk de aandachtswarmtekaart over de afbeelding.",
        "suggestion_repeated_grid": "Je krijgt altijd {grid}",
        "suggestion_repeated_grid_sub": "Open het dashboard om de verdeling te zien.",
        "suggestion_history_hint": "Je geschiedenis groeit",
        "suggestion_history_hint_sub": "Blader door je laatste 20 geanalyseerde afbeeldingen.",
        "suggestion_dashboard_hint": "Je dashboard heeft interessante statistieken",
        "suggestion_dashboard_hint_sub": "Totalen, gemiddelden en verdeling in één oogopslag.",
        "suggestion_settings_hint": "Personaliseer de app",
        "suggestion_settings_hint_sub": "Wijzig taal, thema, lettertype en schakel 2FA in.",
        "suggestion_idle_help": "Een snelle gids nodig?",
        "suggestion_idle_help_sub": "Herhaal de onboarding-rondleiding op elk moment.",
        "image_not_liver_warning": "Deze afbeelding lijkt geen lever-echografie te zijn.\n\nVoer een correcte lever-echografie-afbeelding in.",
        "result_not_confident": "De AI kon deze afbeelding niet met zekerheid identificeren als een lever-echografie.\n\nVoer een correcte lever-echografie-afbeelding in en probeer het opnieuw.",
        "not_liver_title": "Geen lever-echografie",
        "not_liver_hint": "Gebruik een correcte lever-echografie (sonogram) afbeelding",
        "voice_welcome": "Welkom, {user}",
        "voice_goodbye": "Tot ziens",
        "voice_logged_out": "Succesvol uitgelogd",
        "voice_image_loaded": "Afbeelding succesvol geladen",
        "voice_image_rejected": "Dit is geen lever-echografie",
        "voice_inference_result": "{grid_name}, vertrouwen {conf} procent",
        "voice_pdf_saved": "PDF-rapport succesvol opgeslagen",
        "voice_error": "Er is een fout opgetreden",
        "voice_settings_applied": "Instellingen toegepast",
        "voice_language_changed": "Taal gewijzigd naar {lang}",
        "voice_theme_changed": "Thema gewijzigd",
        "voice_2fa_enabled": "Tweefactorauthenticatie ingeschakeld",
        "voice_2fa_disabled": "Tweefactorauthenticatie uitgeschakeld",
        "voice_locked": "Sessie vergrendeld",
        "voice_unlocked": "Sessie ontgrendeld",
        "voice_processing": "Afbeelding verwerken",
        "voice_processing_done": "Verwerking voltooid",
        "voice_voice_enabled": "Spraakfeedback ingeschakeld",
        "voice_voice_disabled": "Spraakfeedback uitgeschakeld",
    },

}


def tr(lang: str, key: str) -> str:
    """Translate a key."""
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["English"])
    return lang_dict.get(key, TRANSLATIONS["English"].get(key, key))

# ================================================================
# ✅ SMART TOOLTIPS (v3.0 — HTML rich tooltips)
# ================================================================

def make_tooltip(title: str, description: str = "", shortcut: str = "",
                 tip: str = "", theme: Dict = None) -> str:
    """
    ساخت HTML tooltip حرفه‌ای.

    Args:
        title: عنوان (مثلاً "📷 Insert Image")
        description: توضیح کوتاه
        shortcut: کلید میانبر (مثلاً "Ctrl+O")
        tip: نکته‌ی اضافی (مثلاً "You can also drag & drop")
        theme: دیکشنری تم (برای رنگ‌ها)

    Returns:
        رشته‌ی HTML برای QToolTip

    Example:
        >>> make_tooltip("📷 Insert Image", "Load a sono image", "Ctrl+O")
        '<b style="color:#00e5ff;font-size:12px;">📷 Insert Image</b>...'
    """
    if theme is None:
        theme = THEMES.get("theme_dark_aqua", {})

    primary = theme.get("primary", "#00e5ff")
    text_dim = theme.get("text_dim", "#94a3b8")
    success = theme.get("success", "#34d399")
    warning = theme.get("warning", "#fbbf24")

    parts = []

    # ✅ عنوان (Bold + رنگ primary)
    if title:
        parts.append(
            f'<b style="color:{primary};font-size:12px;">{title}</b>'
        )

    # ✅ توضیح (رنگ text_dim)
    if description:
        parts.append(
            f'<span style="color:{text_dim};font-size:10px;">{description}</span>'
        )

    # ✅ کلید میانبر (با کادر)
    if shortcut:
        parts.append(
            f'<span style="color:{success};font-size:10px;">'
            f'⌨ {shortcut}</span>'
        )

    # ✅ نکته (با آیکون)
    if tip:
        parts.append(
            f'<span style="color:{warning};font-size:10px;font-style:italic;">'
            f'💡 {tip}</span>'
        )

    # ✅ جدا کردن با خط افقی
    return "<br>".join(parts)


def build_standard_tooltip(widget, tr_lang: str, key_base: str,
                           theme: Dict, include_shortcut: str = "",
                           include_tip: str = "") -> None:
    """
    helper: ساخت tooltip استاندارد از TRANSLATIONS و اعمال به widget.

    Args:
        widget: QPushButton یا هر widget
        tr_lang: زبان فعلی (مثلاً "English")
        key_base: کلید پایه (مثلاً "insert_sono" → tooltip_title_insert_sono، ...)
        theme: دیکشنری تم
        include_shortcut: کلید میانبر (اگه خالی، نمایش داده نمی‌شه)
        include_tip: نکته‌ی اضافی (اگه خالی، نمایش داده نمی‌شه)
    """
    try:
        title = tr(tr_lang, f"tooltip_title_{key_base}")
        desc = tr(tr_lang, f"tooltip_desc_{key_base}")

        # ✅ اگه کلید ترجمه نبود، از عنوان دکمه استفاده کن
        if title == f"tooltip_title_{key_base}":
            title = ""
        if desc == f"tooltip_desc_{key_base}":
            desc = ""

        tip_text = ""
        if include_tip:
            tip_text = tr(tr_lang, include_tip)
            if tip_text == include_tip:
                tip_text = ""

        html = make_tooltip(
            title=title or widget.text(),
            description=desc,
            shortcut=include_shortcut,
            tip=tip_text,
            theme=theme,
        )
        widget.setToolTip(html)
    except Exception as e:
        print(f"⚠️ build_standard_tooltip failed: {e}")

# ================================================================
# ✅ ANIMATED DIALOG TRANSITIONS (v3.0 — fade + scale)
# ================================================================

def animate_dialog_open(dialog: QDialog, duration_ms: int = 280):
    """
    انیمیشن باز شدن دیالوگ:
    - fade: opacity 0.0 → 1.0
    - scale: 0.94 → 1.0 (با QGraphicsOpacityEffect + windowOpacity)
    - position: از پایین-وسط به مرکز (slide up)
    """
    try:
        # ✅ ذخیره‌ی موقعیت نهایی
        final_geom = dialog.geometry()

        # ✅ افکت opacity
        eff = QGraphicsOpacityEffect(dialog)
        eff.setOpacity(0.0)
        dialog.setGraphicsEffect(eff)

        # ✅ انیمیشن opacity
        anim_opacity = QPropertyAnimation(eff, b"opacity", dialog)
        anim_opacity.setDuration(duration_ms)
        anim_opacity.setStartValue(0.0)
        anim_opacity.setEndValue(1.0)
        anim_opacity.setEasingCurve(QEasingCurve.OutCubic)

        # ✅ انیمیشن position (slide up از ۲۰px پایین‌تر)
        start_geom = QRect(
            final_geom.x(),
            final_geom.y() + 20,
            final_geom.width(),
            final_geom.height(),
        )
        dialog.setGeometry(start_geom)
        anim_pos = QPropertyAnimation(dialog, b"geometry", dialog)
        anim_pos.setDuration(duration_ms)
        anim_pos.setStartValue(start_geom)
        anim_pos.setEndValue(final_geom)
        anim_pos.setEasingCurve(QEasingCurve.OutCubic)

        # ✅ اجرای همزمان
        group = QParallelAnimationGroup(dialog)
        group.addAnimation(anim_opacity)
        group.addAnimation(anim_pos)
        group.start()

        # ✅ ذخیره برای garbage collection
        dialog._anim_open_group = group
        dialog._anim_open_opacity = anim_opacity
        dialog._anim_open_pos = anim_pos

        # ✅ بعد از پایان، افکت رو حذف کن (که tooltipها و shadow خراب نشن)
        def _on_finished():
            try:
                dialog.setGraphicsEffect(None)
            except Exception:
                pass
        anim_opacity.finished.connect(_on_finished)
    except Exception as e:
        print(f"⚠️ animate_dialog_open failed: {e}")


def animate_dialog_close(dialog: QDialog, duration_ms: int = 200, on_finished=None):
    """
    انیمیشن بسته شدن دیالوگ (safe wrapper).
    """
    try:
        # ✅ چک کن دیالوگ هنوز زنده‌ست
        if dialog is None:
            if on_finished:
                try:
                    on_finished()
                except Exception:
                    pass
            return

        try:
            if not dialog.isVisible():
                # ✅ اگه دیالوگ بسته شده، مستقیم on_finished
                if on_finished:
                    on_finished()
                return
        except RuntimeError:
            # ✅ دیالوگ حذف شده
            if on_finished:
                try:
                    on_finished()
                except Exception:
                    pass
            return

        # ✅ افکت opacity
        eff = QGraphicsOpacityEffect(dialog)
        eff.setOpacity(1.0)
        dialog.setGraphicsEffect(eff)

        anim_opacity = QPropertyAnimation(eff, b"opacity", dialog)
        anim_opacity.setDuration(duration_ms)
        anim_opacity.setStartValue(1.0)
        anim_opacity.setEndValue(0.0)
        anim_opacity.setEasingCurve(QEasingCurve.InCubic)

        current_geom = dialog.geometry()
        end_geom = QRect(
            current_geom.x(),
            current_geom.y() + 20,
            current_geom.width(),
            current_geom.height(),
        )
        anim_pos = QPropertyAnimation(dialog, b"geometry", dialog)
        anim_pos.setDuration(duration_ms)
        anim_pos.setStartValue(current_geom)
        anim_pos.setEndValue(end_geom)
        anim_pos.setEasingCurve(QEasingCurve.InCubic)

        group = QParallelAnimationGroup(dialog)
        group.addAnimation(anim_opacity)
        group.addAnimation(anim_pos)
        group.start()

        dialog._anim_close_group = group
        dialog._anim_close_opacity = anim_opacity
        dialog._anim_close_pos = anim_pos

        def _on_finished():
            try:
                dialog.setGraphicsEffect(None)
            except Exception:
                pass
            if on_finished:
                try:
                    on_finished()
                except Exception:
                    pass

        anim_opacity.finished.connect(_on_finished)
    except Exception as e:
        print(f"⚠️ animate_dialog_close failed: {e}")
        # ✅ fallback
        if on_finished:
            try:
                on_finished()
            except Exception:
                pass
        else:
            try:
                dialog.accept()
            except Exception:
                pass

# ================================================================
# ✅ SYSTEM THEME DETECTION (Windows / macOS / Linux)
# ================================================================

def detect_system_theme() -> str:
    """
    تشخیص تم سیستم و برگرداندن کلید تم مناسب.
    - Windows: رجیستری (AppsUseLightTheme)
    - macOS: defaults read
    - Linux: gsettings / محیط متغیر
    برمی‌گرداند: "theme_dark_aqua" یا "theme_solarized_white"
    """
    try:
        # Windows
        if sys.platform.startswith("win"):
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                )
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                winreg.CloseKey(key)
                return "theme_solarized_white" if value == 1 else "theme_dark_aqua"
            except Exception:
                pass

        # macOS
        elif sys.platform == "darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["defaults", "read", "-g", "AppleInterfaceStyle"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0 and "Dark" in result.stdout:
                    return "theme_dark_aqua"
                return "theme_solarized_white"
            except Exception:
                pass

        # Linux
        else:
            try:
                import subprocess
                # gsettings (GNOME)
                result = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    if "dark" in result.stdout.lower():
                        return "theme_dark_aqua"
                    if "light" in result.stdout.lower():
                        return "theme_solarized_white"
                # fallback: GTK_THEME
                gtk = os.environ.get("GTK_THEME", "").lower()
                if "dark" in gtk:
                    return "theme_dark_aqua"
                if "light" in gtk:
                    return "theme_solarized_white"
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ detect_system_theme failed: {e}")

    # fallback پیش‌فرض
    return "theme_dark_aqua"

# ================================================================
# ✅ THEMES (3 Themes — Enhanced with multi-color gradients)
# ================================================================

THEMES = {
    "theme_dark_aqua": {
        "name": "Dark Aqua-Purple",
        "bg_gradient_start": "#0f0c29",
        "bg_gradient_mid": "#302b63",
        "bg_gradient_end": "#24243e",
        "primary": "#00e5ff",
        "secondary": "#c084fc",
        "accent": "#f472b6",
        "text": "#f1f5f9",
        "text_dim": "#94a3b8",
        "button_bg": "#00e5ff",
        "button_text": "#0a0a14",
        "button_hover": "#67e8f9",
        "input_bg": "rgba(30, 41, 59, 0.85)",
        "input_border": "#00e5ff",
        "glass_bg": "rgba(15, 20, 40, 0.25)",
        "glass_border": "rgba(0, 229, 255, 0.35)",
        "glass_highlight": "rgba(0, 229, 255, 0.10)",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#f87171",
        "shadow": "rgba(0, 229, 255, 0.5)",
    },
    "theme_solarized_white": {
        "name": "Solarized White",
        "bg_gradient_start": "#fdf6e3",
        "bg_gradient_mid": "#f5efdc",
        "bg_gradient_end": "#eee8d5",
        "primary": "#268bd2",
        "secondary": "#6c71c4",
        "accent": "#d33682",
        "text": "#073642",
        "text_dim": "#586e75",
        "button_bg": "#268bd2",
        "button_text": "#fdf6e3",
        "button_hover": "#2aa198",
        "input_bg": "rgba(255, 255, 255, 0.95)",
        "input_border": "#93a1a1",
        "glass_bg": "rgba(253, 246, 227, 0.35)",
        "glass_border": "rgba(38, 139, 210, 0.35)",
        "glass_highlight": "rgba(38, 139, 210, 0.08)",
        "success": "#859900",
        "warning": "#b58900",
        "danger": "#dc322f",
        "shadow": "rgba(38, 139, 210, 0.3)",
    },
    "theme_solarized_dark": {
        "name": "Solarized Dark",
        "bg_gradient_start": "#001f27",
        "bg_gradient_mid": "#002b36",
        "bg_gradient_end": "#073642",
        "primary": "#2aa198",
        "secondary": "#6c71c4",
        "accent": "#d33682",
        "text": "#93a1a1",
        "text_dim": "#586e75",
        "button_bg": "#2aa198",
        "button_text": "#002b36",
        "button_hover": "#268bd2",
        "input_bg": "rgba(7, 54, 66, 0.9)",
        "input_border": "#2aa198",
        "glass_bg": "rgba(0, 31, 39, 0.35)",
        "glass_border": "rgba(42, 161, 152, 0.4)",
        "glass_highlight": "rgba(42, 161, 152, 0.10)",
        "success": "#859900",
        "warning": "#b58900",
        "danger": "#dc322f",
        "shadow": "rgba(42, 161, 152, 0.4)",
    },
    "theme_dark_forest": {
        "name": "Dark Forest Green",
        "bg_gradient_start": "#0a1f0a",  # سبز خیلی تیره
        "bg_gradient_mid": "#0d3318",  # سبز لجنی تیره
        "bg_gradient_end": "#06120a",  # سبز تقریباً سیاه
        "primary": "#4ade80",  # سبز روشن (نئون)
        "secondary": "#16a34a",  # سبز متوسط
        "accent": "#84cc16",  # سبز-زیتونی
        "text": "#dcfce7",  # سبز خیلی روشن (متن)
        "text_dim": "#86efac",  # سبز ملایم (متن کم‌رنگ)
        "button_bg": "#4ade80",
        "button_text": "#0a1f0a",
        "button_hover": "#86efac",
        "input_bg": "rgba(13, 51, 24, 0.85)",  # سبز لجنی شفاف
        "input_border": "#4ade80",
        "glass_bg": "rgba(10, 31, 10, 0.30)",
        "glass_border": "rgba(74, 222, 128, 0.40)",
        "glass_highlight": "rgba(74, 222, 128, 0.12)",
        "success": "#22c55e",
        "warning": "#eab308",
        "danger": "#ef4444",
        "shadow": "rgba(74, 222, 128, 0.5)",
    },
}

LANGUAGE_ORDER = [
    "Persian", "English", "USA", "France", "Russian",
    "Greece", "Germany", "Arabic", "Ukranian", "Italy",
    "China", "Japan", "SouthKorea", "NorthKorea",
    "India", "Portugal", "Brazil",
    "Argentina", "Morocco", "Turkey", "Switzerland",
    "Sweden", "Wales", "Nederland",
]

THEME_ORDER = [
    "theme_dark_aqua",
    "theme_solarized_white",
    "theme_solarized_dark",
    "theme_dark_forest",   # ← تم جدید
]


# ================================================================
# ✅ SOUND MANAGER (per-click single play, no overlap)
# ================================================================

class SoundManager:
    """Plays UI sound effects exactly once per click.
    Uses pygame.mixer.Sound on dedicated channels — no overlap of the same sound.
    """

    SOUNDS_DIR = SOUNDS_DIR
    EXT = ".mp3"  # ⚠️ اگه فایل‌هات mp3 هستن، به ".mp3" تغییر بده

    NAMES = [
        "DashboadClickButtton",
        "ErrorSound",
        "linkClick",
        "loginSound",
        "logoutSound",
        "muteClick",
        "ok_sound",
        "processButton",
        "signinSound",
        "signoutSound",
        "simpleClick",
        "warningSound",
    ]

    _instance = None
    _sounds = {}
    _channels = {}
    _enabled = False
    _volume = 0.6

    def __init__(self):
        self._init_sounds()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = SoundManager()
        return cls._instance

    def _init_sounds(self):
        if not HAS_PYGAME:
            print("⚠️ SoundManager: pygame not available")
            return

        if not os.path.isdir(self.SOUNDS_DIR):
            print(f"⚠️ SoundManager: sounds dir not found: {self.SOUNDS_DIR}")
            return

        loaded = 0
        for name in self.NAMES:
            path = os.path.join(self.SOUNDS_DIR, name + self.EXT)
            if not os.path.exists(path):
                alt_path = os.path.join(self.SOUNDS_DIR, name + ".mp3")
                if os.path.exists(alt_path):
                    path = alt_path
                else:
                    continue
            try:
                snd = pygame.mixer.Sound(path)
                snd.set_volume(self._volume)
                self._sounds[name] = snd
                self._channels[name] = pygame.mixer.find_channel(True)
                loaded += 1
            except Exception as e:
                print(f"⚠️ SoundManager: failed to load {name}: {e}")

        self._enabled = loaded > 0
        print(f"🔊 SoundManager loaded {loaded}/{len(self.NAMES)} sounds")

    def play(self, name: str):
        """Play a sound exactly once. Stops previous instance if still playing."""
        if not self._enabled or name not in self._sounds:
            return
        try:
            channel = self._channels.get(name)
            if channel is None:
                channel = pygame.mixer.find_channel(True)
                self._channels[name] = channel
            if channel is None:
                return
            if channel.get_busy():
                return  # ← نادیده بگیر
            channel.play(self._sounds[name])
        except Exception:
            pass

    def set_volume(self, vol: float):
        self._volume = max(0.0, min(1.0, vol))
        for snd in self._sounds.values():
            try:
                snd.set_volume(self._volume)
            except Exception:
                pass

    def set_enabled(self, enabled: bool):
        self._enabled = enabled


def attach_sound(button, sound_name: str):
    """Attach a click sound to a QPushButton (plays exactly once per click)."""
    def _on_click(_checked=False):
        try:
            SoundManager.get().play(sound_name)
        except Exception:
            pass
    button.clicked.connect(_on_click)


# ================================================================
# ✅ SOUND-AWARE MESSAGE BOXES
# ================================================================

class SoundMessageBox(QMessageBox):
    """QMessageBox that plays a sound when shown."""

    def __init__(self, icon_type: str, parent=None):
        super().__init__(parent)
        self._icon_type = icon_type

    def showEvent(self, event):
        sound_map = {
            "warning": "warningSound",
            "critical": "ErrorSound",
            "information": "ok_sound",
            "question": "warningSound",
        }
        sound = sound_map.get(self._icon_type, "simpleClick")
        try:
            SoundManager.get().play(sound)
        except Exception:
            pass
        super().showEvent(event)


def show_message(parent, icon: str, title: str, text: str):
    """Show a QMessageBox with sound — safe wrapper (no crash)."""
    try:
        # ✅ اگه parent در حال بسته شدنه، نادیده بگیر
        if parent is not None:
            try:
                if not parent.isVisible():
                    # ✅ اگه parent دیده نمی‌شه، از None استفاده کن
                    parent = None
            except Exception:
                parent = None

        box = SoundMessageBox(icon, parent)
        box.setIcon({
            "warning": QMessageBox.Warning,
            "critical": QMessageBox.Critical,
            "information": QMessageBox.Information,
            "question": QMessageBox.Question,
        }.get(icon, QMessageBox.Information))
        box.setWindowTitle(title or "")
        box.setText(text or "")
        box.setStandardButtons(QMessageBox.Ok)
        return box.exec_()
    except Exception as e:
        print(f"⚠️ show_message failed: {e}")
        return QMessageBox.Ok


def ask_question(parent, title: str, text: str) -> bool:
    """Show a Yes/No question box with sound. Returns True if Yes."""
    box = SoundMessageBox("question", parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    box.setDefaultButton(QMessageBox.No)
    return box.exec_() == QMessageBox.Yes


# ================================================================
# ✅ USER DATABASE (JSON-based, PBKDF2 hashing + migration)
# ================================================================

class UserDB:
    """Simple JSON-based user database with PBKDF2 hashing."""

    PBKDF2_ITERATIONS = 260000

    def __init__(self, path: str = USERS_DB_PATH):
        self.path = path
        # ✅ Use encrypted store
        try:
            from core.crypto import EncryptedJSONStore
            self._store = EncryptedJSONStore(path, default={"users": {}})
        except Exception as e:
            print(f"⚠️ Failed to init EncryptedJSONStore: {e}")
            self._store = None
        self.data = self._load()

    def _load(self) -> Dict:
        if self._store is not None:
            try:
                data = self._store.load()
                if not isinstance(data, dict) or "users" not in data:
                    data = {"users": {}}
                return data
            except Exception as e:
                print(f"⚠️ EncryptedJSONStore.load failed: {e}")
                return {"users": {}}

        # ✅ Fallback: raw JSON
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"users": {}}
        return {"users": {}}

    def _save(self):
        if self._store is not None:
            try:
                self._store.save(self.data)
                return
            except Exception as e:
                print(f"⚠️ EncryptedJSONStore.save failed: {e}")

        # ✅ Fallback: raw JSON
        tmp_path = self.path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, self.path)
        except Exception:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)

    def _hash(self, password: str, salt: Optional[str] = None) -> str:
        if salt is None:
            salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            self.PBKDF2_ITERATIONS,
            dklen=32,
        )
        return f"{salt}${dk.hex()}"

    def _verify_hash(self, password: str, stored: str) -> bool:
        if not stored:
            return False
        if "$" not in stored:
            legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
            return hmac.compare_digest(legacy, stored)
        try:
            salt, _ = stored.split("$", 1)
        except ValueError:
            return False
        return hmac.compare_digest(stored, self._hash(password, salt))

    def user_exists(self, username: str) -> bool:
        return username in self.data.get("users", {})

    def create_user(
        self,
        username: str,
        password: str,
        first_name: str = "",
        last_name: str = "",
        email: str = "",
    ) -> bool:
        if self.user_exists(username):
            return False
        self.data.setdefault("users", {})[username] = {
            "password_hash": self._hash(password),
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "created_at": datetime.now().isoformat(),
        }
        self._save()
        return True

    def verify(self, username: str, password: str) -> bool:
        users = self.data.get("users", {})
        if username not in users:
            return False
        stored = users[username].get("password_hash", "")
        if self._verify_hash(password, stored):
            if "$" not in stored:
                users[username]["password_hash"] = self._hash(password)
                self._save()
            return True
        return False

    def get_user_info(self, username: str) -> Optional[Dict]:
        return self.data.get("users", {}).get(username)

    def update_user(self, username: str, **kwargs) -> bool:
        users = self.data.get("users", {})
        if username not in users:
            return False
        for k, v in kwargs.items():
            if k == "password":
                users[username]["password_hash"] = self._hash(v)
            else:
                users[username][k] = v
        self._save()
        return True

    def delete_user(self, username: str) -> bool:
        users = self.data.get("users", {})
        if username in users:
            del users[username]
            self._save()
            return True
        return False
    # ============================================================
    # ✅ 2FA (Two-Factor Authentication)
    # ============================================================

    def get_2fa_status(self, username: str) -> dict:
        """
        Return 2FA status for a user.

        Returns:
            {
                "enabled": bool,
                "secret": str | "",
                "backup_codes_hashed": list[str],
                "last_used_step": int | None,
                "enabled_at": str | "",
            }
        """
        users = self.data.get("users", {})
        user = users.get(username, {})
        return {
            "enabled": bool(user.get("2fa_enabled", False)),
            "secret": user.get("2fa_secret", ""),
            "backup_codes_hashed": list(user.get("2fa_backup_codes", [])),
            "last_used_step": user.get("2fa_last_step"),
            "enabled_at": user.get("2fa_enabled_at", ""),
        }

    def set_2fa_secret(self, username: str, secret: str) -> bool:
        """
        Store a pending 2FA secret (not yet enabled).
        Called during setup, before user verifies first code.
        """
        users = self.data.get("users", {})
        if username not in users:
            return False
        users[username]["2fa_secret"] = secret
        users[username]["2fa_enabled"] = False
        self._save()
        return True

    def enable_2fa(self, username: str, backup_codes_hashed: list) -> bool:
        """
        Enable 2FA after user successfully verified first code.
        """
        users = self.data.get("users", {})
        if username not in users:
            return False
        user = users[username]
        if not user.get("2fa_secret"):
            return False
        user["2fa_enabled"] = True
        user["2fa_backup_codes"] = list(backup_codes_hashed)
        user["2fa_last_step"] = None
        user["2fa_enabled_at"] = datetime.now().isoformat()
        self._save()
        return True

    def disable_2fa(self, username: str) -> bool:
        """Disable 2FA and remove all 2FA data."""
        users = self.data.get("users", {})
        if username not in users:
            return False
        user = users[username]
        user.pop("2fa_secret", None)
        user.pop("2fa_backup_codes", None)
        user.pop("2fa_enabled", None)
        user.pop("2fa_last_step", None)
        user.pop("2fa_enabled_at", None)
        self._save()
        return True

    def update_2fa_last_step(self, username: str, step: int) -> bool:
        """Store the last accepted time-step (replay protection)."""
        users = self.data.get("users", {})
        if username not in users:
            return False
        users[username]["2fa_last_step"] = int(step)
        self._save()
        return True

    def use_backup_code(self, username: str, index: int) -> bool:
        """Remove a used backup code (single-use)."""
        users = self.data.get("users", {})
        if username not in users:
            return False
        codes = users[username].get("2fa_backup_codes", [])
        if 0 <= index < len(codes):
            codes.pop(index)
            users[username]["2fa_backup_codes"] = codes
            self._save()
            return True
        return False

    def regenerate_2fa_backup_codes(self, username: str, new_hashed: list) -> bool:
        """Replace backup codes (invalidates old ones)."""
        users = self.data.get("users", {})
        if username not in users:
            return False
        if not users[username].get("2fa_enabled"):
            return False
        users[username]["2fa_backup_codes"] = list(new_hashed)
        self._save()
        return True

class TwoFactorSetupDialog(QDialog):
    """
    Two-step setup dialog with full user guidance.

    Step 1: Instructions + QR code + manual secret + code input
    Step 2: Backup codes (one-time display)
    """
    setup_completed = pyqtSignal(str, list)

    def __init__(self, db: UserDB, username: str, theme: Dict,
                 lang: str = "English", parent=None):
        super().__init__(parent)
        self.db = db
        self.username = username
        self.theme = theme
        self.lang = lang
        self._secret = ""
        self._backup_codes_plain = []
        self._step = 1

        self.setWindowTitle(tr(lang, "2fa_setup_title"))
        # ✅ بزرگ‌تر — چون محتوای بیشتری داره (hints)
        self.setFixedSize(580, 820)
        self.setMinimumSize(580, 700)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

        try:
            from gui.dialogs.update_dialogs import _with_backdrop
            _with_backdrop(self, parent, theme)
        except Exception as e:
            print(f"⚠️ TwoFactorSetupDialog backdrop failed: {e}")

        QTimer.singleShot(0, lambda: animate_dialog_open(self))

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        # ✅ ScrollArea — چون محتوا بیشتر از ارتفاع می‌شه
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("tfa_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("tfa_scroll_content")

        layout = QVBoxLayout(self.scroll_content)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(18)

        # ============================================================
        # Title
        # ============================================================
        self.title_label = QLabel("🔐 " + tr(self.lang, "2fa_setup_title"))
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(tr(self.lang, "2fa_setup_subtitle"))
        self.subtitle_label.setObjectName("subtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.subtitle_label)

        # ============================================================
        # ✅ HINT BOX — راهنمای کامل برای کاربر
        # ============================================================
        self.hint_box = QFrame()
        self.hint_box.setObjectName("hint_box")
        hint_layout = QVBoxLayout(self.hint_box)
        hint_layout.setContentsMargins(14, 12, 14, 12)
        hint_layout.setSpacing(8)

        hint_title = QLabel("💡 " + tr(self.lang, "2fa_hint_title"))
        hint_title.setObjectName("hint_title")
        hint_layout.addWidget(hint_title)

        # ۴ مرحله راهنما
        steps = [
            ("1️⃣", "2fa_hint_step1"),
            ("2️⃣", "2fa_hint_step2"),
            ("3️⃣", "2fa_hint_step3"),
            ("4️⃣", "2fa_hint_step4"),
        ]
        for icon, key in steps:
            step_row = QHBoxLayout()
            step_row.setSpacing(8)
            icon_lbl = QLabel(icon)
            icon_lbl.setObjectName("hint_icon")
            icon_lbl.setFixedWidth(24)
            icon_lbl.setAlignment(Qt.AlignTop)
            text_lbl = QLabel(tr(self.lang, key))
            text_lbl.setObjectName("hint_text")
            text_lbl.setWordWrap(True)
            step_row.addWidget(icon_lbl)
            step_row.addWidget(text_lbl, 1)
            hint_layout.addLayout(step_row)

        layout.addWidget(self.hint_box)

        # ============================================================
        # QR Code
        # ============================================================
        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignCenter)
        self.qr_label.setObjectName("qr_label")
        self.qr_label.setFixedSize(280, 280)
        layout.addWidget(self.qr_label, 0, Qt.AlignCenter)

        # ============================================================
        # Manual secret (fallback)
        # ============================================================
        self.manual_hint = QLabel(
            "📋 " + tr(self.lang, "2fa_manual_secret_hint")
        )
        self.manual_hint.setObjectName("manual_hint")
        self.manual_hint.setWordWrap(True)
        layout.addWidget(self.manual_hint)

        secret_row = QHBoxLayout()
        secret_row.setSpacing(8)

        self.secret_label = QLabel(tr(self.lang, "2fa_manual_secret"))
        self.secret_label.setObjectName("field_label")
        secret_row.addWidget(self.secret_label)

        self.secret_display = QLineEdit()
        self.secret_display.setReadOnly(True)
        self.secret_display.setObjectName("secret_input")
        secret_row.addWidget(self.secret_display, 1)

        self.copy_secret_btn = QPushButton("📋")
        self.copy_secret_btn.setObjectName("icon_btn")
        self.copy_secret_btn.setFixedSize(36, 36)
        self.copy_secret_btn.setCursor(Qt.PointingHandCursor)
        self.copy_secret_btn.clicked.connect(self._copy_secret)
        secret_row.addWidget(self.copy_secret_btn)

        layout.addLayout(secret_row)

        # ============================================================
        # Code input
        # ============================================================
        layout.addSpacing(6)

        self.code_label = QLabel(tr(self.lang, "2fa_enter_code"))
        self.code_label.setObjectName("field_label")
        self.code_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.code_label)

        self.code_input = QLineEdit()
        self.code_input.setObjectName("code_input")
        self.code_input.setPlaceholderText("000000")
        self.code_input.setMaxLength(6)
        self.code_input.setAlignment(Qt.AlignCenter)
        self.code_input.setFixedHeight(54)
        self.code_input.returnPressed.connect(self._on_confirm_code)
        layout.addWidget(self.code_input)

        self.error_label = QLabel("")
        self.error_label.setObjectName("error_label")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        # ============================================================
        # Backup codes container (hidden initially)
        # ============================================================
        self.backup_container = QFrame()
        self.backup_container.setObjectName("backup_container")
        backup_layout = QVBoxLayout(self.backup_container)
        backup_layout.setContentsMargins(14, 14, 14, 14)
        backup_layout.setSpacing(8)

        backup_title = QLabel("🔑 " + tr(self.lang, "2fa_backup_codes_title"))
        backup_title.setObjectName("backup_title")
        backup_title.setWordWrap(True)
        backup_layout.addWidget(backup_title)

        backup_hint = QLabel(tr(self.lang, "2fa_backup_codes_hint"))
        backup_hint.setObjectName("backup_hint")
        backup_hint.setWordWrap(True)
        backup_layout.addWidget(backup_hint)

        self.backup_codes_label = QLabel("")
        self.backup_codes_label.setObjectName("backup_codes")
        self.backup_codes_label.setAlignment(Qt.AlignCenter)
        self.backup_codes_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.backup_codes_label.setWordWrap(True)
        backup_layout.addWidget(self.backup_codes_label)

        self.backup_container.setVisible(False)
        layout.addWidget(self.backup_container)

        layout.addStretch()

        self.scroll.setWidget(self.scroll_content)
        container_layout.addWidget(self.scroll)

        # ============================================================
        # ✅ Buttons — بیرون از scroll (همیشه دیده بشه)
        # ============================================================
        button_bar = QFrame()
        button_bar.setObjectName("button_bar")
        button_bar_layout = QHBoxLayout(button_bar)
        button_bar_layout.setContentsMargins(30, 12, 30, 18)
        button_bar_layout.setSpacing(10)

        self.cancel_btn = QPushButton("✕  " + tr(self.lang, "cancel"))
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.setMinimumHeight(44)
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_bar_layout.addWidget(self.cancel_btn)

        self.confirm_btn = QPushButton("✓  " + tr(self.lang, "2fa_verify_btn"))
        self.confirm_btn.setObjectName("primary_btn")
        self.confirm_btn.setMinimumHeight(44)
        self.confirm_btn.setCursor(Qt.PointingHandCursor)
        self.confirm_btn.clicked.connect(self._on_confirm_code)
        button_bar_layout.addWidget(self.confirm_btn, 1)

        container_layout.addWidget(button_bar)

        # ✅ Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self._on_cancel)
        close_btn.setParent(self.container)
        close_btn.move(540, 10)
        close_btn.raise_()

    def _apply_theme(self):
        t = self.theme

        # ============================================================
        # ✅ Container + close button + button_bar (بیرون از scroll)
        # ============================================================
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)

        # ============================================================
        # ✅ ScrollArea — شفاف کامل (بدون سفیدی)
        # ============================================================
        self.scroll.setStyleSheet(f"""
            QScrollArea#tfa_scroll {{
                background: transparent;
                border: none;
            }}
            QScrollArea#tfa_scroll > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollArea#tfa_scroll > QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 0px;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                border: none;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {t['glass_border']};
                border-radius: 4px;
                min-height: 30px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {t['primary']};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
                background: transparent;
                border: none;
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: 0px;
                border: none;
            }}
        """)

        # ============================================================
        # ✅ Scroll content — همه‌ی المان‌های داخل
        # ============================================================
        self.scroll_content.setStyleSheet(f"""
            QWidget#tfa_scroll_content {{
                background: transparent;
            }}
            QLabel {{
                color: {t['text']};
                background: transparent;
            }}
            QLabel#title {{
                font-size: 20px;
                font-weight: bold;
                color: {t['primary']};
                padding-bottom: 2px;
            }}
            QLabel#subtitle {{
                font-size: 12px;
                color: {t['text_dim']};
                padding-bottom: 4px;
            }}

            /* ============ HINT BOX ============ */
            QFrame#hint_box {{
                background: {t['input_bg']};
                border: 2px solid {t['primary']};
                border-left: 5px solid {t['primary']};
                border-radius: 14px;
            }}
            QLabel#hint_title {{
                font-size: 13px;
                font-weight: bold;
                color: {t['primary']};
                padding-bottom: 2px;
            }}
            QLabel#hint_icon {{
                font-size: 16px;
            }}
            QLabel#hint_text {{
                font-size: 12px;
                color: {t['text']};
                line-height: 1.6;
            }}

            /* ============ QR CODE ============ */
            QLabel#qr_label {{
                background: white;
                border: 8px solid white;
                padding: 0px;
                border-radius: 0px;
            }}

            QLabel#manual_hint {{
                font-size: 11px;
                color: {t['warning']};
                font-style: italic;
                padding: 6px 0px;
            }}
            QLabel#field_label {{
                font-size: 12px;
                font-weight: bold;
                color: {t['text']};
                padding-top: 6px;
                padding-bottom: 2px;
            }}

            /* ============ SECRET INPUT ============ */
            QLineEdit#secret_input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px 12px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                font-weight: bold;
                color: {t['primary']};
                letter-spacing: 1px;
            }}
            QLineEdit#secret_input:focus {{
                border: 2px solid {t['primary']};
            }}

            /* ============ CODE INPUT ============ */
            QLineEdit#code_input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 12px;
                padding: 10px;
                font-size: 26px;
                font-weight: bold;
                letter-spacing: 10px;
                color: {t['primary']};
            }}
            QLineEdit#code_input:focus {{
                border: 2px solid {t['primary']};
                background: {t['glass_highlight']};
            }}

            /* ============ ERROR ============ */
            QLabel#error_label {{
                font-size: 11px;
                font-weight: bold;
                color: {t['danger']};
                padding: 4px 0px;
            }}

            /* ============ BACKUP CODES ============ */
            QFrame#backup_container {{
                background: {t['input_bg']};
                border: 2px solid {t['warning']};
                border-radius: 14px;
            }}
            QLabel#backup_title {{
                font-size: 14px;
                font-weight: bold;
                color: {t['warning']};
                padding-bottom: 4px;
            }}
            QLabel#backup_hint {{
                font-size: 11px;
                color: {t['text_dim']};
                padding-bottom: 6px;
            }}
            QLabel#backup_codes {{
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                color: {t['text']};
                padding: 14px;
                background: {t['glass_highlight']};
                border: 1px solid {t['glass_border']};
                border-radius: 10px;
                line-height: 1.6;
            }}

            /* ============ ICON BUTTON (📋) ============ */
            QPushButton#icon_btn {{
                background: {t['input_bg']};
                color: {t['primary']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton#icon_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
                border-color: {t['primary']};
            }}
            QPushButton#icon_btn:pressed {{
                background: {t['secondary']};
            }}
        """)

        # ============================================================
        # ✅ BUTTON BAR — جدا از scroll (همیشه دیده می‌شه)
        # ============================================================
        # ⚠️ button_bar خودش یه QFrame با objectName="button_bar" هست
        #    ولی چون توی container هست (نه توی scroll_content)،
        #    باید از طریق container's stylesheet استایل بدیم
        #    ولی container.setStyleSheet قبلاً ست شده — پس از
        #    findChild استفاده می‌کنیم و مستقیم روش ست می‌کنیم

        if hasattr(self, "confirm_btn"):
            self.confirm_btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 {t['primary']},
                        stop:0.5 {t['accent']},
                        stop:1 {t['secondary']}
                    );
                    color: {t['button_text']};
                    border: none;
                    border-radius: 12px;
                    padding: 12px 24px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 140px;
                }}
                QPushButton:hover {{
                    background: {t['button_hover']};
                }}
                QPushButton:pressed {{
                    background: {t['accent']};
                }}
            """)

        if hasattr(self, "cancel_btn"):
            self.cancel_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {t['input_bg']};
                    color: {t['text']};
                    border: 2px solid {t['input_border']};
                    border-radius: 12px;
                    padding: 12px 24px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 100px;
                }}
                QPushButton:hover {{
                    background: {t['danger']};
                    color: {t['button_text']};
                    border-color: {t['danger']};
                }}
                QPushButton:pressed {{
                    background: {t['danger']};
                }}
            """)

        # ✅ استایل button_bar (خط جداکننده بالاش)
        try:
            button_bar = self.container.findChild(QFrame, "button_bar")
            if button_bar is not None:
                button_bar.setStyleSheet(f"""
                    QFrame#button_bar {{
                        background: transparent;
                        border-top: 1px solid {t['glass_border']};
                        border-bottom: none;
                        border-left: none;
                        border-right: none;
                    }}
                """)
        except Exception as e:
            print(f"⚠️ button_bar style failed: {e}")

    def _generate_secret_and_qr(self):
        """Generate secret + QR on first show."""
        try:
            from core.security.totp import (
                generate_secret, build_provisioning_uri, generate_qr_pixmap,
            )
            self._secret = generate_secret()
            self.db.set_2fa_secret(self.username, self._secret)

            uri = build_provisioning_uri(
                self._secret, self.username,
                email=self.db.get_user_info(self.username).get("email", ""),
            )
            pixmap = generate_qr_pixmap(uri, size=300)
            if pixmap:
                self.qr_label.setPixmap(pixmap)
            else:
                self.qr_label.setText("QR unavailable")
                self.qr_label.setStyleSheet(
                    "color: #f87171; font-size: 14px; background: white;"
                )

            self.secret_display.setText(self._secret)
        except Exception as e:
            print(f"⚠️ _generate_secret_and_qr failed: {e}")
            show_message(
                self, "critical",
                tr(self.lang, "error"),
                f"2FA setup failed:\n{e}",
            )
            self.reject()

    def showEvent(self, event):
        super().showEvent(event)
        if self._step == 1 and not self._secret:
            self._generate_secret_and_qr()

    def _copy_secret(self):
        try:
            QApplication.clipboard().setText(self._secret)
            SoundManager.get().play("ok_sound")
            self._show_error(tr(self.lang, "2fa_secret_copied"), success=True)
        except Exception:
            pass

    def _on_confirm_code(self):
        try:
            from core.security.totp import (
                verify_code, generate_backup_codes, hash_backup_codes,
            )
            code = self.code_input.text().strip()
            if len(code) != 6:
                self._show_error(tr(self.lang, "2fa_invalid_code_format"))
                return

            ok, step = verify_code(self._secret, code)
            if not ok:
                SoundManager.get().play("warningSound")
                self._show_error(tr(self.lang, "2fa_invalid_code"))
                self.code_input.clear()
                self.code_input.setFocus()
                return

            self._backup_codes_plain = generate_backup_codes()
            hashed = hash_backup_codes(self._backup_codes_plain)
            self.db.enable_2fa(self.username, hashed)
            self.db.update_2fa_last_step(self.username, step)

            self._show_backup_codes()
            SoundManager.get().play("ok_sound")
        except Exception as e:
            print(f"⚠️ _on_confirm_code failed: {e}")
            self._show_error(f"Error: {e}")

    def _show_backup_codes(self):
        """Switch UI to step 2 (show backup codes)."""
        self._step = 2

        # مخفی کردن المان‌های step 1
        self.code_input.setVisible(False)
        self.code_label.setVisible(False)
        self.qr_label.setVisible(False)
        self.secret_display.setVisible(False)
        self.secret_label.setVisible(False)
        self.copy_secret_btn.setVisible(False)
        self.subtitle_label.setVisible(False)
        self.hint_box.setVisible(False)
        self.manual_hint.setVisible(False)

        # نمایش backup codes
        codes_text = "\n".join(self._backup_codes_plain)
        self.backup_codes_label.setText(codes_text)
        self.backup_container.setVisible(True)

        # تغییر دکمه
        self.confirm_btn.setText("✓  " + tr(self.lang, "2fa_done_btn"))
        try:
            self.confirm_btn.clicked.disconnect()
        except TypeError:
            pass
        self.confirm_btn.clicked.connect(self._on_done)
        self.cancel_btn.setVisible(False)

        # اسکرول به بالا
        try:
            self.scroll.verticalScrollBar().setValue(0)
        except Exception:
            pass

    def _on_done(self):
        try:
            SoundManager.get().play("ok_sound")
            self.setup_completed.emit(self.username, self._backup_codes_plain)
        except Exception:
            pass
        animate_dialog_close(self, on_finished=self.accept)

    def _on_cancel(self):
        try:
            if self._step == 1:
                self.db.disable_2fa(self.username)
        except Exception:
            pass
        SoundManager.get().play("simpleClick")
        animate_dialog_close(self, on_finished=self.reject)

    def _show_error(self, msg: str, success: bool = False):
        try:
            color = "#34d399" if success else "#f87171"
            self.error_label.setText(("✓  " if success else "⚠️  ") + msg)
            self.error_label.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: bold;"
                "background: transparent;"
            )
            self.error_label.setVisible(True)
            QTimer.singleShot(4000, lambda: self.error_label.setVisible(False))
        except Exception:
            pass

class TwoFactorVerifyDialog(QDialog):
    """
    Shown during login when user has 2FA enabled.
    Accepts either 6-digit TOTP or backup code.
    """
    verified = pyqtSignal(str)  # emits username on success

    def __init__(self, db: UserDB, username: str, theme: Dict,
                 lang: str = "English", parent=None):
        super().__init__(parent)
        self.db = db
        self.username = username
        self.theme = theme
        self.lang = lang
        self._attempts = []
        self._use_backup_mode = False

        self.setWindowTitle(tr(lang, "2fa_verify_title"))
        self.setFixedSize(460, 480)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

        try:
            from gui.dialogs.update_dialogs import _with_backdrop
            _with_backdrop(self, parent, theme)
        except Exception as e:
            print(f"⚠️ TwoFactorVerifyDialog backdrop failed: {e}")

        QTimer.singleShot(0, lambda: animate_dialog_open(self))

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(14)

        # Icon
        icon = QLabel("🔐")
        icon.setObjectName("icon")
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        # Title
        title = QLabel(tr(self.lang, "2fa_verify_title"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        self.subtitle = QLabel(tr(self.lang, "2fa_verify_subtitle"))
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        layout.addSpacing(10)

        # Code input
        self.code_input = QLineEdit()
        self.code_input.setObjectName("code_input")
        self.code_input.setPlaceholderText("000000")
        self.code_input.setMaxLength(9)  # 6 for TOTP, 9 for backup (XXXX-XXXX)
        self.code_input.setAlignment(Qt.AlignCenter)
        self.code_input.setFixedHeight(56)
        self.code_input.returnPressed.connect(self._on_verify)
        layout.addWidget(self.code_input)

        # Error
        self.error_label = QLabel("")
        self.error_label.setObjectName("error_label")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        # Toggle backup mode
        self.backup_toggle_btn = QPushButton(
            "🔑 " + tr(self.lang, "2fa_use_backup_code")
        )
        self.backup_toggle_btn.setObjectName("link_btn")
        self.backup_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.backup_toggle_btn.clicked.connect(self._toggle_backup_mode)
        layout.addWidget(self.backup_toggle_btn, 0, Qt.AlignCenter)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.cancel_btn = QPushButton("✕  " + tr(self.lang, "cancel"))
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.setMinimumHeight(44)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.cancel_btn)

        self.verify_btn = QPushButton("✓  " + tr(self.lang, "2fa_verify_btn"))
        self.verify_btn.setObjectName("primary_btn")
        self.verify_btn.setMinimumHeight(44)
        self.verify_btn.clicked.connect(self._on_verify)
        btn_row.addWidget(self.verify_btn, 1)

        layout.addLayout(btn_row)

        # Close
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self._on_cancel)
        close_btn.setParent(self.container)
        close_btn.move(420, 10)

    def _apply_theme(self):
        t = self.theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#icon {{ font-size: 48px; background: transparent; }}
            QLabel#title {{
                font-size: 20px; font-weight: bold;
                color: {t['primary']}; background: transparent;
            }}
            QLabel#subtitle {{
                font-size: 12px; color: {t['text_dim']};
                background: transparent;
            }}
            QLineEdit#code_input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 12px; padding: 8px;
                font-size: 24px; font-weight: bold;
                letter-spacing: 6px; color: {t['primary']};
            }}
            QLineEdit#code_input:focus {{
                border: 2px solid {t['primary']};
            }}
            QLabel#error_label {{
                font-size: 11px; font-weight: bold;
                color: {t['danger']}; background: transparent;
            }}
            QPushButton#link_btn {{
                background: transparent; color: {t['primary']};
                border: none; font-size: 11px;
                text-decoration: underline;
            }}
            QPushButton#link_btn:hover {{ color: {t['secondary']}; }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:0.5 {t['accent']}, stop:1 {t['secondary']}
                );
                color: {t['button_text']}; border: none;
                border-radius: 12px; padding: 12px 20px;
                font-size: 14px; font-weight: bold;
            }}
            QPushButton#primary_btn:hover {{ background: {t['button_hover']}; }}
            QPushButton#secondary_btn {{
                background: {t['input_bg']}; color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 12px; padding: 12px 20px;
                font-size: 14px; font-weight: bold;
            }}
            QPushButton#secondary_btn:hover {{
                background: {t['danger']}; color: {t['button_text']};
            }}
            QPushButton#close_btn {{
                background: transparent; color: {t['text_dim']};
                border: none; font-size: 18px; font-weight: bold;
            }}
            QPushButton#close_btn:hover {{ color: {t['danger']}; }}
        """)

    def _toggle_backup_mode(self):
        self._use_backup_mode = not self._use_backup_mode
        if self._use_backup_mode:
            self.subtitle.setText(tr(self.lang, "2fa_enter_backup_code"))
            self.code_input.setPlaceholderText("XXXX-XXXX")
            self.code_input.setMaxLength(9)
            self.backup_toggle_btn.setText(
                "📱 " + tr(self.lang, "2fa_use_totp_code")
            )
        else:
            self.subtitle.setText(tr(self.lang, "2fa_verify_subtitle"))
            self.code_input.setPlaceholderText("000000")
            self.code_input.setMaxLength(6)
            self.backup_toggle_btn.setText(
                "🔑 " + tr(self.lang, "2fa_use_backup_code")
            )
        self.code_input.clear()
        self.code_input.setFocus()

    def _on_verify(self):
        import time
        now = time.time()
        self._attempts = [t for t in self._attempts if now - t < 300]
        if len(self._attempts) >= 5:
            self._show_error(tr(self.lang, "too_many_attempts"))
            return

        code = self.code_input.text().strip()
        if not code:
            self._show_error(tr(self.lang, "2fa_enter_code_hint"))
            return

        status = self.db.get_2fa_status(self.username)
        secret = status.get("secret", "")
        last_step = status.get("last_used_step")

        cleaned = code.replace("-", "").replace(" ", "")
        looks_like_backup = ("-" in code) or (len(cleaned) > 6)

        verified = False

        if self._use_backup_mode or looks_like_backup:
            from core.security.totp import verify_backup_code
            ok, idx = verify_backup_code(
                code, status.get("backup_codes_hashed", [])
            )
            if ok:
                self.db.use_backup_code(self.username, idx)
                verified = True
        else:
            from core.security.totp import verify_code
            ok, step = verify_code(secret, code, last_used_step=last_step)
            if ok:
                self.db.update_2fa_last_step(self.username, step)
                verified = True

        if verified:
            SoundManager.get().play("ok_sound")
            self.verified.emit(self.username)
            animate_dialog_close(self, on_finished=self.accept)
            return

        self._attempts.append(now)
        SoundManager.get().play("warningSound")
        self._show_error(tr(self.lang, "2fa_invalid_code"))
        self.code_input.clear()
        self.code_input.setFocus()

    def _on_cancel(self):
        SoundManager.get().play("simpleClick")
        animate_dialog_close(self, on_finished=self.reject)

    def _show_error(self, msg: str):
        try:
            self.error_label.setText(f"⚠️  {msg}")
            self.error_label.setVisible(True)
            QTimer.singleShot(4000, lambda: self.error_label.setVisible(False))
        except Exception:
            pass



# ================================================================
# ✅ USER STATS (JSON-based per-user statistics)
# ================================================================

class UserStats:
    """
    Per-user statistics: total processed, distribution, recent results.
    Stored in stats.json next to users_db.json.

    Structure of stats.json:
    {
        "users": {
            "username": {
                "total": int,
                "counts": {"0": int, "1": int, "2": int, "3": int},
                "confidences": [float, ...],
                "recent": [
                    {"timestamp": ISO, "grid": int, "confidence": float},
                    ...
                ]
            }
        }
    }
    """

    MAX_RECENT = 10         # حداکثر recent نگه‌داری
    MAX_CONFIDENCES = 1000  # حداکثر confidence برای میانگین

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(PROJECT_ROOT, "stats.json")
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "users" not in data:
                        data["users"] = {}
                    return data
            except Exception:
                return {"users": {}}
        return {"users": {}}

    def _save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=4, ensure_ascii=False)
            except Exception:
                pass

    def _get_user(self, username: str) -> Dict:
        """آمار کاربر رو برمی‌گردونه (اگه نبود، می‌سازه)."""
        users = self.data.setdefault("users", {})
        if username not in users:
            users[username] = {
                "total": 0,
                "counts": {"0": 0, "1": 0, "2": 0, "3": 0},
                "confidences": [],
                "recent": [],
            }
        # ✅ اطمینان از ساختار درست (برای فایل‌های قدیمی)
        u = users[username]
        u.setdefault("total", 0)
        u.setdefault("counts", {"0": 0, "1": 0, "2": 0, "3": 0})
        u.setdefault("confidences", [])
        u.setdefault("recent", [])
        for k in ("0", "1", "2", "3"):
            u["counts"].setdefault(k, 0)
        return u

    def add_result(self, username: str, grid: int, confidence: float):
        """یه نتیجه‌ی جدید اضافه می‌کنه."""
        if not username:
            return
        try:
            u = self._get_user(username)
            u["total"] = int(u.get("total", 0)) + 1
            u["counts"][str(grid)] = int(u["counts"].get(str(grid), 0)) + 1
            u["confidences"].append(float(confidence))
            # ✅ recent: جدیدترین اول
            u["recent"].insert(0, {
                "timestamp": datetime.now().isoformat(),
                "grid": int(grid),
                "confidence": float(confidence),
            })
            # ✅ محدودیت recent
            if len(u["recent"]) > self.MAX_RECENT:
                u["recent"] = u["recent"][:self.MAX_RECENT]
            # ✅ محدودیت confidences
            if len(u["confidences"]) > self.MAX_CONFIDENCES:
                u["confidences"] = u["confidences"][-self.MAX_CONFIDENCES:]
            self._save()
        except Exception as e:
            print(f"⚠️ UserStats.add_result failed: {e}")

    def get_stats(self, username: str) -> Dict:
        """
        آمار کامل کاربر:
        {
            "total": int,
            "counts": {"0": int, "1": int, "2": int, "3": int},
            "avg_confidence": float (0.0 اگه خالی),
            "recent": [{"timestamp": str, "grid": int, "confidence": float}, ...],
            "distribution": {0: percent, 1: percent, 2: percent, 3: percent},
        }
        """
        empty = {
            "total": 0,
            "counts": {"0": 0, "1": 0, "2": 0, "3": 0},
            "avg_confidence": 0.0,
            "recent": [],
            "distribution": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        }
        if not username:
            return empty
        try:
            u = self._get_user(username)
            confs = u.get("confidences", [])
            avg = (sum(confs) / len(confs)) if confs else 0.0
            total = int(u.get("total", 0))
            counts = u.get("counts", {"0": 0, "1": 0, "2": 0, "3": 0})

            distribution = {}
            for g in (0, 1, 2, 3):
                c = int(counts.get(str(g), 0))
                distribution[g] = (c / total * 100.0) if total > 0 else 0.0

            return {
                "total": total,
                "counts": {
                    "0": int(counts.get("0", 0)),
                    "1": int(counts.get("1", 0)),
                    "2": int(counts.get("2", 0)),
                    "3": int(counts.get("3", 0)),
                },
                "avg_confidence": float(avg),
                "recent": list(u.get("recent", [])),
                "distribution": distribution,
            }
        except Exception:
            return empty

    def clear_user(self, username: str):
        """پاک کردن آمار یه کاربر."""
        if not username:
            return
        try:
            users = self.data.get("users", {})
            if username in users:
                users[username] = {
                    "total": 0,
                    "counts": {"0": 0, "1": 0, "2": 0, "3": 0},
                    "confidences": [],
                    "recent": [],
                }
                self._save()
        except Exception:
            pass


# ================================================================
# ✅ HISTORY MANAGER (v3.0 — per-user image history with thumbnails)
# ================================================================

class HistoryManager:
    """
    مدیریت تاریخچه‌ی تصاویر per-user.

    ساختار:
        history/
            babak/
                20250115_143000.jpg      ← تصویر اصلی
                20250115_143000.json     ← metadata
                ...
            ali/
                ...

    هر رکورد شامل:
        - timestamp: ISO string
        - grid: int (0-3)
        - confidence: float (0-1)
        - image_path: مسیر نسبی به history
        - model_name: str
        - model_version: str
    """

    MAX_ITEMS = 20  # حداکثر تعداد رکورد per-user

    def __init__(self, base_dir: str = HISTORY_DIR):
        self.base_dir = base_dir
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ HistoryManager: cannot create base dir: {e}")

    # ============================================================
    # Helpers
    # ============================================================
    def _user_dir(self, username: str) -> str:
        """مسیر پوشه‌ی تاریخچه‌ی یه کاربر."""
        # ✅ sanitize: فقط حروف/اعداد/آندرلاین
        safe = "".join(c for c in (username or "guest") if c.isalnum() or c in ("_", "-"))
        if not safe:
            safe = "guest"
        return os.path.join(self.base_dir, safe)

    def _ensure_user_dir(self, username: str) -> str:
        """اطمینان از وجود پوشه‌ی کاربر."""
        d = self._user_dir(username)
        try:
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            print(f"⚠️ HistoryManager: cannot create user dir: {e}")
        return d

    def _timestamp_filename(self) -> str:
        """نام فایل بر اساس timestamp (بدون کاراکترهای غیرمجاز)."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms

    # ============================================================
    # Save
    # ============================================================
    def save_result(
            self,
            username: str,
            image_path: str,
            grid: int,
            confidence: float,
            model_name: str = "ResNet50 KD Student",
            model_version: str = "v3.0",
    ) -> Optional[Dict]:
        """
        ذخیره‌ی یه نتیجه توی تاریخچه.

        - تصویر اصلی رو کپی می‌کنه به history/{username}/{timestamp}.jpg
        - metadata رو توی {timestamp}.json ذخیره می‌کنه
        - رکورد رو برمی‌گردونه (برای اضافه شدن به UI)
        """
        if not username:
            return None
        if not image_path or not os.path.isfile(image_path):
            return None

        try:
            user_dir = self._ensure_user_dir(username)
            ts_name = self._timestamp_filename()
            dest_img = os.path.join(user_dir, f"{ts_name}.jpg")
            dest_json = os.path.join(user_dir, f"{ts_name}.json")

            # ✅ کپی تصویر
            try:
                import shutil
                shutil.copy2(image_path, dest_img)
            except Exception as e:
                print(f"⚠️ HistoryManager: copy image failed: {e}")
                return None

            # ✅ metadata
            meta = {
                "timestamp": datetime.now().isoformat(),
                "grid": int(grid),
                "confidence": float(confidence),
                "image_path": f"history/{os.path.basename(self.base_dir)}/{os.path.basename(user_dir)}/{os.path.basename(dest_img)}"
                if False else dest_img,  # ← مسیر مطلق برای سادگی
                "model_name": model_name,
                "model_version": model_version,
            }
            try:
                with open(dest_json, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ HistoryManager: save json failed: {e}")
                # ✅ اگه json ذخیره نشد، تصویر رو هم پاک کن
                try:
                    os.remove(dest_img)
                except Exception:
                    pass
                return None

            # ✅ هرس کردن (حداکثر MAX_ITEMS)
            self._prune_user(username)

            return meta
        except Exception as e:
            print(f"⚠️ HistoryManager.save_result failed: {e}")
            return None

    def _prune_user(self, username: str):
        """حذف قدیمی‌ترین رکوردها اگه بیش از MAX_ITEMS باشه."""
        try:
            user_dir = self._user_dir(username)
            if not os.path.isdir(user_dir):
                return
            # ✅ همه‌ی فایل‌های json
            jsons = sorted(
                [f for f in os.listdir(user_dir) if f.endswith(".json")],
                key=lambda x: os.path.getmtime(os.path.join(user_dir, x)),
            )
            if len(jsons) <= self.MAX_ITEMS:
                return
            # ✅ حذف قدیمی‌ترها
            to_remove = jsons[: len(jsons) - self.MAX_ITEMS]
            for jname in to_remove:
                jpath = os.path.join(user_dir, jname)
                ipath = jpath[:-5] + ".jpg"
                try:
                    os.remove(jpath)
                except Exception:
                    pass
                try:
                    if os.path.exists(ipath):
                        os.remove(ipath)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ HistoryManager._prune_user failed: {e}")

    # ============================================================
    # Load
    # ============================================================
    def load_history(self, username: str) -> List[Dict]:
        """
        لود تاریخچه‌ی یه کاربر.
        جدیدترین رکوردها اول.
        """
        if not username:
            return []
        try:
            user_dir = self._user_dir(username)
            if not os.path.isdir(user_dir):
                return []

            jsons = [f for f in os.listdir(user_dir) if f.endswith(".json")]
            records = []
            for jname in jsons:
                jpath = os.path.join(user_dir, jname)
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    # ✅ اطمینان از وجود تصویر
                    ipath = meta.get("image_path", "")
                    if not ipath or not os.path.isfile(ipath):
                        # سعی کن با همون نام فایل کنار json
                        alt = jpath[:-5] + ".jpg"
                        if os.path.isfile(alt):
                            meta["image_path"] = alt
                        else:
                            continue
                    records.append(meta)
                except Exception:
                    continue

            # ✅ sort by timestamp (جدیدترین اول)
            def _ts_key(r):
                try:
                    return r.get("timestamp", "")
                except Exception:
                    return ""

            records.sort(key=_ts_key, reverse=True)
            return records
        except Exception as e:
            print(f"⚠️ HistoryManager.load_history failed: {e}")
            return []

    # ============================================================
    # Clear
    # ============================================================
    def clear_user(self, username: str):
        """پاک کردن کل تاریخچه‌ی یه کاربر."""
        if not username:
            return
        try:
            user_dir = self._user_dir(username)
            if not os.path.isdir(user_dir):
                return
            for fname in os.listdir(user_dir):
                fpath = os.path.join(user_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception:
                    continue
        except Exception as e:
            print(f"⚠️ HistoryManager.clear_user failed: {e}")


# ================================================================
# ✅ HISTORY THUMBNAIL (v3.0 — یک thumbnail قابل کلیک)
# ================================================================

class HistoryThumbnail(QFrame):
    """
    یک thumbnail از یه نتیجه‌ی قبلی.

    - تصویر کوچیک (۸۰×۸۰)
    - برچسب grid + confidence
    - Hover effect (border رنگی)
    - Click → سیگنال clicked(record)

    Usage:
        thumb = HistoryThumbnail(record_dict, theme, lang)
        thumb.clicked.connect(lambda rec: ...)
        thumb.set_theme(theme)
        thumb.set_lang(lang)
    """

    # ✅ رنگ هر گرید (هماهنگ با DashboardPanel و PDFExporter)
    GRID_COLORS = {
        0: "#34d399",  # NL — سبز
        1: "#fbbf24",  # G1 — زرد
        2: "#fb923c",  # G2 — نارنجی
        3: "#f87171",  # G3 — قرمز
    }

    clicked = pyqtSignal(dict)  # سیگنال با record

    THUMB_SIZE = 84  # اندازه‌ی تصویر (px)

    def __init__(self, record: Dict, theme: Dict, lang: str = "English", parent=None):
        super().__init__(parent)
        self._record = record
        self._theme = theme
        self._lang = lang
        self._hover = False

        self.setObjectName("history_thumb")
        self.setFixedSize(self.THUMB_SIZE + 8, self.THUMB_SIZE + 36)
        self.setCursor(Qt.PointingHandCursor)

        self._build_ui()
        self._apply_theme()
        self._load_image()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ✅ تصویر
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        self._img_label.setObjectName("history_thumb_img")
        layout.addWidget(self._img_label, 0, Qt.AlignCenter)

        # ✅ برچسب پایین: grid + confidence
        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignCenter)
        self._info_label.setObjectName("history_thumb_info")
        self._info_label.setFixedHeight(16)
        layout.addWidget(self._info_label)

    def _load_image(self):
        """لود و scale تصویر."""
        try:
            ipath = self._record.get("image_path", "")
            if not ipath or not os.path.isfile(ipath):
                self._img_label.setText("🖼️")
                return
            pixmap = QPixmap(ipath)
            if pixmap.isNull():
                self._img_label.setText("🖼️")
                return
            scaled = pixmap.scaled(
                self.THUMB_SIZE, self.THUMB_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._img_label.setPixmap(scaled)
        except Exception:
            self._img_label.setText("🖼️")

    def _update_info(self):
        """به‌روزرسانی برچسب grid + confidence."""
        try:
            grid = int(self._record.get("grid", 0))
            conf = float(self._record.get("confidence", 0.0))
            short = {0: "NL", 1: "G1", 2: "G2", 3: "G3"}.get(grid, f"G{grid}")
            self._info_label.setText(f"{short} · {conf * 100:.0f}%")
            # ✅ رنگ متن با رنگ grid
            grid_color = self.GRID_COLORS.get(grid, "#00e5ff")
            self._info_label.setStyleSheet(f"color: {grid_color};")
        except Exception:
            self._info_label.setText("—")

    def set_tooltips(self, theme: Dict, lang: str):
        """tooltip کامل برای thumbnail."""
        try:
            t = theme or self._theme
            self._lang = lang

            grid = int(self._record.get("grid", 0))
            conf = float(self._record.get("confidence", 0.0))
            ts = self._record.get("timestamp", "")

            time_str = ""
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = ts or "—"

            grid_full = tr(lang, f"grid_{grid}")

            self.setToolTip(
                make_tooltip(
                    title=grid_full,
                    description=f"📅 {time_str}  •  🎯 {conf * 100:.1f}%",
                    tip=tr(lang, "tooltip_tip_thumb_click"),
                    theme=t,
                )
            )
        except Exception as e:
            print(f"⚠️ HistoryThumbnail.set_tooltips failed: {e}")

    # ============================================================
    # ✅ Smart tooltip
    # ============================================================

    def set_tooltips(self, theme: Dict, lang: str):
        """
        tooltip کامل برای thumbnail:
        - عنوان: grid کامل
        - توضیح: تاریخ + زمان + اطمینان
        - نکته: کلیک برای لود مجدد
        """
        try:
            t = theme or self._theme
            self._lang = lang

            grid = int(self._record.get("grid", 0))
            conf = float(self._record.get("confidence", 0.0))
            ts = self._record.get("timestamp", "")

            # ✅ فرمت زمان
            time_str = ""
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = ts or "—"

            grid_full = tr(lang, f"grid_{grid}")

            self.setToolTip(
                make_tooltip(
                    title=grid_full,
                    description=f"📅 {time_str}  •  🎯 {conf * 100:.1f}%",
                    tip=tr(lang, "tooltip_tip_thumb_click"),
                    theme=t,
                )
            )
        except Exception as e:
            print(f"⚠️ HistoryThumbnail.set_tooltips failed: {e}")

    # ============================================================
    # Theme
    # ============================================================
    def set_theme(self, theme: Dict):
        self._theme = theme
        self._apply_theme()

    def set_lang(self, lang: str):
        self._lang = lang
        self._update_info()

    def _apply_theme(self):
        t = self._theme
        try:
            grid = int(self._record.get("grid", 0))
            grid_color = self.GRID_COLORS.get(grid, "#00e5ff")
        except Exception:
            grid_color = "#00e5ff"

        # ✅ اگه hover، border رنگی؛ وگرنه border ملایم
        if self._hover:
            border = f"2px solid {grid_color}"
            bg = t.get("input_bg", "rgba(30, 41, 59, 0.85)")
        else:
            border = f"1px solid {t.get('glass_border', '#00e5ff')}"
            bg = t.get("glass_bg", "rgba(15, 20, 40, 0.45)")

        self.setStyleSheet(f"""
            QFrame#history_thumb {{
                background: {bg};
                border: {border};
                border-radius: 8px;
            }}
            QLabel#history_thumb_img {{
                background: transparent;
                border: none;
            }}
            QLabel#history_thumb_info {{
                background: transparent;
                font-size: 10px;
                font-weight: bold;
                border: none;
            }}
        """)
        self._update_info()

    # ============================================================
    # Events
    # ============================================================
    def enterEvent(self, event):
        self._hover = True
        self._apply_theme()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self._apply_theme()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            try:
                self.clicked.emit(self._record)
            except Exception:
                pass
        super().mousePressEvent(event)

    # ============================================================
    # Accessor
    # ============================================================
    def get_record(self) -> Dict:
        return self._record

# ================================================================
# ✅ HISTORY PANEL (v3.0 — drawer از سمت راست با thumbnails)
# ================================================================

class HistoryPanel(QFrame):
    """
    پنل تاریخچه — نمایش ۲۰ نتیجه‌ی آخر با thumbnail.

    - Header: عنوان + دکمه‌ی clear + دکمه‌ی close
    - ScrollArea: grid ۲ ستونی از HistoryThumbnailها
    - Empty state: وقتی تاریخچه خالیه

    Usage:
        panel = HistoryPanel(theme, lang, username)
        panel.thumbnail_clicked.connect(lambda rec: ...)
        panel.closed.connect(lambda: ...)
        panel.clear_requested.connect(lambda: ...)
        panel.refresh(records_list)
    """

    closed = pyqtSignal()
    clear_requested = pyqtSignal()
    thumbnail_clicked = pyqtSignal(dict)

    def __init__(self, theme: Dict, lang: str = "English", username: str = "",
                 parent=None):
        super().__init__(parent)
        self._theme = theme
        self._lang = lang
        self._username = username
        self._thumbnails = []

        self.setObjectName("history_panel")
        self.setFixedWidth(320)
        self.setMinimumHeight(400)

        self._build_ui()
        self._apply_theme()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # ---------- Header ----------
        header = QHBoxLayout()
        header.setSpacing(6)

        self._title = QLabel("🕐 History")
        self._title.setObjectName("hist_title")
        header.addWidget(self._title)
        header.addStretch()

        self._clear_btn = QPushButton("🗑️")
        self._clear_btn.setObjectName("hist_icon_btn")
        self._clear_btn.setFixedSize(26, 26)
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        header.addWidget(self._clear_btn)

        self._close_btn = QPushButton("✕")
        self._close_btn.setObjectName("hist_icon_btn")
        self._close_btn.setFixedSize(26, 26)
        self._close_btn.clicked.connect(self._on_close_clicked)
        header.addWidget(self._close_btn)

        main_layout.addLayout(header)

        # ---------- Scroll Area ----------
        self._scroll = QScrollArea()
        self._scroll.setObjectName("hist_scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._grid_container = QWidget()
        self._grid_container.setObjectName("hist_grid_container")
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(8)
        self._grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self._scroll.setWidget(self._grid_container)
        main_layout.addWidget(self._scroll, 1)

        # ---------- Empty State ----------
        self._empty_label = QLabel()
        self._empty_label.setObjectName("hist_empty")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setTextFormat(Qt.RichText)
        self._empty_label.setVisible(False)
        self._update_empty_text()
        main_layout.addWidget(self._empty_label, 1)

    def _apply_theme(self):
        t = self._theme
        self.setStyleSheet(f"""
            QFrame#history_panel {{
                background: {t['glass_bg']};
                border-left: 2px solid {t['glass_border']};
            }}
            QLabel#hist_title {{
                color: {t['primary']};
                font-size: 16px;
                font-weight: bold;
                background: transparent;
            }}
            QPushButton#hist_icon_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: 1px solid {t['glass_border']};
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#hist_icon_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QScrollArea#hist_scroll {{
                background: transparent;
                border: none;
            }}
            QWidget#hist_grid_container {{
                background: transparent;
            }}
            QLabel#hist_empty {{
                color: {t['text_dim']};
                font-size: 13px;
                font-style: italic;
                background: transparent;
                padding: 20px;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {t['glass_border']};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {t['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    # ============================================================
    # Public API
    # ============================================================
    def set_theme(self, theme: Dict):
        """رنگ‌ها رو با تم جدید هماهنگ می‌کنه."""
        self._theme = theme
        self._apply_theme()
        # ✅ آپدیت thumbnailها
        for th in self._thumbnails:
            try:
                th.set_theme(theme)
            except Exception:
                pass
        # ✅ آپدیت Empty State با تم جدید
        try:
            self._update_empty_text()
        except Exception:
            pass
        # ✅ به‌روزرسانی tooltipها
        try:
            self.set_tooltips(theme, self._lang)
        except Exception:
            pass

    def set_lang(self, lang: str):
        """متن‌ها رو با زبان جدید هماهنگ می‌کنه."""
        self._lang = lang
        try:
            self._title.setText("🕐 " + tr(lang, "history_title"))
            # ✅ آپدیت Empty State با HTML
            self._update_empty_text()
        except Exception:
            pass
        for th in self._thumbnails:
            try:
                th.set_lang(lang)
            except Exception:
                pass
        # ✅ به‌روزرسانی tooltipها
        try:
            self.set_tooltips(self._theme, lang)
        except Exception:
            pass

    def set_username(self, username: str):
        """تغییر کاربر فعلی (بعد از login/logout)."""
        self._username = username
        # ✅ آپدیت Empty State (ممکنه از "no user" به "empty" یا برعکس بره)
        try:
            self._update_empty_text()
        except Exception:
            pass

    def _update_empty_text(self):
        """آپدیت متن Empty State — متفاوت اگه لاگین نیست."""
        try:
            # ✅ تشخیص: کاربر لاگین هست یا نه
            key = "history_empty" if self._username else "history_no_user"
            html = make_empty_state_html(
                key=key,
                theme=self._theme,
                lang=self._lang,
            )
            if html:
                self._empty_label.setText(html)
            else:
                self._empty_label.setText(tr(self._lang, "history_empty"))
        except Exception as e:
            print(f"⚠️ HistoryPanel._update_empty_text failed: {e}")
            try:
                self._empty_label.setText(tr(self._lang, "history_empty"))
            except Exception:
                pass

    # ============================================================
    # ✅ Smart tooltips
    # ============================================================

    def set_tooltips(self, theme: Dict, lang: str):
        """
        به‌روزرسانی tooltipها با تم و زبان جدید.
        باید از MainWindow._update_all_tooltips صدا زده بشه.
        """
        try:
            t = theme or self._theme
            self._theme = t
            self._lang = lang

            # ---------- دکمه‌ی clear (🗑️) ----------
            try:
                self._clear_btn.setToolTip(
                    make_tooltip(
                        title="🗑️ " + tr(lang, "tooltip_title_hist_clear"),
                        description=tr(lang, "tooltip_desc_hist_clear"),
                        tip=tr(lang, "tooltip_tip_hist_clear")
                            if "tooltip_tip_hist_clear" in TRANSLATIONS.get(lang, {}) else "",
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- دکمه‌ی close (✕) ----------
            try:
                self._close_btn.setToolTip(
                    make_tooltip(
                        title="✕ " + tr(lang, "tooltip_title_hist_close"),
                        description=tr(lang, "tooltip_desc_hist_close"),
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- thumbnailها ----------
            # tooltip هر thumbnail توی set_tooltips توی HistoryThumbnail ست می‌شه
            for th in self._thumbnails:
                try:
                    th.set_tooltips(t, lang)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ HistoryPanel.set_tooltips failed: {e}")

    def refresh(self, records: List[Dict]):
        """
        بازسازی grid از روی لیست رکوردها.
        - records: لیست دیکشنری‌ها (از HistoryManager.load_history)
        """
        # ✅ پاک کردن thumbnailهای قبلی
        self._clear_grid()

        if not records:
            self._show_empty(True)
            return

        self._show_empty(False)

        # ✅ ساخت thumbnailها (۲ ستون)
        for i, record in enumerate(records):
            try:
                thumb = HistoryThumbnail(
                    record=record,
                    theme=self._theme,
                    lang=self._lang,
                    parent=self._grid_container,
                )
                thumb.clicked.connect(self._on_thumb_clicked)
                # ✅ tooltip
                try:
                    thumb.set_tooltips(self._theme, self._lang)
                except Exception:
                    pass
                self._thumbnails.append(thumb)
                row = i // 2
                col = i % 2
                self._grid_layout.addWidget(thumb, row, col)
            except Exception as e:
                print(f"⚠️ HistoryPanel: build thumb failed: {e}")
                continue

        # ✅ spacer در پایین (تا thumbnailها به بالا بچسبن)
        # پیدا کردن آخرین row
        try:
            count = self._grid_layout.count()
            if count > 0:
                last_row = (count - 1) // 2
                self._grid_layout.setRowStretch(last_row + 1, 1)
        except Exception:
            pass

    def _clear_grid(self):
        """پاک کردن همه‌ی thumbnailها از grid."""
        self._thumbnails.clear()
        while self._grid_layout.count() > 0:
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _show_empty(self, show: bool):
        """نمایش/مخفف کردن empty state."""
        self._empty_label.setVisible(show)
        self._scroll.setVisible(not show)

    # ============================================================
    # Events
    # ============================================================
    def _on_thumb_clicked(self, record: Dict):
        try:
            self.thumbnail_clicked.emit(record)
        except Exception:
            pass

    def _on_close_clicked(self):
        try:
            self.closed.emit()
        except Exception:
            pass

    def _on_clear_clicked(self):
        try:
            self.clear_requested.emit()
        except Exception:
            pass

# ================================================================
# ✅ PDF EXPORTER (v3.0 — professional medical report)
# ================================================================

class PDFExporter:
    """
    تولید PDF حرفه‌ای از نتیجه‌ی inference.

    شامل:
    - Header با لوگو و عنوان
    - اطلاعات بیمار (نام کاربر، تاریخ، مدل)
    - تصویر اصلی (کیفیت کامل)
    - نتیجه (Grid + Grade + Confidence)
    - جدول احتمال کلاس‌ها (bar chart)
    - Footer با disclaimer پزشکی

    Usage:
        exporter = PDFExporter(
            theme=THEMES["theme_dark_aqua"],
            lang="English",
        )
        ok = exporter.generate(
            output_path="report.pdf",
            username="babak",
            image_path="scan.png",
            grid=2,
            confidence=0.953,
            all_probs={0: 0.021, 1: 0.015, 2: 0.953, 3: 0.011},
            model_name="ResNet50 KD Student",
            model_version="v3.0",
        )
    """

    def __init__(self, theme: Dict, lang: str = "English"):
        self.theme = theme
        self.lang = lang
        self._fonts_loaded = False
        self._font_regular = "Helvetica"  # fallback
        self._font_bold = "Helvetica-Bold"
        self._font_italic = "Helvetica-Oblique"

        if HAS_REPORTLAB:
            self._load_fonts()

    # ============================================================
    # Font loading
    # ============================================================
    def _load_fonts(self):
        """لود فونت‌های PDF: اول Vazirmatn (برای RTL)، بعد Noto Sans."""
        try:
            # ✅ اگه زبان RTL هست، اول Vazirmatn رو لود کن
            if self.lang in ("Persian", "Arabic", "Morocco"):
                vazir_regular = os.path.join(FONTS_DIR, "Vazirmatn-Regular.ttf")
                vazir_bold = os.path.join(FONTS_DIR, "Vazirmatn-Bold.ttf")
                if os.path.exists(vazir_regular):
                    pdfmetrics.registerFont(TTFont("VazirRegular", vazir_regular))
                    self._font_regular = "VazirRegular"
                if os.path.exists(vazir_bold):
                    pdfmetrics.registerFont(TTFont("VazirBold", vazir_bold))
                    self._font_bold = "VazirBold"
                # ✅ italic از Vazirmatn نیست، از Noto استفاده کن
                self._font_italic = self._font_regular
                self._fonts_loaded = True
                return  # ← برای RTL، Noto لازم نیست

            # ✅ برای بقیه‌ی زبان‌ها، Noto Sans
            regular = os.path.join(FONTS_DIR, "NotoSans-Regular.ttf")
            bold = os.path.join(FONTS_DIR, "NotoSans-Bold.ttf")
            italic = os.path.join(FONTS_DIR, "NotoSans-Italic.ttf")

            if os.path.exists(regular):
                pdfmetrics.registerFont(TTFont("NotoRegular", regular))
                self._font_regular = "NotoRegular"
            if os.path.exists(bold):
                pdfmetrics.registerFont(TTFont("NotoBold", bold))
                self._font_bold = "NotoBold"
            if os.path.exists(italic):
                pdfmetrics.registerFont(TTFont("NotoItalic", italic))
                self._font_italic = "NotoItalic"

            self._fonts_loaded = True
        except Exception as e:
            print(f"⚠️ PDF font loading failed: {e}")

    # ============================================================
    # ✅ Part 2: Header + Patient Info
    # ============================================================

    def _get_grid_label(self, grid: int) -> str:
        """برچسب کامل گرید (مثلاً 'Grid 2 (G2) — Grade 2')."""
        try:
            return tr(self.lang, f"grid_{grid}")
        except Exception:
            return f"Grid {grid}"

    def _get_grid_short(self, grid: int) -> str:
        """برچسب کوتاه گرید (NL, G1, G2, G3)."""
        return {0: "NL", 1: "G1", 2: "G2", 3: "G3"}.get(grid, f"G{grid}")

    def _get_grid_color(self, grid: int) -> Color:
        """رنگ گرید (از theme یا پیش‌فرض)."""
        # ✅ رنگ‌ها هماهنگ با DashboardPanel.GRID_COLORS
        colors = {
            0: "#34d399",   # NL — سبز
            1: "#fbbf24",   # G1 — زرد
            2: "#fb923c",   # G2 — نارنجی
            3: "#f87171",   # G3 — قرمز
        }
        try:
            return HexColor(colors.get(grid, "#00e5ff"))
        except Exception:
            return HexColor("#00e5ff")

    def _parse_color(self, color_str: str, fallback: str = "#1e293b") -> Color:
        """
        تبدیل رشته‌ی رنگ به Color از reportlab.
        پشتیبانی از:
        - "#RRGGBB"
        - "#RGB"
        - "rgba(r, g, b, a)"
        - "rgb(r, g, b)"
        """
        try:
            if not color_str or not isinstance(color_str, str):
                return HexColor(fallback)
            s = color_str.strip()

            # ✅ فرمت rgba(...)
            if s.lower().startswith("rgba("):
                inside = s[5:-1]  # بین "(" و ")"
                parts = [p.strip() for p in inside.split(",")]
                if len(parts) >= 3:
                    r = int(float(parts[0]))
                    g = int(float(parts[1]))
                    b = int(float(parts[2]))
                    a = float(parts[3]) if len(parts) >= 4 else 1.0
                    return Color(r / 255.0, g / 255.0, b / 255.0, alpha=a)

            # ✅ فرمت rgb(...)
            if s.lower().startswith("rgb("):
                inside = s[4:-1]
                parts = [p.strip() for p in inside.split(",")]
                if len(parts) >= 3:
                    r = int(float(parts[0]))
                    g = int(float(parts[1]))
                    b = int(float(parts[2]))
                    return Color(r / 255.0, g / 255.0, b / 255.0)

            # ✅ فرمت hex (#RRGGBB یا #RGB)
            if s.startswith("#"):
                return HexColor(s)

            # ✅ fallback
            return HexColor(fallback)
        except Exception:
            return HexColor(fallback)

    def _fix_rtl(self, text: str) -> str:
        """
        اصلاح متن برای RTL (فارسی/عربی).
        - اگه HAS_RTL و زبان RTL بود، متن رو reshaper + bidi می‌کنه
        - وگرنه، متن رو دست‌نخورده برمی‌گردونه
        """
        if not text:
            return text
        if not HAS_RTL:
            return text
        if self.lang not in ("Persian", "Arabic", "Morocco"):
            return text
        try:
            reshaped = arabic_reshaper.reshape(text)
            return bidi_get_display(reshaped)
        except Exception:
            return text

    def _build_header(self) -> List:
        """ساخت بخش header PDF — لوگو + عنوان + خط جداکننده."""
        story = []
        primary = HexColor(self.theme.get("primary", "#00e5ff"))
        text_dim = HexColor(self.theme.get("text_dim", "#94a3b8"))

        # ✅ عنوان اصلی
        title_style = ParagraphStyle(
            name="TitleStyle",
            fontName=self._font_bold,
            fontSize=22,
            textColor=primary,
            alignment=TA_LEFT,
            spaceAfter=2,
        )
        subtitle_style = ParagraphStyle(
            name="SubtitleStyle",
            fontName=self._font_italic,
            fontSize=11,
            textColor=text_dim,
            alignment=TA_LEFT,
            spaceAfter=8,
        )

        story.append(Paragraph(self._fix_rtl("Liver AI Pipeline"), title_style))
        story.append(Paragraph(
            self._fix_rtl(
                tr(self.lang, "pdf_subtitle") if "pdf_subtitle" in TRANSLATIONS.get(self.lang, {})
                else "Analysis Report"
            ),
            subtitle_style,
        ))

        # ✅ خط جداکننده (با Table چون reportlab خط افقی ساده نداره)
        line = Table(
            [[""]],
            colWidths=[170 * mm],
            rowHeights=[0.8],
        )
        line.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, -1), 1.5, primary),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(line)
        story.append(Spacer(1, 8))

        return story

    def _build_patient_info(self, username: str, timestamp: str,
                            model_name: str, model_version: str) -> List:
        """ساخت بخش اطلاعات بیمار — نام، تاریخ، مدل."""
        story = []
        text_color = HexColor(self.theme.get("text", "#f1f5f9"))
        text_dim = HexColor(self.theme.get("text_dim", "#94a3b8"))
        primary = HexColor(self.theme.get("primary", "#00e5ff"))

        label_style = ParagraphStyle(
            name="LabelStyle",
            fontName=self._font_bold,
            fontSize=10,
            textColor=text_dim,
            alignment=TA_LEFT,
        )
        value_style = ParagraphStyle(
            name="ValueStyle",
            fontName=self._font_regular,
            fontSize=11,
            textColor=text_color,
            alignment=TA_LEFT,
        )

        # ✅ جدول ۲ ستونی: label | value
        rows = [
            [
                Paragraph(self._fix_rtl(
                    tr(self.lang, "pdf_patient") if "pdf_patient" in TRANSLATIONS.get(self.lang, {}) else "Patient:"
                ), label_style),
                Paragraph(self._fix_rtl(username or "—"), value_style),
            ],
            [
                Paragraph(self._fix_rtl(
                    tr(self.lang, "pdf_date") if "pdf_date" in TRANSLATIONS.get(self.lang, {}) else "Date:"
                ), label_style),
                Paragraph(self._fix_rtl(timestamp), value_style),
            ],
            [
                Paragraph(self._fix_rtl(
                    tr(self.lang, "pdf_model") if "pdf_model" in TRANSLATIONS.get(self.lang, {}) else "Model:"
                ), label_style),
                Paragraph(self._fix_rtl(f"{model_name} {model_version}".strip()), value_style),
            ],
        ]

        table = Table(rows, colWidths=[35 * mm, 135 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

        return story

    # ============================================================
    # ✅ Part 3: Image + Result Card
    # ============================================================

    def _build_image_section(self, image_path: str) -> List:
        """ساخت بخش تصویر اصلی — با کیفیت کامل، با کادر و shadow."""
        story = []
        if not image_path or not os.path.isfile(image_path):
            return story

        try:
            # ✅ محاسبه‌ی ابعاد تصویر (حداکثر 170mm × 110mm)
            max_w = 170 * mm
            max_h = 110 * mm
            img = RLImage(image_path)
            iw, ih = img.imageWidth, img.imageHeight
            if iw <= 0 or ih <= 0:
                return story

            ratio = min(max_w / iw, max_h / ih, 1.0)
            new_w = iw * ratio
            new_h = ih * ratio

            img.drawWidth = new_w
            img.drawHeight = new_h

            # ✅ در یک Table برای کادر کردن
            border_color = self._parse_color(self.theme.get("glass_border", "#00e5ff"))
            img_table = Table(
                [[img]],
                colWidths=[new_w + 6],
                rowHeights=[new_h + 6],
            )
            img_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#0a0a14")),
                ("BOX", (0, 0), (-1, -1), 1.2, border_color),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 12))
        except Exception as e:
            print(f"⚠️ PDF image failed: {e}")

        return story

    def _build_result_card(self, grid: int, confidence: float) -> List:
        """ساخت کارت نتیجه — Grid + Grade + Confidence با رنگ‌بندی."""
        story = []
        text_color = HexColor(self.theme.get("text", "#f1f5f9"))
        primary = HexColor(self.theme.get("primary", "#00e5ff"))
        grid_color = self._get_grid_color(grid)

        # ✅ عنوان بخش
        section_style = ParagraphStyle(
            name="SectionStyle",
            fontName=self._font_bold,
            fontSize=10,
            textColor=primary,
            alignment=TA_LEFT,
            spaceAfter=4,
        )
        story.append(Paragraph(
            self._fix_rtl(
                tr(self.lang, "pdf_diagnosis") if "pdf_diagnosis" in TRANSLATIONS.get(self.lang, {})
                else "DIAGNOSIS"
            ),
            section_style,
        ))

        # ✅ کارت نتیجه: ۲ ستون (متن نتیجه | Confidence)
        grid_label = self._get_grid_label(grid)
        grid_short = self._get_grid_short(grid)

        result_text_style = ParagraphStyle(
            name="ResultText",
            fontName=self._font_bold,
            fontSize=16,
            textColor=grid_color,
            alignment=TA_LEFT,
        )
        result_sub_style = ParagraphStyle(
            name="ResultSub",
            fontName=self._font_regular,
            fontSize=10,
            textColor=text_color,
            alignment=TA_LEFT,
        )
        conf_value_style = ParagraphStyle(
            name="ConfValue",
            fontName=self._font_bold,
            fontSize=26,
            textColor=grid_color,
            alignment=TA_RIGHT,
        )
        conf_label_style = ParagraphStyle(
            name="ConfLabel",
            fontName=self._font_regular,
            fontSize=9,
            textColor=HexColor(self.theme.get("text_dim", "#94a3b8")),
            alignment=TA_RIGHT,
        )

        left_cell = [
            Paragraph(self._fix_rtl(grid_short), result_text_style),
            Spacer(1, 2),
            Paragraph(self._fix_rtl(grid_label), result_sub_style),
        ]

        right_cell = [
            Paragraph(f"{confidence * 100:.1f}%", conf_value_style),
            Paragraph(
                self._fix_rtl(
                    tr(self.lang, "pdf_confidence") if "pdf_confidence" in TRANSLATIONS.get(self.lang, {})
                    else "Confidence"
                ),
                conf_label_style,
            ),
        ]

        card = Table(
            [[left_cell, right_cell]],
            colWidths=[110 * mm, 60 * mm],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), self._parse_color(self.theme.get("input_bg", "#1e293b"))),
            ("BACKGROUND", (1, 0), (1, 0), self._parse_color(self.theme.get("input_bg", "#1e293b"))),
            ("BOX", (0, 0), (-1, -1), 1.5, grid_color),
            ("LINEBEFORE", (1, 0), (1, 0), 1.5, grid_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(card)
        story.append(Spacer(1, 12))

        return story

    # ============================================================
    # ✅ Part 4: Bar Chart + Footer
    # ============================================================

    def _build_class_probabilities(self, all_probs: Dict) -> List:
        """
        ساخت جدول احتمال کلاس‌ها — با bar chart افقی.
        all_probs: {0: 0.02, 1: 0.01, 2: 0.95, 3: 0.02}
        """
        story = []
        primary = HexColor(self.theme.get("primary", "#00e5ff"))
        text_color = HexColor(self.theme.get("text", "#f1f5f9"))
        text_dim = HexColor(self.theme.get("text_dim", "#94a3b8"))

        # ✅ عنوان بخش
        section_style = ParagraphStyle(
            name="SectionStyle2",
            fontName=self._font_bold,
            fontSize=10,
            textColor=primary,
            alignment=TA_LEFT,
            spaceAfter=4,
        )
        story.append(Paragraph(
            self._fix_rtl(
                tr(self.lang, "pdf_class_probs") if "pdf_class_probs" in TRANSLATIONS.get(self.lang, {})
                else "CLASS PROBABILITIES"
            ),
            section_style,
        ))

        # ✅ ساخت هر ردیف
        rows = []
        for g in (0, 1, 2, 3):
            p = float(all_probs.get(g, 0.0))
            p = max(0.0, min(1.0, p))
            grid_color = self._get_grid_color(g)
            short = self._get_grid_short(g)

            # ✅ نام کلاس (مثلاً "Grid 2 (G2)")
            name_text = f"Grid {g} ({short})"
            name_para = Paragraph(
                self._fix_rtl(name_text),
                ParagraphStyle(
                    name=f"ProbName{g}",
                    fontName=self._font_bold,
                    fontSize=9,
                    textColor=text_color,
                    alignment=TA_LEFT,
                ),
            )

            # ✅ bar chart افقی (با Drawing)
            # عرض کل: 100mm، ارتفاع: 5mm
            bar_w_full = 100 * mm
            bar_h = 5 * mm
            d = Drawing(bar_w_full, bar_h)

            # ✅ پس‌زمینه‌ی bar (خاکستری)
            bg_rect = Rect(
                0, 0, bar_w_full, bar_h,
                fillColor=self._parse_color(self.theme.get("glass_highlight", "#1e293b")),
                strokeColor=None,
            )
            d.add(bg_rect)

            # ✅ پر شدن bar (رنگ کلاس)
            fill_w = bar_w_full * p
            if fill_w > 0:
                fill_rect = Rect(
                    0, 0, fill_w, bar_h,
                    fillColor=grid_color,
                    strokeColor=None,
                )
                d.add(fill_rect)

            # ✅ درصد (راست bar)
            pct_text = String(
                bar_w_full + 3 * mm,
                bar_h / 2 - 2,
                f"{p * 100:.1f}%",
                fontName=self._font_bold,
                fontSize=9,
                fillColor=grid_color,
                textAnchor="start",
            )
            d.add(pct_text)

            # ✅ ستون‌های جدول: name | bar+percent
            rows.append([name_para, d])

        # ✅ ساخت جدول
        table = Table(rows, colWidths=[35 * mm, 135 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            # ✅ خط افقی ملایم بین ردیف‌ها
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, self._parse_color(self.theme.get("glass_border", "#00e5ff"))),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

        return story

    def _build_footer(self) -> List:
        """ساخت footer — disclaimer + نسخه + صفحه."""
        story = []
        text_dim = HexColor(self.theme.get("text_dim", "#94a3b8"))
        text_color = HexColor(self.theme.get("text", "#f1f5f9"))
        primary = HexColor(self.theme.get("primary", "#00e5ff"))

        story.append(Spacer(1, 8))

        # ✅ خط جداکننده
        line = Table(
            [[""]],
            colWidths=[170 * mm],
            rowHeights=[0.8],
        )
        line.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, -1), 1.0, primary),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(line)
        story.append(Spacer(1, 6))

        # ✅ Disclaimer
        disclaimer_style = ParagraphStyle(
            name="Disclaimer",
            fontName=self._font_italic,
            fontSize=8,
            textColor=text_dim,
            alignment=TA_LEFT,
            leading=11,
        )
        disclaimer_text = (
            tr(self.lang, "pdf_disclaimer") if "pdf_disclaimer" in TRANSLATIONS.get(self.lang, {})
            else (
                "This report was developed by Babak Yousefian under the supervision of experienced physicians. "
                "It is a professional analysis tool and does not require referral to a physician."
            )
        )
        story.append(Paragraph(self._fix_rtl(disclaimer_text), disclaimer_style))
        story.append(Spacer(1, 6))

        # ✅ خط پایانی: نسخه + صفحه (سمت راست)
        version_style = ParagraphStyle(
            name="VersionStyle",
            fontName=self._font_regular,
            fontSize=8,
            textColor=text_dim,
            alignment=TA_RIGHT,
        )
        version_text = "Liver AI Pipeline v3.0  •  Page 1"
        story.append(Paragraph(self._fix_rtl(version_text), version_style))

        return story

    # ============================================================
    # ✅ Part 5: Main generate() — قلب کلاس
    # ============================================================

    def generate(
        self,
        output_path: str,
        username: str,
        image_path: str,
        grid: int,
        confidence: float,
        all_probs: Optional[Dict] = None,
        model_name: str = "ResNet50 KD Student",
        model_version: str = "v3.0",
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        تولید PDF کامل.

        Args:
            output_path: مسیر خروجی PDF
            username: نام کاربر (یا "—")
            image_path: مسیر تصویر اصلی
            grid: کلاس پیش‌بینی‌شده (0-3)
            confidence: اطمینان (0.0-1.0)
            all_probs: {0: p0, 1: p1, 2: p2, 3: p3} — اگه None، فقط grid=confidence
            model_name: نام مدل
            model_version: نسخه
            timestamp: اگه None، الان

        Returns:
            True اگه موفق، False اگه خطا
        """
        if not HAS_REPORTLAB:
            print("⚠️ PDF export: reportlab not available")
            return False

        try:
            # ✅ timestamp
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ✅ اگه all_probs نیست، از grid/confidence بساز
            if all_probs is None:
                all_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                all_probs[grid] = confidence

            # ✅ ساخت Document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                leftMargin=20 * mm,
                rightMargin=20 * mm,
                topMargin=15 * mm,
                bottomMargin=15 * mm,
                title="Liver AI — Analysis Report",
                author="Liver AI Pipeline",
                subject="Medical Image Analysis",
            )

            # ✅ جمع‌آوری story
            story = []

            # 1) Header
            story.extend(self._build_header())

            # 2) اطلاعات بیمار
            story.extend(self._build_patient_info(
                username=username,
                timestamp=timestamp,
                model_name=model_name,
                model_version=model_version,
            ))

            # 3) تصویر
            story.extend(self._build_image_section(image_path))

            # 4) کارت نتیجه
            story.extend(self._build_result_card(grid, confidence))

            # 5) bar chart احتمال کلاس‌ها
            story.extend(self._build_class_probabilities(all_probs))

            # 6) Footer
            story.extend(self._build_footer())

            # ✅ ساخت PDF
            doc.build(story)
            print(f"✅ PDF exported: {output_path}")
            return True

        except Exception as e:
            print(f"⚠️ PDF generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# ================================================================
# ✅ VALIDATION
# ================================================================

import re


def validate_password(password: str) -> Tuple[bool, str]:
    if len(password) < 8 or len(password) > 16:
        return False, "Length must be 8-16 characters"
    if not re.search(r"[a-z]", password):
        return False, "Must contain at least 1 lowercase letter"
    if not re.search(r"[A-Z]", password):
        return False, "Must contain at least 1 uppercase letter"
    if len(re.findall(r"\d", password)) < 4:
        return False, "Must contain at least 4 digits"
    if len(re.findall(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\/;`~]", password)) < 4:
        return False, "Must contain at least 4 special characters"
    return True, "OK"


def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_username(username: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_]{3,32}$", username))


# ================================================================
# ✅ MODEL LOADER (Lazy)
# ================================================================

class ModelLoader:
    _instance = None
    _model = None
    _device = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._load_model()
        return cls._model, cls._device

    @classmethod
    def _load_model(cls):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        if not os.path.exists(FINAL_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {FINAL_MODEL_PATH}")
        from src.models.resnet.resnet50 import create_resnet50
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_resnet50(num_classes=4, pretrained=False, verbose=False)
        ckpt = torch.load(FINAL_MODEL_PATH, map_location=cls._device, weights_only=False)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
        model = model.to(cls._device)
        model.eval()
        cls._model = model


# ================================================================
# ✅ IMAGE PREPROCESSING
# ================================================================

def preprocess_image(image_path: str, input_size: int = 224) -> "torch.Tensor":
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = (img_t - mean) / std
    return img_t.unsqueeze(0)

# ================================================================
# ✅ GRAD-CAM (v3.0 — attention heatmap)
# ================================================================

def generate_gradcam(
    model,
    img_tensor: "torch.Tensor",
    target_class: int,
    device=None,
    layer_name: str = None,
):
    """
    تولید Grad-CAM heatmap.

    Args:
        model: مدل ResNet (باید last conv layer داشته باشه)
        img_tensor: تنسور تصویر (1, 3, 224, 224)
        target_class: کلاسی که heatmap براش ساخته می‌شه
        device: دستگاه (اگه None، از img_tensor.device)
        layer_name: نام لایه (اگه None، خودکار پیدا می‌شه)

    Returns:
        heatmap: numpy array (H, W) با مقادیر 0-255
        یا None اگه خطا داد
    """
    try:
        if not HAS_TORCH:
            return None
        import torch
        import torch.nn.functional as F
        import numpy as np

        if device is None:
            device = img_tensor.device

        model.eval()
        img_tensor = img_tensor.to(device)

        # ✅ پیدا کردن آخرین لایه‌ی conv
        target_layer = None
        if layer_name:
            # اگه اسم لایه داده شده، از getattr استفاده کن
            parts = layer_name.split(".")
            target_layer = model
            for p in parts:
                target_layer = getattr(target_layer, p)
        else:
            # ✅ جستجوی خودکار: آخرین Conv2d
            last_conv = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            target_layer = last_conv

        if target_layer is None:
            print("⚠️ Grad-CAM: no conv layer found")
            return None

        # ✅ hookها
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations["value"] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        try:
            # ✅ forward
            model.zero_grad()
            output = model(img_tensor)

            # ✅ کلاس هدف
            if target_class < 0 or target_class >= output.shape[1]:
                target_class = int(output.argmax(dim=1).item())

            # ✅ score برای کلاس هدف
            score = output[0, target_class]
            score.backward(retain_graph=False)

            # ✅ محاسبه‌ی Grad-CAM
            if "value" not in gradients or "value" not in activations:
                return None

            grads = gradients["value"]  # (1, C, H, W)
            acts = activations["value"]  # (1, C, H, W)

            # ✅ weight = mean of gradients (per channel)
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # ✅ cam = sum(weights * activations)
            cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = F.relu(cam)  # فقط مقادیر مثبت

            # ✅ نرمالایز به 0-1
            cam_min = cam.min()
            cam_max = cam.max()
            if (cam_max - cam_min).item() < 1e-6:
                cam = torch.zeros_like(cam)
            else:
                cam = (cam - cam_min) / (cam_max - cam_min)

            # ✅ resize به 224x224 (اندازه‌ی ورودی)
            cam = F.interpolate(
                cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
                mode="bilinear", align_corners=False,
            )

            # ✅ به numpy
            heatmap = cam[0, 0].cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            return heatmap
        finally:
            h_fwd.remove()
            h_bwd.remove()
    except Exception as e:
        print(f"⚠️ generate_gradcam failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def overlay_heatmap_on_image(
    original_pixmap: QPixmap,
    heatmap: "np.ndarray",
    alpha: float = 0.45,
) -> QPixmap:
    """
    روی هم انداختن heatmap (با colormap JET) روی تصویر اصلی.
    """
    try:
        if not HAS_CV2 or heatmap is None:
            return original_pixmap

        import cv2
        import numpy as np
        from PyQt5.QtGui import QImage

        # ✅ تصویر اصلی رو به QImage تبدیل کن
        qimg = original_pixmap.toImage()
        qimg = qimg.convertToFormat(QImage.Format_RGB888)

        w = qimg.width()
        h = qimg.height()

        # ✅ bpl (bytes per line) رو دقیق بگیر
        bpl = qimg.bytesPerLine()

        # ✅ بافر رو با bpl بگیر
        ptr = qimg.bits()
        ptr.setsize(h * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, bpl))

        # ✅ فقط ستون‌های اول (w*3) رو بردار
        arr = arr[:, : w * 3].reshape((h, w, 3)).copy()

        # ✅ RGB → BGR
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # ✅ heatmap رو resize
        hm = cv2.resize(heatmap, (w, h))

        # ✅ JET colormap
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # ✅ overlay
        overlayed = cv2.addWeighted(arr_bgr, 1.0 - alpha, hm_color, alpha, 0)

        # ✅ BGR → RGB
        overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

        # ✅ به QPixmap (با bpl درست)
        qimg_result = QImage(
            overlayed_rgb.data,
            w, h,
            3 * w,
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(qimg_result)
    except Exception as e:
        print(f"⚠️ overlay_heatmap_on_image failed: {e}")
        return original_pixmap

# ================================================================
# ✅ ANIMATED BACKGROUND (calm, theme-aware, 30s cycle)
# ================================================================

class AnimatedBackground(QWidget):
    """Calm, romantic animated background.

    - 30s forward + 30s reverse (very slow)
    - Gentle ±30° rotation only (no vertigo)
    - Smooth wrap between last and first color (no jump)
    - Theme-aware palette (light themes → light palette)
    """

    CYCLE_MS = 30000
    TICK_MS = 100   # ← از ۵۰ به ۱۰۰

    COLORS_DARK = [
        QColor("#0f172a"),
        QColor("#0a0f26"),
        QColor("#0a3a4a"),
        QColor("#06202e"),
        QColor("#1a0b3a"),
        QColor("#0f0518"),
        QColor("#00161f"),
    ]

    COLORS_LIGHT = [
        QColor("#fdf6e3"),
        QColor("#eee8d5"),
        QColor("#e8f4f8"),
        QColor("#d8eef5"),
        QColor("#ede4f5"),
        QColor("#f0e8f5"),
        QColor("#e0f0ef"),
    ]

    def __init__(self, parent=None, theme_key="theme_dark_aqua"):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._elapsed = 0
        self._theme_key = theme_key
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.TICK_MS)

    def set_theme(self, theme_key: str):
        self._theme_key = theme_key
        self.update()

    def _current_palette(self):
        if self._theme_key == "theme_solarized_white":
            return self.COLORS_LIGHT
        return self.COLORS_DARK

    def _tick(self):
        self._elapsed = (self._elapsed + self.TICK_MS) % (2 * self.CYCLE_MS)
        self.update()

    def _compute_phase(self) -> float:
        t = self._elapsed / self.CYCLE_MS
        if t <= 1.0:
            return t
        return 2.0 - t

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _blend(self, c1: QColor, c2: QColor, t: float) -> QColor:
        t = max(0.0, min(1.0, t))
        return QColor(
            int(c1.red()   + (c2.red()   - c1.red())   * t),
            int(c1.green() + (c2.green() - c1.green()) * t),
            int(c1.blue()  + (c2.blue()  - c1.blue())  * t),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            painter.end()
            return

        palette = self._current_palette()
        n = len(palette)
        raw_phase = self._compute_phase()

        idx = raw_phase * n
        i0 = int(idx) % n
        i1 = (i0 + 1) % n
        i2 = (i0 + 2) % n
        i3 = (i0 + 3) % n
        frac = idx - int(idx)
        frac_smooth = self._smoothstep(frac)

        c0 = self._blend(palette[i0], palette[i1], frac_smooth)
        c1 = self._blend(palette[i1], palette[i2], frac_smooth)
        c2 = self._blend(palette[i2], palette[i3], frac_smooth)
        c3 = self._blend(palette[i3], palette[(i0 + 4) % n], frac_smooth)

        # ✅ rotation ملایم — فقط ±30 درجه
        ang = math.sin(self._smoothstep(raw_phase) * 2.0 * math.pi) * math.radians(30)
        dx = math.cos(ang)
        dy = math.sin(ang)

        grad = QLinearGradient(
            w * (0.5 - dx * 0.5), h * (0.5 - dy * 0.5),
            w * (0.5 + dx * 0.5), h * (0.5 + dy * 0.5),
        )
        # ✅ فقط ۲ stop (کمتر banding، کمتر CPU)
        grad.setColorAt(0.00, c0)
        grad.setColorAt(1.00, c3)
        painter.fillRect(self.rect(), QBrush(grad))

        # ✅ Dithering: نویز خیلی کم برای شکستن banding
        self._apply_dither(painter, w, h)

        painter.end()

    def _apply_dither(self, painter: QPainter, w: int, h: int):
        """
        نویز خیلی کم (۳٪) روی گرادیان برای شکستن banding.
        از یه pixmap نویز ۶۴×۶۴ استفاده می‌کنیم که tile می‌شه.
        """
        try:
            # ✅ ساخت یه pixmap نویز (فقط بار اول)
            if not hasattr(self, "_dither_pixmap") or self._dither_pixmap is None:
                size = 64
                pixmap = QPixmap(size, size)
                pixmap.fill(Qt.transparent)
                p = QPainter(pixmap)
                p.setRenderHint(QPainter.Antialiasing, False)
                # ✅ الگوی نویز شبه‌تصادفی (deterministic)
                for y in range(size):
                    for x in range(size):
                        # فرمول ساده‌ی شبه‌تصادفی
                        val = (x * 73 + y * 151 + (x * y) * 37) % 256
                        # opacity ۰ تا ۸ (خیلی کم)
                        alpha = min(8, val % 9)
                        if alpha > 0:
                            p.setPen(QColor(255, 255, 255, alpha))
                            p.drawPoint(x, y)
                p.end()
                self._dither_pixmap = pixmap

            # ✅ tile کردن pixmap نویز روی کل صفحه
            painter.save()
            painter.setOpacity(1.0)
            brush = QBrush(self._dither_pixmap)
            painter.setBrushOrigin(0, 0)
            painter.fillRect(self.rect(), brush)
            painter.restore()
        except Exception as e:
            print(f"⚠️ _apply_dither failed: {e}")

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()

# ================================================================
# ✅ GLASSMORPHISM BACKGROUND (v3.0 — real blur with cached pixmap)
# ================================================================

class GlassmorphismBackground(QWidget):
    """
    پس‌زمینه‌ی glassmorphism واقعی:
    - AnimatedBackground رو رندر می‌کنه به pixmap
    - pixmap رو blur می‌کنه (QGraphicsBlurEffect)
    - pixmap blur شده رو به‌عنوان background نشون می‌ده
    - هر BLUR_INTERVAL فریم یه‌بار blur (برای performance)
    """

    BLUR_INTERVAL = 15    # هر ۱۵ فریم blur (۷۵۰ms — انیمیشن نرم)
    BLUR_RADIUS = 22      # شدت blur (بالاتر = glassmorphism واضح‌تر)
    DOWNSCALE = 0.4       # pixmap رو ۱/۲.۵ اندازه blur (تعادل کیفیت/سرعت)
    TICK_MS = 50          # هر ۵۰ms

    def __init__(self, parent=None, theme_key="theme_dark_aqua"):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._theme_key = theme_key
        self._frame_count = 0
        self._blurred_pixmap = None
        self._bg_inner = AnimatedBackground(self, theme_key)
        self._bg_inner.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._bg_inner.setVisible(False)

        # ✅ تایمر اصلی
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.TICK_MS)

        # ✅ اجرای اولیه
        QTimer.singleShot(100, self._do_blur)

    def set_theme(self, theme_key: str):
        self._theme_key = theme_key
        self._bg_inner.set_theme(theme_key)
        self._blurred_pixmap = None
        self._do_blur()
        self.update()

    def _tick(self):
        """هر ۵۰ms: فریم رو جلو ببر، هر ۱۵ فریم blur کن."""
        self._frame_count += 1
        if self._frame_count % self.BLUR_INTERVAL == 0:
            self._do_blur()
        self.update()

    def _do_blur(self):
        """رندر AnimatedBackground به pixmap و blur."""
        try:
            w, h = self.width(), self.height()
            if w <= 0 or h <= 0:
                return

            small_w = max(1, int(w * self.DOWNSCALE))
            small_h = max(1, int(h * self.DOWNSCALE))

            # ✅ رندر با QPainter دستی (به‌جای render که همزمان painter ایجاد می‌کنه)
            small_pixmap = QPixmap(small_w, small_h)
            small_pixmap.fill(Qt.transparent)
            self._bg_inner.setGeometry(0, 0, small_w, small_h)
            _p = QPainter(small_pixmap)
            try:
                self._bg_inner.render(_p, QPoint(0, 0), QRegion(0, 0, small_w, small_h))
            finally:
                _p.end()

            # ✅ blur با QGraphicsBlurEffect
            scene = QGraphicsScene()
            item = QGraphicsPixmapItem(small_pixmap)
            blur = QGraphicsBlurEffect()
            blur.setBlurRadius(self.BLUR_RADIUS * self.DOWNSCALE)
            item.setGraphicsEffect(blur)
            scene.addItem(item)

            # ✅ رندر
            blurred_small = QPixmap(small_w, small_h)
            blurred_small.fill(Qt.transparent)
            painter = QPainter(blurred_small)
            scene.render(painter, QRectF(0, 0, small_w, small_h), QRectF(0, 0, small_w, small_h))
            painter.end()

            # ✅ scale up
            self._blurred_pixmap = blurred_small.scaled(
                w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
        except Exception as e:
            print(f"⚠️ _do_blur failed: {e}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        if self._blurred_pixmap is not None and not self._blurred_pixmap.isNull():
            painter.drawPixmap(0, 0, self._blurred_pixmap)
        else:
            painter.fillRect(self.rect(), QColor("#0f172a"))

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._blurred_pixmap = None
        QTimer.singleShot(50, self._do_blur)

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()

# ================================================================
# ✅ DRAG & DROP OVERLAY
# ================================================================

# ================================================================
# ✅ DRAG & DROP OVERLAY (v3.0 — theme-aware, animated, safe)
# ================================================================

class DropOverlay(QWidget):
    """
    Semi-transparent overlay shown during drag to indicate the drop zone.

    - WA_TransparentForMouseEvents  → drop event را نمی‌دزدد (خیلی مهم!)
    - WA_TranslucentBackground      → پس‌زمینه شفاف
    - set_theme()                   → با تغییر تم، رنگ هم عوض می‌شود
    - set_texts()                   → متن قابل ترجمه
    - fade_in() / fade_out()        → انیمیشن نرم ورود/خروج
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setVisible(False)
        self._theme = THEMES["theme_dark_aqua"]
        self._icon_text = "🖼️"
        self._title_text = "Drop image here"
        self._subtitle_text = "PNG, JPG, JPEG, BMP, DCM"
        self._opacity = 0.0

        # ✅ افکت شفافیت برای fade
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(180)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_theme(self, theme: Dict):
        self._theme = theme
        self.update()

    def set_texts(self, title: str, subtitle: str):
        self._title_text = title
        self._subtitle_text = subtitle
        self.update()

    def fade_in(self):
        """نمایش overlay با انیمیشن fade-in."""
        if self.isVisible() and self._opacity_effect.opacity() >= 0.99:
            return
        self.setVisible(True)
        self.raise_()
        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except TypeError:
            pass
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def fade_out(self):
        """محو کردن overlay با انیمیشن fade-out، سپس مخفی کردن."""
        if not self.isVisible():
            return
        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except TypeError:
            pass
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.finished.connect(self._on_fade_out_done)
        self._fade_anim.start()

    def _on_fade_out_done(self):
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_done)
        except TypeError:
            pass
        if self._opacity_effect.opacity() <= 0.01:
            self.setVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        t = self._theme
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            painter.end()
            return

        # ✅ لایه نیمه‌شفاف با رنگ primary تم
        try:
            base = QColor(t["primary"])
            painter.fillRect(self.rect(), QColor(base.red(), base.green(), base.blue(), 70))
        except Exception:
            painter.fillRect(self.rect(), QColor(0, 229, 255, 70))

        # ✅ کادر بزرگ نقطه‌چین
        margin = 60
        rect = QRect(margin, margin, w - 2 * margin, h - 2 * margin)
        pen = QPen(QColor(t["primary"]), 3, Qt.DashLine)
        pen.setDashPattern([10, 6])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 24, 24)

        # ✅ آیکون بزرگ
        painter.setPen(QColor(t["primary"]))
        icon_font = QFont("Segoe UI Emoji", 72)
        painter.setFont(icon_font)
        icon_rect = QRect(0, h // 2 - 140, w, 120)
        painter.drawText(icon_rect, Qt.AlignCenter, self._icon_text)

        # ✅ عنوان
        title_font = QFont("Segoe UI", 24, QFont.Bold)
        painter.setFont(title_font)
        title_rect = QRect(0, h // 2 - 10, w, 40)
        painter.drawText(title_rect, Qt.AlignCenter, self._title_text)

        # ✅ زیرنویس
        sub_font = QFont("Segoe UI", 12)
        painter.setFont(sub_font)
        painter.setPen(QColor(t["text_dim"]))
        sub_rect = QRect(0, h // 2 + 30, w, 30)
        painter.drawText(sub_rect, Qt.AlignCenter, self._subtitle_text)

        painter.end()

# ================================================================
# ✅ ZOOMABLE IMAGE VIEW (v3.0 — Zoom + Pan + Fit)
# ================================================================

class ZoomableImageView(QWidget):
    """
    A widget that displays an image with:
    - Mouse-wheel zoom (centered under the mouse)
    - Pan by dragging with left mouse button (when zoomed)
    - +/-/reset buttons in bottom-right corner
    - Double-click to fit-to-window
    - Placeholder when no image
    - Theme-aware styling

    Usage:
        view = ZoomableImageView(theme=THEMES["theme_dark_aqua"])
        view.set_theme(theme_dict)
        view.set_placeholder_text("🖼️\n\nNo image inserted")
        view.set_image(pixmap)
        view.clear_image()  # back to placeholder
    """

    MIN_ZOOM = 0.1      # 10% — حداقل زوم
    MAX_ZOOM = 20.0     # 2000% — حداکثر زوم
    ZOOM_STEP = 1.15    # ضریب هر چرخ ماوس

    def __init__(self, parent=None, theme=None):
        super().__init__(parent)

        self._theme = theme or THEMES["theme_dark_aqua"]
        self._placeholder_text = "🖼️\n\nNo image inserted"
        self._has_image = False
        self._is_panning = False
        self._pan_start = QPoint(0, 0)
        self._zoom_level = 1.0
        self._fit_zoom = 1.0
        self._user_zoomed = False  # ← اگه کاربر دستی زوم کرد، دیگه auto-fit نکن
        self._lang = "English"  # ✅ برای tooltipها

        # ============================================================
        # Scene + View
        # ============================================================
        self.scene = QGraphicsScene(self)

        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setDragMode(QGraphicsView.NoDrag)  # ← خودمون pan می‌کنیم
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # ← زوم زیر ماوس
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setBackgroundBrush(QBrush(Qt.NoBrush))
        self.view.setStyleSheet("background: transparent; border: none;")

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self._pixmap_item)

        # ============================================================
        # Placeholder (یک QLabel که وقتی تصویری نیست نشون داده میشه)
        # ============================================================
        self._placeholder = QLabel(self._placeholder_text, self)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setObjectName("zoom_placeholder")
        self._placeholder.setWordWrap(True)

        # ============================================================
        # Zoom controls (شناور در گوشه‌ی پایین-راست)
        # ============================================================
        self._controls = QFrame(self)
        self._controls.setObjectName("zoom_controls")
        controls_layout = QHBoxLayout(self._controls)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_layout.setSpacing(4)

        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setObjectName("zoom_btn")
        self._btn_zoom_in.setFixedSize(32, 32)
        self._btn_zoom_in.clicked.connect(self.zoom_in)

        self._btn_zoom_out = QPushButton("−")
        self._btn_zoom_out.setObjectName("zoom_btn")
        self._btn_zoom_out.setFixedSize(32, 32)
        self._btn_zoom_out.clicked.connect(self.zoom_out)

        self._btn_reset = QPushButton("⟲")
        self._btn_reset.setObjectName("zoom_btn")
        self._btn_reset.setFixedSize(32, 32)
        self._btn_reset.clicked.connect(self.fit_to_window)

        controls_layout.addWidget(self._btn_zoom_in)
        controls_layout.addWidget(self._btn_zoom_out)
        controls_layout.addWidget(self._btn_reset)

        # ✅ همه‌ی دکمه‌ها کلیک‌پذیر باشن
        for btn in (self._btn_zoom_in, self._btn_zoom_out, self._btn_reset):
            btn.setCursor(Qt.PointingHandCursor)

        self._controls.setVisible(False)  # فقط وقتی تصویر هست نشون بده

        # ============================================================
        # Layout — view پرکننده، placeholder و controls روی هم
        # ============================================================
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self._apply_theme()

    # ============================================================
    # Theme
    # ============================================================
    def set_theme(self, theme: Dict):
        self._theme = theme
        self._apply_theme()

    def set_lang(self, lang: str):
        """تغییر زبان tooltipها."""
        self._lang = lang
        try:
            self.set_tooltips(self._theme, lang)
        except Exception:
            pass

    def _apply_theme(self):
        t = self._theme
        self._placeholder.setStyleSheet(f"""
            QLabel#zoom_placeholder {{
                color: {t['text_dim']};
                background: transparent;
                border: 2px dashed {t['glass_border']};
                border-radius: 20px;
                padding: 20px;
            }}
        """)
        # ✅ Empty state HTML برای placeholder
        try:
            self._update_placeholder_text()
        except Exception:
            pass
        self._controls.setStyleSheet(f"""
            QFrame#zoom_controls {{
                background: {t['input_bg']};
                border: 1px solid {t['glass_border']};
                border-radius: 12px;
            }}
            QPushButton#zoom_btn {{
                background: transparent;
                color: {t['text']};
                border: 1px solid {t['glass_border']};
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton#zoom_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#zoom_btn:pressed {{
                background: {t['accent']};
            }}
        """)

    # ============================================================
    # Placeholder
    # ============================================================
    def set_placeholder_text(self, text: str):
        self._placeholder_text = text
        # ✅ از HTML زیبا استفاده کن اگه text شامل \n هست
        try:
            self._update_placeholder_text()
        except Exception:
            self._placeholder.setText(text)

    def _update_placeholder_text(self):
        """ساخت HTML زیبا برای placeholder."""
        try:
            html = make_empty_state_html(
                key="image_empty",
                theme=self._theme,
                lang=self._lang,
                custom_title=self._placeholder_text.split("\n")[-1].strip()
                    if "\n" in self._placeholder_text else "",
            )
            if html:
                self._placeholder.setText(html)
                self._placeholder.setTextFormat(Qt.RichText)
            else:
                self._placeholder.setText(self._placeholder_text)
        except Exception as e:
            print(f"⚠️ _update_placeholder_text failed: {e}")
            self._placeholder.setText(self._placeholder_text)

    # ============================================================
    # Image control
    # ============================================================
    def set_image(self, pixmap: QPixmap):
        """بارگذاری تصویر جدید و fit خودکار."""
        if pixmap is None or pixmap.isNull():
            self.clear_image()
            return

        self._has_image = True
        self._user_zoomed = False

        self._pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self._pixmap_item.boundingRect())

        self._placeholder.setVisible(False)
        self._controls.setVisible(True)
        self.view.setVisible(True)

        # ✅ fit بعد از layout شدن
        QTimer.singleShot(0, self.fit_to_window)

    def clear_image(self):
        """پاک کردن تصویر و برگشتن به placeholder."""
        self._has_image = False
        self._user_zoomed = False
        self._pixmap_item.setPixmap(QPixmap())  # خالی
        self.scene.setSceneRect(QRect())

        self._placeholder.setVisible(True)
        self._controls.setVisible(False)

    def has_image(self) -> bool:
        return self._has_image

    # ============================================================
    # Zoom
    # ============================================================
    def fit_to_window(self):
        """تصویر رو کامل داخل view جا بده."""
        if not self._has_image:
            return
        self.view.resetTransform()
        self.view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        # ذخیره‌ی zoom فعلی برای مقایسه
        self._fit_zoom = self.view.transform().m11()
        self._zoom_level = 1.0
        self._user_zoomed = False

    def zoom_in(self):
        self._zoom_by(self.ZOOM_STEP)

    def zoom_out(self):
        self._zoom_by(1.0 / self.ZOOM_STEP)

    def _zoom_by(self, factor: float):
        """زوم با محدودیت MIN/MAX."""
        if not self._has_image:
            return
        current = self.view.transform().m11()
        target = current * factor
        # محدودیت نسبت به fit_zoom
        min_allowed = self._fit_zoom * self.MIN_ZOOM
        max_allowed = self._fit_zoom * self.MAX_ZOOM
        if target < min_allowed:
            factor = min_allowed / current
        elif target > max_allowed:
            factor = max_allowed / current
        if abs(factor - 1.0) < 1e-6:
            return
        self.view.scale(factor, factor)
        self._zoom_level *= factor
        self._user_zoomed = True

    # ============================================================
    # Events — wheel / mouse / resize
    # ============================================================
    def wheelEvent(self, event: QWheelEvent):
        """زوم با چرخ ماوس — حول نقطه‌ی زیر ماوس."""
        if not self._has_image:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = self.ZOOM_STEP if delta > 0 else (1.0 / self.ZOOM_STEP)
        self._zoom_by(factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """شروع pan با کلیک چپ."""
        if not self._has_image:
            return super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            # ✅ بررسی اینکه کلیک روی کنترل‌ها نباشه
            child = self.childAt(event.pos())
            if child is not None and (child is self._controls or self._controls.isAncestorOf(child)):
                return super().mousePressEvent(event)
            self._is_panning = True
            self._pan_start = event.pos()
            self.view.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """pan در حین حرکت."""
        if self._is_panning and self._has_image:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            # ✅ حرکت دادن scrollbarها (بدون scrollbar visible)
            h = self.view.horizontalScrollBar()
            v = self.view.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """پایان pan."""
        if event.button() == Qt.LeftButton and self._is_panning:
            self._is_panning = False
            self.view.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """double-click → fit to window."""
        if event.button() == Qt.LeftButton and self._has_image:
            self.fit_to_window()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        """اگه کاربر زوم نکرده، با تغییر اندازه پنجره دوباره fit کن."""
        super().resizeEvent(event)
        # layout placeholder و controls
        self._placeholder.setGeometry(20, 20, self.width() - 40, self.height() - 40)
        self._controls.adjustSize()
        self._controls.move(
            self.width() - self._controls.width() - 16,
            self.height() - self._controls.height() - 16,
        )
        if self._has_image and not self._user_zoomed:
            self.fit_to_window()

    def showEvent(self, event):
        super().showEvent(event)
        # ✅ اطمینان از geometry درست بعد از نمایش
        self._placeholder.setGeometry(20, 20, self.width() - 40, self.height() - 40)
        if self._has_image and not self._user_zoomed:
            QTimer.singleShot(0, self.fit_to_window)

    # ============================================================
    # ✅ Smart tooltips
    # ============================================================

    def set_tooltips(self, theme: Dict, lang: str):
        """
        به‌روزرسانی tooltipهای دکمه‌های zoom با تم و زبان جدید.
        باید از MainWindow._update_all_tooltips صدا زده بشه.
        """
        try:
            t = theme or self._theme
            self._theme = t

            # ---------- Zoom In (+) ----------
            try:
                self._btn_zoom_in.setToolTip(
                    make_tooltip(
                        title="➕ " + tr(lang, "tooltip_title_zoom_in"),
                        description=tr(lang, "tooltip_desc_zoom_in"),
                        shortcut="Ctrl++ / Wheel ↑",
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- Zoom Out (−) ----------
            try:
                self._btn_zoom_out.setToolTip(
                    make_tooltip(
                        title="➖ " + tr(lang, "tooltip_title_zoom_out"),
                        description=tr(lang, "tooltip_desc_zoom_out"),
                        shortcut="Ctrl+- / Wheel ↓",
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- Reset (⟲) ----------
            try:
                self._btn_reset.setToolTip(
                    make_tooltip(
                        title="⟲ " + tr(lang, "tooltip_title_zoom_reset"),
                        description=tr(lang, "tooltip_desc_zoom_reset"),
                        shortcut="Double-click",
                        tip=tr(lang, "tooltip_tip_zoom_reset"),
                        theme=t,
                    )
                )
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ ZoomableImageView.set_tooltips failed: {e}")

# ================================================================
# ✅ HEATMAP VIEWER (v3.0 — دیالوگ نمایش Grad-CAM)
# ================================================================

class HeatmapViewer(QDialog):
    """
    دیالوگ نمایش Grad-CAM heatmap.
    - تصویر اصلی + heatmap overlay
    - toggle برای نمایش/مخفی کردن heatmap
    - ذخیره‌ی تصویر
    """

    def __init__(self, original_pixmap: QPixmap, heatmap_pixmap: QPixmap,
                 theme: Dict, lang: str = "English", parent=None):
        super().__init__(parent)
        self.original_pixmap = original_pixmap
        self.heatmap_pixmap = heatmap_pixmap
        self.theme = theme
        self.lang = lang
        self._show_overlay = True

        self.setWindowTitle(tr(lang, "gradcam_title"))
        self.setMinimumSize(700, 700)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

        # ✅ انیمیشن باز شدن
        try:
            from GUI_update import animate_dialog_open
            QTimer.singleShot(0, lambda: animate_dialog_open(self))
        except Exception:
            pass

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # ✅ عنوان
        title = QLabel("🔥 " + tr(self.lang, "gradcam_title"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ✅ تصویر
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("heatmap_image")
        self.image_label.setMinimumHeight(500)
        self._update_image()
        layout.addWidget(self.image_label, 1)

        # ✅ دکمه‌ها
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.toggle_btn = QPushButton("👁 " + tr(self.lang, "gradcam_toggle"))
        self.toggle_btn.setObjectName("action_btn")
        self.toggle_btn.clicked.connect(self._toggle_overlay)
        btn_layout.addWidget(self.toggle_btn)

        self.save_btn = QPushButton("💾 " + tr(self.lang, "gradcam_save"))
        self.save_btn.setObjectName("action_btn")
        self.save_btn.clicked.connect(self._save_image)
        btn_layout.addWidget(self.save_btn)

        self.close_btn = QPushButton("✕ " + tr(self.lang, "gradcam_close"))
        self.close_btn.setObjectName("secondary_btn")
        self.close_btn.clicked.connect(self._close_with_anim)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _update_image(self):
        """آپدیت تصویر بر اساس toggle."""
        try:
            if self._show_overlay:
                pix = self.heatmap_pixmap
            else:
                pix = self.original_pixmap

            # ✅ scale برای نمایش
            max_w, max_h = 640, 500
            scaled = pix.scaled(
                max_w, max_h,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
        except Exception as e:
            print(f"⚠️ _update_image failed: {e}")

    def _toggle_overlay(self):
        """Toggle بین تصویر اصلی و heatmap."""
        self._show_overlay = not self._show_overlay
        self._update_image()

    def _save_image(self):
        """ذخیره‌ی heatmap به فایل."""
        try:
            from datetime import datetime
            default_name = f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.isdir(save_dir):
                save_dir = PROJECT_ROOT

            output_path, _ = QFileDialog.getSaveFileName(
                self,
                tr(self.lang, "gradcam_save"),
                os.path.join(save_dir, default_name),
                "Images (*.png *.jpg *.jpeg)",
            )
            if output_path:
                self.heatmap_pixmap.save(output_path, "PNG")
                show_message(
                    self, "information",
                    tr(self.lang, "success"),
                    f"{tr(self.lang, 'gradcam_saved')}:\n{output_path}",
                )
        except Exception as e:
            print(f"⚠️ _save_image failed: {e}")

    def _close_with_anim(self):
        """بستن با انیمیشن."""
        try:
            from GUI_update import animate_dialog_close
            animate_dialog_close(self, on_finished=self.accept)
        except Exception:
            self.accept()

    def _apply_theme(self):
        t = self.theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#title {{
                font-size: 20px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
                padding: 5px;
            }}
            QLabel#heatmap_image {{
                background: {t['input_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 12px;
            }}
            QPushButton#action_btn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:0.5 {t['accent']}, stop:1 {t['secondary']});
                color: {t['button_text']};
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-width: 120px;
            }}
            QPushButton#action_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#secondary_btn {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton#secondary_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
        """)

# ================================================================
# ✅ PROCESSING PROGRESS BAR (v3.0 — real-ish, animated, theme-aware)
# ================================================================

# ================================================================
# ✅ PROCESSING PROGRESS BAR (v3.1 — bigger, glowing, 3.5s)
# ================================================================

class ProcessingProgressBar(QWidget):
    """
    A thin animated progress bar shown during inference.

    States:
        - idle:    hidden
        - running: animated fill (primary color) up to ~90% in ~3.5s
        - success: jump to 100%, green, then fade out
        - error:   jump to 100%, red, then fade out

    Visual highlights:
        - 10px height (visible, not chunky)
        - Outer border + inner glow
        - Shine line on top of the fill
        - Animated pulse while running (subtle)

    Usage:
        bar = ProcessingProgressBar(theme=THEMES["theme_dark_aqua"])
        bar.set_theme(theme_dict)
        bar.start()                # begin animation
        bar.finish_success()       # or finish_error()
        bar.reset()
    """

    # ✅ زمان‌بندی — ۳.۵ ثانیه به ۹۰٪
    TICK_MS = 50                   # هر ۵۰ms یه قدم (نرم‌تر)
    TARGET_MS = 3500               # ۳.۵ ثانیه تا ۹۰٪
    MAX_FAKE_PERCENT = 90          # شبیه‌سازی تا ۹۰٪
    SUCCESS_HOLD_MS = 800          # مدت نمایش ۱۰۰٪ سبز
    ERROR_HOLD_MS = 1000           # مدت نمایش قرمز
    FADE_MS = 350                  # مدت محو شدن

    HEIGHT = 10                    # ← کلفت‌تر از ۶px

    def __init__(self, parent=None, theme=None):
        super().__init__(parent)

        self._theme = theme or THEMES["theme_dark_aqua"]
        self._state = "idle"       # idle | running | success | error
        self._percent = 0.0        # 0.0 تا 1.0
        self._pulse = 0.0          # برای انیمیشن pulse

        # ✅ ظرف پس‌زمینه (track)
        self._track = QFrame(self)
        self._track.setObjectName("progress_track")

        # ✅ نوار پر شده (fill) — داخل track
        self._fill = QFrame(self._track)
        self._fill.setObjectName("progress_fill")

        # ✅ نوار درخشان (shine) — روی بالای fill
        self._shine = QFrame(self._fill)
        self._shine.setObjectName("progress_shine")

        # ✅ افکت شفافیت برای fade
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(self.FADE_MS)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)

        # ✅ تایمر شبیه‌سازی
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_tick)

        # ✅ تایمر pulse (برای حس زنده بودن)
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._on_pulse_tick)

        # ✅ تایمر hold (بعد از success/error)
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._fade_out_and_hide)

        self.setVisible(False)
        self.setFixedHeight(self.HEIGHT)

        self._apply_theme()

    # ============================================================
    # Theme
    # ============================================================
    def set_theme(self, theme: Dict):
        self._theme = theme
        self._apply_theme()
        # ✅ به‌روزرسانی tooltipها
        try:
            self.set_tooltips(theme, self._lang if hasattr(self, "_lang") else "English")
        except Exception:
            pass

    def _apply_theme(self):
        t = self._theme

        # ✅ رنگ fill بر اساس state
        if self._state == "success":
            fill_start = t["success"]
            fill_end = t["primary"]
        elif self._state == "error":
            fill_start = t["danger"]
            fill_end = t["accent"]
        else:
            fill_start = t["primary"]
            fill_end = t["secondary"]

        # ✅ Track: تیره با border واضح
        # اگه تم روشنه، track روشن‌تر؛ اگه تیره، track تیره‌تر
        is_light_theme = (self._theme.get("name") == "Solarized White")
        if is_light_theme:
            track_bg = "rgba(38, 139, 210, 0.18)"
            track_border = "rgba(38, 139, 210, 0.55)"
        else:
            track_bg = "rgba(0, 0, 0, 0.35)"
            track_border = t["glass_border"]

        self._track.setStyleSheet(f"""
            QFrame#progress_track {{
                background: {track_bg};
                border: 1px solid {track_border};
                border-radius: 5px;
            }}
        """)

        # ✅ Fill: gradient پرانرژی + گوشه‌ی گرد
        self._fill.setStyleSheet(f"""
            QFrame#progress_fill {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {fill_start},
                    stop:0.5 {fill_end},
                    stop:1 {fill_start}
                );
                border-radius: 4px;
            }}
        """)

        # ✅ Shine: یه خط نازک روشن بالای fill
        self._shine.setStyleSheet(f"""
            QFrame#progress_shine {{
                background: rgba(255, 255, 255, 0.45);
                border-radius: 2px;
            }}
        """)

    # ============================================================
    # Layout
    # ============================================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # ✅ track کل عرض و ارتفاع
        self._track.setGeometry(0, 0, self.width(), self.height())
        # ✅ fill بر اساس percent
        self._update_fill()

    def _update_fill(self):
        """به‌روزرسانی عرض fill + shine بر اساس percent."""
        w = self._track.width()
        h = self._track.height()
        fill_w = max(0, int(w * self._percent))
        # ✅ یه کم inset برای گردی گوشه‌ها
        inset = 1
        self._fill.setGeometry(
            inset, inset,
            max(0, fill_w - 2 * inset),
            max(0, h - 2 * inset),
        )
        # ✅ shine: خط نازک بالای fill (۲px ارتفاع)
        if self._fill.width() > 4:
            self._shine.setGeometry(
                4, 1,
                self._fill.width() - 8,
                2,
            )
        else:
            self._shine.setGeometry(0, 0, 0, 0)

    # ============================================================
    # State transitions
    # ============================================================
    def start(self):
        """شروع انیمیشن از ۰٪ — ۳.۵ ثانیه تا ۹۰٪."""
        self._tick_timer.stop()
        self._pulse_timer.stop()
        self._hold_timer.stop()
        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except TypeError:
            pass

        self._state = "running"
        self._percent = 0.0
        self._pulse = 0.0
        self._apply_theme()

        self.setVisible(True)
        self.raise_()

        # ✅ fade in
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

        # ✅ شروع شبیه‌سازی و pulse
        self._tick_timer.start(self.TICK_MS)
        self._pulse_timer.start(80)

    def finish_success(self):
        """پایان موفق — پرش به ۱۰۰٪ سبز، بعد محو."""
        if self._state != "running":
            return
        self._tick_timer.stop()
        self._pulse_timer.stop()
        self._state = "success"
        self._percent = 1.0
        self._apply_theme()
        self._update_fill()
        self._hold_timer.start(self.SUCCESS_HOLD_MS)

    def finish_error(self):
        """پایان با خطا — پرش به ۱۰۰٪ قرمز، بعد محو."""
        if self._state != "running":
            return
        self._tick_timer.stop()
        self._pulse_timer.stop()
        self._state = "error"
        self._percent = 1.0
        self._apply_theme()
        self._update_fill()
        self._hold_timer.start(self.ERROR_HOLD_MS)

    def reset(self):
        """برگشت به idle (مخفی)."""
        self._tick_timer.stop()
        self._pulse_timer.stop()
        self._hold_timer.stop()
        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except TypeError:
            pass
        self._state = "idle"
        self._percent = 0.0
        self._pulse = 0.0
        self._apply_theme()
        self._update_fill()
        self._opacity_effect.setOpacity(0.0)
        self.setVisible(False)

    # ============================================================
    # Internals
    # ============================================================
    def _on_tick(self):
        """شبیه‌سازی پیشرفت — ۳.۵ ثانیه تا ۹۰٪."""
        if self._state != "running":
            return

        # ✅ محاسبه‌ی step بر اساس زمان کل
        # تعداد tick‌های لازم: TARGET_MS / TICK_MS
        # درصد لازم در هر tick: (MAX_FAKE_PERCENT/100) / num_ticks
        num_ticks = max(1, self.TARGET_MS // self.TICK_MS)
        per_tick = (self.MAX_FAKE_PERCENT / 100.0) / num_ticks

        # ✅ easing: اول کند، بعد سریع، آخر کند (smooth)
        # با یه factor ساده، mid-range سریع‌تر
        if self._percent < 0.3:
            factor = 1.0
        elif self._percent < 0.75:
            factor = 1.3
        else:
            factor = 0.7

        step = per_tick * factor
        self._percent = min(
            self._percent + step,
            self.MAX_FAKE_PERCENT / 100.0,
        )
        self._update_fill()

    def _on_pulse_tick(self):
        """پالس ملایم روی shine — حس زنده بودن."""
        if self._state != "running":
            return
        import math as _m
        self._pulse = (self._pulse + 0.15) % (2 * _m.pi)
        # shine opacity بین 0.25 و 0.65 نوسان می‌کنه
        alpha = 0.45 + 0.2 * _m.sin(self._pulse)
        try:
            self._shine.setStyleSheet(f"""
                QFrame#progress_shine {{
                    background: rgba(255, 255, 255, {alpha:.2f});
                    border-radius: 2px;
                }}
            """)
        except Exception:
            pass

    def _fade_out_and_hide(self):
        """محو کردن و مخفی کردن بعد از success/error."""
        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except TypeError:
            pass
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.finished.connect(self._on_fade_out_done)
        self._fade_anim.start()

    def _on_fade_out_done(self):
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_done)
        except TypeError:
            pass
        if self._opacity_effect.opacity() <= 0.01:
            self.setVisible(False)
            self._state = "idle"
            self._percent = 0.0

# ================================================================
# ✅ DASHBOARD PANEL (v3.0 — Mini-Stats: UI construction)
# ================================================================

class DashboardPanel(QFrame):
    """
    پنل کناری داشبورد — نمایش آمار کاربر:
    - Total images processed
    - Average confidence (با رنگ‌بندی)
    - Distribution bar chart (NL/G1/G2/G3)
    - Last 5 recent results
    """

    # ✅ رنگ هر گرید
    GRID_COLORS = {
        0: "#34d399",   # NL — سبز
        1: "#fbbf24",   # G1 — زرد
        2: "#fb923c",   # G2 — نارنجی
        3: "#f87171",   # G3 — قرمز
    }

    closed = pyqtSignal()  # سیگنال وقتی کاربر دکمه‌ی ✕ رو می‌زنه

    def __init__(self, parent=None, theme=None, lang="English"):
        super().__init__(parent)
        self._theme = theme or THEMES["theme_dark_aqua"]
        self._lang = lang
        self._stats = None
        self._clear_callback = None

        self.setObjectName("dashboard_panel")
        self.setFixedWidth(300)
        self.setMinimumHeight(400)

        self._build_ui()
        self._apply_theme()

    # ============================================================
    # UI construction
    # ============================================================
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # ---------- Header ----------
        header = QHBoxLayout()
        header.setSpacing(6)

        self._title = QLabel("📊 Dashboard")
        self._title.setObjectName("dash_title")
        header.addWidget(self._title)
        header.addStretch()

        self._clear_btn = QPushButton("🗑️")
        self._clear_btn.setObjectName("dash_icon_btn")
        self._clear_btn.setFixedSize(26, 26)
        header.addWidget(self._clear_btn)

        self._close_btn = QPushButton("✕")
        self._close_btn.setObjectName("dash_icon_btn")
        self._close_btn.setFixedSize(26, 26)
        header.addWidget(self._close_btn)

        layout.addLayout(header)

        # ---------- Total card ----------
        self._total_card = QFrame()
        self._total_card.setObjectName("dash_card")
        total_layout = QVBoxLayout(self._total_card)
        total_layout.setContentsMargins(14, 12, 14, 12)
        total_layout.setSpacing(2)

        self._total_label = QLabel("Total Images")
        self._total_label.setObjectName("dash_card_label")
        self._total_label.setAlignment(Qt.AlignCenter)

        self._total_value = QLabel("0")
        self._total_value.setObjectName("dash_card_value")
        self._total_value.setAlignment(Qt.AlignCenter)

        total_layout.addWidget(self._total_label)
        total_layout.addWidget(self._total_value)
        layout.addWidget(self._total_card)

        # ---------- Average confidence card ----------
        self._avg_card = QFrame()
        self._avg_card.setObjectName("dash_card")
        avg_layout = QVBoxLayout(self._avg_card)
        avg_layout.setContentsMargins(14, 12, 14, 12)
        avg_layout.setSpacing(2)

        self._avg_label = QLabel("Avg. Confidence")
        self._avg_label.setObjectName("dash_card_label")
        self._avg_label.setAlignment(Qt.AlignCenter)

        self._avg_value = QLabel("—")
        self._avg_value.setObjectName("dash_card_value")
        self._avg_value.setAlignment(Qt.AlignCenter)

        avg_layout.addWidget(self._avg_label)
        avg_layout.addWidget(self._avg_value)
        layout.addWidget(self._avg_card)

        # ---------- Distribution ----------
        self._dist_title = QLabel("Distribution")
        self._dist_title.setObjectName("dash_section_title")
        layout.addWidget(self._dist_title)

        self._dist_container = QFrame()
        self._dist_container.setObjectName("dash_dist_container")
        dist_layout = QVBoxLayout(self._dist_container)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(8)

        self._dist_rows = []  # لیست ۴ ردیف
        for g in (0, 1, 2, 3):
            row = self._make_dist_row(g)
            self._dist_rows.append(row)
            dist_layout.addWidget(row["widget"])

        layout.addWidget(self._dist_container)

        # ---------- Recent results ----------
        self._recent_title = QLabel("Last 5 Results")
        self._recent_title.setObjectName("dash_section_title")
        layout.addWidget(self._recent_title)

        self._recent_container = QFrame()
        self._recent_container.setObjectName("dash_recent_container")
        self._recent_layout = QVBoxLayout(self._recent_container)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(5)

        # ✅ برچسب «هنوز نتیجه‌ای نیست» (Empty State)
        self._recent_empty = QLabel()
        self._recent_empty.setObjectName("dash_empty")
        self._recent_empty.setAlignment(Qt.AlignCenter)
        self._recent_empty.setWordWrap(True)
        self._recent_empty.setTextFormat(Qt.RichText)
        self._update_recent_empty()
        self._recent_layout.addWidget(self._recent_empty)

        layout.addWidget(self._recent_container, 1)

        # ---------- اتصالات ----------
        self._close_btn.clicked.connect(self._on_close_clicked)
        self._clear_btn.clicked.connect(self._on_clear_clicked)

    def _make_dist_row(self, grid: int) -> Dict:
        """ساخت یه ردیف توزیع برای یه گرید (نام + مقدار + نوار)."""
        widget = QWidget()
        w_layout = QVBoxLayout(widget)
        w_layout.setContentsMargins(0, 0, 0, 0)
        w_layout.setSpacing(3)

        # ✅ خط بالا: «Grid X» در چپ + «count (percent%)» در راست
        top = QHBoxLayout()
        top.setSpacing(4)

        name = QLabel(f"Grid {grid}")
        name.setObjectName("dash_dist_name")

        value = QLabel("0 (0.0%)")
        value.setObjectName("dash_dist_value")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top.addWidget(name)
        top.addStretch()
        top.addWidget(value)

        # ✅ خط پایین: نوار افقی
        bar_container = QFrame()
        bar_container.setObjectName("dash_bar_container")
        bar_container.setFixedHeight(8)

        bar_fill = QFrame(bar_container)
        bar_fill.setObjectName("dash_bar_fill")
        bar_fill.setGeometry(0, 0, 0, 8)

        # رنگ fill بر اساس grid
        try:
            c = QColor(self.GRID_COLORS.get(grid, "#00e5ff"))
            bar_fill.setStyleSheet(
                f"background: {c.name()}; border-radius: 4px;"
            )
        except Exception:
            pass

        w_layout.addLayout(top)
        w_layout.addWidget(bar_container)

        return {
            "widget": widget,
            "name": name,
            "value": value,
            "bar_container": bar_container,
            "bar_fill": bar_fill,
        }

    # ============================================================
    # Public API — update stats
    # ============================================================
    def update_stats(self, stats: Dict):
        """آمار جدید رو نمایش بده."""
        self._stats = stats or {}
        try:
            total = int(self._stats.get("total", 0))
            avg = float(self._stats.get("avg_confidence", 0.0))
            counts = self._stats.get("counts", {"0": 0, "1": 0, "2": 0, "3": 0})
            dist = self._stats.get("distribution", {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
            recent = self._stats.get("recent", [])

            # ✅ Total (count-up)
            try:
                prev_total = 0
                if hasattr(self, "_last_total"):
                    prev_total = self._last_total
                self._last_total = total
                if prev_total != total:
                    animate_count_up(
                        self._total_value,
                        from_value=float(prev_total),
                        to_value=float(total),
                        duration_ms=600,
                        decimals=0,
                    )
                else:
                    self._total_value.setText(str(total))
            except Exception:
                self._total_value.setText(str(total))

            # ✅ Average confidence + رنگ‌بندی
            if total > 0:
                try:
                    prev_avg = 0.0
                    if hasattr(self, "_last_avg"):
                        prev_avg = self._last_avg
                    self._last_avg = avg * 100
                    if abs(prev_avg - avg * 100) > 0.05:
                        animate_count_up(
                            self._avg_value,
                            from_value=prev_avg,
                            to_value=avg * 100,
                            duration_ms=700,
                            decimals=1,
                            suffix="%",
                        )
                    else:
                        self._avg_value.setText(f"{avg * 100:.1f}%")
                except Exception:
                    self._avg_value.setText(f"{avg * 100:.1f}%")
                if avg >= 0.85:
                    color = "#34d399"  # سبز — عالی
                elif avg >= 0.70:
                    color = "#fbbf24"  # زرد — متوسط
                else:
                    color = "#f87171"  # قرمز — ضعیف
                try:
                    self._avg_value.setStyleSheet(f"color: {color};")
                except Exception:
                    pass
            else:
                self._avg_value.setText("—")
                try:
                    self._avg_value.setStyleSheet("")
                except Exception:
                    pass

            # ✅ Distribution bars
            for g in (0, 1, 2, 3):
                row = self._dist_rows[g]
                count = int(counts.get(str(g), 0))
                percent = float(dist.get(g, 0.0))
                row["value"].setText(f"{count} ({percent:.1f}%)")
                self._set_bar_width(row, percent)

            # ✅ Recent results
            self._update_recent(recent)
        except Exception as e:
            print(f"⚠️ DashboardPanel.update_stats failed: {e}")

    def _update_recent_empty(self):
        """آپدیت متن Empty State برای recent list."""
        try:
            html = make_empty_state_html(
                key="dash_no_results",
                theme=self._theme,
                lang=self._lang,
            )
            if html:
                self._recent_empty.setText(html)
            else:
                self._recent_empty.setText(tr(self._lang, "dash_no_results"))
        except Exception as e:
            print(f"⚠️ _update_recent_empty failed: {e}")
            try:
                self._recent_empty.setText(tr(self._lang, "dash_no_results"))
            except Exception:
                pass

    def _set_bar_width(self, row: Dict, percent: float):
        """عرض نوار افقی رو بر اساس درصد تنظیم می‌کنه."""
        try:
            container_w = row["bar_container"].width()
            if container_w <= 0:
                # ✅ اگه هنوز layout نشده، از عرض پیش‌فرض استفاده کن
                container_w = 240
            fill_w = int(container_w * (percent / 100.0))
            row["bar_fill"].setGeometry(0, 0, max(0, fill_w), 8)
        except Exception:
            pass

    def _update_recent(self, recent: List[Dict]):
        """لیست نتایج اخیر رو بازسازی می‌کنه."""
        # ✅ پاک کردن ردیف‌های قبلی (به‌جز empty)
        while self._recent_layout.count() > 0:
            item = self._recent_layout.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._recent_empty:
                w.deleteLater()

        if not recent:
            self._recent_empty.setVisible(True)
            self._recent_layout.addWidget(self._recent_empty)
            return

        self._recent_empty.setVisible(False)

        # ✅ نمایش حداکثر ۵ مورد
        for entry in recent[:5]:
            try:
                grid = int(entry.get("grid", 0))
                conf = float(entry.get("confidence", 0.0))
                ts = entry.get("timestamp", "")

                # ✅ فرمت زمان
                time_str = ""
                try:
                    dt = datetime.fromisoformat(ts)
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    time_str = ts[:8] if ts else "—"

                # ✅ رنگ background بر اساس grid
                bg_color = self.GRID_COLORS.get(grid, "#00e5ff")

                row = QFrame()
                row.setObjectName("dash_recent_row")
                row.setFixedHeight(34)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(8, 4, 8, 4)
                row_layout.setSpacing(6)

                # ✅ نشانگر رنگی (dot)
                dot = QLabel()
                dot.setFixedSize(8, 8)
                try:
                    dot.setStyleSheet(
                        f"background: {bg_color}; border-radius: 4px;"
                    )
                except Exception:
                    pass

                # ✅ زمان
                time_lbl = QLabel(time_str)
                time_lbl.setObjectName("dash_recent_time")

                # ✅ گرید
                grid_lbl = QLabel(f"G{grid}")
                grid_lbl.setObjectName("dash_recent_grid")
                try:
                    grid_lbl.setStyleSheet(
                        f"color: {bg_color}; font-weight: bold;"
                    )
                except Exception:
                    pass

                # ✅ confidence
                conf_lbl = QLabel(f"{conf * 100:.1f}%")
                conf_lbl.setObjectName("dash_recent_conf")
                conf_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

                row_layout.addWidget(dot)
                row_layout.addWidget(time_lbl)
                row_layout.addStretch()
                row_layout.addWidget(grid_lbl)
                row_layout.addWidget(conf_lbl)

                # ✅ tooltip برای recent row
                try:
                    t = self._theme
                    grid_full = tr(self._lang, f"grid_{grid}")
                    row.setToolTip(
                        make_tooltip(
                            title=grid_full,
                            description=f"🕐 {time_str}  •  🎯 {conf * 100:.1f}%",
                            tip=tr(self._lang, "tooltip_tip_recent"),
                            theme=t,
                        )
                    )
                except Exception:
                    pass

                self._recent_layout.addWidget(row)
            except Exception:
                continue

        # ✅ spacer در انتها
        self._recent_layout.addStretch()

    # ============================================================
    # Public API — theme & language
    # ============================================================
    def set_theme(self, theme: Dict):
        """رنگ‌ها رو با تم جدید هماهنگ می‌کنه."""
        self._theme = theme
        self._apply_theme()
        # ✅ آپدیت رنگ barها
        try:
            for g in (0, 1, 2, 3):
                c = QColor(self.GRID_COLORS.get(g, "#00e5ff"))
                self._dist_rows[g]["bar_fill"].setStyleSheet(
                    f"background: {c.name()}; border-radius: 4px;"
                )
        except Exception:
            pass
        # ✅ آپدیت Empty State با تم جدید
        try:
            self._update_recent_empty()
        except Exception:
            pass

    def set_lang(self, lang: str):
        """متن‌ها رو با زبان جدید هماهنگ می‌کنه."""
        self._lang = lang
        try:
            self._title.setText("📊 " + tr(lang, "dashboard_title"))
            self._total_label.setText(tr(lang, "dash_total"))
            self._avg_label.setText(tr(lang, "dash_avg_conf"))
            self._dist_title.setText(tr(lang, "dash_distribution"))
            self._recent_title.setText(tr(lang, "dash_last_5"))
            # ✅ آپدیت Empty State با HTML
            self._update_recent_empty()
            # ✅ رفرش recent
            if self._stats is not None:
                self._update_recent(self._stats.get("recent", []))
        except Exception:
            pass

    def set_clear_callback(self, callback):
        """تعیین callback برای دکمه‌ی clear."""
        self._clear_callback = callback

    # ============================================================
    # ✅ Smart tooltips
    # ============================================================

    def set_tooltips(self, theme: Dict, lang: str):
        """
        به‌روزرسانی tooltipها با تم و زبان جدید.
        باید از MainWindow._update_all_tooltips صدا زده بشه.
        """
        try:
            t = theme or self._theme
            self._theme = t

            # ---------- دکمه‌ی clear (🗑️) ----------
            try:
                self._clear_btn.setToolTip(
                    make_tooltip(
                        title="🗑️ " + tr(lang, "tooltip_title_dash_clear"),
                        description=tr(lang, "tooltip_desc_dash_clear"),
                        tip=tr(lang, "tooltip_tip_dash_clear")
                            if "tooltip_tip_dash_clear" in TRANSLATIONS.get(lang, {}) else "",
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- دکمه‌ی close (✕) ----------
            try:
                self._close_btn.setToolTip(
                    make_tooltip(
                        title="✕ " + tr(lang, "tooltip_title_dash_close"),
                        description=tr(lang, "tooltip_desc_dash_close"),
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- کارت Total ----------
            try:
                self._total_card.setToolTip(
                    make_tooltip(
                        title="📈 " + tr(lang, "dash_total"),
                        description=tr(lang, "tooltip_desc_dash_total"),
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- کارت Avg ----------
            try:
                self._avg_card.setToolTip(
                    make_tooltip(
                        title="🎯 " + tr(lang, "dash_avg_conf"),
                        description=tr(lang, "tooltip_desc_dash_avg"),
                        theme=t,
                    )
                )
            except Exception:
                pass

            # ---------- Distribution bars ----------
            try:
                for g in (0, 1, 2, 3):
                    row = self._dist_rows[g]
                    # رنگ گرید
                    try:
                        c = QColor(self.GRID_COLORS.get(g, "#00e5ff"))
                    except Exception:
                        c = QColor("#00e5ff")
                    grid_label = tr(lang, f"grid_{g}")
                    row["widget"].setToolTip(
                        make_tooltip(
                            title=grid_label,
                            description=tr(lang, "tooltip_desc_dash_dist"),
                            theme=t,
                        )
                    )
            except Exception:
                pass

            # ---------- Recent rows (اگه ساختن) ----------
            # recent rows توی _update_recent ساخته می‌شن و tooltip خودشون رو دارن
            # (توی Part 4.2 اضافه می‌کنیم)

            # ✅ re-apply theme (چون رنگ tooltip توی QSS نیست، با make_tooltip اومده)
        except Exception as e:
            print(f"⚠️ DashboardPanel.set_tooltips failed: {e}")

    # ============================================================
    # Theme application
    # ============================================================
    def _apply_theme(self):
        t = self._theme
        self.setStyleSheet(f"""
            QFrame#dashboard_panel {{
                background: {t['glass_bg']};
                border-left: 2px solid {t['glass_border']};
            }}
            QLabel#dash_title {{
                color: {t['primary']};
                font-size: 16px;
                font-weight: bold;
                background: transparent;
            }}
            QPushButton#dash_icon_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: 1px solid {t['glass_border']};
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#dash_icon_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QFrame#dash_card {{
                background: {t['input_bg']};
                border: 1px solid {t['glass_border']};
                border-radius: 10px;
            }}
            QLabel#dash_card_label {{
                color: {t['text_dim']};
                font-size: 11px;
                background: transparent;
            }}
            QLabel#dash_card_value {{
                color: {t['primary']};
                font-size: 28px;
                font-weight: bold;
                background: transparent;
            }}
            QLabel#dash_section_title {{
                color: {t['text']};
                font-size: 12px;
                font-weight: bold;
                background: transparent;
                margin-top: 6px;
            }}
            QLabel#dash_dist_name {{
                color: {t['text']};
                font-size: 11px;
                background: transparent;
            }}
            QLabel#dash_dist_value {{
                color: {t['text_dim']};
                font-size: 11px;
                background: transparent;
            }}
            QFrame#dash_bar_container {{
                background: {t['glass_highlight']};
                border-radius: 4px;
            }}
            QFrame#dash_recent_container {{
                background: transparent;
            }}
            QFrame#dash_recent_row {{
                background: {t['input_bg']};
                border: 1px solid {t['glass_border']};
                border-radius: 8px;
            }}
            QLabel#dash_recent_time {{
                color: {t['text_dim']};
                font-size: 10px;
                background: transparent;
            }}
            QLabel#dash_recent_grid {{
                font-size: 11px;
                font-weight: bold;
                background: transparent;
            }}
            QLabel#dash_recent_conf {{
                color: {t['text']};
                font-size: 11px;
                font-weight: bold;
                background: transparent;
            }}
            QLabel#dash_empty {{
                color: {t['text_dim']};
                font-size: 11px;
                font-style: italic;
                background: transparent;
                padding: 15px;
            }}
        """)

    # ============================================================
    # Events
    # ============================================================
    def resizeEvent(self, event):
        """وقتی اندازه عوض شد، عرض barها رو دوباره حساب کن."""
        super().resizeEvent(event)
        if self._stats:
            try:
                dist = self._stats.get("distribution", {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
                for g in (0, 1, 2, 3):
                    row = self._dist_rows[g]
                    percent = float(dist.get(g, 0.0))
                    self._set_bar_width(row, percent)
            except Exception:
                pass

    def _on_close_clicked(self):
        """دکمه‌ی ✕ → سیگنال closed."""
        try:
            self.closed.emit()
        except Exception:
            pass

    def _on_clear_clicked(self):
        """دکمه‌ی 🗑️ → callback."""
        try:
            if self._clear_callback:
                self._clear_callback()
        except Exception:
            pass

# ================================================================
# ✅ RIPPLE OVERLAY (v3.2 — بدون pyqtProperty، تمیز و مطمئن)
# ================================================================

class RippleOverlay(QWidget):
    """
    Material-style ripple overlay — بدون pyqtProperty.
    از QTimer برای انیمیشن دستی استفاده می‌کند تا از هرگونه مشکل جلوگیری شود.
    """

    def __init__(self, parent: QWidget, radius: int = 10):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setVisible(False)

        self._corner_radius = radius
        self._origin = QPoint(0, 0)
        self._current_radius = 0.0
        self._max_radius = 0.0
        self._opacity = 0.0
        self._color = QColor(255, 255, 255)

        # ✅ تایمر برای انیمیشن
        self._duration_ms = 450
        self._elapsed_ms = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._tick_interval = 16  # ~60 FPS

    def set_color(self, color: QColor):
        self._color = QColor(color)

    def set_radius(self, radius: int):
        self._corner_radius = radius

    def trigger(self, origin: QPoint):
        """شروع ripple از نقطه‌ی داده‌شده (نسبت به دکمه)."""
        try:
            parent = self.parent()
            if parent is None:
                return
            self.setGeometry(parent.rect())
            self.raise_()

            self._origin = QPoint(origin)
            self._current_radius = 0.0
            self._opacity = 0.55
            self._elapsed_ms = 0

            # ✅ محاسبه‌ی حداکثر radius
            w = parent.width()
            h = parent.height()
            max_dist = 0.0
            for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]:
                d = math.hypot(cx - origin.x(), cy - origin.y())
                if d > max_dist:
                    max_dist = d
            self._max_radius = max_dist * 1.1

            self.setVisible(True)
            self._timer.start(self._tick_interval)
        except Exception:
            pass

    def _tick(self):
        """یه قدم انیمیشن جلو برو."""
        try:
            self._elapsed_ms += self._tick_interval
            t = min(1.0, self._elapsed_ms / self._duration_ms)
            # ✅ easing (OutCubic)
            eased = 1.0 - (1.0 - t) ** 3
            self._current_radius = self._max_radius * eased
            self._opacity = 0.55 * (1.0 - eased)

            if t >= 1.0:
                self._timer.stop()
                self.setVisible(False)
                self._current_radius = 0.0
                self._opacity = 0.0

            self.update()
        except Exception:
            self._timer.stop()
            self.setVisible(False)

    def paintEvent(self, event):
        if not self.isVisible() or self._opacity <= 0.01 or self._current_radius <= 0.5:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # ✅ clip به شکل دکمه
        clip_path = QPainterPath()
        clip_path.addRoundedRect(
            0, 0, self.width(), self.height(),
            self._corner_radius, self._corner_radius,
        )
        painter.setClipPath(clip_path)

        c = QColor(self._color)
        c.setAlphaF(max(0.0, min(1.0, self._opacity)))
        painter.setBrush(QBrush(c))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self._origin, int(self._current_radius), int(self._current_radius))
        painter.end()

# ================================================================
# ✅ ANIMATED ICON BUTTON (v3.2)
# ================================================================

class AnimatedIconButton(QPushButton):
    """
    QPushButton با ripple Material-style.
    بدون pyqtProperty — همه چیز با QTimer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ripple_radius = 15
        self._ripple = RippleOverlay(self, radius=self._ripple_radius)
        self._ripple.setGeometry(self.rect())
        # ✅ اتصال به pressed (بعد از کلیک، ripple شروع شه)
        self.pressed.connect(self._trigger_ripple)

    def set_ripple_radius(self, radius: int):
        self._ripple_radius = radius
        if hasattr(self, "_ripple"):
            self._ripple.set_radius(radius)

    def set_theme(self, theme: Dict):
        """رنگ ripple رو با تم هماهنگ کن."""
        try:
            c = QColor(theme["primary"])
            c.setAlpha(90)
            self._ripple.set_color(c)
        except Exception:
            pass

    def _trigger_ripple(self):
        try:
            pos = self.mapFromGlobal(QCursor.pos())
            if not self.rect().contains(pos):
                pos = self.rect().center()
            self._ripple.trigger(pos)
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_ripple"):
            self._ripple.setGeometry(self.rect())

# ================================================================
# ✅ HOVER / CLICK ANIMATION HELPERS
# ================================================================

def add_hover_animation(widget, shadow_color: str = "rgba(0, 200, 255, 0.5)"):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(0)
    shadow.setColor(QColor(0, 0, 0, 0))
    shadow.setOffset(0, 0)
    widget.setGraphicsEffect(shadow)
    # ✅ ذخیره‌ی رنگ سایه برای تغییرات بعدی (بدون تغییر رفتار)
    widget._shadow_color = shadow_color
    widget._shadow_effect = shadow

    anim = QPropertyAnimation(shadow, b"blurRadius", widget)
    anim.setDuration(350)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    widget._shadow_anim = anim

    def on_enter(event):
        try:
            c = QColor(shadow_color)
            shadow.setColor(QColor(c.red(), c.green(), c.blue(), 200))
        except Exception:
            shadow.setColor(QColor(0, 200, 255, 200))
        anim.stop()
        anim.setStartValue(shadow.blurRadius())
        anim.setEndValue(28)
        anim.start()
        QPushButton.enterEvent(widget, event)

    def on_leave(event):
        anim.stop()
        anim.setStartValue(shadow.blurRadius())
        anim.setEndValue(0)
        anim.start()
        QPushButton.leaveEvent(widget, event)

    widget.enterEvent = on_enter
    widget.leaveEvent = on_leave

def refresh_button_shadow(widget, theme: Dict):
    """
    رنگ سایه‌ی hover رو با تم جدید هماهنگ می‌کنه.
    بدون تغییر رفتار hover.
    """
    try:
        if not hasattr(widget, "_shadow_color"):
            return
        primary = theme.get("primary", "#00e5ff")
        c = QColor(primary)
        widget._shadow_color = f"rgba({c.red()}, {c.green()}, {c.blue()}, 0.5)"
        # ✅ اگه الان hover فعاله، فوراً اعمال کن
        if hasattr(widget, "_shadow_effect"):
            eff = widget._shadow_effect
            if eff.blurRadius() > 5:
                eff.setColor(QColor(c.red(), c.green(), c.blue(), 200))
    except Exception:
        pass

def add_click_animation(widget):
    def on_press(event):
        g = widget.geometry()
        widget._orig_geom = QRect(g)
        anim = QPropertyAnimation(widget, b"geometry", widget)
        anim.setDuration(120)
        anim.setStartValue(g)
        anim.setEndValue(QRect(g.x() + 3, g.y() + 3, g.width() - 6, g.height() - 6))
        anim.setEasingCurve(QEasingCurve.OutQuad)
        anim.start()
        widget._press_anim = anim
        QPushButton.mousePressEvent(widget, event)

    def on_release(event):
        if hasattr(widget, "_orig_geom"):
            anim = QPropertyAnimation(widget, b"geometry", widget)
            anim.setDuration(180)
            anim.setStartValue(widget.geometry())
            anim.setEndValue(widget._orig_geom)
            anim.setEasingCurve(QEasingCurve.OutBack)
            anim.start()
            widget._release_anim = anim
        QPushButton.mouseReleaseEvent(widget, event)

    widget.mousePressEvent = on_press
    widget.mouseReleaseEvent = on_release


def decorate_button(widget, theme: Dict):
    add_hover_animation(widget, theme.get("shadow", "rgba(0, 200, 255, 0.5)"))
    add_click_animation(widget)
    # ✅ Micro-interaction: press scale
    try:
        button_press_scale(widget, pressed_scale=0.96, duration_ms=90)
    except Exception:
        pass
    # ✅ اگه AnimatedIconButton هست، theme رو ست کن
    try:
        if isinstance(widget, AnimatedIconButton):
            widget.set_theme(theme)
    except Exception:
        pass


# ================================================================
# ✅ SPLASH SCREEN
# ================================================================

# ================================================================
# ✅ SPLASH SCREEN (v3.0 — interactive, real progress, skip)
# ================================================================

class SplashScreen(QWidget):
    finished = pyqtSignal()
    skip_requested = pyqtSignal()
    continue_without_ai = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        screen = QApplication.primaryScreen().geometry()
        w, h = 800, 500
        self.setGeometry(
            (screen.width() - w) // 2,
            (screen.height() - h) // 2,
            w, h,
        )

        self._current_progress = 0
        self._model_ok = None
        self._current_stage = ""
        self._finished_emitted = False
        self._start_time = time.time()   # ← اضافه کن

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setFixedSize(w, h)
        self.container.setStyleSheet("""
            QFrame {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e,
                    stop:0.5 #16213e,
                    stop:1 #0f3460
                );
                border-radius: 30px;
            }
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setAlignment(Qt.AlignCenter)
        container_layout.setContentsMargins(40, 40, 40, 40)

        # ✅ دکمه‌ی Skip (بالا-راست)
        self.skip_btn = QPushButton("Skip →")
        self.skip_btn.setParent(self.container)
        self.skip_btn.setFixedSize(80, 30)
        self.skip_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94a3b8;
                border: 1px solid #475569;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #00d4ff;
                color: #0a0a14;
                border: 1px solid #00d4ff;
            }
        """)
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.clicked.connect(self._on_skip_clicked)
        self.skip_btn.move(w - 100, 15)

        # ✅ لوگو
        self.logo_label = QLabel("🧬")
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("""
            font-size: 140px;
            background: transparent;
        """)
        container_layout.addWidget(self.logo_label)

        # ✅ عنوان
        self.title_label = QLabel("Liver AI-babak")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 42px;
            font-weight: bold;
            color: #00d4ff;
            background: transparent;
            font-family: 'Segoe UI', Arial;
        """)
        container_layout.addWidget(self.title_label)

        # ✅ زیرنویس
        self.subtitle_label = QLabel("Knowledge Distillation Ensemble")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            font-size: 18px;
            color: #a855f7;
            background: transparent;
            font-style: italic;
            font-family: 'Segoe UI', Arial;
        """)
        container_layout.addWidget(self.subtitle_label)

        # ✅ progress container
        self.progress_container = QFrame()
        self.progress_container.setFixedSize(500, 8)
        self.progress_container.setStyleSheet("""
            background: rgba(0, 212, 255, 0.15);
            border-radius: 4px;
        """)
        container_layout.addSpacing(30)
        container_layout.addWidget(self.progress_container, 0, Qt.AlignCenter)

        # ✅ progress bar
        self.progress_bar = QFrame(self.progress_container)
        self.progress_bar.setGeometry(0, 0, 0, 8)
        self.progress_bar.setStyleSheet("""
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #00d4ff,
                stop:1 #a855f7
            );
            border-radius: 4px;
        """)

        # ✅ stage label (متن مرحله)
        self.stage_label = QLabel("Loading...")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setStyleSheet("""
            font-size: 14px;
            color: #94a3b8;
            background: transparent;
            margin-top: 10px;
            font-family: 'Segoe UI', Arial;
        """)
        container_layout.addSpacing(15)
        container_layout.addWidget(self.stage_label)

        # ✅ دکمه‌ی Continue Without AI (اول مخفی)
        self.continue_btn = QPushButton("Continue without AI")
        self.continue_btn.setFixedSize(220, 40)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background: #fbbf24;
                color: #0a0a14;
                border: none;
                border-radius: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #f59e0b;
            }
        """)
        self.continue_btn.setCursor(Qt.PointingHandCursor)
        self.continue_btn.setVisible(False)
        self.continue_btn.clicked.connect(self._on_continue_clicked)
        container_layout.addSpacing(15)
        container_layout.addWidget(self.continue_btn, 0, Qt.AlignCenter)

        # ✅ developer label
        self.developer_label = QLabel("✨ developed by : babak yousefian ✨")
        self.developer_label.setAlignment(Qt.AlignCenter)
        self.developer_label.setStyleSheet("""
            font-size: 12px;
            color: #64748b;
            background: transparent;
            font-style: italic;
            font-family: 'Segoe UI', Arial;
            margin-top: 25px;
        """)
        container_layout.addWidget(self.developer_label)

        layout.addWidget(self.container)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(60)
        shadow.setColor(QColor(0, 212, 255, 100))
        shadow.setOffset(0, 0)
        self.container.setGraphicsEffect(shadow)

        self._start_animations()

    def _start_animations(self):
        # ✅ لوگو pulse
        self.logo_opacity = QGraphicsOpacityEffect(self.logo_label)
        self.logo_label.setGraphicsEffect(self.logo_opacity)
        self.logo_pulse = QPropertyAnimation(self.logo_opacity, b"opacity")
        self.logo_pulse.setDuration(1500)
        self.logo_pulse.setStartValue(0.4)
        self.logo_pulse.setEndValue(1.0)
        self.logo_pulse.setEasingCurve(QEasingCurve.InOutSine)
        self.logo_pulse.setLoopCount(-1)
        self.logo_pulse.start()

    # ============================================================
    # Public API
    # ============================================================

    def set_progress(self, percent: int, stage: str = ""):
        """آپدیت progress bar با درصد و متن."""
        try:
            percent = max(0, min(100, int(percent)))
            self._current_progress = percent
            self._current_stage = stage or self._current_stage

            # ✅ عرض bar
            max_width = 500
            width = int(max_width * (percent / 100.0))
            self.progress_bar.setGeometry(0, 0, width, 8)

            # ✅ اگه درصد ۱۰۰ شد، رنگ سبز
            if percent >= 100:
                self.progress_bar.setStyleSheet("""
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #34d399,
                        stop:1 #10b981
                    );
                    border-radius: 4px;
                """)

            # ✅ متن stage
            if stage:
                self.stage_label.setText(stage)

            # ✅ اگه مدل لود شده
            if "ready ✓" in (stage or ""):
                self.stage_label.setStyleSheet("""
                    font-size: 14px;
                    color: #34d399;
                    background: transparent;
                    margin-top: 10px;
                    font-weight: bold;
                    font-family: 'Segoe UI', Arial;
                """)

            # ✅ اگه خطا
            if "failed ✗" in (stage or ""):
                self.stage_label.setStyleSheet("""
                    font-size: 14px;
                    color: #f87171;
                    background: transparent;
                    margin-top: 10px;
                    font-weight: bold;
                    font-family: 'Segoe UI', Arial;
                """)
        except Exception as e:
            print(f"⚠️ set_progress failed: {e}")

    def on_preload_finished(self, success: bool, message: str):
        """وقتی preload تموم شد."""
        try:
            self._model_ok = success

            # ✅ محاسبه‌ی زمان گذشته
            elapsed = time.time() - self._start_time
            MIN_DISPLAY = 6.0  # حداقل ۶ ثانیه
            remaining = max(0.0, MIN_DISPLAY - elapsed)

            if success:
                self.stage_label.setText("Ready ✓")
                self.stage_label.setStyleSheet("""
                    font-size: 14px;
                    color: #34d399;
                    background: transparent;
                    margin-top: 10px;
                    font-weight: bold;
                    font-family: 'Segoe UI', Arial;
                """)
                # ✅ اگه کمتر از ۶ ثانیه گذشته، صبر کن
                QTimer.singleShot(int(remaining * 1000) + 200, self._finish)
            else:
                # ✅ خطا → دکمه‌ی Continue Without AI
                self.stage_label.setText(message or "Model failed ✗")
                self.stage_label.setStyleSheet("""
                    font-size: 13px;
                    color: #f87171;
                    background: transparent;
                    margin-top: 10px;
                    font-weight: bold;
                    font-family: 'Segoe UI', Arial;
                """)
                self.continue_btn.setVisible(True)
                self.skip_btn.setVisible(False)
                # ✅ برای خطا، نیازی به صبر نیست (کاربر خودش Continue می‌زنه)
        except Exception as e:
            print(f"⚠️ on_preload_finished failed: {e}")

    # ============================================================
    # Events
    # ============================================================

    def _on_skip_clicked(self):
        """کاربر Skip کرد → فوری بسته شو (بدون صبر ۶ ثانیه)."""
        try:
            self.stage_label.setText("Skipped")
            self.skip_requested.emit()
            # ✅ Skip فوری — صبر نمی‌کنیم
            self._finish()
        except Exception:
            self._finish()

    def _on_continue_clicked(self):
        """کاربر Continue Without AI کرد."""
        try:
            self.continue_without_ai.emit()
            self._finish()
        except Exception:
            self._finish()

    def _finish(self):
        """بستن splash و emit finished."""
        if self._finished_emitted:
            return
        self._finished_emitted = True
        try:
            self.logo_pulse.stop()
        except Exception:
            pass
        self.close()
        self.finished.emit()

    def closeEvent(self, event):
        """اگه کاربر splash رو بست، emit finished."""
        if not self._finished_emitted:
            self._finished_emitted = True
            self.finished.emit()
        super().closeEvent(event)


# ================================================================
# ✅ PRELOAD THREAD (v3.0 — load resources async)
# ================================================================

class PreloadThread(QThread):
    """
    لود منابع به‌صورت async:
    - صداها (SoundManager)
    - مدل (ModelLoader)

    سیگنال‌ها:
    - progress(int, str): درصد (0-100) + نام مرحله
    - finished(bool, str): موفق/ناموفق + پیام
    """
    progress = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._skip = False

    def request_skip(self):
        """درخواست Skip (متوقف کردن لود مدل)."""
        self._skip = True

    def run(self):
        try:
            # ============================================================
            # مرحله ۱: صداها (10%)
            # ============================================================
            self.progress.emit(10, "Loading sounds...")
            self.msleep(300)
            try:
                SoundManager.get()
            except Exception as e:
                print(f"⚠️ Sounds preload failed: {e}")
            self.progress.emit(15, "Sounds ready")
            self.msleep(400)

            # ============================================================
            # ✅ مرحله ۱.۵: Image Guard (خیلی سریع — از pkl)
            # ============================================================
            self.progress.emit(18, "Loading image guard...")
            try:
                from core.security.image_guard import _get_liver_model
                guard_model = _get_liver_model()
                if guard_model is not None:
                    print(f"✅ Image Guard ready ({guard_model.n_samples} samples)")
                else:
                    print("⚠️ Image Guard in fail-open mode (no pkl)")
            except Exception as e:
                print(f"⚠️ Image Guard preload failed: {e}")
            self.progress.emit(22, "Image guard ready")

            # ============================================================
            # مرحله ۲: مدل (60%) — از این‌جا به بعد دست نزن
            # ============================================================
            if self._skip:
                self.progress.emit(60, "Skipped model load")
                self.finished_signal.emit(True, "Skipped")
                return

            self.progress.emit(25, "Loading AI model...")  # ← از ۲۰ به ۲۵ (چون guard ۲۲٪ گرفت)
            self.msleep(300)
            # ... بقیه دقیقاً همان‌طور که بود

            if not HAS_TORCH:
                # ✅ PyTorch نیست → Skip کن
                self.progress.emit(60, "PyTorch not available")
                self.finished_signal.emit(False, "PyTorch not available")
                return

            model_ok = False
            model_msg = ""
            try:
                # ✅ ModelLoader رو صدا بزن (توی thread جداست، پس UI block نمی‌شه)
                ModelLoader.get_model()
                model_ok = True
                model_msg = "Model ready"
            except Exception as e:
                model_ok = False
                model_msg = f"Model failed: {str(e)[:50]}"

            if self._skip:
                self.progress.emit(60, "Skipped")
                self.finished_signal.emit(True, "Skipped")
                return

            # ✅ آپدیت progress به ۷۰٪
            self.progress.emit(70, "Model " + ("ready ✓" if model_ok else "failed ✗"))

            # ============================================================
            # مرحله ۳: آماده‌سازی UI (90%)
            # ============================================================
            self.progress.emit(90, "Preparing UI...")

            # ============================================================
            # مرحله ۴: تمام (100%)
            # ============================================================
            self.progress.emit(100, "Ready")

            self.finished_signal.emit(model_ok, model_msg)
        except Exception as e:
            print(f"⚠️ PreloadThread failed: {e}")
            self.finished_signal.emit(False, str(e))

# ================================================================
# ✅ LOGIN DIALOG
# ================================================================

class LoginDialog(QDialog):
    login_success = pyqtSignal(str)

    def __init__(self, db: UserDB, theme: Dict, lang: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.theme = theme
        self.lang = lang

        self.setWindowTitle(tr(lang, "login"))
        self.setFixedSize(450, 480)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

        # ✅ Add backdrop (exactly like UpdateAvailableDialog)
        try:
            from gui.dialogs.update_dialogs import _with_backdrop
            _with_backdrop(self, parent, theme)
        except Exception as e:
            print(f"⚠️ LoginDialog backdrop failed: {e}")

        # ✅ انیمیشن باز شدن
        QTimer.singleShot(0, lambda: animate_dialog_open(self))

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)

        title = QLabel(tr(self.lang, "login"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(20)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText(tr(self.lang, "username"))
        self.username_input.setObjectName("input")
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText(tr(self.lang, "password"))
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("input")
        layout.addWidget(self.password_input)

        self.remember_me = QCheckBox(tr(self.lang, "remember_me"))
        self.remember_me.setObjectName("checkbox")
        layout.addWidget(self.remember_me)

        layout.addSpacing(20)

        self.login_btn = QPushButton(tr(self.lang, "login_btn"))
        self.login_btn.setObjectName("primary_btn")
        self.login_btn.clicked.connect(self._do_login)
        attach_sound(self.login_btn, "loginSound")
        layout.addWidget(self.login_btn)

        signup_btn = QPushButton(f"→ {tr(self.lang, 'signup')}")
        signup_btn.setObjectName("link_btn")
        signup_btn.clicked.connect(self._open_signup)
        attach_sound(signup_btn, "linkClick")
        layout.addWidget(signup_btn)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        attach_sound(close_btn, "simpleClick")
        close_btn.setParent(self.container)
        close_btn.move(410, 10)

    def _apply_theme(self):
        t = self.theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#title {{
                font-size: 32px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLineEdit#input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                color: {t['text']};
            }}
            QLineEdit#input:focus {{
                border: 2px solid {t['secondary']};
            }}
            QCheckBox#checkbox {{
                color: {t['text_dim']};
                font-size: 13px;
                background: transparent;
            }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:0.5 {t['accent']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 12px;
                padding: 14px;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton#primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#link_btn {{
                background: transparent;
                color: {t['primary']};
                border: none;
                padding: 8px;
                font-size: 13px;
                text-decoration: underline;
            }}
            QPushButton#link_btn:hover {{
                color: {t['secondary']};
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)

    def _do_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            show_message(self, "warning", "Error", tr(self.lang, "empty_fields"))
            return

        main_window = self.parent()
        if main_window is not None and hasattr(main_window, "login_attempts"):
            now = time.time()
            attempts = main_window.login_attempts.get(username, [])
            attempts = [t for t in attempts if now - t < 300]
            if len(attempts) >= 5:
                show_message(self, "warning", tr(self.lang, "warning"),
                             tr(self.lang, "too_many_attempts"))
                self.password_input.clear()
                return
            main_window.login_attempts[username] = attempts

        if not self.db.verify(username, password):
            if main_window is not None and hasattr(main_window, "login_attempts"):
                now = time.time()
                main_window.login_attempts.setdefault(username, []).append(now)
            show_message(self, "warning", "Error", tr(self.lang, "invalid_credentials"))
            self.password_input.clear()
            # ✅ Micro-interaction: shake on error
            try:
                highlight_widget_error(self.password_input, self.theme)
                shake_widget(self)
            except Exception:
                pass
            # (forgot-password disabled — 2FA is the recovery method)
            return

        if main_window is not None and hasattr(main_window, "login_attempts"):
            main_window.login_attempts[username] = []
        self.password_input.clear()

        status = self.db.get_2fa_status(username)
        needs_2fa = status.get("enabled", False)

        if needs_2fa:
            self.hide()

            verified = {"ok": False}

            try:
                verify_dlg = TwoFactorVerifyDialog(
                    db=self.db,
                    username=username,
                    theme=self.theme,
                    lang=self.lang,
                    parent=main_window,
                )
                verify_dlg.verified.connect(lambda u: verified.update({"ok": True}))
                verify_dlg.exec_()
            except Exception as e:
                print(f"⚠️ TwoFactorVerifyDialog failed: {e}")

            if not verified["ok"]:
                self.show()
                return

        self.login_success.emit(username)
        animate_dialog_close(self, on_finished=self.accept)

    def _open_signup(self):
        try:
            dialog = SignupDialog(self.db, self.theme, self.lang, self.parent())
            # ✅ اتصال سیگنال signup_success به login_success
            # (که MainWindow هم notify کنه)
            dialog.signup_success.connect(self.login_success.emit)
            if dialog.exec_() == QDialog.Accepted:
                pass
        except Exception as e:
            print(f"⚠️ _open_signup failed: {e}")
            import traceback
            traceback.print_exc()

# ================================================================
# ✅ TOAST (v3.0 — in-app notification)
# ================================================================

class Toast(QFrame):
    """
    یه toast در-app (بالای صفحه).
    - levels: success / info / warning / error
    - auto-hide بعد از duration
    - fade-in/fade-out
    """

    dismissed = pyqtSignal(object)  # سیگنال با خود toast

    LEVEL_COLORS = {
        "success": ("#34d399", "#059669"),
        "info": ("#00e5ff", "#0284c7"),
        "warning": ("#fbbf24", "#d97706"),
        "error": ("#f87171", "#dc2626"),
    }

    LEVEL_ICONS = {
        "success": "✓",
        "info": "ℹ",
        "warning": "⚠",
        "error": "✕",
    }

    def __init__(self, parent, level: str, title: str, message: str = "",
                 duration_ms: int = 3000, theme: Dict = None):
        super().__init__(parent)
        self._theme = theme or THEMES.get("theme_dark_aqua", {})
        self._level = level if level in self.LEVEL_COLORS else "info"
        self._duration = duration_ms
        self._closed = False

        self.setObjectName("toast")
        self.setFixedWidth(400)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # ---------- Layout ----------
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        # آیکون
        self._icon = QLabel(self.LEVEL_ICONS.get(self._level, "ℹ"))
        self._icon.setFixedSize(28, 28)
        self._icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._icon)

        # متن
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self._title_label = QLabel(title or "")
        self._title_label.setWordWrap(True)

        self._msg_label = QLabel(message or "")
        self._msg_label.setWordWrap(True)
        self._msg_label.setVisible(bool(message))

        text_layout.addWidget(self._title_label)
        text_layout.addWidget(self._msg_label)
        layout.addLayout(text_layout, 1)

        # دکمه‌ی بستن
        self._close_btn = QPushButton("✕")
        self._close_btn.setFixedSize(22, 22)
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.clicked.connect(self.dismiss)
        layout.addWidget(self._close_btn)

        self.setFixedHeight(60 if not message else 76)

        self._apply_style()

        # افکت opacity
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        # تایمر auto-hide
        if self._duration > 0:
            QTimer.singleShot(self._duration, self.dismiss)

    def _apply_style(self):
        t = self._theme
        color, color_dark = self.LEVEL_COLORS.get(self._level, ("#00e5ff", "#0284c7"))
        bg = t.get("input_bg", "rgba(30, 41, 59, 0.95)")
        text = t.get("text", "#f1f5f9")
        text_dim = t.get("text_dim", "#94a3b8")

        self.setStyleSheet(f"""
            QFrame#toast {{
                background: {bg};
                border: 2px solid {color};
                border-left: 4px solid {color};
                border-radius: 10px;
            }}
            QLabel {{
                color: {text};
                background: transparent;
                border: none;
            }}
        """)
        self._icon.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background: transparent;
                font-size: 18px;
                font-weight: bold;
                border: none;
            }}
        """)
        self._title_label.setStyleSheet(f"""
            QLabel {{
                color: {text};
                font-size: 13px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        self._msg_label.setStyleSheet(f"""
            QLabel {{
                color: {text_dim};
                font-size: 11px;
                background: transparent;
                border: none;
            }}
        """)
        self._close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {text_dim};
                border: none;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: {color};
            }}
        """)

    def set_theme(self, theme: Dict):
        self._theme = theme
        self._apply_style()

    def fade_in(self, start_pos: QPoint, end_pos: QPoint):
        """fade in + slide."""
        if self._closed:
            return
        try:
            self.setVisible(True)
            self.raise_()
            # انیمیشن opacity
            self._anim_in = QPropertyAnimation(self._opacity_effect, b"opacity", self)
            self._anim_in.setDuration(280)
            self._anim_in.setStartValue(0.0)
            self._anim_in.setEndValue(1.0)
            self._anim_in.setEasingCurve(QEasingCurve.OutCubic)
            self._anim_in.start()
            # انیمیشن position
            self._pos_anim = QPropertyAnimation(self, b"pos", self)
            self._pos_anim.setDuration(280)
            self._pos_anim.setStartValue(start_pos)
            self._pos_anim.setEndValue(end_pos)
            self._pos_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._pos_anim.start()
        except Exception:
            pass

    def fade_out(self):
        """fade out + slide up."""
        if self._closed:
            return
        try:
            self._anim_out = QPropertyAnimation(self._opacity_effect, b"opacity", self)
            self._anim_out.setDuration(220)
            self._anim_out.setStartValue(self._opacity_effect.opacity())
            self._anim_out.setEndValue(0.0)
            self._anim_out.setEasingCurve(QEasingCurve.InCubic)
            self._anim_out.finished.connect(self._on_fade_out_done)
            self._anim_out.start()
            # slide up
            self._pos_out = QPropertyAnimation(self, b"pos", self)
            self._pos_out.setDuration(220)
            self._pos_out.setStartValue(self.pos())
            self._pos_out.setEndValue(QPoint(self.pos().x(), self.pos().y() - 30))
            self._pos_out.start()
        except Exception:
            self._on_fade_out_done()

    def _on_fade_out_done(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.setVisible(False)
            self.dismissed.emit(self)
        except Exception:
            pass

    def dismiss(self):
        if self._closed:
            return
        self.fade_out()

    def is_closed(self) -> bool:
        return self._closed

# ================================================================
# ✅ NOTIFICATION CENTER (v3.0 — manager for toasts)
# ================================================================

def _motion_ok_safe() -> bool:
    """Safe wrapper for motion check."""
    try:
        return not MotionSettings.get().is_reduced()
    except Exception:
        return True

class NotificationCenter(QObject):
    """
    مدیر toastها.
    - هر toast بالای صفحه ظاهر می‌شه
    - اگه چند تا، روی هم می‌چینن (stack)
    - fade-in / fade-out
    - theme-aware
    """

    MAX_TOASTS = 4  # حداکثر toast همزمان
    MARGIN_TOP = 80  # فاصله از بالا (زیر topbar)
    MARGIN_RIGHT = 20
    SPACING = 8

    def __init__(self, parent_widget: QWidget, theme: Dict = None):
        super().__init__(parent_widget)
        self._parent = parent_widget
        self._theme = theme or THEMES.get("theme_dark_aqua", {})
        self._toasts = []  # list[Toast]

    def set_theme(self, theme: Dict):
        self._theme = theme
        for toast in self._toasts:
            try:
                toast.set_theme(theme)
            except Exception:
                pass

    def push(self, level: str, title: str, message: str = "",
             duration_ms: int = 3000):
        """نمایش یه toast جدید."""
        try:
            # اگه از حد گذشته، قدیمی‌ترین رو ببند
            while len(self._toasts) >= self.MAX_TOASTS:
                oldest = self._toasts[0]
                try:
                    oldest._closed = True
                    oldest.setVisible(False)
                    oldest.deleteLater()
                except Exception:
                    pass
                self._toasts.pop(0)
                self._reposition()

            # ساخت toast جدید
            toast = Toast(
                self._parent,
                level=level,
                title=title,
                message=message,
                duration_ms=duration_ms,
                theme=self._theme,
            )
            toast.dismissed.connect(self._on_toast_dismissed)

            # موقعیت اولیه: بالای صفحه (خارج از دید)
            parent_w = self._parent.width()
            x = max(10, parent_w - toast.width() - self.MARGIN_RIGHT)
            start_y = self.MARGIN_TOP - toast.height() - 10
            toast.move(x, start_y)
            toast.setVisible(False)

            self._toasts.append(toast)
            self._reposition()

            # fade-in
            # ✅ Micro-interaction: bounce-in (toast)
            end_pos = QPoint(x, self._get_y_for_index(len(self._toasts) - 1))
            bounce_in(toast, QPoint(x, start_y), end_pos, duration_ms=350)
            # Also fade opacity
            try:
                toast.setVisible(True)
                toast.raise_()
                if hasattr(toast, "_opacity_effect"):
                    toast._opacity_effect.setOpacity(0.0)
                    fade_anim = QPropertyAnimation(toast._opacity_effect, b"opacity", toast)
                    fade_anim.setDuration(280)
                    fade_anim.setStartValue(0.0)
                    fade_anim.setEndValue(1.0)
                    fade_anim.setEasingCurve(QEasingCurve.OutCubic)
                    fade_anim.start()
                    toast._fade_in_anim = fade_anim
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ NotificationCenter.push failed: {e}")

    def _get_y_for_index(self, index: int) -> int:
        """محاسبه‌ی y برای toast با ایندکس داده‌شده."""
        y = self.MARGIN_TOP
        for i in range(index):
            if i < len(self._toasts):
                y += self._toasts[i].height() + self.SPACING
        return y

    def _reposition(self):
        """Smoothly reposition all toasts."""
        try:
            parent_w = self._parent.width()
            y = self.MARGIN_TOP
            for toast in self._toasts:
                if toast.is_closed():
                    continue
                x = max(10, parent_w - toast.width() - self.MARGIN_RIGHT)
                try:
                    # ✅ Smooth move (if position differs)
                    current = toast.pos()
                    target = QPoint(x, y)
                    if current != target:
                        if _motion_ok_safe():
                            move_anim = QPropertyAnimation(toast, b"pos", toast)
                            move_anim.setDuration(220)
                            move_anim.setStartValue(current)
                            move_anim.setEndValue(target)
                            move_anim.setEasingCurve(QEasingCurve.OutCubic)
                            move_anim.start()
                            toast._reposition_anim = move_anim
                        else:
                            toast.move(target)
                except Exception:
                    pass
                y += toast.height() + self.SPACING
        except Exception:
            pass

    def _on_toast_dismissed(self, toast):
        """وقتی یه toast بسته شد."""
        try:
            if toast in self._toasts:
                self._toasts.remove(toast)
            try:
                toast.deleteLater()
            except Exception:
                pass
            self._reposition()
        except Exception:
            pass

    def clear_all(self):
        """بستن همه‌ی toastها."""
        for toast in list(self._toasts):
            try:
                toast._closed = True
                toast.setVisible(False)
                toast.deleteLater()
            except Exception:
                pass
        self._toasts.clear()

# ================================================================
# ✅ SIGNUP DIALOG
# ================================================================

class SignupDialog(QDialog):
    signup_success = pyqtSignal(str)

    def __init__(self, db: UserDB, theme: Dict, lang: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.theme = theme
        self.lang = lang

        self.setWindowTitle(tr(lang, "signup"))
        self.setFixedSize(500, 700)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

        # ✅ Add backdrop (exactly like UpdateAvailableDialog)
        try:
            from gui.dialogs.update_dialogs import _with_backdrop
            _with_backdrop(self, parent, theme)
        except Exception as e:
            print(f"⚠️ SignupDialog backdrop failed: {e}")

        # ✅ انیمیشن باز شدن
        QTimer.singleShot(0, lambda: animate_dialog_open(self))

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # ✅ ScrollArea
        self.scroll = QScrollArea()
        self.scroll.setObjectName("signup_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("signup_scroll_content")

        layout = QVBoxLayout(self.scroll_content)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(10)

        title = QLabel(tr(self.lang, "signup"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(15)

        self.first_name = QLineEdit()
        self.first_name.setPlaceholderText(tr(self.lang, "first_name"))
        self.first_name.setObjectName("input")
        layout.addWidget(self.first_name)

        self.last_name = QLineEdit()
        self.last_name.setPlaceholderText(tr(self.lang, "last_name"))
        self.last_name.setObjectName("input")
        layout.addWidget(self.last_name)

        self.username = QLineEdit()
        self.username.setPlaceholderText(tr(self.lang, "username"))
        self.username.setObjectName("input")
        layout.addWidget(self.username)

        self.email = QLineEdit()
        self.email.setPlaceholderText(tr(self.lang, "email"))
        self.email.setObjectName("input")
        layout.addWidget(self.email)

        self.password = QLineEdit()
        self.password.setPlaceholderText(tr(self.lang, "password"))
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setObjectName("input")
        # ✅ Password row: input + eye button
        pwd_row = QHBoxLayout()
        pwd_row.setSpacing(4)
        pwd_row.addWidget(self.password, 1)

        self._eye_btn = QPushButton("👁")
        self._eye_btn.setObjectName("eye_btn")
        self._eye_btn.setFixedSize(36, 36)
        self._eye_btn.setCursor(Qt.PointingHandCursor)
        self._eye_btn.setCheckable(True)
        self._eye_btn.toggled.connect(self._toggle_password_visibility)
        pwd_row.addWidget(self._eye_btn)

        layout.addLayout(pwd_row)

        # ✅ PasswordStrengthMeter (زیر فیلد پسورد)
        try:
            from gui.widgets.password_strength_meter import PasswordStrengthMeter
            self._strength_meter = PasswordStrengthMeter(
                theme=self.theme,
                lang=self.lang,
            )
            self.password.textChanged.connect(self._strength_meter.set_password)
            layout.addWidget(self._strength_meter)
        except Exception as e:
            print(f"⚠️ PasswordStrengthMeter init failed: {e}")
            self._strength_meter = None

        # ✅ Show password toggle (کنار فیلد پسورد)
        # (اینو می‌تونی توی یه HBoxLayout بذاری، یا یه دکمه‌ی چشم کنار فیلد)

        self.repeat_password = QLineEdit()
        self.repeat_password.setPlaceholderText(tr(self.lang, "repeat_password"))
        self.repeat_password.setEchoMode(QLineEdit.Password)
        self.repeat_password.setObjectName("input")
        layout.addWidget(self.repeat_password)

        # ✅ PasswordMatchIndicator (زیر فیلد repeat)
        try:
            from gui.widgets.password_match_indicator import PasswordMatchIndicator
            self._match_indicator = PasswordMatchIndicator(
                theme=self.theme,
                lang=self.lang,
            )
            self.password.textChanged.connect(self._update_match)
            self.repeat_password.textChanged.connect(self._update_match)
            layout.addWidget(self._match_indicator)
        except Exception as e:
            print(f"⚠️ PasswordMatchIndicator init failed: {e}")
            self._match_indicator = None

        rules = QLabel(tr(self.lang, "password_rule"))
        rules.setObjectName("rules")
        rules.setWordWrap(True)
        layout.addWidget(rules)

        self.remember_me = QCheckBox(tr(self.lang, "remember_me"))
        self.remember_me.setObjectName("checkbox")
        layout.addWidget(self.remember_me)

        layout.addSpacing(15)

        self.signup_btn = QPushButton(tr(self.lang, "signup_btn"))
        self.signup_btn.setObjectName("primary_btn")
        self.signup_btn.clicked.connect(self._do_signup)
        attach_sound(self.signup_btn, "signinSound")
        layout.addWidget(self.signup_btn)

        layout.addStretch()

        self.scroll.setWidget(self.scroll_content)
        container_layout.addWidget(self.scroll)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        attach_sound(close_btn, "simpleClick")
        close_btn.setParent(self.container)
        close_btn.move(460, 10)
        close_btn.raise_()

    def _update_match(self):
        """Update password match indicator."""
        try:
            if getattr(self, "_match_indicator", None) is not None:
                self._match_indicator.set_passwords(
                    self.password.text(),
                    self.repeat_password.text(),
                )
        except Exception:
            pass

    def _toggle_password_visibility(self, checked: bool):
        """Toggle password visibility (mask vs plain)."""
        try:
            mode = QLineEdit.Normal if checked else QLineEdit.Password
            self.password.setEchoMode(mode)
            # ✅ هم repeat_password
            if hasattr(self, "repeat_password"):
                self.repeat_password.setEchoMode(mode)
            # ✅ آیکون رو عوض کن
            self._eye_btn.setText("🙈" if checked else "👁")
        except Exception:
            pass

    def _apply_theme(self):
        t = self.theme

        # ✅ استایل اصلی روی container (برای close_btn و border)
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)

        # ✅ ScrollArea شفاف
        self.scroll.setStyleSheet("""
            QScrollArea#signup_scroll {
                background: transparent;
                border: none;
            }
            QScrollArea#signup_scroll > QWidget > QWidget {
                background: transparent;
            }
        """)

        # ✅ استایل کامل روی scroll_content (همه‌ی widgetهای داخلش)
        self.scroll_content.setStyleSheet(f"""
            QWidget#signup_scroll_content {{
                background: transparent;
            }}
            QLabel#title {{
                font-size: 28px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#rules {{
                font-size: 11px;
                color: {t['text_dim']};
                background: transparent;
                padding: 8px;
                border-left: 3px solid {t['primary']};
            }}
            QLabel {{
                color: {t['text']};
                background: transparent;
            }}
            QLineEdit#input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
                color: {t['text']};
                min-height: 18px;
            }}
            QLineEdit#input:focus {{
                border: 2px solid {t['secondary']};
            }}
            QCheckBox#checkbox {{
                color: {t['text_dim']};
                font-size: 13px;
                background: transparent;
            }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:0.5 {t['accent']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton#primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {t['glass_border']};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {t['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QPushButton#eye_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: 1px solid {t['input_border']};
                border-radius: 8px;
                font-size: 14px;
            }}
            QPushButton#eye_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#eye_btn:checked {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
        """)

    def _do_signup(self):
        try:
            first_name = self.first_name.text().strip()
            last_name = self.last_name.text().strip()
            username = self.username.text().strip()
            email = self.email.text().strip()
            password = self.password.text()
            repeat = self.repeat_password.text()

            if not all([first_name, last_name, username, email, password, repeat]):
                show_message(self, "warning", "Error", tr(self.lang, "empty_fields"))
                return

            if not validate_username(username):
                show_message(self, "warning", tr(self.lang, "warning"),
                             tr(self.lang, "invalid_username"))
                return

            if not validate_email(email):
                show_message(self, "warning", "Error", tr(self.lang, "invalid_email"))
                try:
                    highlight_widget_error(self.email, self.theme)
                except Exception:
                    pass
                return

            if password != repeat:
                show_message(self, "warning", "Error", tr(self.lang, "passwords_dont_match"))
                try:
                    highlight_widget_error(self.repeat_password, self.theme)
                except Exception:
                    pass
                return

            ok, msg = validate_password(password)
            if not ok:
                show_message(
                    self, "warning", tr(self.lang, "password_requirements"),
                    f"{tr(self.lang, 'password_rule')}\n\n❌ {msg}",
                )
                return

            if self.db.user_exists(username):
                show_message(self, "warning", "Error", tr(self.lang, "user_exists"))
                return

            if self.db.create_user(username, password, first_name, last_name, email):
                self.password.clear()
                self.repeat_password.clear()
                self.signup_success.emit(username)
                QTimer.singleShot(0, lambda: self._finish_signup())
        except Exception as e:
            print(f"⚠️ _do_signup failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                show_message(self, "critical", "Error", f"Signup failed:\n{e}")
            except Exception:
                pass

    def _finish_signup(self):
        try:
            animate_dialog_close(self, on_finished=self.accept)
        except Exception:
            try:
                self.accept()
            except Exception:
                pass


# ================================================================
# ✅ SETTINGS DIALOG
# ================================================================

class SettingsDialog(QDialog):
    settings_changed = pyqtSignal(str, str)

    def __init__(self, current_lang: str, current_theme: str,
                 follow_system: bool = False,
                 glassmorphism: bool = True, parent=None):
        super().__init__(parent)
        self.current_lang = current_lang
        self.current_theme = current_theme
        self.current_follow_system = follow_system
        self.current_glassmorphism = glassmorphism

        self.setWindowTitle(tr(current_lang, "settings"))
        self.setFixedSize(560, 720)
        self.setMinimumSize(560, 600)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # ✅ Auto-lock config
        try:
            self._auto_lock_config = load_auto_lock_config()
        except Exception:
            self._auto_lock_config = AutoLockConfig()

        # ✅ Font settings
        try:
            self._font_settings = load_font_settings()
        except Exception:
            self._font_settings = FontSettings()

        self._setup_ui()
        self._apply_theme()

        # ✅ Add backdrop (same as LoginDialog / SignupDialog / UpdateAvailableDialog)
        try:
            from gui.dialogs.update_dialogs import _with_backdrop
            _with_backdrop(
                self, parent,
                THEMES.get(self.current_theme, THEMES["theme_dark_aqua"]),
            )
        except Exception as e:
            print(f"⚠️ SettingsDialog backdrop failed: {e}")

        # ✅ انیمیشن باز شدن
        QTimer.singleShot(0, lambda: animate_dialog_open(self))

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        # ============================================================
        # ✅ Container layout: close button + scroll area
        # ============================================================
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # ✅ ScrollArea (تا اگه محتوا بیشتر از ارتفاع شد، اسکرول بشه)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("settings_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("settings_scroll_content")

        layout = QVBoxLayout(self.scroll_content)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(14)

        # ---------- Title ----------
        self.title_label = QLabel(tr(self.current_lang, "settings"))
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        layout.addSpacing(10)

        # ---------- Language ----------
        self.lang_label = QLabel(tr(self.current_lang, "change_language"))
        self.lang_label.setObjectName("label")
        layout.addWidget(self.lang_label)

        self.lang_combo = QComboBox()
        self.lang_combo.setObjectName("combo")
        for lang_name in LANGUAGE_ORDER:
            if lang_name in TRANSLATIONS:
                self.lang_combo.addItem(lang_name, lang_name)
        idx = self.lang_combo.findData(self.current_lang)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        layout.addWidget(self.lang_combo)

        layout.addSpacing(8)

        # ---------- Theme ----------
        self.theme_label = QLabel(tr(self.current_lang, "change_theme"))
        self.theme_label.setObjectName("label")
        layout.addWidget(self.theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("combo")
        for theme_key in THEME_ORDER:
            if theme_key in THEMES:
                theme_display = tr(self.current_lang, theme_key)
                self.theme_combo.addItem(theme_display, theme_key)
        idx = self.theme_combo.findData(self.current_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        layout.addWidget(self.theme_combo)

        layout.addSpacing(8)

        # ---------- Follow System Theme ----------
        self.follow_system_checkbox = QCheckBox(
            tr(self.current_lang, "follow_system_theme")
        )
        self.follow_system_checkbox.setObjectName("checkbox")
        self.follow_system_checkbox.setChecked(self.current_follow_system)
        self.follow_system_checkbox.setToolTip(
            tr(self.current_lang, "follow_system_theme_desc")
        )

        # ✅ Connect checkbox to enable/disable theme combo
        self.follow_system_checkbox.toggled.connect(self._on_follow_system_toggled)

        # ✅ Initial state
        self._on_follow_system_toggled(self.follow_system_checkbox.isChecked())

        layout.addWidget(self.follow_system_checkbox)

        layout.addSpacing(8)

        # ---------- Glassmorphism ----------
        self.glassmorphism_checkbox = QCheckBox("Glassmorphism (blur)")
        self.glassmorphism_checkbox.setObjectName("checkbox")
        self.glassmorphism_checkbox.setChecked(self.current_glassmorphism)
        self.glassmorphism_checkbox.setToolTip(
            "Real blur effect. Disable if the app feels slow."
        )
        layout.addWidget(self.glassmorphism_checkbox)

        layout.addSpacing(14)

        # ============================================================
        # ✅ Auto-Lock section
        # ============================================================
        autolock_sep = QFrame()
        autolock_sep.setObjectName("update_separator")
        autolock_sep.setFixedHeight(1)
        layout.addWidget(autolock_sep)

        layout.addSpacing(10)

        autolock_title = QLabel("🔒 " + tr(self.current_lang, "auto_lock_section_title"))
        autolock_title.setObjectName("label")
        layout.addWidget(autolock_title)

        self.auto_lock_checkbox = QCheckBox(
            tr(self.current_lang, "auto_lock_enable")
        )
        self.auto_lock_checkbox.setObjectName("checkbox")
        self.auto_lock_checkbox.setChecked(self._auto_lock_config.enabled)
        layout.addWidget(self.auto_lock_checkbox)

        self.auto_lock_timeout_label = QLabel(
            tr(self.current_lang, "auto_lock_timeout_label")
        )
        self.auto_lock_timeout_label.setObjectName("label")
        layout.addWidget(self.auto_lock_timeout_label)

        self.auto_lock_timeout_combo = QComboBox()
        self.auto_lock_timeout_combo.setObjectName("combo")
        timeout_options = [
            (1, "1 min"),
            (2, "2 min"),
            (5, "5 min"),
            (10, "10 min"),
            (15, "15 min"),
            (30, "30 min"),
            (60, "60 min"),
        ]
        for mins, label in timeout_options:
            self.auto_lock_timeout_combo.addItem(label, mins)
        idx = self.auto_lock_timeout_combo.findData(
            self._auto_lock_config.timeout_minutes
        )
        if idx >= 0:
            self.auto_lock_timeout_combo.setCurrentIndex(idx)
        self.auto_lock_timeout_combo.setEnabled(self._auto_lock_config.enabled)
        layout.addWidget(self.auto_lock_timeout_combo)

        self.auto_lock_checkbox.toggled.connect(
            self.auto_lock_timeout_combo.setEnabled
        )

        layout.addSpacing(14)

        # ============================================================
        # ✅ Font section
        # ============================================================
        font_sep = QFrame()
        font_sep.setObjectName("update_separator")
        font_sep.setFixedHeight(1)
        layout.addWidget(font_sep)

        layout.addSpacing(10)

        font_title = QLabel("🔤 " + tr(self.current_lang, "font_section_title"))
        font_title.setObjectName("label")
        layout.addWidget(font_title)

        # ✅ Family
        family_label = QLabel(tr(self.current_lang, "font_family_label"))
        family_label.setObjectName("label")
        layout.addWidget(family_label)

        self.font_family_combo = QComboBox()
        self.font_family_combo.setObjectName("combo")
        self.font_family_combo.setEditable(False)
        try:
            fonts = list_available_fonts()
            for fam in fonts[:80]:  # limit to avoid huge list
                self.font_family_combo.addItem(display_name_for(fam), fam)
            # ✅ Select current
            idx = self.font_family_combo.findData(self._font_settings.family)
            if idx < 0:
                idx = self.font_family_combo.findData(DEFAULT_FAMILY)
            if idx >= 0:
                self.font_family_combo.setCurrentIndex(idx)
        except Exception as e:
            print(f"⚠️ font list failed: {e}")
        self.font_family_combo.currentIndexChanged.connect(
            self._preview_font
        )
        layout.addWidget(self.font_family_combo)

        # ✅ Size
        size_label = QLabel(tr(self.current_lang, "font_size_label"))
        size_label.setObjectName("label")
        layout.addWidget(size_label)

        self.font_size_combo = QComboBox()
        self.font_size_combo.setObjectName("combo")
        for size in range(MIN_SIZE, MAX_SIZE + 1):
            self.font_size_combo.addItem(f"{size} pt", size)
        idx = self.font_size_combo.findData(self._font_settings.size)
        if idx >= 0:
            self.font_size_combo.setCurrentIndex(idx)
        self.font_size_combo.currentIndexChanged.connect(
            self._preview_font
        )
        layout.addWidget(self.font_size_combo)

        # ✅ Scale
        scale_label = QLabel(tr(self.current_lang, "font_scale_label"))
        scale_label.setObjectName("label")
        layout.addWidget(scale_label)

        self.font_scale_combo = QComboBox()
        self.font_scale_combo.setObjectName("combo")
        scale_options = [
            (0.85, tr(self.current_lang, "font_scale_small")),
            (1.0,  tr(self.current_lang, "font_scale_normal")),
            (1.15, tr(self.current_lang, "font_scale_large")),
            (1.30, tr(self.current_lang, "font_scale_xlarge")),
            (1.50, tr(self.current_lang, "font_scale_xxlarge")),
        ]
        for val, label in scale_options:
            self.font_scale_combo.addItem(label, val)
        # find closest
        best_idx = 1
        for i, (val, _lbl) in enumerate(scale_options):
            if abs(val - self._font_settings.scale) < 0.05:
                best_idx = i
                break
        self.font_scale_combo.setCurrentIndex(best_idx)
        self.font_scale_combo.currentIndexChanged.connect(
            self._preview_font
        )
        layout.addWidget(self.font_scale_combo)

        # ✅ Preview label
        self.font_preview = QLabel(
            tr(self.current_lang, "font_preview_text")
        )
        self.font_preview.setObjectName("font_preview")
        self.font_preview.setWordWrap(True)
        self.font_preview.setMinimumHeight(50)
        self.font_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.font_preview)

        # ✅ Reset button
        self.font_reset_btn = QPushButton(
            "↺ " + tr(self.current_lang, "font_reset")
        )
        self.font_reset_btn.setObjectName("secondary_btn")
        self.font_reset_btn.setMinimumHeight(38)
        self.font_reset_btn.setCursor(Qt.PointingHandCursor)
        self.font_reset_btn.clicked.connect(self._reset_font_defaults)
        layout.addWidget(self.font_reset_btn)

        # ✅ Reduced motion toggle
        layout.addSpacing(14)

        motion_sep = QFrame()
        motion_sep.setObjectName("update_separator")
        motion_sep.setFixedHeight(1)
        layout.addWidget(motion_sep)

        layout.addSpacing(10)

        motion_title = QLabel("✨ " + tr(self.current_lang, "motion_section_title"))
        motion_title.setObjectName("label")
        layout.addWidget(motion_title)

        self.reduce_motion_checkbox = QCheckBox(
            tr(self.current_lang, "reduce_motion_label")
        )
        self.reduce_motion_checkbox.setObjectName("checkbox")
        try:
            self.reduce_motion_checkbox.setChecked(
                MotionSettings.get().is_reduced()
            )
        except Exception:
            pass
        layout.addWidget(self.reduce_motion_checkbox)

        motion_hint = QLabel(tr(self.current_lang, "reduce_motion_hint"))
        motion_hint.setObjectName("rules")
        motion_hint.setWordWrap(True)
        layout.addWidget(motion_hint)


        layout.addSpacing(14)

        # ============================================================
        # ✅ Update section
        # ============================================================
        update_separator = QFrame()
        update_separator.setObjectName("update_separator")
        update_separator.setFixedHeight(1)
        layout.addWidget(update_separator)

        layout.addSpacing(10)

        update_title = QLabel("🔄 " + tr(self.current_lang, "check_for_updates"))
        update_title.setObjectName("label")
        layout.addWidget(update_title)

        self.check_update_btn = QPushButton(
            "🔍 " + tr(self.current_lang, "check_for_updates")
        )
        self.check_update_btn.setObjectName("primary_btn")
        self.check_update_btn.setCursor(Qt.PointingHandCursor)
        self.check_update_btn.setMinimumHeight(42)
        self.check_update_btn.clicked.connect(self._check_for_updates)
        attach_sound(self.check_update_btn, "simpleClick")
        layout.addWidget(self.check_update_btn)

        layout.addSpacing(16)

        # ---------- Action buttons ----------
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.apply_btn = QPushButton("✓ " + tr(self.current_lang, "apply"))
        self.apply_btn.setObjectName("primary_btn")
        self.apply_btn.setMinimumHeight(42)
        self.apply_btn.clicked.connect(self._apply_settings)
        attach_sound(self.apply_btn, "ok_sound")
        btn_layout.addWidget(self.apply_btn)

        self.cancel_btn = QPushButton("✕ " + tr(self.current_lang, "cancel"))
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.setMinimumHeight(42)
        self.cancel_btn.clicked.connect(self.reject)
        attach_sound(self.cancel_btn, "simpleClick")
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        layout.addStretch()

        # ✅ Set scroll content
        self.scroll.setWidget(self.scroll_content)
        container_layout.addWidget(self.scroll)

        # ✅ Close button (روی container، نه روی scroll)
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        attach_sound(close_btn, "simpleClick")
        close_btn.setParent(self.container)
        close_btn.move(525, 10)  # ← موقعیت جدید برای عرض ۵۶۰
        close_btn.raise_()

    def _on_follow_system_toggled(self, checked: bool):
        """When follow_system is checked, show system theme in combo (but keep it enabled)."""
        try:
            if checked:
                # ✅ Show current system theme in the combo
                system_theme = detect_system_theme()
                idx = self.theme_combo.findData(system_theme)
                if idx >= 0:
                    self.theme_combo.setCurrentIndex(idx)
            # ✅ DON'T disable the combo — user can still pick a theme
            # (theme will be respected when user clicks Apply)
        except Exception as e:
            print(f"⚠️ _on_follow_system_toggled failed: {e}")

    def _apply_theme(self):
        t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])

        # ✅ Style container (border + close button)
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)

        # ✅ ScrollArea transparent
        self.scroll.setStyleSheet("""
            QScrollArea#settings_scroll {
                background: transparent;
                border: none;
            }
            QScrollArea#settings_scroll > QWidget > QWidget {
                background: transparent;
            }
        """)

        # ✅ Style scroll_content (همه‌ی widgetهای داخلش)
        self.scroll_content.setStyleSheet(f"""
            QWidget#settings_scroll_content {{
                background: transparent;
            }}
            QLabel#title {{
                font-size: 28px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
                padding-bottom: 6px;
            }}
            QLabel#label {{
                font-size: 14px;
                font-weight: bold;
                color: {t['text']};
                background: transparent;
                padding-top: 4px;
                padding-bottom: 2px;
            }}
            QComboBox#combo {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px 12px;
                font-size: 13px;
                color: {t['text']};
                min-height: 24px;
            }}
            QComboBox#combo::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox#combo QAbstractItemView {{
                background: {t['input_bg']};
                color: {t['text']};
                selection-background-color: {t['primary']};
                border: 1px solid {t['input_border']};
                outline: none;
            }}
            QCheckBox#checkbox {{
                color: {t['text']};
                font-size: 13px;
                background: transparent;
                spacing: 8px;
                padding: 4px 0px;
            }}
            QFrame#update_separator {{
                background: {t['glass_border']};
                border: none;
            }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:0.5 {t['accent']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#secondary_btn {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#secondary_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {t['glass_border']};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {t['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def _preview_font(self):
        """Live preview of font selection."""
        try:
            family = self.font_family_combo.currentData() or DEFAULT_FAMILY
            size = self.font_size_combo.currentData() or DEFAULT_SIZE
            scale = self.font_scale_combo.currentData() or DEFAULT_SCALE
            effective = max(6, int(round(int(size) * float(scale))))
            self.font_preview.setStyleSheet(f"""
                QLabel#font_preview {{
                    background: {self._get_theme()['input_bg']};
                    color: {self._get_theme()['text']};
                    border: 2px solid {self._get_theme()['primary']};
                    border-radius: 10px;
                    padding: 10px;
                    font-family: '{family}', 'Noto Sans', sans-serif;
                    font-size: {effective}pt;
                }}
            """)
        except Exception as e:
            print(f"⚠️ _preview_font failed: {e}")

    def _get_theme(self):
        """Return current theme dict (for preview)."""
        try:
            return THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
        except Exception:
            return THEMES["theme_dark_aqua"]

    def _reset_font_defaults(self):
        """Reset font combos to defaults."""
        try:
            idx = self.font_family_combo.findData(DEFAULT_FAMILY)
            if idx >= 0:
                self.font_family_combo.setCurrentIndex(idx)
            idx = self.font_size_combo.findData(DEFAULT_SIZE)
            if idx >= 0:
                self.font_size_combo.setCurrentIndex(idx)
            self.font_scale_combo.setCurrentIndex(1)  # 1.0 = normal
            self._preview_font()
        except Exception as e:
            print(f"⚠️ _reset_font_defaults failed: {e}")


    def _check_for_updates(self):
        """One-click update flow with user confirmation."""
        try:
            from core.updater import CURRENT_VERSION
            from core.updater.models import UpdateConfig
            from gui.dialogs.update_dialog import run_update_flow
            import json
            import os

            # ✅ Load current update config from settings.json
            settings_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "settings.json"
            )
            config = UpdateConfig()
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "update" in data:
                        config = UpdateConfig.from_dict(data["update"])
                except Exception:
                    pass

            # ✅ Run the full flow
            self.check_update_btn.setEnabled(False)
            self.check_update_btn.setText("⏳ " + tr(self.current_lang, "downloading"))

            try:
                installed = run_update_flow(
                    parent=self,
                    config=config,
                    current_version=CURRENT_VERSION,
                    theme=self.theme if hasattr(self, "theme") else THEMES.get(
                        self.current_theme, THEMES["theme_dark_aqua"]
                    ),
                    lang=self.current_lang,
                )
            finally:
                try:
                    self.check_update_btn.setEnabled(True)
                    self.check_update_btn.setText(
                        "🔍 " + tr(self.current_lang, "check_for_updates")
                    )
                except Exception:
                    pass

        except Exception as e:
            print(f"⚠️ _check_for_updates failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                show_message(
                    self, "critical",
                    tr(self.current_lang, "error"),
                    f"Update check failed:\n{e}",
                )
            except Exception:
                pass

    def _apply_settings(self):
        new_lang = self.lang_combo.currentData()
        new_theme = self.theme_combo.currentData()
        follow_system = self.follow_system_checkbox.isChecked()
        glassmorphism = self.glassmorphism_checkbox.isChecked()

        # ✅ Auto-lock settings
        auto_lock_enabled = self.auto_lock_checkbox.isChecked()
        auto_lock_timeout = self.auto_lock_timeout_combo.currentData() or 5
        try:
            new_auto_lock_cfg = AutoLockConfig(
                enabled=auto_lock_enabled,
                timeout_minutes=int(auto_lock_timeout),
            )
            save_auto_lock_config(new_auto_lock_cfg)
        except Exception as e:
            print(f"⚠️ save auto_lock failed: {e}")
            new_auto_lock_cfg = None

        print(f"⚙️ _apply_settings: lang={new_lang}, theme={new_theme}, "
              f"follow_system={follow_system}, glassmorphism={glassmorphism}")

        # ✅ Micro-interaction: success check
        try:
            from core.ui.micro_interactions import show_success_check
            theme_dict = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
            show_success_check(self, color=theme_dict.get("success", "#34d399"))
        except Exception:
            pass

        # ✅ ONLY ONE path — no double emit
        parent = self.parent()
        if parent is not None and hasattr(parent, "_on_settings_applied"):
            try:
                parent._on_settings_applied(new_lang, new_theme, follow_system, glassmorphism)
                # ✅ Apply reduced motion
                try:
                    MotionSettings.get().set_reduced(
                        self.reduce_motion_checkbox.isChecked()
                    )
                except Exception:
                    pass
                # ✅ Save + apply font settings
                try:
                    new_font_settings = FontSettings(
                        family=self.font_family_combo.currentData() or DEFAULT_FAMILY,
                        size=int(self.font_size_combo.currentData() or DEFAULT_SIZE),
                        scale=float(self.font_scale_combo.currentData() or DEFAULT_SCALE),
                    )
                    save_font_settings(new_font_settings)
                    # Apply globally
                    FontManager.get().apply(new_font_settings)
                    # Notify MainWindow
                    if parent is not None and hasattr(parent, "_on_font_changed"):
                        parent._on_font_changed(new_font_settings)
                except Exception as e:
                    print(f"⚠️ font apply failed: {e}")
                # ✅ Apply auto-lock config live
                if new_auto_lock_cfg is not None and parent is not None:
                    try:
                        if hasattr(parent, "_auto_lock_config"):
                            parent._auto_lock_config = new_auto_lock_cfg
                        if hasattr(parent, "_auto_lock") and parent._auto_lock is not None:
                            parent._auto_lock.set_config(new_auto_lock_cfg)
                            if new_auto_lock_cfg.enabled and parent.current_user:
                                if not parent._auto_lock.is_active():
                                    parent._auto_lock.start()
                            elif not new_auto_lock_cfg.enabled:
                                parent._auto_lock.stop()
                    except Exception as e:
                        print(f"⚠️ live auto_lock update failed: {e}")
            except Exception as e:
                print(f"⚠️ _on_settings_applied failed: {e}")
                import traceback
                traceback.print_exc()
        # ✅ انیمیشن بسته شدن + accept
        animate_dialog_close(self, on_finished=self.accept)


# ================================================================
# ✅ ACCOUNT INFO DIALOG
# ================================================================

class AccountInfoDialog(QDialog):
    info_updated = pyqtSignal()

    def __init__(self, db: UserDB, username: str, theme: Dict, lang: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.username = username
        self.theme = theme
        self.lang = lang

        self.setWindowTitle(tr(lang, "account_info"))
        self.setFixedSize(540, 620)
        self.setMinimumSize(540, 560)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

        try:
            from gui.dialogs.update_dialogs import _with_backdrop
            _with_backdrop(self, parent, theme)
        except Exception as e:
            print(f"⚠️ AccountInfoDialog backdrop failed: {e}")

        QTimer.singleShot(0, lambda: animate_dialog_open(self))

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # ✅ ScrollArea
        self.scroll = QScrollArea()
        self.scroll.setObjectName("account_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("account_scroll_content")

        layout = QVBoxLayout(self.scroll_content)
        layout.setContentsMargins(36, 32, 36, 32)
        layout.setSpacing(14)

        # ---------- Title ----------
        title = QLabel(tr(self.lang, "account_info"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(10)

        # ---------- First Name ----------
        self.first_name_label = QLabel(tr(self.lang, "first_name"))
        self.first_name_label.setObjectName("field_label")
        layout.addWidget(self.first_name_label)

        self.first_name = QLineEdit()
        self.first_name.setObjectName("input")
        self.first_name.setMinimumHeight(40)
        layout.addWidget(self.first_name)

        # ---------- Last Name ----------
        self.last_name_label = QLabel(tr(self.lang, "last_name"))
        self.last_name_label.setObjectName("field_label")
        layout.addWidget(self.last_name_label)

        self.last_name = QLineEdit()
        self.last_name.setObjectName("input")
        self.last_name.setMinimumHeight(40)
        layout.addWidget(self.last_name)

        # ---------- Email ----------
        self.email_label = QLabel(tr(self.lang, "email"))
        self.email_label.setObjectName("field_label")
        layout.addWidget(self.email_label)

        self.email = QLineEdit()
        self.email.setObjectName("input")
        self.email.setMinimumHeight(40)
        layout.addWidget(self.email)

        # ✅ پر کردن مقادیر فعلی
        user_info = self.db.get_user_info(self.username) or {}
        self.first_name.setText(user_info.get("first_name", ""))
        self.last_name.setText(user_info.get("last_name", ""))
        self.email.setText(user_info.get("email", ""))

        layout.addSpacing(10)

        # ---------- Save button ----------
        save_btn = QPushButton("💾 " + tr(self.lang, "save"))
        save_btn.setObjectName("primary_btn")
        save_btn.setMinimumHeight(42)
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._save)
        attach_sound(save_btn, "ok_sound")
        layout.addWidget(save_btn)

        # ---------- 2FA Section ----------
        layout.addSpacing(16)

        twofa_sep = QFrame()
        twofa_sep.setObjectName("sep")
        twofa_sep.setFixedHeight(1)
        layout.addWidget(twofa_sep)

        layout.addSpacing(8)

        twofa_title = QLabel("🔐 " + tr(self.lang, "2fa_section_title"))
        twofa_title.setObjectName("section_title")
        layout.addWidget(twofa_title)

        # ✅ 2FA status card
        self._build_2fa_section(layout)

        layout.addStretch()

        self.scroll.setWidget(self.scroll_content)
        container_layout.addWidget(self.scroll)

        # ✅ Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        attach_sound(close_btn, "simpleClick")
        close_btn.setParent(self.container)
        close_btn.move(505, 10)
        close_btn.raise_()

    def _build_2fa_section(self, parent_layout):
        """بخش 2FA با ظاهر تمیز و هماهنگ با تم."""
        # ---------- Status Card ----------
        status_card = QFrame()
        status_card.setObjectName("twofa_status_card")
        status_card.setMinimumHeight(64)

        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(14, 10, 10, 10)
        status_layout.setSpacing(10)

        # آیکون وضعیت
        self.twofa_icon = QLabel("🔐")
        self.twofa_icon.setObjectName("twofa_icon")
        self.twofa_icon.setFixedSize(32, 32)
        self.twofa_icon.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.twofa_icon)

        # متن وضعیت
        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(2)

        self.twofa_title_label = QLabel(tr(self.lang, "2fa_section_title"))
        self.twofa_title_label.setObjectName("twofa_card_title")

        self.twofa_status_label = QLabel("")
        self.twofa_status_label.setObjectName("twofa_status_text")
        self.twofa_status_label.setWordWrap(True)

        text_box.addWidget(self.twofa_title_label)
        text_box.addWidget(self.twofa_status_label)
        status_layout.addLayout(text_box, 1)

        # دکمه
        self.twofa_toggle_btn = QPushButton("")
        self.twofa_toggle_btn.setObjectName("twofa_action_btn")
        self.twofa_toggle_btn.setMinimumHeight(34)
        self.twofa_toggle_btn.setMinimumWidth(110)
        self.twofa_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.twofa_toggle_btn.clicked.connect(self._on_toggle_2fa)
        attach_sound(self.twofa_toggle_btn, "simpleClick")
        status_layout.addWidget(self.twofa_toggle_btn)

        parent_layout.addWidget(status_card)

        self._refresh_2fa_status()

    def _apply_theme(self):
        t = self.theme

        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)

        self.scroll.setStyleSheet("""
            QScrollArea#account_scroll {
                background: transparent;
                border: none;
            }
            QScrollArea#account_scroll > QWidget > QWidget {
                background: transparent;
            }
        """)

        self.scroll_content.setStyleSheet(f"""
            QWidget#account_scroll_content {{
                background: transparent;
            }}
            QLabel {{
                color: {t['text']};
                background: transparent;
            }}
            QLabel#title {{
                font-size: 26px;
                font-weight: bold;
                color: {t['primary']};
            }}
            QLabel#field_label {{
                font-size: 12px;
                font-weight: bold;
                color: {t['text']};
                padding-top: 4px;
            }}
            QLabel#section_title {{
                font-size: 14px;
                font-weight: bold;
                color: {t['primary']};
            }}
            QLineEdit#input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px 12px;
                font-size: 13px;
                color: {t['text']};
            }}
            QLineEdit#input:focus {{
                border: 2px solid {t['secondary']};
            }}
            QFrame#sep {{
                background: {t['glass_border']};
                border: none;
            }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:0.5 {t['accent']}, stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QFrame#twofa_status_card {{
                background: {t['input_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 12px;
            }}
            QLabel#twofa_icon {{
                background: transparent;
                font-size: 22px;
            }}
            QLabel#twofa_card_title {{
                font-size: 13px;
                font-weight: bold;
                color: {t['text']};
                background: transparent;
            }}
            QLabel#twofa_status_text {{
                font-size: 11px;
                color: {t['text_dim']};
                background: transparent;
            }}
            QPushButton#twofa_action_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#twofa_action_btn:hover {{
                background: {t['button_hover']};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {t['glass_border']};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {t['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def _refresh_2fa_status(self):
        try:
            status = self.db.get_2fa_status(self.username)
            enabled = status.get("enabled", False)
            t = self.theme

            if enabled:
                self.twofa_icon.setText("🔓")
                self.twofa_status_label.setText(tr(self.lang, "2fa_enabled_status"))
                self.twofa_status_label.setStyleSheet(
                    f"color: {t['success']}; font-size: 11px; font-weight: bold;"
                    "background: transparent;"
                )
                self.twofa_toggle_btn.setText("🔓 " + tr(self.lang, "2fa_disable_btn"))
                self.twofa_toggle_btn.setStyleSheet(f"""
                    QPushButton#twofa_action_btn {{
                        background: {t['input_bg']};
                        color: {t['danger']};
                        border: 2px solid {t['danger']};
                        border-radius: 8px;
                        padding: 6px 14px;
                        font-size: 12px;
                        font-weight: bold;
                    }}
                    QPushButton#twofa_action_btn:hover {{
                        background: {t['danger']};
                        color: {t['button_text']};
                    }}
                """)
            else:
                self.twofa_icon.setText("🔐")
                self.twofa_status_label.setText(tr(self.lang, "2fa_disabled_status"))
                self.twofa_status_label.setStyleSheet(
                    f"color: {t['text_dim']}; font-size: 11px;"
                    "background: transparent;"
                )
                self.twofa_toggle_btn.setText("🔐 " + tr(self.lang, "2fa_enable_btn"))
                self.twofa_toggle_btn.setStyleSheet(f"""
                    QPushButton#twofa_action_btn {{
                        background: qlineargradient(
                            x1:0, y1:0, x2:1, y2:1,
                            stop:0 {t['primary']},
                            stop:1 {t['secondary']}
                        );
                        color: {t['button_text']};
                        border: none;
                        border-radius: 8px;
                        padding: 6px 14px;
                        font-size: 12px;
                        font-weight: bold;
                    }}
                    QPushButton#twofa_action_btn:hover {{
                        background: {t['button_hover']};
                    }}
                """)
        except Exception as e:
            print(f"⚠️ _refresh_2fa_status failed: {e}")

    def _on_toggle_2fa(self):
        try:
            status = self.db.get_2fa_status(self.username)
            enabled = status.get("enabled", False)

            if enabled:
                ok = ask_question(
                    self,
                    tr(self.lang, "2fa_disable_title"),
                    tr(self.lang, "2fa_disable_confirm"),
                )
                if not ok:
                    return
                self.db.disable_2fa(self.username)
                SoundManager.get().play("ok_sound")
                self._refresh_2fa_status()
                try:
                    if hasattr(self.parent(), "_notify"):
                        self.parent()._notify(
                            "success",
                            tr(self.lang, "2fa_disabled_success"),
                            "",
                            3000,
                        )
                except Exception:
                    pass
            else:
                dlg = TwoFactorSetupDialog(
                    db=self.db,
                    username=self.username,
                    theme=self.theme,
                    lang=self.lang,
                    parent=self,
                )
                dlg.setup_completed.connect(self._on_2fa_setup_done)
                dlg.exec_()
                self._refresh_2fa_status()
        except Exception as e:
            print(f"⚠️ _on_toggle_2fa failed: {e}")

    def _on_2fa_setup_done(self, username, backup_codes):
        try:
            SoundManager.get().play("ok_sound")
            # ✅ Micro-interaction: success check
            show_success_check(self, color=self.theme.get("success", "#34d399"))
            self._refresh_2fa_status()
            try:
                if hasattr(self.parent(), "_notify"):
                    self.parent()._notify(
                        "success",
                        tr(self.lang, "2fa_enabled_success"),
                        "",
                        3000,
                    )
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ _on_2fa_setup_done failed: {e}")

    def _save(self):
        self.db.update_user(
            self.username,
            first_name=self.first_name.text().strip(),
            last_name=self.last_name.text().strip(),
            email=self.email.text().strip(),
        )
        self.info_updated.emit()
        # ✅ Micro-interaction: success check
        try:
            SoundManager.get().play("ok_sound")
            show_success_check(self, color=self.theme.get("success", "#34d399"))
            QTimer.singleShot(700, lambda: animate_dialog_close(self, on_finished=self.accept))
        except Exception:
            animate_dialog_close(self, on_finished=self.accept)


# ================================================================
# ✅ PROCESSING THREAD
# ================================================================

class ProcessThread(QThread):
    result_ready = pyqtSignal(int, float, dict, object)  # + heatmap
    error = pyqtSignal(str)

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            model, device = ModelLoader.get_model()
            img_tensor = preprocess_image(self.image_path, input_size=224)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits.float(), dim=1)
                conf, pred = probs.max(dim=1)
                all_probs = {
                    i: float(probs[0, i].item())
                    for i in range(probs.shape[1])
                }

            # ✅ Grad-CAM (بعد از inference)
            heatmap = None
            try:
                heatmap = generate_gradcam(
                    model=model,
                    img_tensor=img_tensor,
                    target_class=int(pred.item()),
                    device=device,
                )
            except Exception as e:
                print(f"⚠️ Grad-CAM generation failed: {e}")

            self.result_ready.emit(
                int(pred.item()),
                float(conf.item()),
                all_probs,
                heatmap,
            )
        except Exception as e:
            self.error.emit(str(e))


# ================================================================
# ✅ MAIN WINDOW
# ================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # State
        self.db = UserDB()
        self.stats = UserStats()  # ← اضافه شد
        self.current_lang = "English"
        self.current_theme = "theme_dark_aqua"
        self.current_user = None
        self.current_image_path = None
        self.is_muted = False
        self.music_playing = False
        self._music_pool = []
        self._music_queue = []
        self._current_track = None
        self.login_attempts = {}
        self._last_result_grid = None
        self._last_result_confidence = None
        # ✅ Dashboard state
        self.dashboard = None
        self._dashboard_visible = False
        self._dashboard_anim = None
        # ✅ PDF export state
        self._last_all_probs = None  # ← برای ذخیره‌ی همه‌ی احتمالات
        # ✅ Grad-CAM state
        self._last_heatmap = None

        # ✅ History state
        self.history_mgr = HistoryManager()
        self.history_panel = None
        self._history_visible = False
        self._history_anim = None
        self.notification_center = None   # ← این خط
        self.follow_system_theme = False   # ← این خط
        self.use_glassmorphism = False   # ← این خط

        # Window setup
        self.setWindowTitle(tr(self.current_lang, "app_title"))
        self.setMinimumSize(1400, 900)
        # ✅ فعال‌سازی Drag & Drop روی کل پنجره
        self.setAcceptDrops(True)
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

        # ✅ بارگذاری تنظیمات ذخیره‌شده
        self._load_saved_settings()

        # ✅ Font settings (loaded here so MainWindow knows them)
        self._font_settings = load_font_settings()
        # Apply immediately (though main() already did)
        try:
            FontManager.get().apply(self._font_settings)
        except Exception as e:
            print(f"⚠️ Font apply failed: {e}")

        # ✅ اگه follow_system فعاله، تم سیستم رو تشخیص بده
        if self.follow_system_theme:
            self.current_theme = detect_system_theme()

        self._setup_ui()
        self._apply_theme()
        self._decorate_buttons()
        self._update_all_tooltips()   # ← اضافه شد
        self._setup_shortcuts()      # ← اضافه شد

        # ✅ Auto-lock (replaces session timeout)
        self._auto_lock = None
        self._lock_screen = None
        self._auto_lock_config = load_auto_lock_config()
        self._setup_auto_lock()

        # ✅ Onboarding Tour state
        self._onboarding_storage = OnboardingStorage()
        self._onboarding_tour = None
        self._guest_tour_shown = False

        # ✅ Smart Suggestions state
        self._suggestion_storage = SuggestionStorage()
        self._suggestion_engine = None
        self._suggestion_bar = None
        self._user_activity_times = {}  # برای track activity
        self._image_loaded_at = None
        self._result_at = None
        self._consecutive_same_grid = 0
        self._history_opened_ever = False
        self._dashboard_opened_ever = False
        self._settings_opened_ever = False
        self._session_start_at = 0.0

        # ✅ راه‌اندازی Smart Suggestions (بعد از تعریف state)
        self._setup_smart_suggestions()

        self._start_music()

        # ✅ Initialize sound manager (loading sounds)
        SoundManager.get()

        # ✅ Voice Feedback
        self._voice_enabled = self._load_voice_enabled_setting()
        if HAS_VOICE:
            try:
                self._voice = VoiceManager.get()
                self._voice.set_enabled(self._voice_enabled)
                self._voice.set_language(self.current_lang)
                print(f"🔊 Voice Feedback initialized (enabled={self._voice_enabled})")
            except Exception as e:
                print(f"⚠️ VoiceManager init failed: {e}")
                self._voice = None
        else:
            self._voice = None

        # در __init__:
        self.update_config = UpdateConfig()
        self._update_info = None  # UpdateInfo | None
        self._update_checker = None  # UpdateChecker | None
        self._update_badge = None  # UpdateBadge | None
        self._update_check_timer = None  # QTimer for periodic checks

        # ✅ بارگذاری update config از settings
        self._load_update_config()

        # ✅ Log app start
        if audit_log is not None:
            try:
                import platform
                audit_log.info(
                    action="app_start",
                    category="SYSTEM",
                    details={"os": platform.platform(), "python": sys.version.split()[0]},
                )
            except Exception:
                pass

    def _load_update_config(self):
        """Read update config from settings.json."""
        try:
            settings_path = os.path.join(PROJECT_ROOT, "settings.json")
            if not os.path.exists(settings_path):
                return
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "update" in data and isinstance(data["update"], dict):
                self.update_config = UpdateConfig.from_dict(data["update"])
        except Exception as e:
            print(f"⚠️ _load_update_config failed: {e}")

    def _save_update_config(self):
        """Persist update config into settings.json (merge)."""
        try:
            settings_path = os.path.join(PROJECT_ROOT, "settings.json")
            data = {}
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            data["update"] = self.update_config.to_dict()
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ _save_update_config failed: {e}")

    def _setup_update_checker(self):
        """Called from _setup_ui or __init__ after UI is ready."""
        # ✅ Badge on the dashboard_btn (or a dedicated update_btn)
        try:
            self._update_badge = UpdateBadge(self.dashboard_btn, THEMES[self.current_theme])
            self._update_badge.reposition(self.dashboard_btn)
            self.dashboard_btn.installEventFilter(self)  # to reposition on resize
        except Exception as e:
            print(f"⚠️ update badge setup failed: {e}")

        # ✅ Manual check button (in topbar, next to dashboard_btn)
        try:
            self.update_btn = QPushButton("🔄")
            self.update_btn.setObjectName("dashboard_btn")  # reuse style
            self.update_btn.setFixedSize(36, 36)
            self.update_btn.clicked.connect(self._manual_check_for_updates)
            attach_sound(self.update_btn, "simpleClick")
            self.update_btn.setToolTip(
                make_tooltip(
                    title="🔄 " + tr(self.current_lang, "check_for_updates"),
                    description=tr(self.current_lang, "check_for_updates_desc"),
                    theme=THEMES[self.current_theme],
                )
            )
            # Insert after dashboard_btn
            parent_layout = self.dashboard_btn.parent().layout()
            if parent_layout:
                idx = parent_layout.indexOf(self.dashboard_btn)
                parent_layout.insertWidget(idx + 1, self.update_btn)
        except Exception as e:
            print(f"⚠️ update_btn setup failed: {e}")

        # ✅ Auto-check on startup (if enabled)
        if self.update_config.auto_check:
            QTimer.singleShot(3000, self._auto_check_for_updates)

        # ✅ Periodic check
        if self.update_config.auto_check:
            interval_ms = max(1, self.update_config.check_interval_hours) * 3600 * 1000
            self._update_check_timer = QTimer(self)
            self._update_check_timer.timeout.connect(self._auto_check_for_updates)
            self._update_check_timer.start(interval_ms)

    def _auto_check_for_updates(self):
        """Silent background check — no error dialogs."""
        self._check_for_updates(silent=True)

    def _manual_check_for_updates(self):
        """User-initiated check — shows dialogs."""
        self._check_for_updates(silent=False)

    def _check_for_updates(self, silent: bool):
        if self._update_checker is not None and self._update_checker.isRunning():
            return

        # ✅ Reload config (user might have changed source)
        self._load_update_config()

        self._update_checker = UpdateChecker(
            config=self.update_config,
            current_version=CURRENT_VERSION,
            force=not silent,  # manual check = force
            parent=self,
        )
        self._update_checker.update_available.connect(
            lambda info: self._on_update_available(info, silent)
        )
        self._update_checker.up_to_date.connect(
            lambda: self._on_up_to_date(silent)
        )
        self._update_checker.check_failed.connect(
            lambda reason: self._on_update_check_failed(reason, silent)
        )
        self._update_checker.start()

    def _on_update_available(self, info, silent: bool):
        self._update_info = info
        # ✅ Save last_check
        self._save_update_config()

        # ✅ Show badge
        if self._update_badge:
            self._update_badge.show_badge()

        if silent:
            # Toast notification
            self._notify(
                "info",
                tr(self.current_lang, "update_available"),
                f"v{info.version}",
                duration_ms=5000,
            )
            # Add click handler to badge? Already handled via update_btn
        else:
            # Full dialog (auto-download for critical)
            self._show_update_dialog(info)

    def _on_up_to_date(self, silent: bool):
        self._save_update_config()
        if self._update_badge:
            self._update_badge.hide_badge()
        if not silent:
            show_message(
                self, "information",
                tr(self.current_lang, "up_to_date"),
                tr(self.current_lang, "you_have_latest"),
            )

    def _on_update_check_failed(self, reason: str, silent: bool):
        print(f"⚠️ Update check failed: {reason}")
        if not silent:
            show_message(
                self, "warning",
                tr(self.current_lang, "update_check_failed"),
                tr(self.current_lang, "check_internet"),
            )

    def _show_update_dialog(self, info):
        dialog = UpdateDialog(
            info=info,
            current_version=CURRENT_VERSION,
            theme=THEMES[self.current_theme],
            lang=self.current_lang,
            parent=self,
            auto_download=info.is_critical,
        )
        result = dialog.exec_()

        # ✅ Handle custom result codes
        if result == 2:  # skip
            self.update_config.skipped_version = info.version
            self._save_update_config()
            if self._update_badge:
                self._update_badge.hide_badge()
        elif result == 3:  # later
            pass
        elif result == QDialog.Accepted:
            if self._update_badge:
                self._update_badge.hide_badge()

    def eventFilter(self, obj, event):
        """Reposition badge on dashboard_btn resize."""
        from PyQt5.QtCore import QEvent
        if obj is getattr(self, "dashboard_btn", None) and event.type() == QEvent.Resize:
            try:
                if self._update_badge:
                    self._update_badge.reposition(self.dashboard_btn)
            except Exception:
                pass
        return super().eventFilter(obj, event)

    def _setup_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        # ✅ Micro-interaction: focus glow on all QLineEdits
        QTimer.singleShot(150, self._decorate_all_inputs)

        # ✅ Glassmorphism background (blur) یا AnimatedBackground ساده
        if getattr(self, "use_glassmorphism", True):
            self._bg_animation = GlassmorphismBackground(central, self.current_theme)
        else:
            self._bg_animation = AnimatedBackground(central, self.current_theme)
        self._bg_animation.setGeometry(central.rect())
        self._bg_animation.lower()

        # ✅ Overlay برای Drag & Drop
        # ✅ Overlay برای Drag & Drop
        self._drop_overlay = DropOverlay(central)
        self._drop_overlay.setGeometry(central.rect())
        self._drop_overlay.set_theme(THEMES[self.current_theme])
        self._drop_overlay.set_texts(
            tr(self.current_lang, "drop_here"),
            tr(self.current_lang, "drop_subtitle"),
        )
        # نکته: اینجا raise نمی‌کنیم — overlay وقتی لازم شد خودش raise می‌شود
        # (تا در حالت عادی جلوی کلیک روی دکمه‌ها را نگیرد)

        # ✅ هم‌سایز شدن overlay با پنجره
        # ✅ دوباره وصل کردن resizeEvent (از تابع مشترک)
        self._rehook_resize_events(central)


        # ✅ Notification Center
        self.notification_center = NotificationCenter(
            central,
            theme=THEMES[self.current_theme],
        )

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._setup_topbar(main_layout)
        self._setup_center(main_layout)
        self._setup_bottombar(main_layout)
        self._setup_footer(main_layout)

    # ============================================================
    # ✅ SMART SUGGESTIONS
    # ============================================================

    def _setup_smart_suggestions(self):
        """راه‌اندازی Smart Suggestions."""
        try:
            central = self.centralWidget()
            if central is None:
                return

            # ✅ Suggestion bar
            self._suggestion_bar = SmartSuggestionBar(
                central,
                theme=THEMES[self.current_theme],
                lang=self.current_lang,
            )
            self._suggestion_bar.action_clicked.connect(
                self._on_suggestion_action
            )
            self._suggestion_bar.dismissed.connect(
                self._on_suggestion_dismissed
            )
            self._suggestion_bar.setVisible(False)

            # ✅ Engine
            self._suggestion_engine = SuggestionEngine(
                storage=self._suggestion_storage,
                lang=self.current_lang,   # ← ← ← اضافه شد
                parent=self,
            )
            self._suggestion_engine.suggestion_ready.connect(
                self._on_suggestion_ready
            )

            # ✅ Activity tracker
            self._activity_timer = QTimer(self)
            self._activity_timer.timeout.connect(self._update_suggestion_context)
            self._activity_timer.start(2000)

            # ✅ نصب event filter برای track activity
            QApplication.instance().installEventFilter(self)

            print("✅ Smart Suggestions initialized")
        except Exception as e:
            print(f"⚠️ _setup_smart_suggestions failed: {e}")
            import traceback
            traceback.print_exc()

    def _update_suggestion_context(self):
        """آپدیت context موتور با وضعیت فعلی."""
        try:
            if self._suggestion_engine is None:
                return
            now = datetime.now().timestamp()
            self._suggestion_engine.update_context(
                username=self.current_user or "",
                has_image=bool(self.current_image_path),
                image_loaded_at=self._image_loaded_at,
                has_result=self._last_result_grid is not None,
                result_at=self._result_at,
                has_pdf=getattr(self, "_pdf_exported_ever", False),
                has_gradcam=getattr(self, "_gradcam_shown_ever", False),
                last_grid=self._last_result_grid,
                consecutive_same_grid=self._consecutive_same_grid,
                results_count=self._get_total_results_count(),
                history_opened=self._history_opened_ever,
                dashboard_opened=self._dashboard_opened_ever,
                settings_opened=self._settings_opened_ever,
                session_start_at=self._session_start_at,  # ← ← ← اضافه شد
                last_activity_at=self._user_activity_times.get("last", 0.0),
                current_time=now,
            )
        except Exception as e:
            print(f"⚠️ _update_suggestion_context failed: {e}")

    def _get_total_results_count(self) -> int:
        """تعداد کل نتایج کاربر."""
        try:
            if not self.current_user:
                return 0
            stats = self.stats.get_stats(self.current_user)
            return int(stats.get("total", 0))
        except Exception:
            return 0

    def _on_suggestion_ready(self, suggestion: Suggestion):
        """وقتی یه پیشنهاد آماده‌ست، نشونش بده."""
        try:
            if self._suggestion_bar is None:
                return
            # اگه Tour بازه، پیشنهاد نشون نده
            if self._onboarding_tour is not None:
                return
            self._suggestion_bar.show_suggestion(suggestion)
        except Exception as e:
            print(f"⚠️ _on_suggestion_ready failed: {e}")

    def _on_suggestion_action(self, rule_id: str):
        """کاربر روی دکمه‌ی action کلیک کرد."""
        try:
            if rule_id == "image_ready":
                self._process_image()
            elif rule_id == "pdf_ready":
                self._export_pdf()
            elif rule_id == "gradcam_hint":
                self._show_gradcam()
            elif rule_id == "repeated_grid":
                self._show_dashboard()
            elif rule_id == "history_hint":
                self._show_history()
            elif rule_id == "dashboard_hint":
                self._show_dashboard()
            elif rule_id == "settings_hint":
                self._open_settings()
            elif rule_id == "idle_help":
                if self.current_user:
                    self._show_tour_again()
            # dismiss
            if self._suggestion_engine is not None:
                self._suggestion_engine.dismiss_current()
        except Exception as e:
            print(f"⚠️ _on_suggestion_action failed: {e}")

    def _on_suggestion_dismissed(self, rule_id: str):
        """کاربر پیشنهاد رو dismiss کرد."""
        try:
            if self.current_user:
                self._suggestion_storage.mark_ignored(
                    self.current_user, rule_id
                )
            if self._suggestion_engine is not None:
                self._suggestion_engine.dismiss_current()
        except Exception as e:
            print(f"⚠️ _on_suggestion_dismissed failed: {e}")

    def eventFilter(self, obj, event):
        """ردیابی فعالیت کاربر."""
        try:
            from PyQt5.QtCore import QEvent
            if event.type() in (
                QEvent.MouseButtonPress,
                QEvent.KeyPress,
                QEvent.Wheel,
            ):
                self._user_activity_times["last"] = datetime.now().timestamp()
        except Exception:
            pass
        return super().eventFilter(obj, event)


    def _decorate_all_inputs(self):
        """Add focus glow + press scale to all existing widgets."""
        try:
            theme = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
            # ✅ QLineEdits
            for w in self.findChildren(QLineEdit):
                try:
                    add_focus_glow(w, theme)
                except Exception:
                    pass
            # ✅ QPushButtons (non-icon ones)
            for w in self.findChildren(QPushButton):
                try:
                    button_press_scale(w, pressed_scale=0.96, duration_ms=90)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ _decorate_all_inputs failed: {e}")

    def _setup_topbar(self, parent_layout):
        topbar = QFrame()
        topbar.setFixedHeight(70)
        topbar.setObjectName("topbar")

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(30, 10, 30, 10)

        self.account_btn = QPushButton()
        self.account_btn.setObjectName("account_btn")
        self.account_btn.setVisible(False)
        self.account_btn.clicked.connect(self._show_account_menu)
        attach_sound(self.account_btn, "DashboadClickButtton")
        layout.addWidget(self.account_btn)

        # ✅ Dashboard toggle button
        self.dashboard_btn = QPushButton("📊")
        self.dashboard_btn.setObjectName("dashboard_btn")
        self.dashboard_btn.setFixedSize(36, 36)
        # ✅ tooltip در _update_texts ست می‌شه (وابسته به زبان)
        self.dashboard_btn.clicked.connect(self._toggle_dashboard)
        attach_sound(self.dashboard_btn, "simpleClick")
        layout.addWidget(self.dashboard_btn)

        # ✅ History toggle button
        self.history_btn = QPushButton("🕐")
        self.history_btn.setObjectName("history_btn")
        self.history_btn.setFixedSize(36, 36)
        # ✅ tooltip در _update_texts ست می‌شه
        self.history_btn.clicked.connect(self._toggle_history)
        attach_sound(self.history_btn, "simpleClick")
        layout.addWidget(self.history_btn)

        layout.addStretch()

        self.login_btn = QPushButton(tr(self.current_lang, "login"))
        self.login_btn.setObjectName("login_btn")
        self.login_btn.clicked.connect(self._open_login)
        attach_sound(self.login_btn, "simpleClick")
        layout.addWidget(self.login_btn)

        self.signup_btn = QPushButton(tr(self.current_lang, "signup"))
        self.signup_btn.setObjectName("signup_btn")
        self.signup_btn.clicked.connect(self._open_signup)
        attach_sound(self.signup_btn, "simpleClick")
        layout.addWidget(self.signup_btn)

        parent_layout.addWidget(topbar)

    def _setup_center(self, parent_layout):
        center_container = QFrame()
        center_container.setObjectName("center_container")

        # ✅ Layout افقی: glass_window (چپ) + dashboard (راست)
        h_layout = QHBoxLayout(center_container)
        h_layout.setContentsMargins(60, 60, 20, 60)
        h_layout.setSpacing(15)

        # ============================================================
        # ✅ سمت چپ: glass_window + progress bar
        # ============================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.glass_window = QFrame()
        self.glass_window.setObjectName("glass_window")
        glass_layout = QVBoxLayout(self.glass_window)
        glass_layout.setContentsMargins(30, 30, 30, 30)
        glass_layout.setSpacing(10)

        # ✅ image_view با stretch=1 (همه‌ی فضای باقی‌مانده)
        self.image_view = ZoomableImageView(
            theme=THEMES[self.current_theme],
        )
        self.image_view.setMinimumHeight(350)
        self.image_view.set_placeholder_text("🖼️\n\n" + tr(self.current_lang, "no_image"))
        glass_layout.addWidget(self.image_view, stretch=1)

        self.image_preview = self.image_view

        # ✅ result_label (زیر تصویر)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setObjectName("result_label")
        self.result_label.setVisible(False)
        glass_layout.addWidget(self.result_label, 0)

        # ✅ دکمه‌ها توی یه HBoxLayout (کنار هم، زیر نتیجه)
        self.action_buttons_layout = QHBoxLayout()
        self.action_buttons_layout.setSpacing(10)
        self.action_buttons_layout.setAlignment(Qt.AlignCenter)

        self.export_pdf_btn = QPushButton("📄 " + tr(self.current_lang, "export_pdf"))
        self.export_pdf_btn.setObjectName("export_pdf_btn")
        self.export_pdf_btn.setFixedHeight(38)
        self.export_pdf_btn.setCursor(Qt.PointingHandCursor)
        self.export_pdf_btn.setVisible(False)
        self.export_pdf_btn.clicked.connect(self._export_pdf)
        attach_sound(self.export_pdf_btn, "ok_sound")
        self.action_buttons_layout.addWidget(self.export_pdf_btn)

        self.gradcam_btn = QPushButton("🔥 " + tr(self.current_lang, "gradcam_show"))
        self.gradcam_btn.setObjectName("gradcam_btn")
        self.gradcam_btn.setFixedHeight(38)
        self.gradcam_btn.setCursor(Qt.PointingHandCursor)
        self.gradcam_btn.setVisible(False)
        self.gradcam_btn.clicked.connect(self._show_gradcam)
        attach_sound(self.gradcam_btn, "ok_sound")
        self.action_buttons_layout.addWidget(self.gradcam_btn)

        glass_layout.addLayout(self.action_buttons_layout, 0)

        left_layout.addWidget(self.glass_window)

        # ✅ فاصله‌ی کوچیک بین کادر و progress bar
        left_layout.addSpacing(6)

        # ✅ Progress bar
        self.progress_bar = ProcessingProgressBar(
            theme=THEMES[self.current_theme],
        )
        left_layout.addWidget(self.progress_bar)

        h_layout.addWidget(left_widget, stretch=1)

        # ============================================================
        # ✅ سمت راست: Dashboard (اول مخفی)
        # ============================================================
        self.dashboard = DashboardPanel(
            theme=THEMES[self.current_theme],
            lang=self.current_lang,
            parent=center_container,
        )
        # ✅ callback برای دکمه‌ی clear
        self.dashboard.set_clear_callback(self._clear_dashboard_stats)
        # ✅ سیگنال close
        self.dashboard.closed.connect(self._hide_dashboard)
        # ✅ اول مخفی
        self.dashboard.setVisible(False)
        h_layout.addWidget(self.dashboard)

        # ============================================================
        # ✅ History Panel (drawer از سمت راست — بعد از dashboard)
        # ============================================================
        self.history_panel = HistoryPanel(
            theme=THEMES[self.current_theme],
            lang=self.current_lang,
            username=self.current_user or "",
            parent=center_container,
        )
        self.history_panel.closed.connect(self._hide_history)
        self.history_panel.clear_requested.connect(self._clear_history)
        self.history_panel.thumbnail_clicked.connect(self._on_history_thumbnail_clicked)
        # ✅ اول مخفی
        self.history_panel.setVisible(False)
        h_layout.addWidget(self.history_panel)

        parent_layout.addWidget(center_container, stretch=1)

    def _setup_bottombar(self, parent_layout):
        bottombar = QFrame()
        bottombar.setFixedHeight(100)
        bottombar.setObjectName("bottombar")

        layout = QHBoxLayout(bottombar)
        layout.setContentsMargins(30, 15, 30, 15)

        self.visit_btn = QPushButton("🌐 " + tr(self.current_lang, "visit_webpage"))
        self.visit_btn.setObjectName("bottom_btn")
        self.visit_btn.clicked.connect(self._visit_webpage)
        attach_sound(self.visit_btn, "linkClick")
        layout.addWidget(self.visit_btn)

        layout.addStretch()

        self.insert_btn = AnimatedIconButton("📷 " + tr(self.current_lang, "insert_sono"))
        self.insert_btn.setObjectName("action_btn")
        self.insert_btn.set_ripple_radius(15)
        self.insert_btn.clicked.connect(self._insert_image)
        attach_sound(self.insert_btn, "simpleClick")
        layout.addWidget(self.insert_btn)

        self.process_btn = AnimatedIconButton("⚡ " + tr(self.current_lang, "process"))
        self.process_btn.setObjectName("action_btn")
        self.process_btn.set_ripple_radius(15)
        self.process_btn.clicked.connect(self._process_image)
        attach_sound(self.process_btn, "processButton")
        layout.addWidget(self.process_btn)

        layout.addStretch()

        self.mute_btn = AnimatedIconButton("🔊 " + tr(self.current_lang, "mute"))
        self.mute_btn.setObjectName("bottom_btn")
        self.mute_btn.set_ripple_radius(15)
        self.mute_btn.clicked.connect(self._toggle_mute)
        attach_sound(self.mute_btn, "muteClick")
        layout.addWidget(self.mute_btn)

        parent_layout.addWidget(bottombar)

    def _setup_footer(self, parent_layout):
        footer = QFrame()
        footer.setFixedHeight(28)
        footer.setObjectName("footer")

        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setAlignment(Qt.AlignCenter)

        self.developer_footer = QLabel("developed by : babak yousefian")
        self.developer_footer.setAlignment(Qt.AlignCenter)
        self.developer_footer.setObjectName("developer_footer")
        footer_layout.addWidget(self.developer_footer)

        parent_layout.addWidget(footer)

    # ============================================================
    # ✅ KEYBOARD SHORTCUTS
    # ============================================================

    def _load_saved_settings(self):
        """بارگذاری تنظیمات ذخیره‌شده از فایل settings.json."""
        try:
            settings_path = os.path.join(PROJECT_ROOT, "settings.json")
            if not os.path.exists(settings_path):
                return
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "lang" in data and data["lang"] in TRANSLATIONS:
                self.current_lang = data["lang"]
            if "theme" in data and data["theme"] in THEMES:
                self.current_theme = data["theme"]
            if "follow_system" in data:
                self.follow_system_theme = bool(data["follow_system"])
            if "glassmorphism" in data:
                self.use_glassmorphism = bool(data["glassmorphism"])
        except Exception as e:
            print(f"⚠️ _load_saved_settings failed: {e}")

    def _load_voice_enabled_setting(self) -> bool:
        """لود تنظیم voice از settings.json (پیشفرض: True)."""
        try:
            settings_path = os.path.join(PROJECT_ROOT, "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return bool(data.get("voice_enabled", True))
        except Exception:
            pass
        return True

    def _save_voice_enabled_setting(self):
        """ذخیرهی تنظیم voice در settings.json."""
        try:
            settings_path = os.path.join(PROJECT_ROOT, "settings.json")
            data = {}
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["voice_enabled"] = bool(self._voice_enabled)
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ _save_voice_enabled_setting failed: {e}")

    def _speak(self, key: str, **kwargs):
        """
        Helper: صداگذاری یک key با ترجمهی زبان فعلی.

        Usage:
            self._speak("voice_welcome", user="Babak")
        """
        try:
            if not HAS_VOICE or self._voice is None:
                return
            if not self._voice_enabled:
                return
            text = tr(self.current_lang, key)
            if text == key:
                return  # key پیدا نشد
            # ✅ format کردن
            try:
                text = text.format(**kwargs)
            except Exception:
                pass
            self._voice.set_language(self.current_lang)
            self._voice.speak(text)
        except Exception as e:
            print(f"⚠️ _speak failed: {e}")

    def _toggle_voice(self):
        """Toggle voice on/off (از Account menu یا Settings)."""
        try:
            self._voice_enabled = not self._voice_enabled
            if self._voice is not None:
                self._voice.set_enabled(self._voice_enabled)
            self._save_voice_enabled_setting()
            # ✅ فیدبک صوتی
            self._speak("voice_voice_enabled" if self._voice_enabled else "voice_voice_disabled")
            # ✅ Toast
            self._notify(
                "success" if self._voice_enabled else "info",
                "🔊 Voice",
                tr(self.current_lang, "voice_voice_enabled" if self._voice_enabled else "voice_voice_disabled"),
                duration_ms=2500,
            )
        except Exception as e:
            print(f"⚠️ _toggle_voice failed: {e}")


    def _save_settings(self):
        """ذخیره‌ی تنظیمات به فایل settings.json."""
        try:
            settings_path = os.path.join(PROJECT_ROOT, "settings.json")
            data = {
                "lang": self.current_lang,
                "theme": self.current_theme,
                "follow_system": self.follow_system_theme,
                "glassmorphism": getattr(self, "use_glassmorphism", True),
            }
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ _save_settings failed: {e}")

    def _setup_shortcuts(self):
        """میانبرهای کیبورد برای دسترسی سریع."""
        shortcuts = []

        # ✅ Ctrl+O → Insert Image
        sc_insert = QShortcut(QKeySequence("Ctrl+O"), self)
        sc_insert.setContext(Qt.ApplicationShortcut)
        sc_insert.activated.connect(self._insert_image)
        shortcuts.append(sc_insert)

        # ✅ Ctrl+Enter → Process
        sc_process = QShortcut(QKeySequence("Ctrl+Return"), self)
        sc_process.setContext(Qt.ApplicationShortcut)
        sc_process.activated.connect(self._process_image)
        shortcuts.append(sc_process)

        # ✅ Ctrl+L → Login
        sc_login = QShortcut(QKeySequence("Ctrl+L"), self)
        sc_login.setContext(Qt.ApplicationShortcut)
        sc_login.activated.connect(self._open_login)
        shortcuts.append(sc_login)

        # ✅ Ctrl+, → Settings
        sc_settings = QShortcut(QKeySequence("Ctrl+,"), self)
        sc_settings.setContext(Qt.ApplicationShortcut)
        sc_settings.activated.connect(self._open_settings)
        shortcuts.append(sc_settings)

        # ✅ Esc → Close active dialog
        sc_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        sc_esc.setContext(Qt.ApplicationShortcut)
        sc_esc.activated.connect(self._close_active_dialog)
        shortcuts.append(sc_esc)

        # ✅ F1 → Help / About
        sc_help = QShortcut(QKeySequence("F1"), self)
        sc_help.setContext(Qt.ApplicationShortcut)
        sc_help.activated.connect(self._show_help)
        shortcuts.append(sc_help)

        # ✅ ذخیره در self برای جلوگیری از garbage collection
        self._shortcuts = shortcuts

    def _setup_auto_lock(self):
        """Initialize auto-lock manager."""
        try:
            # ✅ Force default timeout to 2 min if config has bad value
            if not isinstance(self._auto_lock_config.timeout_minutes, int) \
                    or self._auto_lock_config.timeout_minutes < 1 \
                    or self._auto_lock_config.timeout_minutes > 60:
                self._auto_lock_config.timeout_minutes = 2

            self._auto_lock = AutoLockManager(
                config=self._auto_lock_config,
                parent=self,
            )
            self._auto_lock.timeout_reached.connect(self._lock_session)
            self._auto_lock.timeout_warning.connect(self._on_auto_lock_warning)
            print(f"✅ AutoLock manager ready (timeout={self._auto_lock_config.timeout_minutes} min)")

            # ✅ If user is already logged in (e.g. restored session), start immediately
            if self.current_user is not None:
                self._auto_lock.start()
                print(f"🔄 AutoLock started for existing session ({self.current_user})")
        except Exception as e:
            print(f"⚠️ _setup_auto_lock failed: {e}")
            import traceback
            traceback.print_exc()

    def showEvent(self, event):
        """Start auto-lock whenever main window becomes visible."""
        super().showEvent(event)
        try:
            if (self.current_user is not None
                    and self._auto_lock is not None
                    and not self._auto_lock.is_active()):
                self._auto_lock.set_config(self._auto_lock_config)
                self._auto_lock.start()
                print(f"🔄 AutoLock restarted (timeout={self._auto_lock_config.timeout_minutes} min)")
        except Exception as e:
            print(f"⚠️ showEvent auto_lock start failed: {e}")

    def _on_auto_lock_warning(self, seconds_left: int):
        """Called 60s / 30s / 10s before lock."""
        try:
            if self._lock_screen is not None and self._lock_screen.isVisible():
                return
            self._notify(
                "warning",
                tr(self.current_lang, "auto_lock_warning_title"),
                tr(self.current_lang, "auto_lock_warning_body").format(
                    seconds=seconds_left
                ),
                duration_ms=5000,
            )
        except Exception:
            pass

    def _lock_session(self):
        """Called when auto-lock timeout is reached."""
        # ✅ اگه Tour بازه، ببندش
        if self._onboarding_tour is not None:
            try:
                self._onboarding_tour._finish(skipped=True)
            except Exception:
                pass
            self._onboarding_tour = None

        if not self.current_user:
            return
        if self._lock_screen is not None and self._lock_screen.isVisible():
            return

        if audit_log is not None:
            try:
                audit_log.warning(
                    action="session_locked",
                    category="AUTH",
                    user=self.current_user,
                )
            except Exception:
                pass

        try:
            from gui.widgets.lock_screen import LockScreen

            # ✅ Pause auto-lock while lock screen is open
            if self._auto_lock is not None:
                self._auto_lock.set_lock_screen_open(True)

            self._lock_screen = LockScreen(
                username=self.current_user,
                db=self.db,
                theme=THEMES[self.current_theme],
                lang=self.current_lang,
                parent=self,
            )
            self._lock_screen.unlock_successful.connect(self._on_unlock_success)
            self._lock_screen.logout_requested.connect(self._on_lock_logout)
            self._lock_screen.exec_()
        except Exception as e:
            print(f"⚠️ _lock_session failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if self._lock_screen is not None:
                    self._lock_screen.deleteLater()
            except Exception:
                pass
            self._lock_screen = None
            # ✅ Resume auto-lock after lock screen closes
            if self._auto_lock is not None:
                self._auto_lock.set_lock_screen_open(False)

    def _on_unlock_success(self):
        """Called after successful unlock."""
        if self._auto_lock is not None:
            self._auto_lock.resume()

        self._notify(
            "success",
            tr(self.current_lang, "session_unlocked"),
            "",
            duration_ms=2000,
        )

        if audit_log is not None:
            try:
                audit_log.info(
                    action="session_unlocked",
                    category="AUTH",
                )
            except Exception:
                pass

    def _on_lock_logout(self):
        """Called when user clicks Logout on lock screen."""
        try:
            # ✅ Logout but DON'T delete account (fix old bug)
            if self.current_user:
                pass  # Don't delete — just log out
            self.current_user = None
            self.account_btn.setVisible(False)
            self.login_btn.setVisible(True)
            self.signup_btn.setVisible(True)
            self._update_all_tooltips()

            if self.history_panel is not None:
                self.history_panel.set_username("")
                self.history_panel.refresh([])

            if self._auto_lock is not None:
                self._auto_lock.stop()

            self._notify(
                "info",
                tr(self.current_lang, "logout"),
                "",
                duration_ms=2500,
            )
        except Exception as e:
            print(f"⚠️ _on_lock_logout failed: {e}")


    # ============================================================
    # ✅ DASHBOARD — Toggle, Show, Hide, Clear, Refresh
    # ============================================================

    def _toggle_dashboard(self):
        """باز/بستن داشبورد با انیمیشن."""
        if self.dashboard is None:
            return
        if self._dashboard_visible:
            self._hide_dashboard()
        else:
            self._show_dashboard()

    def _show_dashboard(self):
        """نمایش داشبورد با انیمیشن اسلاید."""
        self._dashboard_opened_ever = True
        if self.dashboard is None:
            return
        try:
            # ✅ آمار رو تازه کن (اگه کاربر لاگین هست)
            self._refresh_dashboard_stats()

            self.dashboard.setVisible(True)
            self.dashboard.setMaximumWidth(0)
            self.dashboard.setMinimumWidth(0)

            # ✅ انیمیشن باز شدن: از ۰ به ۳۰۰
            anim = QPropertyAnimation(self.dashboard, b"maximumWidth", self)
            anim.setDuration(280)
            anim.setStartValue(0)
            anim.setEndValue(300)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.valueChanged.connect(self._dashboard_anim_step)
            anim.finished.connect(lambda: self._dashboard_anim_finished(300))
            anim.start()
            self._dashboard_anim = anim

            self._dashboard_visible = True
            # ✅ آیکون دکمه رو عوض کن (حس باز بودن)
            try:
                self.dashboard_btn.setText("📊")
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ show_dashboard failed: {e}")

    def _hide_dashboard(self):
        """مخفی کردن داشبورد با انیمیشن اسلاید."""
        if self.dashboard is None or not self.dashboard.isVisible():
            self._dashboard_visible = False
            return
        try:
            anim = QPropertyAnimation(self.dashboard, b"maximumWidth", self)
            anim.setDuration(240)
            anim.setStartValue(self.dashboard.width())
            anim.setEndValue(0)
            anim.setEasingCurve(QEasingCurve.InCubic)
            anim.valueChanged.connect(self._dashboard_anim_step)
            anim.finished.connect(lambda: self._dashboard_anim_finished(0))
            anim.start()
            self._dashboard_anim = anim

            self._dashboard_visible = False
        except Exception as e:
            print(f"⚠️ hide_dashboard failed: {e}")

    def _dashboard_anim_step(self, value):
        """در حین انیمیشن، minimum و maximum رو هم‌زمان تغییر بده."""
        try:
            if self.dashboard is not None:
                self.dashboard.setMinimumWidth(int(value))
                self.dashboard.setMaximumWidth(int(value))
        except Exception:
            pass

    def _dashboard_anim_finished(self, final_width: int):
        """بعد از پایان انیمیشن."""
        try:
            if self.dashboard is None:
                return
            if final_width <= 0:
                # ✅ کاملاً بسته شد → مخفی کن، عرض رو ری‌ست کن
                self.dashboard.setVisible(False)
                self.dashboard.setMaximumWidth(300)
                self.dashboard.setMinimumWidth(300)
            else:
                # ✅ کاملاً باز شد → عرض ثابت
                self.dashboard.setMaximumWidth(300)
                self.dashboard.setMinimumWidth(300)
        except Exception:
            pass

    def _clear_dashboard_stats(self):
        """پاک کردن آمار کاربر (با تأیید)."""
        if not self.current_user:
            # ✅ اگه لاگین نیست، فقط داشبورد رو ری‌ست کن
            try:
                if self.dashboard is not None:
                    empty = {
                        "total": 0,
                        "counts": {"0": 0, "1": 0, "2": 0, "3": 0},
                        "avg_confidence": 0.0,
                        "recent": [],
                        "distribution": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                    }
                    self.dashboard.update_stats(empty)
            except Exception:
                pass
            return

        # ✅ تأیید از کاربر
        ok = ask_question(
            self,
            tr(self.current_lang, "dashboard_title"),
            tr(self.current_lang, "dash_clear_confirm"),
        )
        if not ok:
            return

        try:
            self.stats.clear_user(self.current_user)
            self._refresh_dashboard_stats()
            self._notify(
                "success",
                tr(self.current_lang, "dash_cleared"),
                "",
                duration_ms=2500,
            )
            # ✅ Audit log
            if audit_log is not None:
                try:
                    audit_log.warning(
                        action="stats_cleared",
                        category="DATA",
                    )
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ clear_dashboard_stats failed: {e}")

    def _refresh_dashboard_stats(self):
        """آمار رو از `UserStats` بخون و به `dashboard` بده."""
        if self.dashboard is None:
            return
        try:
            if self.current_user:
                stats = self.stats.get_stats(self.current_user)
            else:
                # ✅ اگه لاگین نیست، آمار خالی
                stats = {
                    "total": 0,
                    "counts": {"0": 0, "1": 0, "2": 0, "3": 0},
                    "avg_confidence": 0.0,
                    "recent": [],
                    "distribution": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                }
            self.dashboard.update_stats(stats)
        except Exception as e:
            print(f"⚠️ refresh_dashboard_stats failed: {e}")

    # ============================================================
    # ✅ HISTORY PANEL — Toggle, Show, Hide, Refresh, Click, Clear
    # ============================================================

    def _toggle_history(self):
        """باز/بستن History Panel با انیمیشن."""
        if self.history_panel is None:
            return
        if self._history_visible:
            self._hide_history()
        else:
            self._show_history()

    def _show_history(self):
        """نمایش History Panel با انیمیشن اسلاید."""
        self._history_opened_ever = True
        if self.history_panel is None:
            return
        try:
            # ✅ لود و رفرش تاریخچه (اگه کاربر لاگین هست)
            self._refresh_history_panel()

            self.history_panel.setVisible(True)
            self.history_panel.setMaximumWidth(0)
            self.history_panel.setMinimumWidth(0)

            # ✅ انیمیشن باز شدن: از ۰ به ۳۲۰
            anim = QPropertyAnimation(self.history_panel, b"maximumWidth", self)
            anim.setDuration(280)
            anim.setStartValue(0)
            anim.setEndValue(320)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.valueChanged.connect(self._history_anim_step)
            anim.finished.connect(lambda: self._history_anim_finished(320))
            anim.start()
            self._history_anim = anim

            self._history_visible = True
        except Exception as e:
            print(f"⚠️ show_history failed: {e}")

    def _hide_history(self):
        """مخفی کردن History Panel با انیمیشن اسلاید."""
        if self.history_panel is None or not self.history_panel.isVisible():
            self._history_visible = False
            return
        try:
            anim = QPropertyAnimation(self.history_panel, b"maximumWidth", self)
            anim.setDuration(240)
            anim.setStartValue(self.history_panel.width())
            anim.setEndValue(0)
            anim.setEasingCurve(QEasingCurve.InCubic)
            anim.valueChanged.connect(self._history_anim_step)
            anim.finished.connect(lambda: self._history_anim_finished(0))
            anim.start()
            self._history_anim = anim

            self._history_visible = False
        except Exception as e:
            print(f"⚠️ hide_history failed: {e}")

    def _history_anim_step(self, value):
        """در حین انیمیشن، minimum و maximum رو هم‌زمان تغییر بده."""
        try:
            if self.history_panel is not None:
                self.history_panel.setMinimumWidth(int(value))
                self.history_panel.setMaximumWidth(int(value))
        except Exception:
            pass

    def _history_anim_finished(self, final_width: int):
        """بعد از پایان انیمیشن."""
        try:
            if self.history_panel is None:
                return
            if final_width <= 0:
                # ✅ کاملاً بسته شد
                self.history_panel.setVisible(False)
                self.history_panel.setMaximumWidth(320)
                self.history_panel.setMinimumWidth(320)
            else:
                # ✅ کاملاً باز شد
                self.history_panel.setMaximumWidth(320)
                self.history_panel.setMinimumWidth(320)
        except Exception:
            pass

    def _refresh_history_panel(self):
        """لود تاریخچه از `HistoryManager` و refresh پنل."""
        if self.history_panel is None:
            return
        try:
            if self.current_user:
                records = self.history_mgr.load_history(self.current_user)
            else:
                records = []
            self.history_panel.set_username(self.current_user or "")
            self.history_panel.refresh(records)
        except Exception as e:
            print(f"⚠️ refresh_history_panel failed: {e}")

    def _on_history_thumbnail_clicked(self, record: dict):
        """
        کاربر روی یه thumbnail کلیک کرد → لود تصویر و نتیجه‌ی قبلی.
        """
        if not record:
            return
        try:
            # ✅ لود تصویر
            img_path = record.get("image_path", "")
            if img_path and os.path.isfile(img_path):
                self._load_image_from_path(img_path)

            # ✅ نمایش نتیجه
            grid = int(record.get("grid", 0))
            conf = float(record.get("confidence", 0.0))
            self._last_result_grid = grid
            self._last_result_confidence = conf
            # ✅ برای PDF، همه‌ی احتمالات رو نداریم (فقط grid/conf) — پس فقط همون رو بذار
            self._last_all_probs = {grid: conf}

            grid_text = tr(self.current_lang, f"grid_{grid}")
            conf_pct = f"{conf * 100:.2f}%"
            is_rtl = self.current_lang in ("Persian", "Arabic", "Morocco")
            if is_rtl:
                conf_pct = "\u200f" + conf_pct

            result_text = f"🎯 {tr(self.current_lang, 'result')}: {grid_text}\n"
            result_text += f"📊 {tr(self.current_lang, 'confidence')}: {conf_pct}"
            self.result_label.setText(result_text)
            self.result_label.setVisible(True)

            # ✅ نمایش دکمه‌ی Export PDF
            try:
                if HAS_REPORTLAB:
                    self.export_pdf_btn.setVisible(True)
            except Exception:
                pass

            # ✅ صدای OK
            try:
                SoundManager.get().play("ok_sound")
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ history thumbnail click failed: {e}")

    def _clear_history(self):
        """پاک کردن کل تاریخچه (با تأیید)."""
        if not self.current_user:
            # ✅ اگه لاگین نیست، فقط پنل رو خالی کن
            try:
                if self.history_panel is not None:
                    self.history_panel.refresh([])
            except Exception:
                pass
            return

        # ✅ تأیید
        ok = ask_question(
            self,
            tr(self.current_lang, "history_title"),
            tr(self.current_lang, "history_clear_confirm"),
        )
        if not ok:
            return

        try:
            self.history_mgr.clear_user(self.current_user)
            self._refresh_history_panel()
            try:
                self._notify(
                    "success",
                    tr(self.current_lang, "history_cleared"),
                    "",
                    duration_ms=2500,
                )
            except Exception:
                pass
            # ✅ Audit log
            if audit_log is not None:
                try:
                    audit_log.warning(
                        action="history_cleared",
                        category="DATA",
                    )
                except Exception:
                    pass

        except Exception as e:
            print(f"⚠️ clear_history failed: {e}")

    # ============================================================
    # ✅ ICON PULSE ANIMATION (برای دکمه‌های Process و Insert)
    # ============================================================

    # ============================================================
    # ✅ ICON PULSE ANIMATION (برای دکمه‌های Process و Insert)
    # ============================================================

    def _start_icon_pulse(self, button, temp_icon: str, duration_ms: int = 1500):
        """
        آیکون دکمه رو موقتاً عوض می‌کنه (blink) و بعد از duration برمی‌گردونه.
        """
        try:
            # ✅ متن اصلی رو ذخیره کن (اگه قبلاً ذخیره نشده)
            if not hasattr(button, "_pulse_original_text"):
                button._pulse_original_text = button.text()

            # ✅ اگه یه pulse دیگه در جریانه، متوقفش کن
            if hasattr(button, "_pulse_timer") and button._pulse_timer.isActive():
                button._pulse_timer.stop()

            # ✅ متن رو با آیکون موقت جایگزین کن
            rest = button._pulse_original_text.split(" ", 1)
            rest_text = rest[1] if len(rest) > 1 else ""
            button.setText(temp_icon + rest_text)

            # ✅ بعد از duration، برگردون
            timer = QTimer(button)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._restore_icon(button))
            timer.start(duration_ms)
            button._pulse_timer = timer
        except Exception:
            pass

    def _restore_icon(self, button):
        """آیکون رو به حالت اصلی برمی‌گردونه."""
        try:
            if hasattr(button, "_pulse_original_text"):
                button.setText(button._pulse_original_text)
        except Exception:
            pass

    def _close_active_dialog(self):
        """بستن دیالوگ فعال (اگه باز باشه)."""
        try:
            modal = QApplication.activeModalWidget()
            if modal is not None:
                modal.close()
        except Exception:
            pass

    def _show_help(self):
        """نمایش دیالوگ Help / About."""
        try:
            SoundManager.get().play("simpleClick")
        except Exception:
            pass
        show_message(
            self, "information",
            tr(self.current_lang, "help_about_title"),
            tr(self.current_lang, "help_about_text"),
        )

    def _decorate_buttons(self):
        t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
        for btn in (
            self.login_btn, self.signup_btn, self.account_btn,
            self.insert_btn, self.process_btn, self.visit_btn, self.mute_btn,
        ):
            try:
                decorate_button(btn, t)
            except Exception:
                pass

    # ============================================================
    # ✅ TOOLTIP STYLE (Theme-aware)
    # ============================================================

    def _apply_tooltip_style(self):
        """
        استایل‌دهی به QToolTip با QPalette (نه stylesheet).
        QPalette.ToolTipBase = background
        QPalette.ToolTipText = text color
        """
        try:
            t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
            primary = t.get("primary", "#00e5ff")
            input_bg = t.get("input_bg", "rgba(30, 41, 59, 0.85)")
            text = t.get("text", "#f1f5f9")

            # ✅ پارس rgba به QColor
            bg_color = self._parse_rgba_to_qcolor(input_bg, fallback="#1e293b")
            text_color = QColor(text)
            border_color = QColor(primary)

            # ✅ QPalette
            palette = QApplication.palette()
            palette.setColor(QPalette.ToolTipBase, bg_color)
            palette.setColor(QPalette.ToolTipText, text_color)
            palette.setColor(QPalette.ToolTipBase, bg_color)
            # ✅ border و padding رو نمی‌شه با QPalette ست کرد، ولی background + text کافیه
            QApplication.setPalette(palette)
        except Exception as e:
            print(f"⚠️ _apply_tooltip_style failed: {e}")

    def _parse_rgba_to_qcolor(self, rgba: str, fallback: str = "#1e293b") -> QColor:
        """تبدیل رشته‌ی rgba/rgb/hex به QColor."""
        try:
            s = (rgba or "").strip()
            if s.startswith("rgba("):
                inside = s[5:-1]
                parts = [p.strip() for p in inside.split(",")]
                if len(parts) >= 3:
                    r = int(float(parts[0]))
                    g = int(float(parts[1]))
                    b = int(float(parts[2]))
                    a = float(parts[3]) if len(parts) >= 4 else 1.0
                    c = QColor(r, g, b)
                    c.setAlphaF(a)
                    return c
            if s.startswith("rgb("):
                inside = s[4:-1]
                parts = [p.strip() for p in inside.split(",")]
                if len(parts) >= 3:
                    r = int(float(parts[0]))
                    g = int(float(parts[1]))
                    b = int(float(parts[2]))
                    return QColor(r, g, b)
            if s.startswith("#"):
                return QColor(s)
        except Exception:
            pass
        return QColor(fallback)

    def _strip_tooltip_qss(self, qss: str) -> str:
        """حذف بلوک QToolTip از یه QSS رشته."""
        try:
            # ✅ پیدا کردن QToolTip { ... }
            import re
            return re.sub(r"QToolTip\s*\{[^}]*\}", "", qss).strip()
        except Exception:
            return qss

    def _apply_theme(self):
        t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])

        self.setStyleSheet(f"""
            QMainWindow {{
                background: transparent;
            }}
            QWidget#centralWidget {{
                background: transparent;
            }}
            QFrame#topbar {{
                background: transparent;
                border-bottom: 1px solid {t['glass_border']};
            }}
            QFrame#center_container {{
                background: transparent;
            }}
            QFrame#glass_window {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['glass_bg']},
                    stop:1 {t['glass_highlight']});
                border: 2px solid {t['glass_border']};
                border-radius: 30px;
            }}
            QLabel#image_preview {{
                color: {t['text_dim']};
                font-size: 24px;
                border: 2px dashed {t['glass_border']};
                border-radius: 20px;
                background: transparent;
            }}
            QLabel#result_label {{
                color: {t['primary']};
                font-size: 22px;
                font-weight: bold;
                background: transparent;
                padding: 20px;
            }}
            QFrame#bottombar {{
                background: transparent;
                border-top: 1px solid {t['glass_border']};
            }}
            QPushButton#login_btn, QPushButton#signup_btn {{
                background: transparent;
                color: {t['primary']};
                border: 2px solid {t['primary']};
                border-radius: 20px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#login_btn:hover, QPushButton#signup_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#account_btn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:0.5 {t['accent']}, stop:1 {t['secondary']});
                color: {t['button_text']};
                border: none;
                border-radius: 20px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#account_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#dashboard_btn {{
                background: {t['input_bg']};
                color: {t['primary']};
                border: 1px solid {t['primary']};
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton#dashboard_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#history_btn {{
                background: {t['input_bg']};
                color: {t['primary']};
                border: 1px solid {t['primary']};
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton#history_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#action_btn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:0.5 {t['accent']}, stop:1 {t['secondary']});
                color: {t['button_text']};
                border: none;
                border-radius: 15px;
                padding: 15px 30px;
                font-size: 15px;
                font-weight: bold;
                min-width: 180px;
            }}
            QPushButton#action_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#bottom_btn {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 15px;
                padding: 12px 20px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#bottom_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#export_pdf_btn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['success']},
                    stop:1 {t['primary']});
                color: {t['button_text']};
                border: none;
                border-radius: 12px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: bold;
                min-width: 160px;
            }}
            QPushButton#export_pdf_btn:hover {{
                background: {t['primary']};
            }}
            QFrame#footer {{
                background: transparent;
                border-top: 1px solid {t['glass_border']};
            }}
            QLabel#developer_footer {{
                font-size: 11px;
                color: {t['text_dim']};
                background: transparent;
                font-style: italic;
                padding: 4px;
            }}
        """)
        # ✅ استایل tooltip با تم فعلی
        self._apply_tooltip_style()

    # ============================================================
    # ✅ DRAG & DROP — رویدادهای پنجره
    # ============================================================

    def _is_valid_image_file(self, path: str) -> bool:
        """بررسی اینکه مسیر داده‌شده یک فایل تصویری با پسوند مجاز است."""
        if not path or not isinstance(path, str):
            return False
        if not os.path.isfile(path):
            return False
        _, ext = os.path.splitext(path)
        return ext.lower() in SUPPORTED_IMAGE_EXTS

    def _extract_image_path_from_mime(self, mime: QMimeData) -> Optional[str]:
        """اولین فایل تصویری معتبر را از MIME data بیرون می‌کشد."""
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            try:
                if not url.isLocalFile():
                    continue
                local = url.toLocalFile()
                if self._is_valid_image_file(local):
                    return local
            except Exception:
                continue
        return None

    def dragEnterEvent(self, event: QDragEnterEvent):
        """وقتی فایل وارد پنجره می‌شود."""
        if event.mimeData().hasUrls() and self._extract_image_path_from_mime(event.mimeData()):
            event.acceptProposedAction()
            # ✅ نمایش overlay با متن ترجمه‌شده و تم فعلی
            try:
                self._drop_overlay.set_theme(THEMES[self.current_theme])
                self._drop_overlay.set_texts(
                    tr(self.current_lang, "drop_here"),
                    tr(self.current_lang, "drop_subtitle"),
                )
                self._drop_overlay.setGeometry(self.centralWidget().rect())
                self._drop_overlay.fade_in()
            except Exception:
                pass
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """وقتی فایل روی پنجره حرکت می‌کند."""
        if event.mimeData().hasUrls() and self._extract_image_path_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """وقتی فایل از پنجره خارج می‌شود."""
        try:
            self._drop_overlay.fade_out()
        except Exception:
            pass
        event.accept()

    def dropEvent(self, event: QDropEvent):
        """وقتی کاربر فایل را رها می‌کند."""
        try:
            self._drop_overlay.fade_out()
        except Exception:
            pass

        path = self._extract_image_path_from_mime(event.mimeData())
        if not path:
            # ✅ فایل غیرتصویری — هشدار صوتی + پیام
            try:
                SoundManager.get().play("warningSound")
            except Exception:
                pass
            show_message(
                self, "warning",
                tr(self.current_lang, "warning"),
                tr(self.current_lang, "drop_invalid"),
            )
            event.ignore()
            return

        event.acceptProposedAction()
        self._load_image_from_path(path)

    def _start_music(self):
        """Load all music files in MUSIC_PATH's folder, shuffle, and play."""
        if not HAS_PYGAME:
            return

        music_dir = os.path.dirname(MUSIC_PATH)
        if not os.path.isdir(music_dir):
            print(f"⚠️ Music dir not found: {music_dir}")
            return

        # ✅ جمع‌آوری همه‌ی فایل‌های صوتی
        all_tracks = sorted([
            os.path.join(music_dir, f)
            for f in os.listdir(music_dir)
            if f.lower().endswith((".mp3", ".wav", ".ogg", ".flac"))
        ])
        if not all_tracks:
            print(f"⚠️ No music files in: {music_dir}")
            return

        print(f"🎵 Found {len(all_tracks)} music files")

        # ✅ صف پخش (shuffled، بدون تکرار)
        self._music_pool = list(all_tracks)
        self._music_queue = []
        self._current_track = None
        self._refill_music_queue()

        # ✅ تایمر چک پایان آهنگ (هر ۵۰۰ms)
        self._music_check_timer = QTimer(self)
        self._music_check_timer.timeout.connect(self._check_music_finished)
        self._music_check_timer.start(500)

        # ✅ اولین آهنگ
        self._play_next_track()

    def _refill_music_queue(self):
        """Refill the shuffled queue with all tracks (no repeats within a cycle)."""
        self._music_queue = list(self._music_pool)
        random.shuffle(self._music_queue)
        print(f"🔀 Music queue refilled with {len(self._music_queue)} tracks")

    def _play_next_track(self):
        """Play the next track from the queue. Refill if empty."""
        if not HAS_PYGAME:
            return
        if not self._music_queue:
            self._refill_music_queue()
            if not self._music_queue:
                return

        track = self._music_queue.pop(0)
        try:
            pygame.mixer.music.load(track)
            pygame.mixer.music.set_volume(0.3)
            pygame.mixer.music.play()
            self._current_track = track
            self.music_playing = True
            print(f"🎵 Now playing: {os.path.basename(track)}")
        except Exception as e:
            print(f"⚠️ Music play failed: {e}")
            # اگه این یکی خراب بود، برو سراغ بعدی
            QTimer.singleShot(100, self._play_next_track)

    def _check_music_finished(self):
        """If music stopped (ended naturally), play the next one."""
        if not HAS_PYGAME:
            return
        if not self.music_playing:
            return
        if self.is_muted:
            return  # موقع mute، آهنگ pause شده — نباید رد بشه
        try:
            if not pygame.mixer.music.get_busy():
                # ✅ آهنگ تموم شد → بعدی
                print("🎵 Track ended, loading next...")
                self._play_next_track()
        except Exception:
            pass

    def _toggle_mute(self):
        if not HAS_PYGAME:
            return
        # ✅ ripple خودش کار می‌کنه (AnimatedIconButton)
        # تغییر متن زیر، آیکون رو عوض می‌کنه
        if self.is_muted:
            # unmute
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.unpause()
                else:
                    self._play_next_track()
            except Exception:
                self._play_next_track()
            self.mute_btn.setText("🔊 " + tr(self.current_lang, "mute"))
            self.is_muted = False
        else:
            # mute
            try:
                pygame.mixer.music.pause()
            except Exception:
                pass
            self.mute_btn.setText("🔇 " + tr(self.current_lang, "unmute"))
            self.is_muted = True
        # ✅ به‌روزرسانی tooltip mute_btn (چون متن عوض شد)
        self._update_all_tooltips()

    def _open_login(self):
        dialog = LoginDialog(self.db, THEMES[self.current_theme], self.current_lang, self)
        dialog.login_success.connect(self._on_login_success)
        dialog.exec_()

    def _open_signup(self):
        dialog = SignupDialog(self.db, THEMES[self.current_theme], self.current_lang, self)
        dialog.exec_()

    def _on_login_success(self, username: str):
        self.current_user = username

        # ✅ ثبت زمان شروع session (برای rule 6 - settings_hint)
        try:
            self._session_start_at = datetime.now().timestamp()   # ← ← ← اضافه شد
        except Exception:
            self._session_start_at = 0.0

        # ✅ کاربر لاگین شد — دیگه Tour مهمان لازم نیست
        self._guest_tour_shown = True

        self.account_btn.setText(f"👤 {username} account")
        self.account_btn.setVisible(True)
        self.login_btn.setVisible(False)
        self.signup_btn.setVisible(False)

        # ✅ به‌روزرسانی tooltipها (نام کاربر در account_btn)
        self._update_all_tooltips()

        # ✅ به‌روزرسانی history panel با کاربر جدید
        try:
            if self.history_panel is not None:
                self.history_panel.set_username(username)
                if self._history_visible:
                    self._refresh_history_panel()
        except Exception as e:
            print(f"⚠️ Failed to update history for user: {e}")

        # ✅ Toast
        self._notify(
            "success",
            tr(self.current_lang, "welcome_user").format(user=username),
            "",
            duration_ms=3500,
        )
        # ✅ Voice feedback
        self._speak("voice_welcome", user=username)

        # ✅ 🎉 Confetti خوش‌آمدگویی (فقط اگه اولین بار این session نباشه)
        try:
            if not getattr(self, "_login_celebrated_once", False):
                self._login_celebrated_once = True
                QTimer.singleShot(300, lambda: self._celebrate("welcome"))
        except Exception:
            pass

        # ✅ Start session timeout
        if self._auto_lock is not None and username:
            self._auto_lock.set_config(self._auto_lock_config)
            self._auto_lock.start()
        # ✅ Onboarding Tour — فقط برای کاربر جدید
        try:
            if not self._onboarding_storage.has_completed(username):
                # کاربر جدیده → Tour کامل
                QTimer.singleShot(900, lambda u=username: self._start_full_tour(u))
            else:
                print(f"🎓 Onboarding Tour already completed for {username}")
        except Exception as e:
            print(f"⚠️ onboarding start failed: {e}")
        # ✅ Smart Suggestions — شروع موتور
        try:
            if self._suggestion_engine is not None:
                self._suggestion_engine.set_username(username)
                self._suggestion_engine.start()
            if self._suggestion_bar is not None:
                self._suggestion_bar.set_lang(self.current_lang)
        except Exception as e:
            print(f"⚠️ suggestions start failed: {e}")
        # ✅ Audit log
        if audit_log is not None:
            try:
                audit_log.set_user(username)
                audit_log.info(
                    action="login_success",
                    category="AUTH",
                    user=username,
                    details={"method": "password"},
                )
            except Exception:
                pass

    # ============================================================
    # ✅ ONBOARDING TOUR
    # ============================================================

    def _start_full_tour(self, username: str):
        """شروع Tour کامل برای کاربر جدید."""
        try:
            # اگه Tour قبلی بازه، ببندش
            if self._onboarding_tour is not None:
                try:
                    self._onboarding_tour._finish(skipped=True)
                except Exception:
                    pass
                self._onboarding_tour = None

            print(f"🎓 Starting full onboarding tour for {username}")

            self._onboarding_tour = OnboardingTour(
                main_window=self,
                steps=TOUR_STEPS_FULL,
                theme=THEMES[self.current_theme],
                lang=self.current_lang,
                parent=self,
            )
            self._onboarding_tour.finished.connect(
                lambda skipped, u=username: self._on_tour_finished(u, skipped)
            )
            self._onboarding_tour.start()
        except Exception as e:
            print(f"⚠️ _start_full_tour failed: {e}")
            import traceback
            traceback.print_exc()

    def _start_guest_tour(self):
        """شروع Tour کوتاه برای مهمان (2 مرحله: login + signup)."""
        try:
            if self._guest_tour_shown:
                return
            self._guest_tour_shown = True

            print("🎓 Starting guest onboarding tour")

            self._onboarding_tour = OnboardingTour(
                main_window=self,
                steps=TOUR_STEPS_GUEST,
                theme=THEMES[self.current_theme],
                lang=self.current_lang,
                parent=self,
            )
            self._onboarding_tour.finished.connect(
                lambda skipped: self._on_guest_tour_finished(skipped)
            )
            self._onboarding_tour.start()
        except Exception as e:
            print(f"⚠️ _start_guest_tour failed: {e}")

    def _on_tour_finished(self, username: str, skipped: bool):
        """وقتی Tour کامل شد (یا skip شد)."""
        try:
            # ✅ ثبت در فایل
            self._onboarding_storage.mark_completed(username, skipped=skipped)
            print(f"🎓 Onboarding tour {'skipped' if skipped else 'completed'} for {username}")

            # ✅ Toast
            try:
                msg = (tr(self.current_lang, "tour_completed_msg")
                       if not skipped
                       else tr(self.current_lang, "tour_skipped_msg"))
                if msg == "tour_completed_msg" or msg == "tour_skipped_msg":
                    msg = "Tour completed!" if not skipped else "Tour skipped"
                self._notify("success", "🎓", msg, duration_ms=3000)
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ _on_tour_finished failed: {e}")
        finally:
            self._onboarding_tour = None

    def _on_guest_tour_finished(self, skipped: bool):
        """وقتی Tour مهمان تموم شد."""
        self._onboarding_tour = None

    def _show_tour_again(self):
        """نمایش مجدد Tour (از Account Menu)."""
        if not self.current_user:
            return
        try:
            # reset storage
            self._onboarding_storage.reset(self.current_user)
        except Exception:
            pass
        self._start_full_tour(self.current_user)

    def _show_guest_tooltips(self):
        """
        برای مهمان: یه راهنمای خیلی ساده فقط روی login/signup.
        (tooltip ساده، نه Tour)
        """
        try:
            if self._guest_tour_shown:
                return
            # ✅ یه بار tooltip ساده نشون بده
            self._guest_tour_shown = True
        except Exception:
            pass

    def _show_account_menu(self):
        menu = QMenu(self)
        t = THEMES[self.current_theme]

        menu.setStyleSheet(f"""
            QMenu {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['primary']};
                border-radius: 10px;
                padding: 8px;
                font-size: 13px;
            }}
            QMenu::item {{
                padding: 10px 25px;
                border-radius: 6px;
            }}
            QMenu::item:selected {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
        """)

        # ✅ Lock Now
        action_lock = menu.addAction(
            "🔒 " + tr(self.current_lang, "lock_now"),
            self._lock_session_now,
        )
        action_lock.triggered.connect(
            lambda: SoundManager.get().play("logoutSound")
        )

        # ✅ Show Tour Again (فقط اگه کاربر لاگینه)
        if self.current_user:
            try:
                action_tour = menu.addAction(
                    "🎓 " + tr(self.current_lang, "tour_show_again"),
                    self._show_tour_again,
                )
                action_tour.triggered.connect(
                    lambda: SoundManager.get().play("simpleClick")
                )
            except Exception:
                pass

        # ✅ Settings (فقط یک بار!)
        action_settings = menu.addAction(
            "⚙️ " + tr(self.current_lang, "settings"),
            self._open_settings,
        )
        action_settings.triggered.connect(
            lambda: SoundManager.get().play("simpleClick")
        )
        
        # ✅ Voice toggle (فقط اگه VoiceManager موجود باشه)
        if HAS_VOICE:
            voice_icon = "🔊" if self._voice_enabled else "🔇"
            voice_label_key = "voice_voice_disabled" if self._voice_enabled else "voice_voice_enabled"
            # (برچسب دکمه: اگه فعاله، بزن «غیرفعال کن»)
            action_voice = menu.addAction(
                f"{voice_icon} {tr(self.current_lang, voice_label_key)}",
                self._toggle_voice,
            )
            action_voice.triggered.connect(
                lambda: SoundManager.get().play("simpleClick")
            )

        # ✅ Account Info
        action_info = menu.addAction(
            "ℹ️ " + tr(self.current_lang, "account_info"),
            self._open_account_info,
        )
        action_info.triggered.connect(
            lambda: SoundManager.get().play("simpleClick")
        )

        # ✅ Audit Log
        action_audit = menu.addAction(
            "📋 " + tr(self.current_lang, "audit_log_title"),
            self._open_audit_log,
        )
        action_audit.triggered.connect(
            lambda: SoundManager.get().play("simpleClick")
        )

        menu.addSeparator()

        # ✅ Logout
        action_logout = menu.addAction(
            "🚪 " + tr(self.current_lang, "logout"),
            self._logout,
        )
        action_logout.triggered.connect(
            lambda: SoundManager.get().play("logoutSound")
        )

        # ✅ Exit
        action_exit = menu.addAction(
            "❌ " + tr(self.current_lang, "exit"),
            self._exit_app,
        )
        action_exit.triggered.connect(
            lambda: SoundManager.get().play("signoutSound")
        )

        menu.exec_(self.account_btn.mapToGlobal(self.account_btn.rect().bottomLeft()))

    def _open_settings(self):
        self._settings_opened_ever = True
        dialog = SettingsDialog(
            self.current_lang,
            self.current_theme,
            follow_system=self.follow_system_theme,
            glassmorphism=getattr(self, "use_glassmorphism", True),
            parent=self,
        )
        # ❌ DON'T connect settings_changed — _on_settings_applied handles everything
        dialog.exec_()

    def _open_audit_log(self):
        """Open the audit log dialog."""
        try:
            from gui.dialogs.audit_log_dialog import AuditLogDialog
            dialog = AuditLogDialog(
                theme=THEMES[self.current_theme],
                lang=self.current_lang,
                current_user=self.current_user or "",
                parent=self,
            )
            dialog.exec_()
        except Exception as e:
            print(f"⚠️ _open_audit_log failed: {e}")
            import traceback
            traceback.print_exc()

    def _on_settings_changed(self, lang: str, theme: str):
        print(f"🎨 _on_settings_changed: lang={lang}, theme={theme}")
        print(f"   Current theme before: {self.current_theme}")

        self.current_lang = lang
        # ✅ RTL / LTR بر اساس زبان
        try:
            from PyQt5.QtWidgets import QApplication as _QApp
            if lang in ("Persian", "Arabic", "Morocco"):
                _QApp.setLayoutDirection(Qt.RightToLeft)
            else:
                _QApp.setLayoutDirection(Qt.LeftToRight)
        except Exception:
            pass
        self.current_theme = theme
        print(f"   Current theme after: {self.current_theme}")
        print(f"   Themes available: {list(THEMES.keys())}")

        self._apply_theme()
        print(f"   _apply_theme done")

        self._update_texts()
        print(f"   _update_texts done")

        # ✅ استایل tooltip با تم جدید
        self._apply_tooltip_style()
        # ✅ هماهنگی پس‌زمینه با تم جدید
        if hasattr(self, "_bg_animation"):
            self._bg_animation.set_theme(theme)
        # ✅ re-blur glassmorphism
        if getattr(self, "use_glassmorphism", False) and isinstance(self._bg_animation, GlassmorphismBackground):
            try:
                self._bg_animation._blurred_pixmap = None
                QTimer.singleShot(50, self._bg_animation._do_blur)
            except Exception:
                pass
        # ✅ هماهنگی ZoomableImageView با تم جدید
        if hasattr(self, "image_view"):
            self.image_view.set_theme(THEMES[self.current_theme])
        # ✅ هماهنگی progress bar با تم جدید
        if hasattr(self, "progress_bar"):
            self.progress_bar.set_theme(THEMES[self.current_theme])
        # ✅ هماهنگی dashboard با تم جدید
        if hasattr(self, "dashboard") and self.dashboard is not None:
            self.dashboard.set_theme(THEMES[self.current_theme])
        # ✅ هماهنگی history با تم جدید
        if hasattr(self, "history_panel") and self.history_panel is not None:
            self.history_panel.set_theme(THEMES[self.current_theme])
        # ✅ هماهنگی notification center با تم جدید
        # ✅ هماهنگی smart suggestion bar با تم جدید
        if hasattr(self, "_suggestion_bar") and self._suggestion_bar is not None:
            self._suggestion_bar.set_theme(THEMES[self.current_theme])
            self._suggestion_bar.set_lang(self.current_lang)
            self._suggestion_bar.reposition()
        # ✅ آپدیت زبان موتور (تا متن‌های تولیدشده ترجمه شوند)
        if hasattr(self, "_suggestion_engine") and self._suggestion_engine is not None:
            try:
                self._suggestion_engine.set_lang(self.current_lang)   # ← ← ← اضافه شد
            except Exception:
                pass
        if hasattr(self, "notification_center") and self.notification_center is not None:
            self.notification_center.set_theme(THEMES[self.current_theme])
        # ✅ refresh shadow رنگ برای همه‌ی دکمه‌ها + set theme برای AnimatedIconButton
        t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
        for btn in (
                self.login_btn, self.signup_btn, self.account_btn,
                self.insert_btn, self.process_btn, self.visit_btn, self.mute_btn,
        ):
            try:
                refresh_button_shadow(btn, t)
            except Exception:
                pass
            try:
                if isinstance(btn, AnimatedIconButton):
                    btn.set_theme(t)
            except Exception:
                pass
        self._decorate_buttons()
        self._update_all_tooltips()  # ← برای تم جدید
        self.update()
        self.repaint()
        # ✅ Toast
        self._notify(
            "success",
            tr(self.current_lang, "settings"),
            tr(self.current_lang, "settings_applied"),
            duration_ms=3000,
        )
        # ✅ Audit log
        if audit_log is not None:
            try:
                audit_log.info(
                    action="settings_changed",
                    category="SETTINGS",
                    details={"lang": lang, "theme": theme},
                )
            except Exception:
                pass

    def _on_font_changed(self, font_settings: FontSettings):
        """Called when font settings change (from Settings dialog)."""
        try:
            self._font_settings = font_settings
            # Re-apply theme (which may reference font metrics)
            self._apply_theme()
            # Re-apply tooltips (they may use font)
            self._apply_tooltip_style()
            self._update_all_tooltips()
            # Notify user
            self._notify(
                "success",
                tr(self.current_lang, "font_applied"),
                "",
                duration_ms=2500,
            )
        except Exception as e:
            print(f"⚠️ _on_font_changed failed: {e}")
            import traceback
            traceback.print_exc()

    def _on_settings_applied(self, lang: str, theme: str, follow_system: bool,
                             glassmorphism: bool = True):
        """Apply new settings (including follow_system + glassmorphism)."""
        try:
            print(f"⚙️ _on_settings_applied: lang={lang}, theme={theme}, "
                  f"follow_system={follow_system}, glassmorphism={glassmorphism}")

            old_theme = self.current_theme
            print(f"   old_theme = {old_theme}")

            self.follow_system_theme = bool(follow_system)

            # ✅ If user explicitly changed the theme, respect their choice
            # and turn OFF follow_system (so it doesn't override)
            if self.follow_system_theme and theme != old_theme:
                print(f"🎨 User explicitly chose theme: {theme} → disabling follow_system")
                self.follow_system_theme = False

            # ✅ If follow_system is still ON, apply system theme
            if self.follow_system_theme:
                system_theme = detect_system_theme()
                if system_theme != theme:
                    print(f"🔄 Follow system theme: {theme} → {system_theme}")
                    theme = system_theme

            # ✅ glassmorphism toggle
            old_glass = getattr(self, "use_glassmorphism", True)
            self.use_glassmorphism = bool(glassmorphism)
            if old_glass != self.use_glassmorphism:
                self._recreate_background()

            # ✅ Now apply the theme
            print(f"   → applying theme: {theme}")
            self._on_settings_changed(lang, theme)
            self._save_settings()
        except Exception as e:
            print(f"⚠️ _on_settings_applied failed: {e}")
            import traceback
            traceback.print_exc()

    def _recreate_background(self):
        """حذف و بازسازی پس‌زمینه (برای toggle glassmorphism)."""
        try:
            central = self.centralWidget()
            if central is None:
                return

            # 1️⃣ حذف background قدیمی
            try:
                if hasattr(self, "_bg_animation") and self._bg_animation is not None:
                    self._bg_animation.stop()
                    self._bg_animation.deleteLater()
                    self._bg_animation = None
            except Exception:
                pass

            # 2️⃣ ساخت background جدید
            if self.use_glassmorphism:
                self._bg_animation = GlassmorphismBackground(central, self.current_theme)
            else:
                self._bg_animation = AnimatedBackground(central, self.current_theme)

            # 3️⃣ تنظیم geometry و z-order
            self._bg_animation.setGeometry(central.rect())
            self._bg_animation.lower()

            # 4️⃣ دوباره وصل کردن resizeEvent (چون closure به _bg_animation قدیمی اشاره می‌کرد)
            #    ⚠️ باید resizeEvent رو دوباره از اول تعریف کنیم
            self._rehook_resize_events(central)

        except Exception as e:
            print(f"⚠️ _recreate_background failed: {e}")
            import traceback
            traceback.print_exc()

    def _rehook_resize_events(self, central):
        """
        دوباره وصل کردن resizeEvent به central (بعد از recreate background).
        این کار رو باید توی _setup_ui هم صدا بزنیم (به‌جای closure مستقیم).
        """
        def _resize_all(e):
            try:
                if hasattr(self, "_bg_animation") and self._bg_animation is not None:
                    self._bg_animation.setGeometry(central.rect())
                if hasattr(self, "_drop_overlay") and self._drop_overlay is not None:
                    self._drop_overlay.setGeometry(central.rect())
                    if self._drop_overlay.isVisible():
                        self._drop_overlay.raise_()
                if hasattr(self, "_suggestion_bar") and self._suggestion_bar is not None:
                    self._suggestion_bar.reposition()
            except Exception:
                pass
            # ✅ صدای resize اصلی QWidget
            QWidget.resizeEvent(central, e)

        central.resizeEvent = _resize_all

    def _update_texts(self):
        self.setWindowTitle(tr(self.current_lang, "app_title"))
        # ✅ آپدیت smart suggestion bar
        try:
            if self._suggestion_bar is not None:
                self._suggestion_bar.set_lang(self.current_lang)
        except Exception:
            pass
        self.login_btn.setText(tr(self.current_lang, "login"))
        self.signup_btn.setText(tr(self.current_lang, "signup"))

        if self.current_user:
            self.account_btn.setText(
                f"👤 {self.current_user} {tr(self.current_lang, 'account')}"
            )

        self.insert_btn.setText("📷 " + tr(self.current_lang, "insert_sono"))
        self.process_btn.setText("⚡ " + tr(self.current_lang, "process"))
        self.visit_btn.setText("🌐 " + tr(self.current_lang, "visit_webpage"))


        mute_icon = "🔇 " if self.is_muted else "🔊 "
        mute_key = "unmute" if self.is_muted else "mute"
        self.mute_btn.setText(mute_icon + tr(self.current_lang, mute_key))

        # ✅ هماهنگی dashboard با زبان جدید
        try:
            if hasattr(self, "dashboard") and self.dashboard is not None:
                self.dashboard.set_lang(self.current_lang)
        except Exception:
            pass
        # ✅ هماهنگی History با زبان جدید
        try:
            if hasattr(self, "history_panel") and self.history_panel is not None:
                self.history_panel.set_lang(self.current_lang)
        except Exception:
            pass

        # ✅ هماهنگی image_view با زبان جدید
        try:
            if hasattr(self, "image_view") and self.image_view is not None:
                self.image_view.set_lang(self.current_lang)
        except Exception:
            pass
        # ✅ هماهنگی دکمه‌ی Export PDF با زبان جدید
        # ✅ هماهنگی دکمه‌ی Export PDF با زبان جدید
        try:
            if hasattr(self, "export_pdf_btn"):
                self.export_pdf_btn.setText("📄 " + tr(self.current_lang, "export_pdf"))
        except Exception:
            pass
        # ✅ رفرش pulse text ذخیره‌شده (برای اینکه متن جدید حفظ بشه)
        for btn in (self.insert_btn, self.process_btn):
            try:
                if hasattr(btn, "_pulse_original_text"):
                    if btn is self.insert_btn:
                        btn._pulse_original_text = "📷 " + tr(self.current_lang, "insert_sono")
                    elif btn is self.process_btn:
                        btn._pulse_original_text = "⚡ " + tr(self.current_lang, "process")
            except Exception:
                pass

        if not self.current_image_path:
            self.image_preview.set_placeholder_text(
                "🖼️\n\n" + tr(self.current_lang, "no_image")
            )

        if self.result_label.isVisible() and self._last_result_grid is not None:
            grid = self._last_result_grid
            conf = self._last_result_confidence
            grid_text = tr(self.current_lang, f"grid_{grid}")
            result_text = (
                f"🎯 {tr(self.current_lang, 'result')}: {grid_text}\n"
                f"📊 {tr(self.current_lang, 'confidence')}: {conf * 100:.2f}%"
            )
            self.result_label.setText(result_text)

        # ✅ به‌روزرسانی همه‌ی tooltipها با زبان جدید
        self._update_all_tooltips()
        # ✅ If lock screen is open, close and reopen it with new theme/lang
        if self._lock_screen is not None and self._lock_screen.isVisible():
            try:
                self._lock_screen.reject()
            except Exception:
                pass

    # ============================================================
    # ✅ SMART TOOLTIPS — Update all in one place
    # ============================================================

    def _update_all_tooltips(self):
        """
        به‌روزرسانی همه‌ی tooltipهای MainWindow (زبان + تم فعلی).
        باید توی __init__ و _update_texts و _on_settings_changed صدا زده بشه.
        """
        try:
            t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
            lang = self.current_lang

            # ---------- Topbar ----------
            if hasattr(self, "login_btn"):
                self.login_btn.setToolTip(
                    make_tooltip(
                        title="🔑 " + tr(lang, "tooltip_title_login"),
                        description=tr(lang, "tooltip_desc_login"),
                        shortcut="Ctrl+L",
                        tip=tr(lang, "tooltip_tip_login") if "tooltip_tip_login" in TRANSLATIONS.get(lang, {}) else "",
                        theme=t,
                    )
                )

            if hasattr(self, "signup_btn"):
                self.signup_btn.setToolTip(
                    make_tooltip(
                        title="✨ " + tr(lang, "tooltip_title_signup"),
                        description=tr(lang, "tooltip_desc_signup"),
                        theme=t,
                    )
                )

            if hasattr(self, "account_btn"):
                # اگه کاربر لاگین هست، نامش رو نشون بده
                user_part = ""
                if self.current_user:
                    user_part = tr(lang, "tooltip_desc_account_user").format(user=self.current_user)
                else:
                    user_part = tr(lang, "tooltip_desc_account")
                self.account_btn.setToolTip(
                    make_tooltip(
                        title="👤 " + tr(lang, "tooltip_title_account"),
                        description=user_part,
                        theme=t,
                    )
                )

            if hasattr(self, "dashboard_btn"):
                self.dashboard_btn.setToolTip(
                    make_tooltip(
                        title="📊 " + tr(lang, "tooltip_title_dashboard"),
                        description=tr(lang, "tooltip_desc_dashboard"),
                        theme=t,
                    )
                )

            if hasattr(self, "history_btn"):
                self.history_btn.setToolTip(
                    make_tooltip(
                        title="🕐 " + tr(lang, "tooltip_title_history"),
                        description=tr(lang, "tooltip_desc_history"),
                        theme=t,
                    )
                )

            # ---------- Bottombar ----------
            if hasattr(self, "visit_btn"):
                self.visit_btn.setToolTip(
                    make_tooltip(
                        title="🌐 " + tr(lang, "tooltip_title_visit"),
                        description=tr(lang, "tooltip_desc_visit"),
                        tip=tr(lang, "tooltip_tip_visit") if "tooltip_tip_visit" in TRANSLATIONS.get(lang, {}) else "",
                        theme=t,
                    )
                )

            if hasattr(self, "insert_btn"):
                self.insert_btn.setToolTip(
                    make_tooltip(
                        title="📷 " + tr(lang, "tooltip_title_insert"),
                        description=tr(lang, "tooltip_desc_insert"),
                        shortcut="Ctrl+O",
                        tip=tr(lang, "tooltip_tip_insert") if "tooltip_tip_insert" in TRANSLATIONS.get(lang, {}) else "",
                        theme=t,
                    )
                )

            if hasattr(self, "process_btn"):
                self.process_btn.setToolTip(
                    make_tooltip(
                        title="⚡ " + tr(lang, "tooltip_title_process"),
                        description=tr(lang, "tooltip_desc_process"),
                        shortcut="Ctrl+Enter",
                        tip=tr(lang, "tooltip_tip_process") if "tooltip_tip_process" in TRANSLATIONS.get(lang, {}) else "",
                        theme=t,
                    )
                )

            if hasattr(self, "mute_btn"):
                if self.is_muted:
                    title = "🔇 " + tr(lang, "tooltip_title_unmute")
                    desc = tr(lang, "tooltip_desc_unmute")
                else:
                    title = "🔊 " + tr(lang, "tooltip_title_mute")
                    desc = tr(lang, "tooltip_desc_mute")
                self.mute_btn.setToolTip(
                    make_tooltip(title=title, description=desc, theme=t)
                )

            # ---------- Export PDF ----------
            if hasattr(self, "export_pdf_btn"):
                self.export_pdf_btn.setToolTip(
                    make_tooltip(
                        title="📄 " + tr(lang, "tooltip_title_export"),
                        description=tr(lang, "tooltip_desc_export"),
                        tip=tr(lang, "tooltip_tip_export") if "tooltip_tip_export" in TRANSLATIONS.get(lang, {}) else "",
                        theme=t,
                    )
                )

            # ---------- Dashboard & History panels (اگه ساخته شدن) ----------
            if hasattr(self, "dashboard") and self.dashboard is not None:
                try:
                    self.dashboard.set_tooltips(t, lang)
                except Exception:
                    pass
            if hasattr(self, "history_panel") and self.history_panel is not None:
                try:
                    self.history_panel.set_tooltips(t, lang)
                except Exception:
                    pass
            # ---------- Zoom controls ----------
            if hasattr(self, "image_view") and self.image_view is not None:
                try:
                    self.image_view.set_tooltips(t, lang)
                except Exception:
                    pass
            # ✅ Re-decorate inputs (theme may have changed)
            self._decorate_all_inputs()
        except Exception as e:
            print(f"⚠️ _update_all_tooltips failed: {e}")

    def _lock_session_now(self):
        """Called from account menu — immediately lock."""
        if not self.current_user:
            return
        try:
            if self._auto_lock is not None:
                self._auto_lock.pause()
        except Exception:
            pass
        self._lock_session()

    def _open_account_info(self):
        if not self.current_user:
            return
        dialog = AccountInfoDialog(
            self.db, self.current_user, THEMES[self.current_theme],
            self.current_lang, self,
        )
        dialog.exec_()

    def _logout(self):
        if ask_question(
                self, tr(self.current_lang, "logout"),
                tr(self.current_lang, "logout_confirm"),
        ):
            if audit_log is not None and self.current_user:
                try:
                    audit_log.info(
                        action="logout",
                        category="AUTH",
                        user=self.current_user,
                    )
                    audit_log.clear_user()
                except Exception:
                    pass

            # ✅ Don't delete the account — just log out
            self.current_user = None
            # ✅ توقف Smart Suggestions
            try:
                if self._suggestion_engine is not None:
                    self._suggestion_engine.stop()
                    self._suggestion_engine.set_username("")
                if self._suggestion_bar is not None:
                    self._suggestion_bar.hide_suggestion()
                self._image_loaded_at = None
                self._result_at = None
                self._consecutive_same_grid = 0
                self._session_start_at = 0.0   # ← ← ← اضافه شد
            except Exception:
                pass
            self.account_btn.setVisible(False)
            self.login_btn.setVisible(True)
            self.signup_btn.setVisible(True)
            self._update_all_tooltips()

            try:
                if self.history_panel is not None:
                    self.history_panel.set_username("")
                    self.history_panel.refresh([])
            except Exception as e:
                print(f"⚠️ Failed to clear history panel on logout: {e}")

            try:
                self._notify(
                    "info",
                    tr(self.current_lang, "logout"),
                    "",
                    duration_ms=2500,
                )
                # ✅ Voice feedback
                self._speak("voice_logged_out")
            except Exception:
                pass

            if self._auto_lock is not None:
                self._auto_lock.stop()

    def _exit_app(self):
        if ask_question(
            self, tr(self.current_lang, "exit"),
            tr(self.current_lang, "exit_confirm"),
        ):
            self.close()

    def _load_image_from_path(self, path: str):
        """بارگذاری تصویر، کراپ هوشمند، حذف لیبل، و نمایش تصویر تمیز."""
        if not path or not os.path.isfile(path):
            show_message(self, "warning", tr(self.current_lang, "warning"),
                         tr(self.current_lang, "drop_invalid"))
            return

        if not self._is_valid_image_file(path):
            show_message(self, "warning", tr(self.current_lang, "warning"),
                         tr(self.current_lang, "drop_invalid"))
            return

        # ============================================================
        # ✅ STEP 1: Preprocess with AUTO-DETECT
        # (preprocessor خودش تشخیص می‌دهد header دارد یا نه)
        # ============================================================
        cleaned_path = path
        preprocess_info = None

        if HAS_PREPROCESSOR:
            try:
                # ✅ already_cropped=None → auto-detect header/panels
                result = preprocess_medical_image(
                    path,
                    already_cropped=None,
                )
                if result.success and result.image is not None:
                    preprocess_info = result
                    import tempfile
                    tmp_dir = tempfile.gettempdir()
                    cleaned_path = os.path.join(
                        tmp_dir,
                        f"liver_clean_{os.getpid()}_{int(datetime.now().timestamp() * 1000)}.png"
                    )

                    if result.save(cleaned_path):
                        print(f"🧹 Preprocessed: {result.original_shape} → {result.final_shape}")
                        print(f"   Decision: {result.crop_decision}")
                        print(f"   Cropped: {result.cropped}, "
                              f"Labels removed: {result.labels_removed}, "
                              f"Has header: {result.has_header}, "
                              f"Has panels: {result.has_side_panels}")
                    else:
                        cleaned_path = path
                        preprocess_info = None
            except Exception as e:
                print(f"⚠️ Preprocessing exception: {e}")
                import traceback
                traceback.print_exc()

        # ============================================================
        # ✅ STEP 2: Load pixmap (ONCE)
        # ============================================================
        pixmap = QPixmap(cleaned_path)
        if pixmap.isNull() and HAS_CV2:
            try:
                img = cv2.imread(cleaned_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
            except Exception:
                pixmap = QPixmap()

        if pixmap.isNull():
            SoundManager.get().play("ErrorSound")
            show_message(self, "critical", tr(self.current_lang, "error"),
                         f"Could not load image:\n{path}")
            return

        # ============================================================
        # ✅ STEP 3: Guard check (ONCE)
        # ============================================================
        if HAS_IMAGE_GUARD:
            try:
                guard = guard_analyze_image(cleaned_path)
                is_liver = guard.get("is_liver_like", True)
                score = guard.get("score", 0.0)

                print(f"🛡️ Guard: is_liver={is_liver}, score={score:.3f}, "
                      f"reason={guard.get('reason', '')}")

                if not is_liver:
                    SoundManager.get().play("warningSound")
                    msg = tr(self.current_lang, "image_not_liver_warning")
                    if msg == "image_not_liver_warning":
                        msg = ("This image does not look like a liver ultrasound.\n\n"
                               "Please insert a proper liver ultrasound image.")
                    show_message(self, "warning", tr(self.current_lang, "warning"),
                                 f"{msg}\n\nSimilarity: {score * 100:.1f}%")
                    self._speak("voice_image_rejected")
                    return
            except Exception as e:
                print(f"⚠️ Image guard failed (fail-open): {e}")

        # ============================================================
        # ✅ STEP 4: UI
        # ============================================================
        self.current_image_path = cleaned_path
        self._image_loaded_at = datetime.now().timestamp()
        self.image_view.set_image(pixmap)

        self.result_label.setVisible(False)
        self._last_result_grid = None
        self._last_result_confidence = None
        self._last_all_probs = None
        try:
            self.export_pdf_btn.setVisible(False)
            self.gradcam_btn.setVisible(False)
        except Exception:
            pass

        try:
            SoundManager.get().play("ok_sound")
        except Exception:
            pass

        # ============================================================
        # ✅ STEP 5: Notify (ONCE)
        # ============================================================
        success_title = tr(self.current_lang, "image_loaded")
        success_detail = os.path.basename(path)
        extras = []
        if preprocess_info:
            if preprocess_info.cropped:
                extras.append("cropped")
            if preprocess_info.labels_removed > 0:
                extras.append(f"{preprocess_info.labels_removed} label(s) removed")
            if not preprocess_info.cropped and preprocess_info.labels_removed == 0:
                extras.append("no change")
        if extras:
            success_detail += f" ({', '.join(extras)})"

        self._notify("success", success_title, success_detail, duration_ms=3500)
        self._speak("voice_image_loaded")

    def _insert_image(self):
        """دیالوگ باز کردن فایل — از Ctrl+O یا دکمه."""
        # ✅ blink آیکون
        self._start_icon_pulse(self.insert_btn, "⏳ ", 1500)
        path, _ = QFileDialog.getOpenFileName(
            self, tr(self.current_lang, "insert_sono"),
            "", "Images (*.png *.jpg *.jpeg *.bmp *.dcm *.tif *.tiff)",
        )
        # ✅ برگردوندن آیکون
        self._restore_icon(self.insert_btn)
        if path:
            self._load_image_from_path(path)

    def _process_image(self):
        if not self.current_image_path:
            show_message(self, "warning", "Warning",
                         tr(self.current_lang, "insert_image_first"))
            return

        if not HAS_TORCH:
            show_message(self, "critical", "Error", "PyTorch not available")
            return

        SoundManager.get().play("processButton")
        self.result_label.setText(tr(self.current_lang, "please_wait"))
        self.result_label.setVisible(True)

        # ✅ شروع progress bar
        try:
            self.progress_bar.start()
            # ✅ Voice feedback
            self._speak("voice_processing")
            # ✅ انیمیشن آیکون: ⚡ → ⏳
            self._start_icon_pulse(self.process_btn, "⏳ ", 1500)
        except Exception:
            pass

        self.thread = ProcessThread(self.current_image_path)
        self.thread.result_ready.connect(self._on_result_ready)
        self.thread.error.connect(self._on_error)
        self.thread.start()

    def _on_result_ready(self, grid: int, confidence: float, all_probs: dict = None, heatmap=None):
        # ✅ برگردوندن آیکون
        self._restore_icon(self.process_btn)

        # ✅ 🛡️ چک نهایی: آیا نتیجه معتبره؟
        if HAS_IMAGE_GUARD and self.current_image_path:
            try:
                is_valid, reason, guard_metrics = guard_is_confident_result(
                    image_path=self.current_image_path,
                    model_confidence=confidence,
                    all_probs=all_probs,
                )

                if not is_valid:
                    # ✅ نتیجه رو نشون نده
                    SoundManager.get().play("warningSound")

                    # ✅ progress bar → خطا
                    try:
                        self.progress_bar.finish_error()
                    except Exception:
                        pass

                    # ✅ پیام
                    msg = tr(self.current_lang, "result_not_confident")
                    if msg == "result_not_confident":
                        msg = (
                            "The AI could not confidently identify this image as a liver ultrasound.\n\n"
                            "Please insert a proper liver ultrasound image and try again."
                        )

                    detail = f"\n\n🔍 Reason: {reason}"

                    # ✅ اگه مدل اطمینان کمی داشت، درصد رو هم نشون بده
                    if "confidence too low" in reason.lower():
                        detail += f"\n📊 Confidence: {confidence * 100:.1f}%"

                    show_message(
                        self, "warning",
                        tr(self.current_lang, "warning"),
                        msg + detail,
                    )

                    # ✅ مخفی کردن نتیجه
                    self.result_label.setVisible(False)
                    try:
                        self.export_pdf_btn.setVisible(False)
                        self.gradcam_btn.setVisible(False)
                    except Exception:
                        pass

                    # ✅ Audit log
                    if audit_log is not None and self.current_user:
                        try:
                            audit_log.warning(
                                action="result_rejected_low_confidence",
                                category="DATA",
                                details={
                                    "confidence": round(float(confidence), 3),
                                    "reason": reason,
                                    "metrics": guard_metrics,
                                },
                            )
                        except Exception:
                            pass

                    # ✅ پاک کردن state
                    self._last_result_grid = None
                    self._last_result_confidence = None
                    self._last_all_probs = None
                    self._last_heatmap = None
                    return
            except Exception as e:
                print(f"⚠️ Result validation failed (fail-open): {e}")

        self._last_result_grid = grid
        self._last_result_confidence = confidence
        self._last_all_probs = all_probs if all_probs else {}
        # ✅ ذخیره‌ی heatmap
        self._last_heatmap = heatmap
        # ✅ ثبت زمان نتیجه برای Smart Suggestions
        try:
            self._result_at = datetime.now().timestamp()
            self._image_loaded_at = None  # دیگه image_ready لازم نیست
            # consecutive grid
            prev_grid = getattr(self, "_prev_grid", None)
            if prev_grid == grid:
                self._consecutive_same_grid = getattr(self, "_consecutive_same_grid", 0) + 1
            else:
                self._consecutive_same_grid = 1
            self._prev_grid = grid
        except Exception:
            pass
        SoundManager.get().play("ok_sound")

        # ✅ progress bar → پرش به ۱۰۰٪ سبز
        try:
            self.progress_bar.finish_success()
        except Exception:
            pass

        # ✅ ذخیره‌ی نتیجه توی آمار (اگه کاربر لاگین هست)
        if self.current_user:
            try:
                self.stats.add_result(self.current_user, grid, confidence)
            except Exception as e:
                print(f"⚠️ Failed to save result: {e}")
            try:
                if self._dashboard_visible:
                    self._refresh_dashboard_stats()
            except Exception:
                pass

            try:
                if self.current_image_path and self.history_mgr:
                    self.history_mgr.save_result(
                        username=self.current_user,
                        image_path=self.current_image_path,
                        grid=grid,
                        confidence=confidence,
                        model_name="ResNet50 KD Student",
                        model_version="v3.0",
                    )
                if self._history_visible and self.history_panel is not None:
                    self._refresh_history_panel()
            except Exception as e:
                print(f"⚠️ Failed to save history: {e}")

        grid_text = tr(self.current_lang, f"grid_{grid}")
        conf_pct = f"{confidence * 100:.2f}%"
        is_rtl = self.current_lang in ("Persian", "Arabic", "Morocco")
        if is_rtl:
            conf_pct = "\u200f" + conf_pct

        result_text = f"🎯 {tr(self.current_lang, 'result')}: {grid_text}\n"
        result_text += f"📊 {tr(self.current_lang, 'confidence')}: {conf_pct}"
        self.result_label.setText(result_text)
        self.result_label.setVisible(True)

        # ✅ Voice feedback (نتیجه)
        try:
            grid_name = tr(self.current_lang, f"grid_{grid}")
            self._speak(
                "voice_inference_result",
                grid_name=grid_name,
                conf=f"{confidence * 100:.0f}",
            )
        except Exception:
            pass

        # ✅ نمایش دکمه‌ی Export PDF
        try:
            if HAS_REPORTLAB:
                self.export_pdf_btn.setVisible(True)
        except Exception:
            pass

        # ✅ نمایش دکمه‌ی Show Attention Map (اگه heatmap هست)
        try:
            if heatmap is not None and HAS_CV2:
                self.gradcam_btn.setVisible(True)
            else:
                self.gradcam_btn.setVisible(False)
        except Exception:
            pass

        # ✅ 🎉 Confetti Animation
        try:
            self._celebrate("success")
        except Exception:
            pass

        # ✅ Toast موفقیت
        try:
            grid_short = {0: "NL", 1: "G1", 2: "G2", 3: "G3"}.get(grid, f"G{grid}")
            self._notify(
                "success",
                tr(self.current_lang, "notif_inference_ok"),
                f"{grid_short} — {confidence * 100:.1f}%",
                duration_ms=4000,
            )
        except Exception:
            pass

        # ✅ Audit log
        if audit_log is not None and self.current_user:
            try:
                audit_log.info(
                    action="inference_completed",
                    category="DATA",
                    details={
                        "grid": int(grid),
                        "confidence": round(float(confidence), 4),
                        "image": os.path.basename(self.current_image_path or ""),
                    },
                )
            except Exception:
                pass

    def _on_error(self, error: str):
        # ✅ برگردوندن آیکون
        self._restore_icon(self.process_btn)
        SoundManager.get().play("ErrorSound")
        # ✅ progress bar → قرمز
        try:
            self.progress_bar.finish_error()
            # ✅ Voice feedback
            self._speak("voice_error")
        except Exception:
            pass
        # ✅ Toast خطا
        try:
            self._notify(
                "error",
                tr(self.current_lang, "inference_failed"),
                str(error)[:120],
                duration_ms=5000,
            )
        except Exception:
            pass
        self.result_label.setVisible(False)

    # ============================================================
    # ✅ PDF EXPORT
    # ============================================================

    def _export_pdf(self):
        """خروجی PDF از نتیجه‌ی فعلی."""
        if not HAS_REPORTLAB:
            show_message(
                self, "warning",
                tr(self.current_lang, "error"),
                "reportlab not available. Please install: pip install reportlab"
            )
            return

        if self._last_result_grid is None or not self.current_image_path:
            show_message(
                self, "warning",
                tr(self.current_lang, "warning"),
                tr(self.current_lang, "insert_image_first"),
            )
            return

        # ✅ دیالوگ انتخاب مسیر ذخیره
        try:
            default_name = f"liver_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            save_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.isdir(save_dir):
                save_dir = PROJECT_ROOT

            output_path, _ = QFileDialog.getSaveFileName(
                self,
                tr(self.current_lang, "export_pdf"),
                os.path.join(save_dir, default_name),
                "PDF Files (*.pdf)",
            )
        except Exception as e:
            print(f"⚠️ File dialog failed: {e}")
            return

        if not output_path:
            return  # کاربر لغو کرد

        # ✅ اطمینان از پسوند .pdf
        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"

        # ✅ نمایش progress
        try:
            self.progress_bar.start()
        except Exception:
            pass

        # ✅ ساخت PDF
        try:
            exporter = PDFExporter(
                theme=THEMES[self.current_theme],
                lang=self.current_lang,
            )
            ok = exporter.generate(
                output_path=output_path,
                username=self.current_user or "—",
                image_path=getattr(self, "_display_image_path", self.current_image_path),
                grid=self._last_result_grid,
                confidence=self._last_result_confidence or 0.0,
                all_probs=self._last_all_probs,
                model_name="ResNet50 KD Student",
                model_version="v3.0",
            )

            if ok:
                # ✅ موفق
                try:
                    self.progress_bar.finish_success()
                except Exception:
                    pass
                SoundManager.get().play("ok_sound")
                # ✅ باز کردن PDF (اختیاری)
                try:
                    import webbrowser
                    webbrowser.open(f"file:///{output_path}")
                except Exception:
                    pass
                self._notify(
                    "success",
                    tr(self.current_lang, "pdf_saved"),
                    output_path,
                    duration_ms=4000,
                )
                # ✅ Voice feedback
                self._speak("voice_pdf_saved")
                # ✅ 🎉 Confetti (gentle — چون inference قبلاً جشن گرفته)
                try:
                    self._celebrate("gentle")
                except Exception:
                    pass
                # ✅ ثبت pdf برای Smart Suggestions
                try:
                    self._pdf_exported_ever = True
                except Exception:
                    pass
                # ✅ Audit log
                if audit_log is not None:
                    try:
                        audit_log.info(
                            action="pdf_exported",
                            category="EXPORT",
                            details={"path": output_path},
                        )
                    except Exception:
                        pass
            else:
                # ✅ خطا
                try:
                    self.progress_bar.finish_error()
                except Exception:
                    pass
                SoundManager.get().play("ErrorSound")
                self._notify(
                    "error",
                    tr(self.current_lang, "pdf_failed"),
                    "",
                    duration_ms=4000,
                )
        except Exception as e:
            try:
                self.progress_bar.finish_error()
            except Exception:
                pass
            SoundManager.get().play("ErrorSound")
            print(f"⚠️ PDF export failed: {e}")
            import traceback
            traceback.print_exc()
            show_message(
                self, "critical",
                tr(self.current_lang, "error"),
                f"PDF failed:\n{e}",
            )

    # ============================================================
    # ✅ GRAD-CAM
    # ============================================================

    def _show_gradcam(self):
        """نمایش Grad-CAM heatmap."""
        try:
            if not hasattr(self, "_last_heatmap") or self._last_heatmap is None:
                self._notify(
                    "warning",
                    tr(self.current_lang, "gradcam_no_heatmap"),
                    "",
                    duration_ms=3000,
                )
                return
            # ✅ ثبت نمایش gradcam برای Smart Suggestions
            try:
                self._gradcam_shown_ever = True
            except Exception:
                pass

            if not self.current_image_path or not os.path.isfile(self.current_image_path):
                return

            # ✅ تصویر اصلی
            _display_path = getattr(self, "_display_image_path", self.current_image_path)
            original_pixmap = QPixmap(_display_path)
            if original_pixmap.isNull():
                return

            # ✅ heatmap overlay
            heatmap_pixmap = overlay_heatmap_on_image(
                original_pixmap,
                self._last_heatmap,
                alpha=0.45,
            )

            # ✅ دیالوگ
            viewer = HeatmapViewer(
                original_pixmap=original_pixmap,
                heatmap_pixmap=heatmap_pixmap,
                theme=THEMES[self.current_theme],
                lang=self.current_lang,
                parent=self,
            )
            viewer.exec_()
        except Exception as e:
            print(f"⚠️ _show_gradcam failed: {e}")
            import traceback
            traceback.print_exc()

    def _update_gradcam_theme(self):
        """آپدیت تم دکمه‌ی gradcam (اگه لازم بود)."""
        pass

    def _notify(self, level: str, title: str, message: str = "", duration_ms: int = 3000):
        """
        نمایش یه toast notification.
        levels: success / info / warning / error
        """
        try:
            if self.notification_center is not None:
                self.notification_center.push(
                    level=level,
                    title=title,
                    message=message,
                    duration_ms=duration_ms,
                )
        except Exception as e:
            print(f"⚠️ _notify failed: {e}")

    # ============================================================
    # ✅ CONFETTI ANIMATION
    # ============================================================

    def _celebrate(
        self,
        event_type: str = "success",
        pattern: Optional[str] = None,
        particle_count: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ):
        """
        نمایش انیمیشن confetti.

        Args:
            event_type: "success" | "gentle" | "achievement" | "welcome"
            pattern: اگه داده بشه، از event_type استفاده نمی‌شه
            particle_count: تعداد ذرات (پیش‌فرض بر اساس event_type)
            duration_ms: مدت (پیش‌فرض بر اساس event_type)
        """
        try:
            central = self.centralWidget()
            if central is None:
                return

            # ✅ انتخاب pattern
            if pattern is None:
                pattern = get_pattern_for(event_type)

            # ✅ تعداد و مدت بر اساس event_type
            if particle_count is None:
                particle_count = {
                    "success": 150,
                    "gentle": 70,
                    "achievement": 130,
                    "welcome": 100,
                }.get(event_type, 120)

            if duration_ms is None:
                duration_ms = {
                    "success": 2600,
                    "gentle": 1600,
                    "achievement": 2800,
                    "welcome": 2200,
                }.get(event_type, 2200)

            celebrate(
                parent=central,
                theme=THEMES.get(self.current_theme, {}),
                pattern=pattern,
                particle_count=particle_count,
                duration_ms=duration_ms,
            )
        except Exception as e:
            print(f"⚠️ _celebrate failed: {e}")

    def _visit_webpage(self):
        import webbrowser
        webbrowser.open("https://github.com/babakyousefian/babakyousefian/")

    def closeEvent(self, event):
        # ✅ توقف Smart Suggestions
        try:
            if self._suggestion_engine is not None:
                self._suggestion_engine.stop()
            if hasattr(self, "_activity_timer") and self._activity_timer.isActive():
                self._activity_timer.stop()
            QApplication.instance().removeEventFilter(self)
        except Exception:
            pass

        # ✅ بستن Tour اگه بازه
        try:
            if self._onboarding_tour is not None:
                self._onboarding_tour._finish(skipped=True)
                self._onboarding_tour = None
        except Exception:
            pass

        # ✅ توقف انیمیشن پس‌زمینه
        try:
            if hasattr(self, "_bg_animation"):
                self._bg_animation.stop()
        except Exception:
            pass

        # ✅ توقف تایمر چک موزیک
        try:
            if hasattr(self, "_music_check_timer") and self._music_check_timer.isActive():
                self._music_check_timer.stop()
        except Exception:
            pass

        # ✅ توقف موزیک
        try:
            if HAS_PYGAME and self.music_playing:
                pygame.mixer.music.stop()
        except Exception:
            pass

        # ✅ Audit log
        if audit_log is not None:
            try:
                audit_log.info(
                    action="app_close",
                    category="SYSTEM",
                )
                audit_log.flush()
            except Exception:
                pass

        event.accept()

        # ✅ Stop session timeout manager
        try:
            if self._auto_lock is not None:
                self._auto_lock.stop()
        except Exception:
            pass

# ================================================================
# ✅ FONT LOADER (Noto Sans for multilingual support)
# ================================================================

def load_application_fonts():
    """
    Load Noto Sans fonts from assets/fonts/ and set as application default.
    Returns the font family name that was successfully loaded, or None.
    """
    if not os.path.isdir(FONTS_DIR):
        print(f"⚠️ Fonts dir not found: {FONTS_DIR}")
        return None

    # ✅ لیست فایل‌های فونت (به ترتیب اولویت)
    font_files = [
        "NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf",
        "NotoSans-Italic.ttf",
    ]

    loaded_families = []
    for fname in font_files:
        path = os.path.join(FONTS_DIR, fname)
        if not os.path.exists(path):
            print(f"⚠️ Font file not found: {path}")
            continue
        font_id = QFontDatabase.addApplicationFont(path)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                loaded_families.extend(families)
                print(f"✅ Loaded font: {fname} → {families[0]}")
        else:
            print(f"⚠️ Failed to load font: {fname}")

    if loaded_families:
        # ✅ اولین family رو به‌عنوان default ست کن
        family = loaded_families[0]
        print(f"🎨 Setting application font to: {family}")
        return family
    return None

# ================================================================
# ✅ ENTRY POINT
# ================================================================

def main():
    # ============================================================
    # ✅ STEP 1: Check for pending update (before ANY UI)
    # ============================================================
    pending = check_update_pending()
    if pending:
        print("⚠️ Pending update detected — attempting rollback")
        rolled_back = rollback_if_needed(pending)
        if rolled_back:
            # Show a simple message box (QApplication not created yet, use raw)
            try:
                # ⚠️ QApplication و QMessageBox از global import میان — دوباره import نکن!
                _app = QApplication(sys.argv)
                QMessageBox.warning(
                    None,
                    "Update Failed",
                    "The previous update failed to start.\n\n"
                    "Your app has been automatically rolled back to the previous version.\n"
                    "Please try updating again from Settings.",
                )
                # Don't call sys.exit here — continue to normal startup
            except Exception as e:
                print(f"⚠️ Failed to show rollback message: {e}")

    # ============================================================
    # ✅ STEP 2: Normal startup (High DPI + QApplication)
    # ============================================================
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ✅ Global Exception Handler (جلوگیری از کرش و هنگ)
    def _global_exception_handler(exc_type, exc_value, exc_tb):
        try:
            import traceback
            error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"❌ Unhandled exception:\n{error_text}")

            # ✅ Audit log
            if audit_log is not None:
                try:
                    audit_log.critical(
                        action="unhandled_exception",
                        category="SYSTEM",
                        details={"type": exc_type.__name__, "message": str(exc_value)[:500]},
                    )
                except Exception:
                    pass

            try:
                if QApplication.instance() is not None:
                    box = QMessageBox()
                    box.setIcon(QMessageBox.Critical)
                    box.setWindowTitle("Unexpected Error")
                    box.setText("An unexpected error occurred. The app will continue.")
                    box.setDetailedText(error_text)
                    box.setStandardButtons(QMessageBox.Ok)
                    box.exec_()
            except Exception:
                pass
        except Exception:
            pass

    sys.excepthook = _global_exception_handler

    # ✅ لود فونت‌ها
    loaded_family = load_application_fonts()
    if loaded_family:
        font = QFont(loaded_family, 10)
        app.setFont(font)
        print(f"✅ Application font set to: {loaded_family}")
    else:
        fallback = QFont("Segoe UI", 10)
        app.setFont(fallback)

    # ✅ Apply saved font settings (overrides default)
    try:
        from core.ui.font_manager import load_font_settings, FontManager
        saved_font = load_font_settings()
        if saved_font.family or saved_font.size:
            FontManager.get().apply(saved_font)
            print(f"🎨 Saved font applied: {saved_font.family} {saved_font.size}pt")
    except Exception as e:
        print(f"⚠️ Failed to apply saved font: {e}")

    # ✅ ساخت splash و preload thread
    splash = SplashScreen()
    splash.show()

    preload = PreloadThread()

    # ✅ اتصال سیگنال‌ها
    def on_progress(percent, stage):
        splash.set_progress(percent, stage)

    def on_preload_finished(success, message):
        splash.on_preload_finished(success, message)

    def on_skip_requested():
        # ✅ اگه کاربر Skip زد، preload thread رو متوقف کن
        preload.request_skip()

    preload.progress.connect(on_progress)
    preload.finished_signal.connect(on_preload_finished)
    splash.skip_requested.connect(on_skip_requested)

    # ✅ شروع preload
    preload.start()

    # ✅ ساخت main window (بعد از splash)
    main_window = MainWindow()

    def on_splash_finished():
        try:
            main_window.show()
            # ✅ Confirm update success after 2 seconds (UI is definitely visible)
            QTimer.singleShot(2000, confirm_update_success)
        except Exception as e:
            print(f"⚠️ main_window.show() failed: {e}")
            # ✅ اگه main_window.error داشت، یه پیام
            try:
                if not HAS_TORCH:
                    show_message(
                        main_window, "warning",
                        "AI Model",
                        "AI model not loaded. You can insert images but cannot process them."
                    )
            except Exception:
                pass

    splash.finished.connect(on_splash_finished)
    splash.continue_without_ai.connect(on_splash_finished)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()