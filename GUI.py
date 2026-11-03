# GUI.py
# -*- coding: utf-8 -*-

"""
Liver AI Pipeline - GUI (ULTIMATE EDITION)

Beautiful, animated, user-friendly GUI for the Liver AI Pipeline.

Features:
- Animated splash screen (5-7 seconds)
- Login / Signup with password validation
- Account dropdown menu
- Settings (language + theme)
- Image classification with the KD Student model
- 14 languages
- 3 themes
- Music player with mute/unmute
- Beautiful animations

Author: Babak Yousefian
Version: 2.0.0
"""

import os
import sys
import json
import hashlib
import random
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

warnings.filterwarnings("ignore")

# ================================================================
# ✅ PyQt5 Imports
# ================================================================

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QDialog, QMessageBox, QFileDialog, QFrame, QGraphicsOpacityEffect,
    QSizePolicy, QSpacerItem, QScrollArea, QCheckBox, QComboBox,
    QStackedWidget, QToolButton, QMenu, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize,
    QPoint, pyqtSignal, QParallelAnimationGroup, QSequentialAnimationGroup,
    QThread,
)
from PyQt5.QtGui import (
    QPixmap, QIcon, QFont, QColor, QPalette, QPainter, QBrush,
    QLinearGradient, QRadialGradient, QPen, QFontDatabase, QTransform,
)

from PyQt5.QtWidgets import QAction  # ✅ در PyQt5 اینجاست

# ================================================================
# ✅ Optional: Music
# ================================================================

import os as _os_pg
_os_pg.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False
    print("⚠️ Pygame not available — Music disabled")

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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "finalReports", "selfSupervisedLearning", "final_models",
    "final_model.pth",
)

# Users database
USERS_DB_PATH = os.path.join(PROJECT_ROOT, "users_db.json")

# Assets directory
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
ICON_PATH = os.path.join(ASSETS_DIR, "favicon.ico")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
MUSIC_PATH = os.path.join(PROJECT_ROOT, "packages", "facilities", "assets", "musics", "music3.mp3")

os.makedirs(ASSETS_DIR, exist_ok=True)


# ================================================================
# ✅ TRANSLATIONS (14 Languages)
# ================================================================

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
    },
}


def tr(lang: str, key: str) -> str:
    """Translate a key."""
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["English"])
    return lang_dict.get(key, TRANSLATIONS["English"].get(key, key))


# ================================================================
# ✅ THEMES (3 Themes)
# ================================================================

THEMES = {
    "theme_dark_aqua": {
        "name": "Dark Aqua-Purple",
        "bg_gradient_start": "#1a1a2e",
        "bg_gradient_end": "#16213e",
        "primary": "#00d4ff",        # Aqua
        "secondary": "#a855f7",      # Purple
        "text": "#e8f4f8",
        "text_dim": "#94a3b8",
        "button_bg": "#00d4ff",
        "button_text": "#0a0a14",
        "button_hover": "#22d3ee",
        "input_bg": "#1e293b",
        "input_border": "#00d4ff",
        "glass_bg": "rgba(30, 41, 59, 0.7)",
        "glass_border": "rgba(0, 212, 255, 0.3)",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
    },
    "theme_solarized_white": {
        "name": "Solarized White",
        "bg_gradient_start": "#fdf6e3",
        "bg_gradient_end": "#eee8d5",
        "primary": "#268bd2",
        "secondary": "#6c71c4",
        "text": "#073642",
        "text_dim": "#586e75",
        "button_bg": "#268bd2",
        "button_text": "#fdf6e3",
        "button_hover": "#2aa198",
        "input_bg": "#ffffff",
        "input_border": "#93a1a1",
        "glass_bg": "rgba(255, 255, 255, 0.85)",
        "glass_border": "rgba(38, 139, 210, 0.3)",
        "success": "#859900",
        "warning": "#b58900",
        "danger": "#dc322f",
    },
    "theme_solarized_dark": {
        "name": "Solarized Dark",
        "bg_gradient_start": "#002b36",
        "bg_gradient_end": "#073642",
        "primary": "#2aa198",
        "secondary": "#6c71c4",
        "text": "#93a1a1",
        "text_dim": "#586e75",
        "button_bg": "#2aa198",
        "button_text": "#002b36",
        "button_hover": "#268bd2",
        "input_bg": "#073642",
        "input_border": "#2aa198",
        "glass_bg": "rgba(7, 54, 66, 0.75)",
        "glass_border": "rgba(42, 161, 152, 0.3)",
        "success": "#859900",
        "warning": "#b58900",
        "danger": "#dc322f",
    },
}

# ================================================================
# ✅ LANGUAGE & THEME ORDER
# ================================================================

LANGUAGE_ORDER = [
    "Persian",
    "English",
    "USA",
    "France",
    "Russian",
    "Greece",
    "Germany",
    "Arabic",
    "Ukranian",
    "Italy",
]

THEME_ORDER = [
    "theme_dark_aqua",
    "theme_solarized_white",
    "theme_solarized_dark",
]

# ================================================================
# ✅ USER DATABASE (JSON-based)
# ================================================================

class UserDB:
    """Simple JSON-based user database."""

    def __init__(self, path: str = USERS_DB_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"users": {}}
        return {"users": {}}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def _hash(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

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
        return users[username]["password_hash"] == self._hash(password)

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


# ================================================================
# ✅ PASSWORD VALIDATION
# ================================================================

import re


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password according to rules:
    - 8-16 characters
    - At least 1 lowercase letter
    - At least 1 uppercase letter
    - At least 4 digits
    - At least 4 special characters
    """
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
    """Simple email validation."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# ================================================================
# ✅ MODEL LOADER (Lazy)
# ================================================================

class ModelLoader:
    """Lazy loader for the final_model.pth."""

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
        # ✅ همه‌ی printها حذف شدن


# ================================================================
# ✅ IMAGE PREPROCESSING
# ================================================================

def preprocess_image(image_path: str, input_size: int = 224) -> "torch.Tensor":
    """Preprocess an image for model input."""
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
# ✅ ANIMATED SPLASH SCREEN
# ================================================================

class SplashScreen(QWidget):
    """Beautiful animated splash screen (5-7 seconds)."""

    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Screen center
        screen = QApplication.primaryScreen().geometry()
        w, h = 800, 500
        self.setGeometry(
            (screen.width() - w) // 2,
            (screen.height() - h) // 2,
            w, h,
        )

        # Main layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container with rounded corners + gradient background
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

        # Logo Label
        self.logo_label = QLabel("🧬")
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("""
            font-size: 140px;
            background: transparent;
        """)
        container_layout.addWidget(self.logo_label)

        # Title
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

        # Subtitle
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

        # Progress bar (animated)
        self.progress_container = QFrame()
        self.progress_container.setFixedSize(500, 8)
        self.progress_container.setStyleSheet("""
            background: rgba(0, 212, 255, 0.15);
            border-radius: 4px;
        """)
        container_layout.addSpacing(30)
        container_layout.addWidget(self.progress_container, 0, Qt.AlignCenter)

        # Animated progress bar
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

        # Loading text
        self.loading_label = QLabel("Loading...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            font-size: 14px;
            color: #94a3b8;
            background: transparent;
            margin-top: 10px;
            font-family: 'Segoe UI', Arial;
        """)
        container_layout.addSpacing(15)
        # ---- Developed by ----
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

        # Apply drop shadow to container
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(60)
        shadow.setColor(QColor(0, 212, 255, 100))
        shadow.setOffset(0, 0)
        self.container.setGraphicsEffect(shadow)

        # Animations
        self._start_animations()

    def _start_animations(self):
        # 1. Progress bar animation (0 to 500 over 6 seconds)
        self.progress_anim = QPropertyAnimation(self.progress_bar, b"geometry")
        self.progress_anim.setDuration(6000)
        self.progress_anim.setStartValue(QRect(0, 0, 0, 8))
        self.progress_anim.setEndValue(QRect(0, 0, 500, 8))
        self.progress_anim.setEasingCurve(QEasingCurve.InOutQuad)

        # 2. Logo pulse animation
        self.logo_opacity = QGraphicsOpacityEffect(self.logo_label)
        self.logo_label.setGraphicsEffect(self.logo_opacity)
        self.logo_pulse = QPropertyAnimation(self.logo_opacity, b"opacity")
        self.logo_pulse.setDuration(1500)
        self.logo_pulse.setStartValue(0.4)
        self.logo_pulse.setEndValue(1.0)
        self.logo_pulse.setEasingCurve(QEasingCurve.InOutSine)
        self.logo_pulse.setLoopCount(-1)  # Infinite

        # 3. Title slide-in animation
        self.title_slide = QPropertyAnimation(self.title_label, b"pos")
        self.title_slide.setDuration(1000)
        self.title_slide.setEasingCurve(QEasingCurve.OutBack)

        # 4. Loading dots animation
        self.dots_count = 0
        self.dots_timer = QTimer()
        self.dots_timer.timeout.connect(self._update_dots)

        # Start all
        self.progress_anim.start()
        self.logo_pulse.start()
        self.dots_timer.start(400)

        # Finish after 6.5 seconds
        QTimer.singleShot(6500, self._finish)

    def _update_dots(self):
        self.dots_count = (self.dots_count + 1) % 4
        self.loading_label.setText("Loading" + "." * self.dots_count)

    def _finish(self):
        self.dots_timer.stop()
        self.close()
        self.finished.emit()


# ================================================================
# ✅ LOGIN DIALOG
# ================================================================

class LoginDialog(QDialog):
    """Beautiful login dialog."""

    login_success = pyqtSignal(str)  # username

    def __init__(self, db: UserDB, theme: Dict, lang: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.theme = theme
        self.lang = lang

        self.setWindowTitle(tr(lang, "login"))
        self.setFixedSize(450, 400)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)

        # Title
        title = QLabel(tr(self.lang, "login"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(20)

        # Username
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText(tr(self.lang, "username"))
        self.username_input.setObjectName("input")
        layout.addWidget(self.username_input)

        # Password
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText(tr(self.lang, "password"))
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("input")
        layout.addWidget(self.password_input)

        # Remember me
        self.remember_me = QCheckBox(tr(self.lang, "remember_me"))
        self.remember_me.setObjectName("checkbox")
        layout.addWidget(self.remember_me)

        layout.addSpacing(20)

        # Login button
        self.login_btn = QPushButton(tr(self.lang, "login_btn"))
        self.login_btn.setObjectName("primary_btn")
        self.login_btn.clicked.connect(self._do_login)
        layout.addWidget(self.login_btn)

        # Signup link
        signup_btn = QPushButton(f"→ {tr(self.lang, 'signup')}")
        signup_btn.setObjectName("link_btn")
        signup_btn.clicked.connect(self._open_signup)
        layout.addWidget(signup_btn)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
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
            QMessageBox.warning(self, "Error", tr(self.lang, "empty_fields"))
            return

        if self.db.verify(username, password):
            self.login_success.emit(username)
            self.accept()
        else:
            QMessageBox.warning(self, "Error", tr(self.lang, "invalid_credentials"))

    def _open_signup(self):
        dialog = SignupDialog(self.db, self.theme, self.lang, self.parent())
        if dialog.exec_() == QDialog.Accepted:
            # Auto-fill username
            pass


# ================================================================
# ✅ SIGNUP DIALOG
# ================================================================

class SignupDialog(QDialog):
    """Beautiful signup dialog with password validation."""

    signup_success = pyqtSignal(str)

    def __init__(self, db: UserDB, theme: Dict, lang: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.theme = theme
        self.lang = lang

        self.setWindowTitle(tr(lang, "signup"))
        self.setFixedSize(500, 620)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(10)

        # Title
        title = QLabel(tr(self.lang, "signup"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(15)

        # First name
        self.first_name = QLineEdit()
        self.first_name.setPlaceholderText(tr(self.lang, "first_name"))
        self.first_name.setObjectName("input")
        layout.addWidget(self.first_name)

        # Last name
        self.last_name = QLineEdit()
        self.last_name.setPlaceholderText(tr(self.lang, "last_name"))
        self.last_name.setObjectName("input")
        layout.addWidget(self.last_name)

        # Username
        self.username = QLineEdit()
        self.username.setPlaceholderText(tr(self.lang, "username"))
        self.username.setObjectName("input")
        layout.addWidget(self.username)

        # Email
        self.email = QLineEdit()
        self.email.setPlaceholderText(tr(self.lang, "email"))
        self.email.setObjectName("input")
        layout.addWidget(self.email)

        # Password
        self.password = QLineEdit()
        self.password.setPlaceholderText(tr(self.lang, "password"))
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setObjectName("input")
        layout.addWidget(self.password)

        # Repeat password
        self.repeat_password = QLineEdit()
        self.repeat_password.setPlaceholderText(tr(self.lang, "repeat_password"))
        self.repeat_password.setEchoMode(QLineEdit.Password)
        self.repeat_password.setObjectName("input")
        layout.addWidget(self.repeat_password)

        # Password rules hint
        rules = QLabel(tr(self.lang, "password_rule"))
        rules.setObjectName("rules")
        rules.setWordWrap(True)
        layout.addWidget(rules)

        # Remember me
        self.remember_me = QCheckBox(tr(self.lang, "remember_me"))
        self.remember_me.setObjectName("checkbox")
        layout.addWidget(self.remember_me)

        layout.addSpacing(15)

        # Signup button
        self.signup_btn = QPushButton(tr(self.lang, "signup_btn"))
        self.signup_btn.setObjectName("primary_btn")
        self.signup_btn.clicked.connect(self._do_signup)
        layout.addWidget(self.signup_btn)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        close_btn.setParent(self.container)
        close_btn.move(460, 10)

    def _apply_theme(self):
        t = self.theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
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
            QLineEdit#input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
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

    def _do_signup(self):
        first_name = self.first_name.text().strip()
        last_name = self.last_name.text().strip()
        username = self.username.text().strip()
        email = self.email.text().strip()
        password = self.password.text()
        repeat = self.repeat_password.text()

        # Validate
        if not all([first_name, last_name, username, email, password, repeat]):
            QMessageBox.warning(self, "Error", tr(self.lang, "empty_fields"))
            return

        if not validate_email(email):
            QMessageBox.warning(self, "Error", tr(self.lang, "invalid_email"))
            return

        if password != repeat:
            QMessageBox.warning(self, "Error", tr(self.lang, "passwords_dont_match"))
            return

        ok, msg = validate_password(password)
        if not ok:
            QMessageBox.warning(
                self, tr(self.lang, "password_requirements"),
                f"{tr(self.lang, 'password_rule')}\n\n❌ {msg}",
            )
            return

        if self.db.user_exists(username):
            QMessageBox.warning(self, "Error", tr(self.lang, "user_exists"))
            return

        # Create user
        if self.db.create_user(username, password, first_name, last_name, email):
            QMessageBox.information(self, "Success", tr(self.lang, "signup_success"))
            self.signup_success.emit(username)
            self.accept()


# ================================================================
# ✅ SETTINGS DIALOG
# ================================================================

class SettingsDialog(QDialog):
    """Settings dialog: language + theme."""

    settings_changed = pyqtSignal(str, str)  # lang, theme_key

    def __init__(self, current_lang: str, current_theme: str, parent=None):
        super().__init__(parent)
        self.current_lang = current_lang
        self.current_theme = current_theme

        self.setWindowTitle(tr(current_lang, "settings"))
        self.setFixedSize(520, 480)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)

        # Title
        self.title_label = QLabel(tr(self.current_lang, "settings"))
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        layout.addSpacing(15)

        # Change Language
        self.lang_label = QLabel(tr(self.current_lang, "change_language"))
        self.lang_label.setObjectName("label")
        layout.addWidget(self.lang_label)

        self.lang_combo = QComboBox()
        self.lang_combo.setObjectName("combo")
        # ✅ Add all languages in defined order
        for lang_name in LANGUAGE_ORDER:
            if lang_name in TRANSLATIONS:
                self.lang_combo.addItem(lang_name, lang_name)
        # Set current
        idx = self.lang_combo.findData(self.current_lang)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        layout.addWidget(self.lang_combo)

        layout.addSpacing(10)

        # Change Theme
        self.theme_label = QLabel(tr(self.current_lang, "change_theme"))
        self.theme_label.setObjectName("label")
        layout.addWidget(self.theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("combo")
        # ✅ Add all themes with their translated display names
        for theme_key in THEME_ORDER:
            if theme_key in THEMES:
                theme_display = tr(self.current_lang, theme_key)
                self.theme_combo.addItem(theme_display, theme_key)
        # Set current
        idx = self.theme_combo.findData(self.current_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        layout.addWidget(self.theme_combo)

        layout.addSpacing(25)

        # Buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("✓ " + tr(self.current_lang, "apply"))
        self.apply_btn.setObjectName("primary_btn")
        self.apply_btn.clicked.connect(self._apply_settings)
        btn_layout.addWidget(self.apply_btn)

        self.cancel_btn = QPushButton("✕ " + tr(self.current_lang, "cancel"))
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        close_btn.setParent(self.container)
        close_btn.move(480, 10)

    def _apply_theme(self):
        t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#title {{
                font-size: 28px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#label {{
                font-size: 14px;
                font-weight: bold;
                color: {t['text']};
                background: transparent;
            }}
            QComboBox#combo {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
                color: {t['text']};
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
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
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

    def _apply_settings(self):
        new_lang = self.lang_combo.currentData()
        new_theme = self.theme_combo.currentData()
        self.settings_changed.emit(new_lang, new_theme)
        self.accept()


# ================================================================
# ✅ ACCOUNT INFO DIALOG
# ================================================================

class AccountInfoDialog(QDialog):
    """Account info dialog with edit capability."""

    info_updated = pyqtSignal()

    def __init__(self, db: UserDB, username: str, theme: Dict, lang: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.username = username
        self.theme = theme
        self.lang = lang

        self.setWindowTitle(tr(lang, "account_info"))
        self.setFixedSize(500, 500)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)

        # Title
        title = QLabel(tr(self.lang, "account_info"))
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(20)

        user_info = self.db.get_user_info(self.username) or {}

        # First name
        layout.addWidget(QLabel(tr(self.lang, "first_name")))
        self.first_name = QLineEdit(user_info.get("first_name", ""))
        self.first_name.setObjectName("input")
        layout.addWidget(self.first_name)

        # Last name
        layout.addWidget(QLabel(tr(self.lang, "last_name")))
        self.last_name = QLineEdit(user_info.get("last_name", ""))
        self.last_name.setObjectName("input")
        layout.addWidget(self.last_name)

        # Email
        layout.addWidget(QLabel(tr(self.lang, "email")))
        self.email = QLineEdit(user_info.get("email", ""))
        self.email.setObjectName("input")
        layout.addWidget(self.email)

        layout.addSpacing(20)

        # Save button
        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("primary_btn")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.reject)
        close_btn.setParent(self.container)
        close_btn.move(460, 10)

    def _apply_theme(self):
        t = self.theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel {{
                color: {t['text']};
                font-size: 13px;
                background: transparent;
            }}
            QLabel#title {{
                font-size: 26px;
                font-weight: bold;
                color: {t['primary']};
            }}
            QLineEdit#input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
                color: {t['text']};
            }}
            QPushButton#primary_btn {{
                background: qlinearGradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:1 {t['secondary']});
                color: {t['button_text']};
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
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

    def _save(self):
        self.db.update_user(
            self.username,
            first_name=self.first_name.text().strip(),
            last_name=self.last_name.text().strip(),
            email=self.email.text().strip(),
        )
        self.info_updated.emit()
        self.accept()


# ================================================================
# ✅ PROCESSING THREAD
# ================================================================

class ProcessThread(QThread):
    """Thread for running model inference."""

    result_ready = pyqtSignal(int, float)  # grid, confidence
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

            self.result_ready.emit(int(pred.item()), float(conf.item()))
        except Exception as e:
            self.error.emit(str(e))


# ================================================================
# ✅ MAIN WINDOW
# ================================================================

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # State
        self.db = UserDB()
        self.current_lang = "English"
        self.current_theme = "theme_dark_aqua"
        self.current_user = None
        self.current_image_path = None
        self.is_muted = False
        self.music_playing = False

        # Window setup
        self.setWindowTitle(tr(self.current_lang, "app_title"))
        self.setMinimumSize(1400, 900)
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

        # Setup UI
        self._setup_ui()
        self._apply_theme()

        # Start music
        self._start_music()

    # ============================================================
    # UI Setup
    # ============================================================

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout (3 horizontal zones: left, center, right)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar (header)
        self._setup_topbar(main_layout)

        # Center zone (glass window)
        self._setup_center(main_layout)

        # Bottom bar (buttons)
        self._setup_bottombar(main_layout)

        # ---- Footer (developed by) ----
        self._setup_footer(main_layout)

    def _setup_topbar(self, parent_layout):
        """Top bar with login/signup and account buttons."""
        topbar = QFrame()
        topbar.setFixedHeight(70)
        topbar.setObjectName("topbar")

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(30, 10, 30, 10)

        # Left: Account button (initially hidden)
        self.account_btn = QPushButton()
        self.account_btn.setObjectName("account_btn")
        self.account_btn.setVisible(False)
        self.account_btn.clicked.connect(self._show_account_menu)
        layout.addWidget(self.account_btn)

        layout.addStretch()

        # Right: Login/Signup buttons
        self.login_btn = QPushButton(tr(self.current_lang, "login"))
        self.login_btn.setObjectName("login_btn")
        self.login_btn.clicked.connect(self._open_login)
        layout.addWidget(self.login_btn)

        self.signup_btn = QPushButton(tr(self.current_lang, "signup"))
        self.signup_btn.setObjectName("signup_btn")
        self.signup_btn.clicked.connect(self._open_signup)
        layout.addWidget(self.signup_btn)

        parent_layout.addWidget(topbar)

    def _setup_center(self, parent_layout):
        """Glass-like center window."""
        center_container = QFrame()
        center_container.setObjectName("center_container")

        layout = QVBoxLayout(center_container)
        layout.setContentsMargins(80, 60, 80, 60)

        # Glass window
        self.glass_window = QFrame()
        self.glass_window.setObjectName("glass_window")
        glass_layout = QVBoxLayout(self.glass_window)
        glass_layout.setContentsMargins(50, 50, 50, 50)

        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(400)
        self.image_preview.setObjectName("image_preview")
        self.image_preview.setText("🖼️\n\n" + tr(self.current_lang, "no_image"))
        glass_layout.addWidget(self.image_preview)

        # Result display
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setObjectName("result_label")
        self.result_label.setVisible(False)
        glass_layout.addWidget(self.result_label)

        layout.addWidget(self.glass_window)
        parent_layout.addWidget(center_container, stretch=1)

    def _setup_bottombar(self, parent_layout):
        """Bottom bar with action buttons."""
        bottombar = QFrame()
        bottombar.setFixedHeight(100)
        bottombar.setObjectName("bottombar")

        layout = QHBoxLayout(bottombar)
        layout.setContentsMargins(30, 15, 30, 15)

        # Left: Visit webpage
        self.visit_btn = QPushButton("🌐 " + tr(self.current_lang, "visit_webpage"))
        self.visit_btn.setObjectName("bottom_btn")
        self.visit_btn.clicked.connect(self._visit_webpage)
        layout.addWidget(self.visit_btn)

        layout.addStretch()

        # Center: Insert + Process
        self.insert_btn = QPushButton("📷 " + tr(self.current_lang, "insert_sono"))
        self.insert_btn.setObjectName("action_btn")
        self.insert_btn.clicked.connect(self._insert_image)
        layout.addWidget(self.insert_btn)

        self.process_btn = QPushButton("⚡ " + tr(self.current_lang, "process"))
        self.process_btn.setObjectName("action_btn")
        self.process_btn.clicked.connect(self._process_image)
        layout.addWidget(self.process_btn)

        layout.addStretch()

        # Right: Mute/Unmute
        self.mute_btn = QPushButton("🔊 " + tr(self.current_lang, "mute"))
        self.mute_btn.setObjectName("bottom_btn")
        self.mute_btn.clicked.connect(self._toggle_mute)
        layout.addWidget(self.mute_btn)

        parent_layout.addWidget(bottombar)

    def _setup_footer(self, parent_layout):
        """Footer with developer credit."""
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
    # Theme
    # ============================================================

    def _apply_theme(self):
        t = THEMES.get(self.current_theme, THEMES["theme_dark_aqua"])

        self.setStyleSheet(f"""
            QMainWindow {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['bg_gradient_start']},
                    stop:1 {t['bg_gradient_end']});
            }}
            QFrame#topbar {{
                background: transparent;
                border-bottom: 1px solid {t['glass_border']};
            }}
            QFrame#center_container {{
                background: transparent;
            }}
            QFrame#glass_window {{
                background: {t['glass_bg']};
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
                    stop:0 {t['primary']}, stop:1 {t['secondary']});
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
            QPushButton#action_btn {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:1 {t['secondary']});
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

    # ============================================================
    # Music
    # ============================================================

    def _start_music(self):
        if not HAS_PYGAME:
            return
        if not os.path.exists(MUSIC_PATH):
            print(f"⚠️ Music not found: {MUSIC_PATH}")
            return
        try:
            pygame.mixer.music.load(MUSIC_PATH)
            pygame.mixer.music.set_volume(0.3)
            pygame.mixer.music.play(-1)
            self.music_playing = True
        except Exception as e:
            print(f"⚠️ Music failed: {e}")

    def _toggle_mute(self):
        if not HAS_PYGAME:
            return
        if self.is_muted:
            pygame.mixer.music.unpause()
            self.mute_btn.setText("🔊 " + tr(self.current_lang, "mute"))
            self.is_muted = False
        else:
            pygame.mixer.music.pause()
            self.mute_btn.setText("🔇 " + tr(self.current_lang, "unmute"))
            self.is_muted = True

    # ============================================================
    # Login / Signup
    # ============================================================

    def _open_login(self):
        dialog = LoginDialog(self.db, THEMES[self.current_theme], self.current_lang, self)
        dialog.login_success.connect(self._on_login_success)
        dialog.exec_()

    def _open_signup(self):
        dialog = SignupDialog(self.db, THEMES[self.current_theme], self.current_lang, self)
        dialog.exec_()

    def _on_login_success(self, username: str):
        self.current_user = username
        self.account_btn.setText(f"👤 {username} account")
        self.account_btn.setVisible(True)
        self.login_btn.setVisible(False)
        self.signup_btn.setVisible(False)
        QMessageBox.information(self, "Success", f"Welcome, {username}!")

    # ============================================================
    # Account Menu
    # ============================================================

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

        menu.addAction("⚙️ " + tr(self.current_lang, "settings"), self._open_settings)
        menu.addAction("ℹ️ " + tr(self.current_lang, "account_info"), self._open_account_info)
        menu.addSeparator()
        menu.addAction("🚪 " + tr(self.current_lang, "logout"), self._logout)
        menu.addAction("❌ " + tr(self.current_lang, "exit"), self._exit_app)

        menu.exec_(self.account_btn.mapToGlobal(self.account_btn.rect().bottomLeft()))

    def _open_settings(self):
        dialog = SettingsDialog(self.current_lang, self.current_theme, self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec_()

    def _on_settings_changed(self, lang: str, theme: str):
        """Apply new language and theme across the WHOLE app."""
        # Update state
        self.current_lang = lang
        self.current_theme = theme

        # 1. Apply theme to main window (all frames, buttons, labels)
        self._apply_theme()

        # 2. Update all texts in main window
        self._update_texts()

        # 3. Force repaint to make sure theme fully applies
        self.update()
        self.repaint()

        # 4. Show confirmation
        QMessageBox.information(
            self,
            tr(self.current_lang, "settings"),
            tr(self.current_lang, "settings_applied"),
        )

    def _update_texts(self):
        """Update ALL UI texts after language change."""
        # Window title
        self.setWindowTitle(tr(self.current_lang, "app_title"))

        # Top bar buttons
        self.login_btn.setText(tr(self.current_lang, "login"))
        self.signup_btn.setText(tr(self.current_lang, "signup"))

        # Account button
        if self.current_user:
            self.account_btn.setText(
                f"👤 {self.current_user} {tr(self.current_lang, 'account')}"
            )

        # Bottom bar buttons
        self.insert_btn.setText("📷 " + tr(self.current_lang, "insert_sono"))
        self.process_btn.setText("⚡ " + tr(self.current_lang, "process"))
        self.visit_btn.setText("🌐 " + tr(self.current_lang, "visit_webpage"))

        # Mute button (respect current state)
        mute_icon = "🔇 " if self.is_muted else "🔊 "
        mute_key = "unmute" if self.is_muted else "mute"
        self.mute_btn.setText(mute_icon + tr(self.current_lang, mute_key))

        # Image preview placeholder (only if no image)
        if not self.current_image_path:
            self.image_preview.setText("🖼️\n\n" + tr(self.current_lang, "no_image"))

        # Result label (if visible)
        if self.result_label.isVisible() and self.result_label.text():
            # Re-render result if we have stored it
            if hasattr(self, "_last_result_grid"):
                grid = self._last_result_grid
                conf = self._last_result_confidence
                grid_key = f"grid_{grid}"
                grid_text = tr(self.current_lang, grid_key)
                result_text = (
                    f"🎯 {tr(self.current_lang, 'result')}: {grid_text}\n"
                    f"📊 {tr(self.current_lang, 'confidence')}: {conf * 100:.2f}%"
                )
                self.result_label.setText(result_text)

    def _open_account_info(self):
        if not self.current_user:
            return
        dialog = AccountInfoDialog(
            self.db, self.current_user, THEMES[self.current_theme],
            self.current_lang, self,
        )
        dialog.exec_()

    def _logout(self):
        reply = QMessageBox.question(
            self, tr(self.current_lang, "logout"),
            tr(self.current_lang, "logout_confirm"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            if self.current_user:
                self.db.delete_user(self.current_user)
            self.current_user = None
            self.account_btn.setVisible(False)
            self.login_btn.setVisible(True)
            self.signup_btn.setVisible(True)

    def _exit_app(self):
        reply = QMessageBox.question(
            self, tr(self.current_lang, "exit"),
            tr(self.current_lang, "exit_confirm"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.close()

    # ============================================================
    # Image Processing
    # ============================================================

    def _insert_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr(self.current_lang, "insert_sono"),
            "", "Images (*.png *.jpg *.jpeg *.bmp *.dcm)",
        )
        if path:
            self.current_image_path = path
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self.image_preview.setPixmap(scaled)
            self.result_label.setVisible(False)

    def _process_image(self):
        if not self.current_image_path:
            QMessageBox.warning(
                self, "Warning", tr(self.current_lang, "insert_image_first"),
            )
            return

        if not HAS_TORCH:
            QMessageBox.critical(self, "Error", "PyTorch not available")
            return

        # Show "please wait"
        self.result_label.setText(tr(self.current_lang, "please_wait"))
        self.result_label.setVisible(True)

        # Run in thread
        self.thread = ProcessThread(self.current_image_path)
        self.thread.result_ready.connect(self._on_result_ready)
        self.thread.error.connect(self._on_error)
        self.thread.start()

    def _on_result_ready(self, grid: int, confidence: float):
        # Store for re-rendering on language change
        self._last_result_grid = grid
        self._last_result_confidence = confidence

        grid_key = f"grid_{grid}"
        grid_text = tr(self.current_lang, grid_key)
        result_text = f"🎯 {tr(self.current_lang, 'result')}: {grid_text}\n"
        result_text += f"📊 {tr(self.current_lang, 'confidence')}: {confidence * 100:.2f}%"
        self.result_label.setText(result_text)
        self.result_label.setVisible(True)

    def _on_error(self, error: str):
        QMessageBox.critical(self, "Error", f"Inference failed: {error}")
        self.result_label.setVisible(False)

    # ============================================================
    # Visit Webpage
    # ============================================================

    def _visit_webpage(self):
        import webbrowser
        webbrowser.open("https://github.com/babakyousefian/babakyousefian/")


# ================================================================
# ✅ ENTRY POINT
# ================================================================

def main():
    # High DPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Show splash screen
    splash = SplashScreen()
    splash.show()

    # Main window (hidden initially)
    main_window = MainWindow()

    # When splash finishes, show main window
    def on_splash_finished():
        main_window.show()

    splash.finished.connect(on_splash_finished)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
