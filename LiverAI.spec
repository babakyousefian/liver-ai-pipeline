# -*- mode: python ; coding: utf-8 -*-
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(SPEC))
VENV_SITE_PACKAGES = os.path.join(PROJECT_ROOT, '.venv', 'Lib', 'site-packages')

# ================================================================
#  TORCH BINARIES
# ================================================================
torch_path = os.path.join(VENV_SITE_PACKAGES, 'torch')
torch_binaries = []
torch_datas = []

if os.path.exists(torch_path):
    torch_lib = os.path.join(torch_path, 'lib')
    if os.path.exists(torch_lib):
        for dll in os.listdir(torch_lib):
            if dll.endswith('.dll') or dll.endswith('.pyd'):
                torch_binaries.append((os.path.join(torch_lib, dll), 'torch/lib'))
    for f in os.listdir(torch_path):
        if f.startswith('_C') and (f.endswith('.pyd') or f.endswith('.dll')):
            torch_binaries.append((os.path.join(torch_path, f), 'torch'))
    for f in ['version.py', 'VERSION']:
        p = os.path.join(torch_path, f)
        if os.path.exists(p):
            torch_datas.append((p, 'torch'))

# ================================================================
#  ✅ جمع‌آوری همه‌ی فایل‌های .py (بازگشتی)
# ================================================================
def collect_all_py_files(directory, dest_prefix):
    """
    جمع‌آوری همه‌ی فایل‌های .py از یک پوشه (شامل زیرپوشه‌ها).
    با حفظ ساختار پوشه‌ها.
    """
    result = []
    if not os.path.isdir(directory):
        print(f"⚠️ Directory not found: {directory}")
        return result

    for root, dirs, files in os.walk(directory):
        # __pycache__ رو skip کن
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                result.append((full_path, os.path.dirname(rel_path)))
    return result

# ================================================================
#  ✅ جمع‌آوری فایل‌های غیر .py (داده‌ها)
# ================================================================
def collect_data_files(directory, dest_prefix, extensions=None):
    """
    جمع‌آوری فایل‌های داده (غیر .py) از یک پوشه.
    اگر extensions داده شود، فقط آن‌ها را جمع می‌کند.
    """
    result = []
    if not os.path.isdir(directory):
        return result

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', '.venv')]
        for f in files:
            if extensions and not any(f.endswith(ext) for ext in extensions):
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)
            result.append((full_path, os.path.dirname(rel_path)))
    return result

# ================================================================
#  ✅ فایل‌های اصلی و پوشه‌های ضروری
# ================================================================

project_datas = [
    # ---- Assets ----
    ('assets/favicon.ico', 'assets'),
    ('assets/logo.png', 'assets'),
    ('assets/sounds', 'assets/sounds'),

    # ---- finalReports: مدل + labels ----
    ('finalReports/selfSupervisedLearning/final_models/final_model.pth',
     'finalReports/selfSupervisedLearning/final_models'),
    ('finalReports/labels.csv', 'finalReports'),

    # ---- packages: موزیک ----
    ('packages/facilities/assets/musics', 'packages/facilities/assets/musics'),

    # ---- فایل‌های ریشه ----
    ('users_db.json', '.'),

    # ⚠️ اگر settings.json و stats.json و onboarding.json و ... نیازند، این‌جا اضافه کنید
    ('settings.json', '.') if os.path.exists('settings.json') else None,
    ('stats.json', '.') if os.path.exists('stats.json') else None,
    ('smart_suggestions.json', '.') if os.path.exists('smart_suggestions.json') else None,
    ('onboarding.json', '.') if os.path.exists('onboarding.json') else None,
]

# ✅ حذف None ها
project_datas = [d for d in project_datas if d is not None]

# ================================================================
#  ✅ پوشه‌های ضروری (core, gui, src)
# ================================================================

# ---- core/ (تمام فایل‌های .py + .pkl + .json) ----
core_py_files = collect_all_py_files('core', 'core')
project_datas += core_py_files
print(f"✅ core/: {len(core_py_files)} .py files")

# ---- core/security/models/liver_model.pkl (مهم‌ترین فایل!) ----
if os.path.exists('core/security/models/liver_model.pkl'):
    project_datas.append(
        ('core/security/models/liver_model.pkl', 'core/security/models')
    )
    print("✅ core/security/models/liver_model.pkl")

# ---- core/security/liver_reference/ (اگر وجود دارد، برای training آینده) ----
if os.path.isdir('core/security/liver_reference'):
    liver_ref_files = collect_data_files(
        'core/security/liver_reference',
        'core/security/liver_reference',
        extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    )
    # ⚠️ اگر حجم خیلی زیاد است، این بلاک را کامنت کنید (فقط برای training، نه اجرا)
    # project_datas += liver_ref_files
    print(f"ℹ️ core/security/liver_reference/: {len(liver_ref_files)} images (NOT bundled — only needed for training)")

# ---- gui/ (تمام فایل‌های .py) ----
gui_py_files = collect_all_py_files('gui', 'gui')
project_datas += gui_py_files
print(f"✅ gui/: {len(gui_py_files)} .py files")

# ---- src/ (تمام فایل‌های .py) ----
src_py_files = collect_all_py_files('src', 'src')
project_datas += src_py_files
print(f"✅ src/: {len(src_py_files)} .py files")

# ---- assets/fonts/ (اگر فونت‌ها در assets هستن) ----
if os.path.isdir('assets/fonts'):
    font_files = collect_data_files('assets/fonts', 'assets/fonts')
    project_datas += font_files
    print(f"✅ assets/fonts/: {len(font_files)} font files")

# ================================================================
#  HIDDEN IMPORTS
# ================================================================
hidden = [
    # ---- PyQt5 ----
    'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'PyQt5.QtSvg', 'PyQt5.QtPrintSupport', 'PyQt5.QtNetwork',

    # ---- Core libs ----
    'pygame', 'cv2', 'numpy',

    # ---- Torch ----
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.serialization',
    'torch._C', 'torch._utils', 'torch.storage', 'torch.cuda',
    'torch.cuda.amp', 'torch.backends', 'torch.backends.cuda',
    'torch.backends.cudnn', 'torch.backends.mkl', 'torch.backends.mkldnn',
    'torch.backends.openmp', 'torch.nn.modules', 'torch.nn.modules.conv',
    'torch.nn.modules.linear', 'torch.nn.modules.batchnorm',
    'torch.nn.modules.activation', 'torch.nn.modules.pooling',
    'torch.nn.modules.container',

    # ---- Security / Crypto ----
    'cryptography', 'cryptography.fernet', 'cryptography.hazmat',
    'cryptography.hazmat.primitives', 'cryptography.hazmat.primitives.kdf',
    'cryptography.hazmat.primitives.kdf.pbkdf2',
    'cryptography.hazmat.primitives.hashes',

    # ---- Image Guard (sklearn for PCA/Mahalanobis) ----
    'sklearn', 'sklearn.decomposition', 'sklearn.covariance',
    'sklearn.utils', 'sklearn.utils.extmath',
    'scipy', 'scipy.linalg', 'scipy.sparse', 'scipy.sparse.linalg',

    # ---- TOTP ----
    'pyotp', 'qrcode', 'qrcode.constants', 'qrcode.image',
    'qrcode.image.pil', 'PIL', 'PIL.Image',

    # ---- ReportLab (PDF) ----
    'reportlab', 'reportlab.lib', 'reportlab.lib.pagesizes',
    'reportlab.lib.units', 'reportlab.lib.colors', 'reportlab.lib.styles',
    'reportlab.lib.enums', 'reportlab.platypus',
    'reportlab.pdfbase', 'reportlab.pdfbase.ttfonts',
    'reportlab.graphics', 'reportlab.graphics.shapes',
    'reportlab.graphics.renderPDF',

    # ---- RTL (Persian/Arabic) ----
    'arabic_reshaper', 'bidi', 'bidi.algorithm',

    # ---- Voice Feedback ----
    'pyttsx3', 'pyttsx3.drivers', 'pyttsx3.drivers.sapi5',

    # ---- Updater ----
    'requests', 'urllib3',

    # ---- Local modules ----
    'src.models.resnet.resnet50',

    # ---- System ----
    'unittest', 'unittest.mock',
    'platform', 'socket', 'uuid',
]

# ================================================================
#  EXCLUDES (همه چیز اضافی)
# ================================================================
excludes = [
    # ---- onnx (crash میکنه) ----
    'onnx', 'onnxscript', 'onnx_ir', 'onnxruntime', 'torch.onnx',

    # ---- استفاده نمیشن ----
    'torchvision', 'torchaudio',
    'matplotlib', 'tkinter',
    'IPython', 'jupyter', 'notebook',
    'pytest', 'pluggy', 'iniconfig',
    'sphinx', 'docutils',

    # ---- سنگین‌ها (ولی sklearn و scipy رو نگه داریم) ----
    'pandas', 'sqlalchemy',
    'numba', 'llvmlite',
    'tensorboard',
    'wandb', 'sentry_sdk',
    'fitz', 'pymupdf',
    'captum', 'shap', 'lime',
    'easyocr', 'medpy',
    'SimpleITK', 'itk',
    'kornia', 'imgaug', 'albumentations',
    'optuna',

    # ⚠️ 'unittest' و 'test' رو کامنت نکن!
]

# ================================================================
#  ANALYSIS
# ================================================================
a = Analysis(
    ['GUI_update.py'],
    pathex=[PROJECT_ROOT],
    binaries=torch_binaries,
    datas=project_datas + torch_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LiverAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/favicon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LiverAI',
)
