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
#  ⚠️ فقط فایل‌های .py از src (نه دیتاها!)
# ================================================================
def collect_py_files(src_dir, dest_prefix):
    """فقط فایل‌های .py رو جمع می‌کنه، نه دیتاها"""
    result = []
    for root, dirs, files in os.walk(src_dir):
        # __pycache__ رو skip کن
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                result.append((full_path, os.path.dirname(rel_path)))
    return result

# ================================================================
#  DATA FILES (فقط ضروری‌ها)
# ================================================================
project_datas = [
    # ---- Assets (فقط آیکون و لوگو) ----
    ('assets/favicon.ico', 'assets'),
    ('assets/logo.png', 'assets'),

    # ---- finalReports: فقط مدل + labels ----
    ('finalReports/selfSupervisedLearning/final_models/final_model.pth',
     'finalReports/selfSupervisedLearning/final_models'),
    ('finalReports/labels.csv', 'finalReports'),

    # ---- packages: فقط موزیک ----
    ('packages/facilities/assets/musics', 'packages/facilities/assets/musics'),

    # ---- src: فقط فایل‌های .py ----
    # (اینا با تابع collect_py_files اضافه میشن)

    # ---- فایل‌های ریشه ----
    ('users_db.json', '.'),
    ('assets/sounds', 'assets/sounds'),
]

# اضافه کردن فایل‌های .py از src
project_datas += collect_py_files('src', 'src')

# ================================================================
#  HIDDEN IMPORTS
# ================================================================
hidden = [
    'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'pygame', 'cv2', 'numpy',
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.serialization',
    'torch._C', 'torch._utils', 'torch.storage', 'torch.cuda',
    'torch.cuda.amp', 'torch.backends', 'torch.backends.cuda',
    'torch.backends.cudnn', 'torch.backends.mkl', 'torch.backends.mkldnn',
    'torch.backends.openmp', 'torch.nn.modules', 'torch.nn.modules.conv',
    'torch.nn.modules.linear', 'torch.nn.modules.batchnorm',
    'torch.nn.modules.activation', 'torch.nn.modules.pooling',
    'torch.nn.modules.container',
    'src.models.resnet.resnet50',
    'unittest', 'unittest.mock',
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
    # ---- سنگین‌ها ----
    'pandas', 'sqlalchemy',
    'sklearn', 'scipy',
    'numba', 'llvmlite',
    'tensorboard',
    'wandb', 'sentry_sdk',
    'reportlab', 'fitz', 'pymupdf',
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
    ['GUI.py'],
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
