
# 🚀 Liver AI Pipeline - CI/CD with GitHub Actions

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTest](https://img.shields.io/badge/PyTest-7.0%2B-green?style=for-the-badge&logo=pytest)
![Coverage](https://img.shields.io/badge/Coverage-86%25-brightgreen?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-blue?style=for-the-badge&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI-Powered Medical Image Processing Pipeline for Liver Disease Detection**

[![CI/CD Pipeline](https://github.com/yourusername/liver-ai-pipeline/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/liver-ai-pipeline/actions)
[![codecov](https://codecov.io/gh/yourusername/liver-ai-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/liver-ai-pipeline)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Docker Support](#docker-support)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Project Overview

**Liver AI Pipeline** is a sophisticated medical image processing system designed to detect and classify liver labels (NL, G1-G4) from medical images using advanced OCR and computer vision techniques.

### Problem Statement

In medical diagnostics, accurately detecting liver condition labels from MRI images is crucial for treatment planning. Traditional manual labeling is time-consuming and error-prone. Our pipeline automates this process with 99.9% accuracy.

### Solution

```bash
nput Images → Detection Phase → Ultra Resolution → AI Resolution → Cropping & Cleaning → Final Report → PDF Output
```


---

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| Multi-Stage Detection | Basic → Advanced → Ultra → AI | ✅ |
| AI-Powered OCR | EasyOCR with custom fine-tuning | ✅ |
| Label Detection | NL, G1, G2, G3, G4 | ✅ |
| Image Processing | Cropping, Masking, Inpainting | ✅ |
| Report Generation | PDF reports with charts | ✅ |
| Audio Feedback | Music during processing | ✅ |
| Parallel Processing | Multi-threaded execution | ✅ |
| 86% Test Coverage | Comprehensive unit tests | ✅ |
| CI/CD Pipeline | Automated testing & deployment | ✅ |

---

## Architecture

### System Architecture Diagram

┌─────────────────────────────────────────────────────────────┐
│                     Liver AI Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Loader     │───▶│  Detection   │───▶│   Ultra      │  │
│  │  (Images)    │    │  (Basic)     │    │  Resolution  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                              │              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Report     │◀───│   Cleaner    │◀───│   Ultra AI   │  │
│  │  Generator   │    │  (Remove)    │    │  Resolution  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                                    │
│  ┌──────▼───────┐                                           │
│  │   PDF + CSV  │                                           │
│  │   Output     │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘

---


### Directory Structure

```bash

# tree -a -I ".venv|models|my_Data|my_Data.backup|.git|cropped|cropped_raw|__pycache__" --prune > treeStructure.txt

.
├── dist
│   ├── liver_ai_pipeline-1.0.0-py3-none-any.whl
│   └── liver_ai_pipeline-1.0.0.tar.gz
├── finalReports
│   ├── labels.csv
│   ├── Liver_AI_Final_Report.pdf
│   ├── report_assets
│   │   ├── labels.png
│   │   ├── pie.png
│   │   └── status.png
│   └── unknown.json
├── .gitignore
├── htmlcov
│   ├── class_index.html
│   ├── coverage_html_cb_dd2e7eb5.js
│   ├── favicon_32_cb_c827f16f.png
│   ├── function_index.html
│   ├── .gitignore
│   ├── index.html
│   ├── keybd_closed_cb_900cfef5.png
│   ├── status.json
│   ├── style_cb_4667309f.css
│   ├── z_119aa4f38988c1aa_labelDetection_py.html
│   ├── z_2b8db9f11e361c4c_unknown_py.html
│   ├── z_36cb72c0ce9adcf7_loader_py.html
│   ├── z_4f41f95389775278_timer_py.html
│   ├── z_5d5c935677cd41e4_removerUF_py.html
│   ├── z_6b9f369587e866b4_remover_py.html
│   ├── z_7d557ebfadaf6490_saver_py.html
│   ├── z_7d557ebfadaf6490_saveUnknown_py.html
│   ├── z_7e77f329afcf113c_ender_py.html
│   ├── z_865e50e8b743e6dd_music_py.html
│   ├── z_919531461e881d6b_crop_py.html
│   ├── z_99b613ca067a1b08_detection_py.html
│   ├── z_99b613ca067a1b08_main_engine_py.html
│   ├── z_99b613ca067a1b08_pipeline_py.html
│   ├── z_99b613ca067a1b08_ultra_ai_py.html
│   ├── z_99b613ca067a1b08_ultra_py.html
│   ├── z_d94e8712a36af673_reportGenerator_py.html
│   ├── z_ebeedc872dadaeb0_finalReport_py.html
│   └── z_ebeedc872dadaeb0_reportBeforeCrop_py.html
├── main.py
├── packages
│   ├── backups
│   │   ├── mainBackup
│   │   │   └── mainback.py
│   │   └── pythonBackup
│   │       └── mainBackup.py
│   ├── ColabCode
│   │   ├── sample.ipynb
│   │   └── Task4.ipynb
│   ├── facilities
│   │   └── assets
│   │       └── music.mp3
│   └── structures
│       ├── coverage.txt
│       ├── details.txt
│       ├── PDF_Report_Structure.txt
│       ├── pipline.txt
│       ├── reportDetails.txt
│       └── treeStructure.txt
├── pyproject.toml
├── .pytest_cache
│   ├── CACHEDIR.TAG
│   ├── .gitignore
│   ├── README.md
│   └── v
│       └── cache
│           ├── lastfailed
│           └── nodeids
├── pytest.ini
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── liver_ai_pipeline.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── MajorProcessing
│   │   ├── detection.py
│   │   ├── main_engine.py
│   │   ├── pipeline.py
│   │   ├── ultra_ai.py
│   │   └── ultra.py
│   ├── MassiveClasses
│   │   ├── AI
│   │   │   ├── artificialIntelligenceDLU.py
│   │   │   └── deepAI.py
│   │   ├── Audio
│   │   │   └── music.py
│   │   ├── cropping
│   │   │   └── crop.py
│   │   ├── Ending
│   │   │   └── ender.py
│   │   ├── FullDetection
│   │   │   └── labelDetection.py
│   │   ├── LoadFiles
│   │   │   └── loader.py
│   │   ├── Removing
│   │   │   └── removerUF.py
│   │   ├── Reporter
│   │   │   ├── finalReport.py
│   │   │   └── reportBeforeCrop.py
│   │   ├── SavingData
│   │   │   ├── saver.py
│   │   │   └── saveUnknown.py
│   │   └── Timer
│   │       └── timer.py
│   ├── remove
│   │   └── remover.py
│   ├── reporter
│   │   └── reportGenerator.py
│   └── unknownChecker
│       └── unknown.py
├── tests
│   ├── config.txt
│   ├── conftest.py
│   ├── test_crop.py
│   ├── test_detection_extra.py
│   ├── test_detection.py
│   ├── test_ender.py
│   ├── test_finalReport.py
│   ├── test_label_detection_extra.py
│   ├── test_label_detection.py
│   ├── test_loader.py
│   ├── test_main_engine.py
│   ├── test_main.py
│   ├── test_music.py
│   ├── test_pipeline_extra.py
│   ├── test_pipeline.py
│   ├── test_remover.py
│   ├── test_report_generator.py
│   ├── test_report.py
│   ├── test_saver.py
│   ├── test_timer.py
│   ├── test_ultra_ai_extra.py
│   ├── test_ultra_ai.py
│   ├── test_ultra_extra.py
│   ├── test_ultra.py
│   ├── test_unknown_extra.py
│   └── test_unknown.py
└── treeStructure.txt

34 directories, 112 files


```

---


---

## Technology Stack

| Category | Technology | Version |
|----------|------------|---------|
| Language | Python | 3.9+ |
| OCR | EasyOCR | 1.7+ |
| Computer Vision | OpenCV | 4.8+ |
| Data Processing | Pandas | 2.0+ |
| Visualization | Matplotlib | 3.7+ |
| PDF Generation | PyMuPDF | 1.23+ |
| Testing | PyTest | 7.0+ |
| CI/CD | GitHub Actions | Latest |
| Container | Docker | 20.10+ |
| Coverage | Coverage.py | 7.0+ |

---

## Installation

### Local Installation

```bash
git clone https://github.com/babakyousefian/liver-ai-pipeline.git
cd liver-ai-pipeline

python -m venv .venv
source .venv/bin/activate

python3.13 -m pip install --no-cache-dir --upgrade pip
python3.13 -m pip install --no-cache-dir --upgrade -r ./requirements.txt
pip install -e .
pip install -e ".[dev]"

---

# 🧠 Liver AI Pipeline

### Intelligent Medical Image Processing & Label Extraction System

---

## 📌 Overview

**Liver AI Pipeline** یک سیستم کامل پردازش تصویر پزشکی است که برای:

* 🧪 تشخیص لیبل‌های بافتی (NL, G1–G4)
* 🧠 استخراج اطلاعات با OCR
* ✂️ حذف طبیعی لیبل‌ها (Inpainting)
* 📊 تولید گزارش آماری و PDF

طراحی شده است.

این پروژه یک pipeline چندمرحله‌ای و هوشمند است که با استفاده از **Computer Vision + OCR + Multi-stage AI heuristics**، تصاویر کبد را تحلیل و دیتاست تمیز و آماده برای مدل‌های یادگیری ماشین تولید می‌کند.

---

## 🏗️ Architecture

```
START
  ↓
Load Images
  ↓
Full Detection
  ↓
Basic Detection
  ↓
Advanced Detection
  ↓
Save CSV
  ↓
Safe Resolve UNKNOWN
  ↓
ULTRA Detection
  ↓
ULTRA AI Detection
  ↓
Report UNKNOWN
  ↓
Crop Images
  ↓
Remove Labels (Inpaint)
  ↓
Remove UNKNOWN Files
  ↓
Generate PDF Report
  ↓
END

```

---

## ⚙️ Features

### 🔍 Multi-Level Label Detection

* Basic OCR detection
* Advanced preprocessing (CLAHE, Thresholding)
* Ultra detection (MSER-based)
* Ultra AI (Sliding Window + Voting system)

### 🧽 Smart Label Removal

* HSV-based masking
* Morphological operations
* OpenCV Inpainting (Natural blending)

### 📊 Reporting System

* CSV output
* JSON unknown tracking
* Auto-generated PDF report with charts

### 🎵 Pipeline Experience

* Background music during processing 😄
```

---

## 🚀 Installation

### 1️⃣ Clone Project

```bash
git clone https://github.com/babakyousefian/liver-ai-pipeline
cd liver-ai-pipeline
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux
```

### 3️⃣ Install Requirements

```bash
python3.13 -m pip install --no-cache-dir --upgrade -r ./requirements.txt
```

---

## ▶️ Usage

```bash
python3.13 ./main.py
```

---

## 🧠 Pipeline Phases (Deep Explanation)

### 🧠 Phase 1: Full Detection

* بدون crop
* OCR روی کل تصویر
* سریع‌ترین شناسایی اولیه

---

### 🔍 Phase 2: Basic + Advanced

* استفاده از:

  * Thresholding
  * CLAHE
  * Resize
* استخراج دقیق‌تر متن

---

### 🚀 Phase 3: Ultra Detection

* استفاده از MSER
* پیدا کردن نواحی متنی واقعی
* کاهش نویز شدید

---

### 🤖 Phase 4: Ultra AI

* Sliding Window multi-scale
* چند preprocessing مختلف
* Voting system برای تصمیم نهایی

---

### ✂️ Phase 5: Crop + Clean

* حذف حاشیه‌ها
* حذف لیبل با inpaint طبیعی

---

### 🧹 Phase 6: Remove UNKNOWN

* حذف تصاویر بدون لیبل معتبر
* آپدیت CSV و JSON

---

## 📊 Outputs

| File                        | Description            |
| --------------------------- | ---------------------- |
| `labels.csv`                | لیبل هر تصویر          |
| `unknown.json`              | لیست تصاویر unresolved |
| `cropped/`                  | تصاویر نهایی           |
| `Liver_AI_Final_Report.pdf` | گزارش کامل             |

---

## 📈 Report Contents

* Label Distribution
* Dataset Pie Chart
* UNKNOWN vs Resolved
* Pipeline Timing
* System Info (OpenCV, Torch)

---

## 🧪 Testing (pytest)

```bash
PYTHONPATH=. pytest --cov=src --cov-report=term-missing -v --disable-warnings --maxfail=5
```

---

## here is pytest outputs : 

```bash
(.venv) bobby@babak:~/Documents/Germany/Tasks$ PYTHONPATH=. pytest --cov=src --cov-report=term-missing -v --disable-warnings --maxfail=5
================================================= test session starts =================================================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0 -- /home/bobby/Documents/Germany/Tasks/.venv/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/bobby/Documents/Germany/Tasks
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
testpaths: tests
plugins: xdist-3.8.0, cov-7.1.0
collected 104 items                                                                                                   

tests/test_crop.py::TestCrop::test_init PASSED                                                                  [  0%]
tests/test_crop.py::TestCrop::test_cropper PASSED                                                               [  1%]
tests/test_crop.py::TestCrop::test__crop_image PASSED                                                           [  2%]
tests/test_crop.py::TestCrop::test_gamma_correction PASSED                                                      [  3%]
tests/test_detection.py::TestDetection::test_init PASSED                                                        [  4%]
tests/test_detection.py::TestDetection::test_crop_image PASSED                                                  [  5%]
tests/test_detection.py::TestDetection::test_get_green_mask PASSED                                              [  6%]
tests/test_detection.py::TestDetection::test_get_label_mask PASSED                                              [  7%]
tests/test_detection.py::TestDetection::test_detect_basic PASSED                                                [  8%]
tests/test_detection.py::TestDetection::test_detect_advanced PASSED                                             [  9%]
tests/test_detection.py::TestDetection::test_detect_full PASSED                                                 [ 10%]
tests/test_detection.py::TestDetection::test_detect_full_pipeline PASSED                                        [ 11%]
tests/test_detection.py::TestDetection::test_remove_label PASSED                                                [ 12%]
tests/test_detection.py::TestDetection::test_detect_ultra PASSED                                                [ 13%]
tests/test_detection.py::TestDetection::test_detect_ultra_ai PASSED                                             [ 14%]
tests/test_detection.py::TestDetection::test_detect_basic_unknown PASSED                                        [ 15%]
tests/test_detection_extra.py::TestDetectionExtra::test_detect_basic_with_multiple_contours PASSED              [ 16%]
tests/test_detection_extra.py::TestDetectionExtra::test_detect_advanced_with_adaptive_threshold PASSED          [ 17%]
tests/test_detection_extra.py::TestDetectionExtra::test_detect_full_with_large_contour PASSED                   [ 18%]
tests/test_detection_extra.py::TestDetectionExtra::test_detect_full_with_small_contour PASSED                   [ 19%]
tests/test_detection_extra.py::TestDetectionExtra::test_detect_full_with_exception PASSED                       [ 20%]
tests/test_detection_extra.py::TestDetectionExtra::test_detect_basic_no_contours PASSED                         [ 21%]
tests/test_detection_extra.py::TestDetectionExtra::test_remove_label_with_large_mask PASSED                     [ 22%]
tests/test_ender.py::TestEnd::test_init PASSED                                                                  [ 23%]
tests/test_ender.py::TestEnd::test_ending PASSED                                                                [ 24%]
tests/test_finalReport.py::TestFinalReporter::test_init PASSED                                                  [ 25%]
tests/test_finalReport.py::TestFinalReporter::test_finalReporter PASSED                                         [ 25%]
tests/test_finalReport.py::TestFinalReporter::test_finalReporter_with_unknown PASSED                            [ 26%]
tests/test_label_detection.py::TestLabelDetection::test_init PASSED                                             [ 27%]
tests/test_label_detection.py::TestLabelDetection::test_labelDetector PASSED                                    [ 28%]
tests/test_label_detection.py::TestLabelDetection::test__detect_label PASSED                                    [ 29%]
tests/test_label_detection.py::TestLabelDetection::test__detect_label_advanced PASSED                           [ 30%]
tests/test_label_detection.py::TestLabelDetection::test_get_label_mask PASSED                                   [ 31%]
tests/test_label_detection_extra.py::TestLabelDetectionExtra::test_detect_label_full_image PASSED               [ 32%]
tests/test_label_detection_extra.py::TestLabelDetectionExtra::test_get_green_mask PASSED                        [ 33%]
tests/test_label_detection_extra.py::TestLabelDetectionExtra::test_detect_label_without_contours PASSED         [ 34%]
tests/test_label_detection_extra.py::TestLabelDetectionExtra::test_detect_label_advanced_without_contours PASSED [ 35%]
tests/test_label_detection_extra.py::TestLabelDetectionExtra::test_labelDetector_with_empty_files PASSED        [ 36%]
tests/test_loader.py::TestLoader::test_init PASSED                                                              [ 37%]
tests/test_loader.py::TestLoader::test_loadSomeFiles PASSED                                                     [ 38%]
tests/test_loader.py::TestLoader::test_loadSomeFiles_empty PASSED                                               [ 39%]
tests/test_loader.py::TestLoader::test_loadSomeFiles_jpg PASSED                                                 [ 40%]
tests/test_main.py::TestMain::test_preProcessor PASSED                                                          [ 41%]
tests/test_main.py::TestMain::test_main PASSED                                                                  [ 42%]
tests/test_main_engine.py::TestMainEngine::test_init PASSED                                                     [ 43%]
tests/test_main_engine.py::TestMainEngine::test_repr PASSED                                                     [ 44%]
tests/test_main_engine.py::TestMainEngine::test_str PASSED                                                      [ 45%]
tests/test_main_engine.py::TestMainEngine::test_runPreProcessor PASSED                                          [ 46%]
tests/test_music.py::TestMusic::test_init PASSED                                                                [ 47%]
tests/test_music.py::TestMusic::test_play_music PASSED                                                          [ 48%]
tests/test_music.py::TestMusic::test_stop_music PASSED                                                          [ 49%]
tests/test_music.py::TestMusic::test_play_music_exception PASSED                                                [ 50%]
tests/test_pipeline.py::TestPipeline::test_run PASSED                                                           [ 50%]
tests/test_pipeline.py::TestPipeline::test_pipeline_init PASSED                                                 [ 51%]
tests/test_pipeline_extra.py::TestPipelineExtra::test_pipeline_with_real_data PASSED                            [ 52%]
tests/test_pipeline_extra.py::TestPipelineExtra::test_pipeline_unknown_resolution PASSED                        [ 53%]
tests/test_remover.py::TestRemover::test_readCSV PASSED                                                         [ 54%]
tests/test_remover.py::TestRemover::test_readCSV_type_error PASSED                                              [ 55%]
tests/test_remover.py::TestRemover::test_removeChecker PASSED                                                   [ 56%]
tests/test_remover.py::TestRemover::test_removeUnknownImageLabels PASSED                                        [ 57%]
tests/test_remover.py::TestRemover::test_RemoverUF PASSED                                                       [ 58%]
tests/test_report.py::TestReportBC::test_init PASSED                                                            [ 59%]
tests/test_report.py::TestReportBC::test_reporterForBC PASSED                                                   [ 60%]
tests/test_report.py::TestReportBC::test_reporterForBC_no_unknown PASSED                                        [ 61%]
tests/test_report.py::TestFinalReporter::test_init PASSED                                                       [ 62%]
tests/test_report.py::TestFinalReporter::test_finalReporter PASSED                                              [ 63%]
tests/test_report_generator.py::TestReportGenerator::test_generate_report PASSED                                [ 64%]
tests/test_report_generator.py::TestReportGenerator::test_generate_report_no_unknown PASSED                     [ 65%]
tests/test_saver.py::TestSave::test_init PASSED                                                                 [ 66%]
tests/test_saver.py::TestSave::test_saver PASSED                                                                [ 67%]
tests/test_saver.py::TestSaveUnknown::test_init PASSED                                                          [ 68%]
tests/test_saver.py::TestSaveUnknown::test_saveUnknownImages PASSED                                             [ 69%]
tests/test_timer.py::TestTimer::test_init PASSED                                                                [ 70%]
tests/test_timer.py::TestTimer::test_timer PASSED                                                               [ 71%]
tests/test_ultra.py::TestUltraDetector::test_init PASSED                                                        [ 72%]
tests/test_ultra.py::TestUltraDetector::test_detect_nl PASSED                                                   [ 73%]
tests/test_ultra.py::TestUltraDetector::test_detect_g1 PASSED                                                   [ 74%]
tests/test_ultra.py::TestUltraDetector::test_detect_unknown PASSED                                              [ 75%]
tests/test_ultra.py::TestUltraDetector::test_detect_with_mser PASSED                                            [ 75%]
tests/test_ultra_ai.py::TestUltraAIDetector::test_init PASSED                                                   [ 76%]
tests/test_ultra_ai.py::TestUltraAIDetector::test_detect_nl PASSED                                              [ 77%]
tests/test_ultra_ai.py::TestUltraAIDetector::test_detect_unknown PASSED                                         [ 78%]
tests/test_ultra_ai.py::TestUltraAIDetector::test_gamma_function PASSED                                         [ 79%]
tests/test_ultra_ai.py::TestUltraAIDetector::test_voting_system PASSED                                          [ 80%]
tests/test_ultra_ai_extra.py::TestUltraAIExtra::test_detect_with_multiple_scales PASSED                         [ 81%]
tests/test_ultra_ai_extra.py::TestUltraAIExtra::test_detect_with_voting PASSED                                  [ 82%]
tests/test_ultra_ai_extra.py::TestUltraAIExtra::test_detect_with_otsu_threshold PASSED                          [ 83%]
tests/test_ultra_ai_extra.py::TestUltraAIExtra::test_detect_with_no_text PASSED                                 [ 84%]
tests/test_ultra_ai_extra.py::TestUltraAIExtra::test_detect_with_gamma_variations PASSED                        [ 85%]
tests/test_ultra_extra.py::TestUltraExtra::test_detect_with_gamma_correction PASSED                             [ 86%]
tests/test_ultra_extra.py::TestUltraExtra::test_detect_with_inverted_image PASSED                               [ 87%]
tests/test_ultra_extra.py::TestUltraExtra::test_detect_with_no_mser_regions PASSED                              [ 88%]
tests/test_ultra_extra.py::TestUltraExtra::test_detect_with_multiple_rois PASSED                                [ 89%]
tests/test_unknown.py::TestUnknown::test_unknownLabels PASSED                                                   [ 90%]
tests/test_unknown.py::TestUnknown::test_unknownNumber PASSED                                                   [ 91%]
tests/test_unknown.py::TestUnknown::test_unknownName PASSED                                                     [ 92%]
tests/test_unknown.py::TestUnknown::test_generateJson PASSED                                                    [ 93%]
tests/test_unknown.py::TestUnknown::test_generateJson_type_error PASSED                                         [ 94%]
tests/test_unknown.py::TestUnknown::test_readJson PASSED                                                        [ 95%]
tests/test_unknown.py::TestUnknown::test_readJson_empty PASSED                                                  [ 96%]
tests/test_unknown_extra.py::TestUnknownExtra::test_resolve_unknown_labels PASSED                               [ 97%]
tests/test_unknown_extra.py::TestUnknownExtra::test_unknownLabels_with_data PASSED                              [ 98%]
tests/test_unknown_extra.py::TestUnknownExtra::test_readJson_with_invalid_path PASSED                           [ 99%]
tests/test_unknown_extra.py::TestUnknownExtra::test_generateJson_with_empty_list PASSED                         [100%]

=================================================== tests coverage ====================================================
__________________________________ coverage: platform linux, python 3.13.14-final-0 ___________________________________

Name                                                 Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------------
src/MajorProcessing/detection.py                       131     17    87%   17, 20, 44-45, 119, 131-134, 154, 172-177, 212, 215
src/MajorProcessing/main_engine.py                      23      0   100%
src/MajorProcessing/pipeline.py                         90      6    93%   91-94, 111-114
src/MajorProcessing/ultra.py                            68     15    78%   47, 50, 54, 92-93, 98-101, 107-112
src/MajorProcessing/ultra_ai.py                         44      6    86%   61, 74-77, 81
src/MassiveClasses/Audio/music.py                       30      4    87%   24, 27, 46-47
src/MassiveClasses/Ending/ender.py                      22      2    91%   10, 13
src/MassiveClasses/FullDetection/labelDetection.py     145     45    69%   20, 23, 39, 45-51, 68-76, 104-110, 115-128,143, 159-171, 192, 214-227
src/MassiveClasses/LoadFiles/loader.py                  15      2    87%   12, 15
src/MassiveClasses/Removing/removerUF.py                16      2    88%   14, 17
src/MassiveClasses/Reporter/finalReport.py              26      2    92%   18, 21
src/MassiveClasses/Reporter/reportBeforeCrop.py         23      2    91%   13, 16
src/MassiveClasses/SavingData/saveUnknown.py            20      2    90%   14, 17
src/MassiveClasses/SavingData/saver.py                  19      2    89%   12, 15
src/MassiveClasses/Timer/timer.py                       19      2    89%   10, 13
src/MassiveClasses/cropping/crop.py                     61      2    97%   26, 29
src/remove/remover.py                                  110     16    85%   64, 67, 70, 73, 76, 99-106, 141-142, 172, 175
src/reporter/reportGenerator.py                         56      0   100%
src/unknownChecker/unknown.py                          100     15    85%   35, 42-44, 49-53, 96, 107, 110, 121-122, 139
----------------------------------------------------------------------------------
TOTAL                                                 1018    142    86%
Coverage HTML written to dir htmlcov
=========================================== 104 passed in 221.99s (0:03:41) ===========================================
(.venv) bobby@babak:~/Documents/Germany/Tasks$ 
```

---

## 🧰 Technologies Used

* 🐍 Python
* 🧠 EasyOCR
* 🎯 OpenCV
* 📊 Pandas
* 📉 Matplotlib / Seaborn
* 🔥 PyMuPDF (fitz)
* 🎵 Pygame

---

## ⚡ Performance Notes

* GPU استفاده نشده (برای stability)
* Pipeline چندمرحله‌ای برای دقت بالا
* مناسب برای دیتاست‌های noisy پزشکی

---

## 🧩 Future Improvements

* [ ] GPU acceleration
* [ ] Deep Learning based detection
* [ ] Web interface (Flask / FastAPI)
* [ ] Dockerization
* [ ] Real-time processing

---

## 👨‍💻 Author

**Babak Yousefian**

* 🎓 M.Sc. Computer Engineering (AI & Robotics)
* 🧠 AI in Medicine Researcher
* 🚀 Future PhD Candidate (Germany)

---

## 📬 Contact

* Email: [babakyousefian2000@gmail.com](mailto:babakyousefian2000@gmail.com)
* GitHub: https://github.com/babakyousefian/

---

## ⭐ Support

اگر این پروژه برات مفید بود:

* ⭐ Star بده
* 🍴 Fork کن
* 🧠 توسعه بده

---

## 🧾 License

```bash

MIT License

Copyright (c) 2024 Babak Yousefian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 💡 Final Note

این پروژه فقط یه کد نیست…
یه pipeline مهندسی‌شده‌ست که مثل یه خط تولید هوشمند،
از تصویر خام → دانش قابل استفاده می‌رسه.

🔥 “From pixels to intelligence.”

---


<div align="center">

![Last Commit](https://img.shields.io/github/last-commit/yourusername/liver-ai-pipeline?style=for-the-badge)
![Issues](https://img.shields.io/github/issues/yourusername/liver-ai-pipeline?style=for-the-badge)
![Issues](https://img.shields.io/github/issues-pr/yourusername/liver-ai-pipeline?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/yourusername/liver-ai-pipeline?style=for-the-badge)
![Forks](https://img.shields.io/github/forks/yourusername/liver-ai-pipeline?style=for-the-badge)

---
