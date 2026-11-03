import matplotlib

matplotlib.use("Agg")  # 🔥 GUI warnings


import os
import time
from datetime import datetime

import cv2
import fitz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def generate_report(csv_path, output_pdf, start_time, end_time, input_count):
    REPORT_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\report_assets"

    os.makedirs(REPORT_DIR, exist_ok=True)

    # ============================
    # LOAD DATA
    # ============================

    df = pd.read_csv(csv_path)

    total = len(df)

    label_counts = df["liverLabel"].value_counts()

    unknown_count = (df["liverLabel"] == "UNKNOWN").sum()

    resolved = total - unknown_count

    # ============================
    # CREATE GRAPH 1
    # LABEL DISTRIBUTION
    # ============================

    plt.figure(figsize=(8, 5))

    sns.barplot(x=label_counts.index, y=label_counts.values)

    plt.title("Liver Label Distribution")

    plt.xlabel("Label")
    plt.ylabel("Count")

    plt.tight_layout()

    label_chart = os.path.join(REPORT_DIR, "labels.png")

    plt.savefig(label_chart, dpi=300)

    plt.close()

    # ============================
    # PIE CHART
    # ============================

    plt.figure(figsize=(7, 7))

    plt.pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%")

    plt.title("Dataset Distribution")

    pie_chart = os.path.join(REPORT_DIR, "pie.png")

    plt.savefig(pie_chart, dpi=300)

    plt.close()

    # ============================
    # UNKNOWN GRAPH
    # ============================

    plt.figure(figsize=(6, 4))

    sns.barplot(x=["Resolved", "UNKNOWN"], y=[resolved, unknown_count])

    plt.title("Final Resolution Status")

    status_chart = os.path.join(REPORT_DIR, "status.png")

    plt.savefig(status_chart, dpi=300)

    plt.close()

    # ============================
    # CREATE PDF
    # ============================

    doc = fitz.open()

    page = doc.new_page()

    text = f"""

LIVER AI PIPELINE REPORT
========================


Generated:
{datetime.now()}


AUTHOR:
Babak Yousefian


PIPELINE TIME

Start:
{start_time}


End:
{end_time}



DATASET

Input Images:
{input_count}


Final Images:
{total}



LABEL STATISTICS

{label_counts.to_string()}



UNKNOWN STATUS

Resolved:
{resolved}


Remaining UNKNOWN:
{unknown_count}



SYSTEM

OpenCV:
{cv2.__version__}


Torch:
{torch.__version__}


========================


"""

    page.insert_text((50, 50), text, fontsize=11)

    # Page 2

    page = doc.new_page()

    page.insert_text((50, 50), "LABEL DISTRIBUTION GRAPH", fontsize=14)

    page.insert_image(fitz.Rect(50, 100, 550, 450), filename=label_chart)

    # Page 3

    page = doc.new_page()

    page.insert_text((50, 50), "DATASET PIE CHART", fontsize=14)

    page.insert_image(fitz.Rect(50, 100, 550, 450), filename=pie_chart)

    # Page 4

    page = doc.new_page()

    page.insert_text((50, 50), "FINAL STATUS", fontsize=14)

    page.insert_image(fitz.Rect(50, 100, 550, 450), filename=status_chart)

    doc.save(output_pdf)

    print("\n📄 PDF REPORT CREATED:")

    print(output_pdf)
