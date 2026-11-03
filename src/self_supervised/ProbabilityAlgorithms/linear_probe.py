# src/self_supervised/linear_probe.py

"""
Linear Probe Evaluation for SSL Methods.
این فایل به ما می‌گوید کدام SSL encoder واقعاً بهتر یاد گرفته.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_with_linear_probe(
    encoder: nn.Module,
    labeled_loader: DataLoader,
    device: str = "cpu",
    num_classes: int = 4,
    test_size: float = 0.3,
) -> dict:
    """
    Evaluate SSL encoder quality using Linear Probe.

    ✅ FIX: uses class_weight='balanced' for imbalanced datasets.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    encoder.eval()
    encoder.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(labeled_loader, desc="Extracting embeddings", leave=False):
            images = images.to(device)

            if hasattr(encoder, "get_embedding"):
                emb = encoder.get_embedding(images)
            else:
                emb = encoder(images)
                if emb.dim() > 2:
                    emb = emb.view(emb.size(0), -1)

            all_embeddings.append(emb.cpu().float().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ✅ FIX: class_weight='balanced' for imbalanced classes (G3 has only 70)
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",  # ← مهم!
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return {
        "linear_probe_accuracy": float(accuracy_score(y_test, y_pred)),
        "linear_probe_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "linear_probe_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
    }


def build_labeled_eval_loader(
    labels_csv: str = "finalReports/labels_extended.csv",
    image_dir: str = "finalReports/cropped_extended",
    input_size: int = 224,
    batch_size: int = 32,
    max_per_class: int = 200,
) -> DataLoader:
    """
    Build labeled DataLoader for Linear Probe.

    ✅ FIX: Default uses extended dataset (real + synthetic).
    Falls back to original if extended not available.
    """

    import os
    import cv2
    import pandas as pd
    from PIL import Image
    import torchvision.transforms as T

    # ✅ FIX: اگر دیتاست گسترده نیست، به اصلی برگرد
    if not os.path.exists(labels_csv):
        print(f"⚠️ {labels_csv} not found. Falling back to labels.csv")
        labels_csv = "finalReports/labels.csv"
    if not os.path.exists(image_dir):
        print(f"⚠️ {image_dir} not found. Falling back to cropped/")
        image_dir = "finalReports/cropped"

    CLASS_NAMES = ["NL", "G1", "G2", "G3"]

    df = pd.read_csv(labels_csv)

    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # گروه‌بندی بر اساس کلاس
    class_to_paths = {c: [] for c in CLASS_NAMES}
    for _, row in df.iterrows():
        label = row["liverLabel"]
        if label in CLASS_NAMES:
            img_name = row["liverName"]
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                class_to_paths[label].append(img_path)

    # نمونه‌گیری متعادل
    eval_paths = []
    eval_labels = []
    for class_name, paths in class_to_paths.items():
        selected = paths[:max_per_class]
        for p in selected:
            eval_paths.append(p)
            eval_labels.append(CLASS_NAMES.index(class_name))

    class _DS(torch.utils.data.Dataset):
        def __init__(self, paths, labels, tf):
            self.paths = paths
            self.labels = labels
            self.tf = tf

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = cv2.imread(self.paths[idx])
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            return self.tf(img), self.labels[idx]

    dataset = _DS(eval_paths, eval_labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"✅ Labeled Eval Loader: {len(eval_paths)} samples")
    for c in CLASS_NAMES:
        print(f"   {c}: {len(class_to_paths[c][:max_per_class])}")

    return loader
