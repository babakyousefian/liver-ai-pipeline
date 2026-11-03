# src/reporter/lime_reporter.py

"""
LIME Reporter for Model Interpretability (ULTIMATE EDITION)

This module provides LIME (Local Interpretable Model-agnostic Explanations)
for explaining model predictions on ultrasound images.

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# ================================================================
# ✅ TRY TO IMPORT LIME
# ================================================================

try:
    from lime import lime_image
    from lime.lime_image import LimeImageExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Install with: pip install lime")

# ================================================================
# ✅ LIME EXPLANATION FUNCTION
# ================================================================


def explain_prediction(
    model: torch.nn.Module,
    image: np.ndarray,
    label: int,
    class_names: List[str],
    output_dir: str,
    device: str = "cpu",
    num_features: int = 10,
    top_labels: int = 4,
    hide_color: int = 0,
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Explain a single prediction using LIME.

    Args:
        model: PyTorch model
        image: Input image as numpy array (H, W, C) or (H, W)
        label: True label index
        class_names: List of class names
        output_dir: Directory to save explanations
        device: Device to run on
        num_features: Number of features to show
        top_labels: Number of top labels to consider
        hide_color: Color to hide (0 = black)
        num_samples: Number of samples for LIME

    Returns:
        Dictionary with explanation results
    """

    if not LIME_AVAILABLE:
        print("⚠️ LIME not available. Skipping explanation.")
        return {"status": "failed", "error": "LIME not installed"}

    os.makedirs(output_dir, exist_ok=True)

    # ✅ Set model to evaluation mode
    model.eval()
    model.to(device)

    # ✅ Ensure image is numpy array
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
    else:
        image_np = image.copy()

    # ✅ Ensure image is 3-channel RGB
    if image_np.ndim == 2:
        image_np = np.stack([image_np, image_np, image_np], axis=2)
    elif image_np.ndim == 3 and image_np.shape[2] == 1:
        image_np = np.concatenate([image_np, image_np, image_np], axis=2)

    # ✅ Normalize to 0-255 if needed
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # ✅ Resize to 224x224 if needed
    if image_np.shape[:2] != (224, 224):
        image_np = cv2.resize(image_np, (224, 224))

    # ✅ Create prediction function for LIME
    def predict_fn(images: np.ndarray) -> np.ndarray:
        """Predict function for LIME."""
        # Convert numpy images to torch tensor
        if images.ndim == 3:
            images = images[np.newaxis, ...]

        # Convert to tensor and transpose
        images_tensor = torch.tensor(images, dtype=torch.float32)
        if images_tensor.ndim == 4:
            images_tensor = images_tensor.permute(0, 3, 1, 2)

        # Normalize
        images_tensor = images_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_tensor = (images_tensor - mean) / std

        # Move to device
        images_tensor = images_tensor.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(images_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    # ✅ Create LIME explainer
    explainer = LimeImageExplainer()

    # ✅ Generate explanation
    try:
        explanation = explainer.explain_instance(
            image_np.astype(np.double),
            predict_fn,
            top_labels=top_labels,
            num_features=num_features,
            hide_color=hide_color,
            num_samples=num_samples,
        )
    except Exception as e:
        print(f"⚠️ LIME explanation failed: {e}")
        return {"status": "failed", "error": str(e)}

    # ✅ Get explanation for the predicted label
    pred_label = label if label is not None else 0

    # ✅ Get image and mask
    try:
        temp, mask = explanation.get_image_and_mask(
            pred_label, positive_only=True, num_features=num_features, hide_rest=False
        )
    except Exception as e:
        print(f"⚠️ Could not get image and mask: {e}")
        temp, mask = image_np, None

    # ✅ Save visualization
    plt.figure(figsize=(12, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np.astype(np.uint8))
    plt.title(f'Original Image\nTrue: {class_names[label] if label is not None else "Unknown"}')
    plt.axis("off")

    # LIME explanation
    plt.subplot(1, 3, 2)
    plt.imshow(temp)
    plt.title(f"LIME Explanation\nPositive Features: {num_features}")
    plt.axis("off")

    # Mask overlay
    if mask is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(image_np.astype(np.uint8))
        plt.imshow(mask, cmap="Reds", alpha=0.3)
        plt.title("LIME Mask Overlay")
        plt.axis("off")

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, f"lime_explanation_label_{label}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ LIME explanation saved: {output_path}")

    return {
        "status": "success",
        "explanation": explanation,
        "output_path": output_path,
        "label": label,
        "class_name": class_names[label] if label is not None else "Unknown",
        "top_labels": (explanation.top_labels if hasattr(explanation, "top_labels") else []),
    }


# ================================================================
# ✅ BATCH EXPLANATION FOR DATALOADER
# ================================================================


def explain_predictions_lime(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_dir: str,
    class_names: List[str],
    device: str = "cpu",
    num_samples: int = 5,
    num_features: int = 10,
) -> Dict[str, Any]:
    """
    Generate LIME explanations for a batch of predictions.

    Returns:
        dict with keys: lime_path, num_explained, status
    """
    if not LIME_AVAILABLE:
        return {"lime_path": None, "num_explained": 0, "status": "lime_unavailable"}

    os.makedirs(output_dir, exist_ok=True)

    results = []
    last_saved_path = None

    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break

        # ✅ Handle different batch formats
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images = batch
            labels = torch.zeros(images.size(0), dtype=torch.long)

        # Get first image
        image = images[0].cpu()
        label = int(labels[0].item()) if hasattr(labels[0], "item") else int(labels[0])

        # Convert to numpy
        image_np = image.numpy()
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        # Explain
        try:
            result = explain_prediction(
                model=model,
                image=image_np,
                label=label,
                class_names=class_names,
                output_dir=output_dir,
                device=device,
                num_features=num_features,
            )
            if result.get("status") == "success":
                last_saved_path = result.get("output_path")
            result["sample_idx"] = idx
            results.append(result)
        except Exception as e:
            print(f"⚠️ LIME failed for sample {idx}: {e}")

    return {
        "lime_path": last_saved_path,
        "num_explained": len([r for r in results if r.get("status") == "success"]),
        "status": "success" if results else "failed",
        "all_results": results,
    }

# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing LIME Reporter")
    print("=" * 80)

    if LIME_AVAILABLE:
        # Create dummy model
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.fc = torch.nn.Linear(64 * 224 * 224, 4)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = DummyModel()
        class_names = ["NL", "G1", "G2", "G3"]

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Test explanation
        print("\n📌 Testing LIME explanation...")
        result = explain_prediction(
            model=model,
            image=dummy_image,
            label=0,
            class_names=class_names,
            output_dir="test_lime_output",
            device="cpu",
        )

        print(f"   Status: {result.get('status')}")
        print(f"   Output: {result.get('output_path')}")

        print("\n✅ LIME tests passed!")
    else:
        print("\n⚠️ LIME not available. Skipping tests.")

    print("=" * 80)
