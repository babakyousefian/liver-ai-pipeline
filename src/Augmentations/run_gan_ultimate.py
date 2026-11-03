# run_gan_ultimate.py
"""
Ultimate GAN Training Script

Usage:
    # Single class with default params
    python run_gan_ultimate.py --class_name G3 --epochs 300

    # With custom params
    python run_gan_ultimate.py --class_name G3 --epochs 300 --feature_maps 32 --num_blocks 5

    # Skip FID
    python run_gan_ultimate.py --class_name G3 --epochs 300 --skip_fid

Author: Babak Yousefian
Version: 1.0.0
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.self_supervised.OptimizationsAlgorithms.Augmentations.gan_trainer import create_ultimate_gan_trainer


# ================================================================
# ✅ DATASET
# ================================================================


class ClassDataset(Dataset):
    """Dataset for a specific class."""

    def __init__(self, image_paths, target_size=224, augment=True):
        self.image_paths = image_paths
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))

        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, 0


# ================================================================
# ✅ MAIN
# ================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, default="G3", choices=["NL", "G1", "G2", "G3"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--feature_maps", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--labels_csv", type=str, default="finalReports/labels.csv")
    parser.add_argument("--image_dir", type=str, default="finalReports/cropped")
    parser.add_argument("--output_dir", type=str, default="finalReports/selfSupervisedLearning/GAN_Ultimate")
    parser.add_argument("--num_synthetic", type=int, default=1000)
    parser.add_argument("--num_fid_real", type=int, default=200)
    parser.add_argument("--num_fid_fake", type=int, default=200)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--target_fid", type=float, default=None, help="Target FID for early stopping")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # ✅ CLASS-SPECIFIC CONFIG
    # ============================================================
    CLASS_CONFIGS = {
        "G3": {"feature_maps": 32, "num_blocks": 5, "latent_dim": 128, "batch_size": 8},
        "G2": {"feature_maps": 48, "num_blocks": 5, "latent_dim": 128, "batch_size": 16},
        "G1": {"feature_maps": 64, "num_blocks": 6, "latent_dim": 128, "batch_size": 32},
        "NL": {"feature_maps": 64, "num_blocks": 6, "latent_dim": 128, "batch_size": 32},
    }

    config = CLASS_CONFIGS.get(args.class_name, {})

    # Override with command-line args if provided
    feature_maps = args.feature_maps if args.feature_maps else config.get("feature_maps", 32)
    num_blocks = args.num_blocks if args.num_blocks else config.get("num_blocks", 5)
    latent_dim = args.latent_dim if args.latent_dim else config.get("latent_dim", 128)
    batch_size = args.batch_size if args.batch_size else config.get("batch_size", 16)

    print(f"\n⚙️ Config for {args.class_name}:")
    print(f"   Feature Maps: {feature_maps}")
    print(f"   Num Blocks: {num_blocks}")
    print(f"   Latent Dim: {latent_dim}")
    print(f"   Batch Size: {batch_size}")

    # ============================================================
    # ✅ LOAD DATA
    # ============================================================
    print(f"\n📂 Loading labels from: {args.labels_csv}")
    df = pd.read_csv(args.labels_csv)

    class_paths = []
    for _, row in df.iterrows():
        if row["liverLabel"] == args.class_name:
            img_name = row["liverName"]
            for ext in ["", ".png", ".jpg", ".jpeg"]:
                img_path = os.path.join(args.image_dir, img_name + ext)
                if os.path.exists(img_path):
                    class_paths.append(img_path)
                    break

    print(f"\n📊 Class: {args.class_name}")
    print(f"   Images found: {len(class_paths)}")

    if len(class_paths) < 10:
        print(f"❌ Not enough images for {args.class_name}")
        return

    # ============================================================
    # ✅ DATALOADER
    # ============================================================
    dataset = ClassDataset(class_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n   Device: {device}")
    print(f"   Batches: {len(dataloader)}")

    # ============================================================
    # ✅ CREATE TRAINER
    # ============================================================
    trainer = create_ultimate_gan_trainer(
        latent_dim=latent_dim,
        img_shape=(3, 224, 224),
        device=device,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        feature_maps=feature_maps,
        num_blocks=num_blocks,
        use_feature_matching=True,
        feature_matching_weight=10.0,
        use_r1=True,
        r1_weight=10.0,
        use_ada=True,
        use_ema=True,
        loss_type="hinge",
        n_critic=1,
    )

    # ============================================================
    # ✅ TRAIN
    # ============================================================
    save_dir = os.path.join(args.output_dir, args.class_name)
    os.makedirs(save_dir, exist_ok=True)

    # Optional: FID function for early stopping
    fid_fn = None
    if args.target_fid is not None:
        try:
            from src.self_supervised.OptimizationsAlgorithms.Augmentations.fid_score import calculate_fid_from_paths

            def fid_fn(trainer_instance):
                # Generate 200 images
                fake_images = trainer_instance.generate_images(200, use_ema=True)
                # Save temporarily
                temp_dir = os.path.join(save_dir, "temp_fid")
                os.makedirs(temp_dir, exist_ok=True)
                for i, img in enumerate(fake_images):
                    cv2.imwrite(
                        os.path.join(temp_dir, f"temp_{i:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    )

                # Compute FID
                real_sample = random.sample(class_paths, min(200, len(class_paths)))
                fake_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
                fid = calculate_fid_from_paths(
                    real_paths=real_sample,
                    fake_paths=fake_paths[:200],
                    device=device,
                    verbose=False,
                )

                # Cleanup
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)

                return fid

        except ImportError:
            print("⚠️ FID function not available")

    # Train
    history = trainer.train(
        dataloader=dataloader,
        epochs=args.epochs,
        save_interval=25,
        save_dir=save_dir,
        generate_interval=25,
        num_generate=16,
        target_fid=args.target_fid,
        fid_fn=fid_fn,
    )

    # ============================================================
    # ✅ GENERATE SYNTHETIC IMAGES
    # ============================================================
    print(f"\n🎨 Generating {args.num_synthetic} synthetic images...")
    synthetic = trainer.generate_images(args.num_synthetic, use_ema=True)

    synthetic_dir = os.path.join(save_dir, "synthetic")
    os.makedirs(synthetic_dir, exist_ok=True)

    for i, img in enumerate(synthetic):
        save_path = os.path.join(synthetic_dir, f"synthetic_{args.class_name}_{i:04d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved {len(synthetic)} images to: {synthetic_dir}")

    # ============================================================
    # ✅ FID
    # ============================================================
    if args.skip_fid:
        print("\n⏭️ FID skipped")
        return

    try:
        from src.self_supervised.OptimizationsAlgorithms.Augmentations.fid_score import calculate_fid_from_paths

        random.seed(42)
        real_sample = random.sample(class_paths, min(args.num_fid_real, len(class_paths)))

        synthetic_sample = [
            os.path.join(synthetic_dir, f)
            for f in sorted(os.listdir(synthetic_dir))
            if f.endswith(".png")
        ][: args.num_fid_fake]

        print(f"\n📊 Computing FID...")
        print(f"   Real: {len(real_sample)}")
        print(f"   Fake: {len(synthetic_sample)}")

        fid_value = calculate_fid_from_paths(
            real_paths=real_sample,
            fake_paths=synthetic_sample,
            device=device,
            batch_size=16,
            verbose=True,
        )

        # Save FID
        with open(os.path.join(save_dir, "fid_score.txt"), "w") as f:
            f.write(f"Class: {args.class_name}\n")
            f.write(f"FID: {fid_value:.4f}\n")
            f.write(f"Real samples: {len(real_sample)}\n")
            f.write(f"Synthetic samples: {len(synthetic_sample)}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Feature Maps: {feature_maps}\n")
            f.write(f"Num Blocks: {num_blocks}\n")

        with open(os.path.join(save_dir, "fid_score.json"), "w") as f:
            json.dump({
                "class_name": args.class_name,
                "fid": fid_value,
                "epochs": args.epochs,
                "feature_maps": feature_maps,
                "num_blocks": num_blocks,
                "latent_dim": latent_dim,
            }, f, indent=4)

        print(f"\n✅ FID: {fid_value:.2f}")
        print(f"   Saved to: {save_dir}/fid_score.txt")

    except Exception as e:
        print(f"\n⚠️ FID failed: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # ✅ SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print(f"✅ ULTIMATE GAN TRAINING COMPLETE FOR {args.class_name}")
    print("=" * 70)
    print(f"   Class: {args.class_name}")
    print(f"   Original images: {len(class_paths)}")
    print(f"   Synthetic images: {len(synthetic)}")
    print(f"   Checkpoints: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    