# run_gan.py
"""
Standalone GAN training script with FID-based best checkpoint selection.

Strategy:
- Train GAN with standard gan.train()
- Evaluate FID on all saved checkpoints
- Load best checkpoint by FID
- Generate final synthetic images from best checkpoint

Usage:
    python run_gan.py --class_name G3 --epochs 300
    python run_gan.py --class_name G2 --epochs 400
    python run_gan.py --class_name G1 --epochs 500
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.self_supervised.OptimizationsAlgorithms.Augmentations.gan import create_gan_trainer, DCGANGenerator


# ================================================================
# ✅ DATASET
# ================================================================
class ClassDataset(Dataset):
    """Dataset for a specific class."""

    def __init__(self, image_paths, target_size=224):
        self.image_paths = image_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))

        img = img.astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, 0


# ================================================================
# ✅ FID HELPER
# ================================================================
def compute_fid_for_checkpoint(
    checkpoint_path,
    real_paths,
    latent_dim,
    feature_maps,
    device,
    num_real=100,
    num_fake=100,
    temp_dir="temp_fid_eval",
):
    """
    Compute FID for a saved checkpoint.

    Returns:
        fid: float (or float('inf') on failure)
    """
    from src.self_supervised.OptimizationsAlgorithms.Augmentations.fid_score import calculate_fid_from_paths

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create generator
        gen = DCGANGenerator(
            latent_dim=latent_dim,
            img_channels=3,
            feature_maps=feature_maps,
        ).to(device)
        gen.load_state_dict(checkpoint["generator_state_dict"])
        gen.eval()

        # Generate fake images
        with torch.no_grad():
            z = torch.randn(num_fake, latent_dim).to(device)
            fake_images = gen(z)
            fake_images = (fake_images + 1) / 2
            fake_images = fake_images.cpu().numpy()
            fake_images = np.transpose(fake_images, (0, 2, 3, 1))
            fake_images = (fake_images * 255).astype(np.uint8)

        # Save fake images to temp dir
        os.makedirs(temp_dir, exist_ok=True)
        for i, img in enumerate(fake_images):
            save_path = os.path.join(temp_dir, f"temp_{i:04d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Sample real images
        random.seed(42)
        real_sample = random.sample(real_paths, min(num_real, len(real_paths)))

        # Fake paths
        fake_paths = [
            os.path.join(temp_dir, f)
            for f in sorted(os.listdir(temp_dir))
            if f.endswith(".png")
        ][:num_fake]

        # Compute FID
        fid = calculate_fid_from_paths(
            real_paths=real_sample,
            fake_paths=fake_paths,
            device=device,
            batch_size=16,
            verbose=False,
        )

        # Cleanup
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

        # Cleanup GPU
        del gen
        torch.cuda.empty_cache()

        return fid

    except Exception as e:
        print(f"   ⚠️ FID failed for {os.path.basename(checkpoint_path)}: {e}")
        return float("inf")


# ================================================================
# ✅ MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, default="G3",
                        choices=["NL", "G1", "G2", "G3"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--labels_csv", type=str, default="finalReports/labels.csv")
    parser.add_argument("--image_dir", type=str, default="finalReports/cropped")
    parser.add_argument("--output_dir", type=str,
                        default="finalReports/selfSupervisedLearning/GAN")
    parser.add_argument("--num_synthetic", type=int, default=1000)
    parser.add_argument("--num_fid_real", type=int, default=100)
    parser.add_argument("--num_fid_fake", type=int, default=100)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--feature_maps", type=int, default=64)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--use_wgan", action="store_true", default=False)
    parser.add_argument("--save_interval", type=int, default=25,
                        help="Save checkpoint every N epochs (default: 25)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # ✅ LOAD LABELS
    # ================================================================
    print(f"\n📂 Loading labels from: {args.labels_csv}")
    df = pd.read_csv(args.labels_csv)

    class_paths = []
    for _, row in df.iterrows():
        if row["liverLabel"] == args.class_name:
            img_name = row["liverName"]
            img_path = os.path.join(args.image_dir, img_name)
            if os.path.exists(img_path):
                class_paths.append(img_path)
            else:
                img_path_png = os.path.join(args.image_dir, f"{img_name}.png")
                if os.path.exists(img_path_png):
                    class_paths.append(img_path_png)
                else:
                    base = os.path.splitext(img_name)[0]
                    img_path_no_ext = os.path.join(args.image_dir, base)
                    if os.path.exists(img_path_no_ext):
                        class_paths.append(img_path_no_ext)

    print(f"\n📊 Class: {args.class_name}")
    print(f"   Images found: {len(class_paths)}")

    if len(class_paths) < 10:
        print(f"❌ Not enough images for {args.class_name}. Need at least 10.")
        return

    # ================================================================
    # ✅ DATALOADER
    # ================================================================
    dataset = ClassDataset(class_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    print(f"   Batches: {len(dataloader)}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")

    save_dir = os.path.join(args.output_dir, args.class_name)
    os.makedirs(save_dir, exist_ok=True)

    # ================================================================
    # ✅ CREATE GAN
    # ================================================================
    gan = create_gan_trainer(
        latent_dim=args.latent_dim,
        img_shape=(3, 224, 224),
        device=device,
        lr_g=args.lr * 2,
        lr_d=args.lr,
        use_wgan=args.use_wgan,
        feature_maps=args.feature_maps,
        n_critic=args.n_critic,
    )

    # ================================================================
    # ✅ TRAIN
    # ================================================================
    print("\n" + "=" * 70)
    print("🚀 Training GAN...")
    print("=" * 70)

    gan.train(
        dataloader=dataloader,
        epochs=args.epochs,
        save_interval=args.save_interval,
        save_dir=save_dir,
        generate_interval=args.save_interval,
        num_generate=16,
    )

    # ================================================================
    # ✅ SKIP_FID CHECK
    # ================================================================
    synthetic_dir = os.path.join(save_dir, "synthetic")
    os.makedirs(synthetic_dir, exist_ok=True)

    if args.skip_fid:
        print("\n⏭️ Skipping FID computation (--skip_fid)")
        print("   Using final checkpoint for synthetic generation")

        synthetic = gan.generate_images(args.num_synthetic)
        for f in os.listdir(synthetic_dir):
            os.remove(os.path.join(synthetic_dir, f))
        for i, img in enumerate(synthetic):
            save_path = os.path.join(synthetic_dir, f"synthetic_{args.class_name}_{i:04d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"✅ Saved {len(synthetic)} images to: {synthetic_dir}")
        return

    # ================================================================
    # ✅ FIND BEST CHECKPOINT BY FID
    # ================================================================
    print("\n" + "=" * 70)
    print("🔍 Finding best checkpoint by FID...")
    print("=" * 70)

    # List all saved checkpoints (exclude gan_final.pth)
    checkpoint_files = sorted([
        f for f in os.listdir(save_dir)
        if f.startswith("gan_epoch_") and f.endswith(".pth")
    ], key=lambda x: int(x.split("_")[-1].split(".")[0]))

    print(f"Found {len(checkpoint_files)} checkpoints:")
    for f in checkpoint_files:
        print(f"   - {f}")

    if len(checkpoint_files) == 0:
        print("⚠️ No checkpoint files found. Using final model.")
        checkpoint_files = []

    # Evaluate FID for each checkpoint
    best_fid = float("inf")
    best_checkpoint = None
    best_epoch = 0
    fid_results = {}

    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(save_dir, ckpt_file)
        epoch_num = int(ckpt_file.split("_")[-1].split(".")[0])

        print(f"\n📊 Evaluating {ckpt_file} (epoch {epoch_num})...")

        fid = compute_fid_for_checkpoint(
            checkpoint_path=ckpt_path,
            real_paths=class_paths,
            latent_dim=args.latent_dim,
            feature_maps=args.feature_maps,
            device=device,
            num_real=args.num_fid_real,
            num_fake=args.num_fid_fake,
            temp_dir=os.path.join(save_dir, f"temp_fid_{epoch_num}"),
        )

        fid_results[epoch_num] = fid
        print(f"   ✅ Epoch {epoch_num}: FID = {fid:.2f}")

        if fid < best_fid:
            best_fid = fid
            best_checkpoint = ckpt_file
            best_epoch = epoch_num

    # ================================================================
    # ✅ LOAD BEST CHECKPOINT
    # ================================================================
    if best_checkpoint is None:
        print("⚠️ No valid checkpoint found. Using final model.")
        synthetic = gan.generate_images(args.num_synthetic)
    else:
        print("\n" + "=" * 70)
        print(f"🏆 BEST CHECKPOINT: {best_checkpoint}")
        print(f"   Epoch: {best_epoch}")
        print(f"   FID: {best_fid:.2f}")
        print("=" * 70)

        best_path = os.path.join(save_dir, best_checkpoint)
        best_data = torch.load(best_path, map_location=device)

        # Load into gan
        gan.generator.load_state_dict(best_data["generator_state_dict"])
        gan.discriminator.load_state_dict(best_data["discriminator_state_dict"])

        # Generate synthetic from best
        print(f"\n🎨 Generating {args.num_synthetic} synthetic images from BEST checkpoint...")
        synthetic = gan.generate_images(args.num_synthetic)

    # ================================================================
    # ✅ SAVE SYNTHETIC IMAGES
    # ================================================================
    for f in os.listdir(synthetic_dir):
        os.remove(os.path.join(synthetic_dir, f))

    for i, img in enumerate(synthetic):
        save_path = os.path.join(synthetic_dir, f"synthetic_{args.class_name}_{i:04d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved {len(synthetic)} images to: {synthetic_dir}")

    # ================================================================
    # ✅ SAVE FID REPORT
    # ================================================================
    fid_report = {
        "class_name": args.class_name,
        "best_fid": best_fid if best_checkpoint else None,
        "best_epoch": best_epoch if best_checkpoint else None,
        "best_checkpoint": best_checkpoint,
        "epochs_trained": args.epochs,
        "fid_history": {str(k): float(v) for k, v in fid_results.items()},
        "latent_dim": args.latent_dim,
        "feature_maps": args.feature_maps,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(save_dir, "fid_score.json"), "w") as f:
        json.dump(fid_report, f, indent=4)

    with open(os.path.join(save_dir, "fid_score.txt"), "w") as f:
        f.write(f"Class: {args.class_name}\n")
        f.write(f"Best FID: {best_fid:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Checkpoint: {best_checkpoint}\n")
        f.write(f"Total Epochs: {args.epochs}\n")
        f.write(f"\nFID History:\n")
        for epoch_num, fid in sorted(fid_results.items()):
            f.write(f"  Epoch {epoch_num}: {fid:.2f}\n")

    # ================================================================
    # ✅ FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print(f"✅ GAN TRAINING COMPLETE FOR {args.class_name}")
    print("=" * 70)
    print(f"   Class: {args.class_name}")
    print(f"   Original images: {len(class_paths)}")
    print(f"   Synthetic images: {len(synthetic)}")
    if best_checkpoint:
        print(f"   🏆 Best FID: {best_fid:.2f} (epoch {best_epoch})")
    print(f"   📁 Checkpoints: {save_dir}")
    print(f"   📁 Synthetic: {synthetic_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()