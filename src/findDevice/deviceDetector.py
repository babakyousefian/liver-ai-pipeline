# src/AI/device_utils.py

import warnings

import torch


def get_optimal_device():
    """
    Detect optimal device for your system.
    For NVIDIA GeForce 930M (2GB), recommends CPU.
    """
    if not torch.cuda.is_available():
        print("✅ CUDA not available. Using CPU.")
        return torch.device("cpu")

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

    print(f"🔍 GPU Detected: {gpu_name}")
    print(f"   Memory: {gpu_memory:.2f} GB")

    # Check if GPU is suitable
    if "930M" in gpu_name or gpu_memory < 3:
        print("⚠️ GPU with limited memory detected. Using CPU for stability.")
        print("   (GPU 930M with 2GB is not recommended for training)")
        return torch.device("cpu")

    # For better GPUs
    print("✅ Using GPU for training.")
    return torch.device("cuda")


def get_optimal_batch_size(model_type="resnet18"):
    """
    Get optimal batch size based on device.
    """
    device = get_optimal_device()

    if device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory == 0:
            return 0
        if gpu_memory < 1:
            return 1
        if gpu_memory == 1:
            return 4
        if gpu_memory == 2:
            return 8
        if gpu_memory == 3:
            return 16
        if gpu_memory == 4:
            return 32
        if gpu_memory == 5:
            return 48
        if gpu_memory == 6:
            return 64
        if gpu_memory == 7:
            return 96
        if gpu_memory == 8:
            return 128
        else:
            return 256
    else:
        # CPU
        return 16


def _dd_():
    device = get_optimal_device()
    batch = get_optimal_batch_size()
    print(f"✅ Device: {device}")
    print(f"✅ Recommended batch size: {batch}")


def main():
    _dd_()


# test
if __name__ == "__main__":
    main()
