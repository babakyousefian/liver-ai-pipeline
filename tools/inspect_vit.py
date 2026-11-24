# tools/inspect_vit.py
# -*- coding: utf-8 -*-
"""
Deep inspection of Vision Transformer (ViT) checkpoint.
Target: C:\\Users\\liana\\Documents\\pythonProjects\\FinalTasks\\finalReports\\bestModels\\VisionTransformer\\visiontransformerBEST.pth
"""

import os
from collections import OrderedDict
import torch
import torch.nn as nn

CKPT_PATH = (
    r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
    r"\bestModels\VisionTransformer\visiontransformerBEST.pth"
)


def extract_state_dict(ckpt_obj):
    """Extract state_dict from various checkpoint formats."""
    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict(), "nn.Module", {}
    if isinstance(ckpt_obj, dict):
        meta = {k: v for k, v in ckpt_obj.items()
                if k not in ("state_dict", "model_state_dict", "model", "weights", "net")}
        for key in ("state_dict", "model_state_dict", "model", "weights", "net"):
            if key in ckpt_obj:
                inner = ckpt_obj[key]
                if isinstance(inner, nn.Module):
                    return inner.state_dict(), f"dict[{key} -> nn.Module]", meta
                if isinstance(inner, dict):
                    return inner, f"dict[{key}]", meta
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj, "raw_state_dict", {}
    return {}, type(ckpt_obj).__name__, {}


def detect_vit_family(state_dict):
    """
    Detect if checkpoint is from timm, torchvision, or custom.
    Returns (family, details).
    """
    keys = list(state_dict.keys())

    # timm-style ViT keys:
    #   patch_embed.proj.weight
    #   blocks.0.norm1.weight
    #   blocks.0.attn.qkv.weight
    #   head.weight
    timm_markers = [
        "patch_embed.proj.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn.qkv.weight",
        "head.weight",
    ]
    timm_score = sum(1 for m in timm_markers if any(m in k for k in keys))

    # torchvision-style ViT keys:
    #   conv_proj.weight
    #   encoder.layers.encoder_layer_0.ln_1.weight
    #   encoder.layers.encoder_layer_0.self_attention.in_proj_weight
    #   heads.head.weight
    tv_markers = [
        "conv_proj.weight",
        "encoder.layers.encoder_layer_0.ln_1.weight",
        "encoder.layers.encoder_layer_0.self_attention.in_proj_weight",
        "heads.head.weight",
    ]
    tv_score = sum(1 for m in tv_markers if any(m in k for k in keys))

    if timm_score >= 3:
        return "timm", {"score": timm_score}
    if tv_score >= 3:
        return "torchvision", {"score": tv_score}
    if timm_score > tv_score:
        return "timm(partial)", {"score": timm_score}
    if tv_score > timm_score:
        return "torchvision(partial)", {"score": tv_score}
    return "unknown", {"timm_score": timm_score, "tv_score": tv_score}


def detect_vit_params(state_dict, family):
    """Extract embed_dim, depth, num_heads, patch_size, num_classes."""
    keys = list(state_dict.keys())
    info = {
        "embed_dim": None,
        "depth": 0,
        "num_heads": None,
        "patch_size": None,
        "num_classes": None,
        "mlp_ratio": None,
    }

    # embed_dim + patch_size
    for k in keys:
        if "patch_embed.proj.weight" in k:  # timm
            w = state_dict[k]
            info["embed_dim"] = w.shape[0]
            info["patch_size"] = (w.shape[2], w.shape[3])
            break
        if "conv_proj.weight" in k:  # torchvision
            w = state_dict[k]
            info["embed_dim"] = w.shape[0]
            info["patch_size"] = (w.shape[2], w.shape[3])
            break

    # depth
    max_block = -1
    for k in keys:
        if "blocks." in k:
            try:
                idx = int(k.split("blocks.")[1].split(".")[0])
                max_block = max(max_block, idx)
            except (ValueError, IndexError):
                pass
        elif "encoder_layer_" in k:
            try:
                idx = int(k.split("encoder_layer_")[1].split(".")[0])
                max_block = max(max_block, idx)
            except (ValueError, IndexError):
                pass
    info["depth"] = max_block + 1

    # num_classes
    for k in keys:
        if k in ("head.weight", "heads.head.weight", "classifier.weight"):
            info["num_classes"] = state_dict[k].shape[0]
            break

    # num_heads (from qkv weight shape)
    for k in keys:
        if "attn.qkv.weight" in k:  # timm
            qkv_dim = state_dict[k].shape[0]
            # qkv_dim = 3 * embed_dim
            if info["embed_dim"]:
                info["num_heads"] = qkv_dim // (3 * (info["embed_dim"] // 12)) if info["embed_dim"] % 12 == 0 else None
            break
        if "self_attention.in_proj_weight" in k:  # torchvision
            in_proj_dim = state_dict[k].shape[0]
            # in_proj_dim = 3 * embed_dim
            break

    # mlp_ratio (from mlp.fc1 weight)
    for k in keys:
        if "mlp.fc1.weight" in k:  # timm
            mlp_dim = state_dict[k].shape[0]
            if info["embed_dim"]:
                info["mlp_ratio"] = mlp_dim / info["embed_dim"]
            break
        if "mlp.linear_1.weight" in k:  # torchvision
            mlp_dim = state_dict[k].shape[0]
            if info["embed_dim"]:
                info["mlp_ratio"] = mlp_dim / info["embed_dim"]
            break

    return info


def try_build_timm(state_dict, info):
    """Try to build ViT with timm using detected params."""
    try:
        import timm
    except ImportError:
        return None

    candidates = []

    # Direct name candidates based on (embed_dim, depth)
    arch_map = {
        (192, 12): "vit_tiny_patch16_224",
        (384, 12): "vit_small_patch16_224",
        (768, 12): "vit_base_patch16_224",
        (1024, 24): "vit_large_patch16_224",
        (1280, 32): "vit_huge_patch14_224",
    }

    key = (info["embed_dim"], info["depth"])
    if key in arch_map:
        candidates.append(arch_map[key])

    # Also try common timm names
    candidates.extend([
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_base_patch32_224",
        "vit_small_patch16_224",
        "vit_tiny_patch16_224",
        "vit_large_patch16_224",
    ])

    # Deduplicate
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for name in candidates:
        try:
            model = timm.create_model(
                name,
                pretrained=False,
                num_classes=info["num_classes"] or 4,
                img_size=224,
            )
            model_keys = set(model.state_dict().keys())
            sd_keys = set(state_dict.keys())
            missing = model_keys - sd_keys
            unexpected = sd_keys - model_keys
            score = len(missing) + len(unexpected)
            if score == 0:
                return name, model, 0, 0
            if score <= 5:
                return name, model, len(missing), len(unexpected)
        except Exception:
            continue

    # Try broader search
    try:
        all_models = timm.list_models(pretrained=False)
        vit_models = [m for m in all_models if "vit" in m.lower()]
        for name in vit_models[:50]:  # limit to first 50
            try:
                model = timm.create_model(name, pretrained=False,
                                          num_classes=info["num_classes"] or 4)
                model_keys = set(model.state_dict().keys())
                sd_keys = set(state_dict.keys())
                missing = model_keys - sd_keys
                unexpected = sd_keys - model_keys
                score = len(missing) + len(unexpected)
                if score == 0:
                    return name, model, 0, 0
                if score <= 5:
                    return name, model, len(missing), len(unexpected)
            except Exception:
                continue
    except Exception:
        pass

    return None


def try_build_torchvision(state_dict, info):
    """Try to build ViT with torchvision."""
    try:
        import torchvision.models as tv_models
    except ImportError:
        return None

    arch_map = {
        (768, 12): "vit_b_16",
        (1024, 24): "vit_l_16",
        (1280, 32): "vit_h_14",
    }
    key = (info["embed_dim"], info["depth"])
    candidates = [arch_map[key]] if key in arch_map else ["vit_b_16", "vit_l_16", "vit_h_14"]

    for name in candidates:
        try:
            builder = getattr(tv_models, name)
            model = builder(weights=None, num_classes=info["num_classes"] or 4)
            model_keys = set(model.state_dict().keys())
            sd_keys = set(state_dict.keys())
            missing = model_keys - sd_keys
            unexpected = sd_keys - model_keys
            score = len(missing) + len(unexpected)
            if score == 0:
                return name, model, 0, 0
            if score <= 5:
                return name, model, len(missing), len(unexpected)
        except Exception:
            continue

    return None


def main():
    print("=" * 80)
    print("VISION TRANSFORMER (ViT) — DEEP INSPECTOR")
    print("=" * 80)
    print(f"Path   : {CKPT_PATH}")
    print(f"Exists : {os.path.exists(CKPT_PATH)}")
    if not os.path.exists(CKPT_PATH):
        return

    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"Size   : {size_mb:.2f} MB")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict, container_type, meta = extract_state_dict(ckpt)

    print(f"\nType   : {container_type}")
    if isinstance(ckpt, dict):
        print(f"Top keys: {list(ckpt.keys())}")
    if meta:
        print("\nMetadata / Extra keys:")
        for k, v in meta.items():
            preview = str(v)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  - {k}: {preview}")

    total_tensors = len(state_dict)
    total_params = sum(v.numel() for v in state_dict.values()
                       if isinstance(v, torch.Tensor))
    print(f"\nState dict tensors : {total_tensors}")
    print(f"Total parameters   : {total_params:,}")

    # ── Detect family ──
    family, family_info = detect_vit_family(state_dict)
    print(f"\nFamily : {family}")
    print(f"Info   : {family_info}")

    # ── Detect params ──
    info = detect_vit_params(state_dict, family)
    print(f"\n--- Detected Architecture Params ---")
    print(f"  embed_dim   : {info['embed_dim']}")
    print(f"  depth       : {info['depth']}")
    print(f"  num_classes : {info['num_classes']}")
    print(f"  patch_size  : {info['patch_size']}")
    print(f"  num_heads   : {info['num_heads']}")
    print(f"  mlp_ratio   : {info['mlp_ratio']}")

    # ── Print sample keys ──
    print(f"\n--- First 15 Keys ---")
    for i, (k, v) in enumerate(state_dict.items()):
        if i >= 15:
            break
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        print(f"  {k:70s} | {shape}")

    print(f"\n--- Last 10 Keys ---")
    items = list(state_dict.items())
    for k, v in items[-10:]:
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        print(f"  {k:70s} | {shape}")

    # ── Try to build ──
    print(f"\n--- Attempting timm build ---")
    timm_result = try_build_timm(state_dict, info)
    if timm_result:
        name, model, missing, unexpected = timm_result
        print(f"  ✅ timm.{name} (missing={missing}, unexpected={unexpected})")
    else:
        print(f"  ❌ timm build failed")

    print(f"\n--- Attempting torchvision build ---")
    tv_result = try_build_torchvision(state_dict, info)
    if tv_result:
        name, model, missing, unexpected = tv_result
        print(f"  ✅ torchvision.{name} (missing={missing}, unexpected={unexpected})")
    else:
        print(f"  ❌ torchvision build failed")

    # ── Best result ──
    print(f"\n--- BEST MATCH ---")
    if timm_result and (not tv_result or timm_result[2] + timm_result[3] <= tv_result[2] + tv_result[3]):
        name, model, missing, unexpected = timm_result
        print(f"  📦 timm.create_model('{name}', pretrained=False, num_classes={info['num_classes']})")
        print(f"  Missing keys: {missing}, Unexpected keys: {unexpected}")
    elif tv_result:
        name, model, missing, unexpected = tv_result
        print(f"  📦 torchvision.models.{name}(weights=None, num_classes={info['num_classes']})")
        print(f"  Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        print(f"  ❌ No match found")
        print(f"\n  💡 Suggestion: run `pip install timm --upgrade` and retry")


if __name__ == "__main__":
    main()
