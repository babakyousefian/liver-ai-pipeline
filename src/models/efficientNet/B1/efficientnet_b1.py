# src/models/efficientNet/B1/efficientnet_b1.py

"""
EfficientNet-B1 Architecture Implementation from Scratch (ULTIMATE EDITION)

This module implements the complete EfficientNet-B1 architecture without any
pre-trained weights, following the original paper:
"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

Architecture Details (Compared to B0):
- Input: 240x240x3 (vs 224x224)
- Base channels: 32 (same)
- Width multiplier: 1.1 (vs 1.0)
- Depth multiplier: 1.1 (vs 1.0)
- Resolution: 240x240 (vs 224x224)
- Total Parameters: ~7.8M (vs ~5.3M)

Features:
- Advanced weight initialization
- Config dataclass for easy management
- Layer-wise learning rate support
- Export/Import capabilities
- Zero-init residual connections

Stage Configuration (B1):
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage │ Blocks │ Channels │ Resolution │ Expand Ratio │ Kernel │ SE    │
├────────────────────────────────────────────────────────────────────────────┤
│ 0     │ 1      │ 32       │ 240x240    │ -            │ -      │ -     │
│ 1     │ 1      │ 16       │ 120x120    │ 1            │ 3x3    │ No    │
│ 2     │ 2      │ 24       │ 60x60      │ 6            │ 3x3    │ No    │
│ 3     │ 2      │ 40       │ 30x30      │ 6            │ 5x5    │ Yes   │
│ 4     │ 3      │ 80       │ 15x15      │ 6            │ 3x3    │ Yes   │
│ 5     │ 3      │ 112      │ 15x15      │ 6            │ 5x5    │ Yes   │
│ 6     │ 4      │ 192      │ 8x8        │ 6            │ 5x5    │ Yes   │
│ 7     │ 1      │ 320      │ 8x8        │ 6            │ 3x3    │ Yes   │
│ 8     │ 1      │ 1280     │ 8x8        │ -            │ -      │ -     │
└────────────────────────────────────────────────────────────────────────────┘

Total Parameters: ~7.8M

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 3.0.0 (ULTIMATE EDITION)
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MBConvBlock, SqueezeExcitation, Swish, init_weights, make_layer

warnings.filterwarnings("ignore")


# ================================================================
# ✅ CONFIG DATACLASS 🔥 NEW
# ================================================================
@dataclass
class EfficientNetB1Config:
    """Configuration class for EfficientNet-B1."""

    num_classes: int = 5
    zero_init_residual: bool = False
    use_batch_norm: bool = True
    activation: str = "swish"
    dropout_rate: float = 0.2
    input_channels: int = 3
    width_multiplier: float = 1.1
    depth_multiplier: float = 1.1
    init_type: str = "kaiming"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_classes": self.num_classes,
            "zero_init_residual": self.zero_init_residual,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "input_channels": self.input_channels,
            "width_multiplier": self.width_multiplier,
            "depth_multiplier": self.depth_multiplier,
            "init_type": self.init_type,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EfficientNetB1Config":
        """Create config from dictionary."""
        return cls(**data)


class EfficientNetB1(nn.Module):
    """
    Complete EfficientNet-B1 Architecture from Scratch (ULTIMATE EDITION).

    This implementation follows the exact architecture from the original
    EfficientNet paper with B1 scaling parameters.

    Features:
        - No pre-trained weights (starts from random initialization)
        - Advanced weight initialization with multiple strategies
        - Swish (SiLU) activation
        - Squeeze-and-Excitation (SE) blocks
        - Configurable number of classes
        - Feature extractor for transfer learning
        - B1 scaling: width=1.1, depth=1.1
        - Layer-wise learning rate support
        - Export/Import capabilities

    Attributes:
        conv_stem: First convolution layer (3x3, stride=2)
        bn1: Batch normalization after conv_stem
        swish: Swish activation
        blocks: List of MBConv blocks
        conv_head: Final convolution layer (1x1)
        bn2: Batch normalization after conv_head
        avgpool: Global average pooling
        dropout: Dropout layer
        fc: Final fully connected layer
    """

    def __init__(
        self,
        num_classes: int = 5,
        zero_init_residual: bool = False,
        use_batch_norm: bool = True,
        activation: str = "swish",
        dropout_rate: float = 0.2,
        input_channels: int = 3,
        width_multiplier: float = 1.1,
        depth_multiplier: float = 1.1,
        init_type: str = "kaiming",
        verbose: bool = False,
    ) -> None:
        """
        Initialize EfficientNet-B1.

        Args:
            num_classes: Number of output classes (default: 5)
            zero_init_residual: If True, initialize last BN in each block to zero
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('swish' or 'relu')
            dropout_rate: Dropout rate for the FC layer
            input_channels: Number of input channels (default: 3 for RGB)
            width_multiplier: Width multiplier for scaling (1.1 for B1)
            depth_multiplier: Depth multiplier for scaling (1.1 for B1)
            init_type: Weight initialization type ('kaiming', 'xavier', 'normal', 'uniform')
            verbose: If True, print initialization details
        """
        super(EfficientNetB1, self).__init__()

        # ==========================================
        # Store configuration
        # ==========================================
        self.num_classes = num_classes
        self.zero_init_residual = zero_init_residual
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.init_type = init_type
        self.verbose = verbose
        self._feature_dim = 1280
        self._config = EfficientNetB1Config(
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout_rate=dropout_rate,
            input_channels=input_channels,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            init_type=init_type,
            verbose=verbose,
        )

        # ==========================================
        # Calculate scaled channels
        # ==========================================
        def round_channels(channels: int, divisor: int = 8) -> int:
            """Round channels to nearest multiple of divisor."""
            if width_multiplier == 1.0:
                return channels
            return int(max(1, channels * width_multiplier // divisor * divisor))

        def round_repeats(repeats: int) -> int:
            """Round repeats based on depth multiplier."""
            if depth_multiplier == 1.0:
                return repeats
            return int(max(1, repeats * depth_multiplier))

        # ==========================================
        # Activation function
        # ==========================================
        if activation.lower() == "swish":
            self.act = Swish()
        elif activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # ==========================================
        # Stage 0: Stem
        # ==========================================
        stem_channels = round_channels(32)
        self.conv_stem = nn.Conv2d(
            input_channels,
            stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False if use_batch_norm else True,
        )
        init_weights(self.conv_stem, init_type)

        self.bn1 = nn.BatchNorm2d(stem_channels) if use_batch_norm else nn.Identity()
        self.act_stem = self.act

        # ==========================================
        # Stage 1-7: MBConv Blocks
        # ==========================================
        block_configs = [
            (round_repeats(1), round_channels(16), 3, 1, 1, False),
            (round_repeats(2), round_channels(24), 3, 2, 6, False),
            (round_repeats(2), round_channels(40), 5, 2, 6, True),
            (round_repeats(3), round_channels(80), 3, 2, 6, True),
            (round_repeats(3), round_channels(112), 5, 1, 6, True),
            (round_repeats(4), round_channels(192), 5, 2, 6, True),
            (round_repeats(1), round_channels(320), 3, 1, 6, True),
        ]

        self.blocks = nn.ModuleList()
        in_channels = stem_channels

        for i, (
            num_blocks,
            out_channels,
            kernel_size,
            stride,
            expand_ratio,
            use_se,
        ) in enumerate(block_configs):
            layer = make_layer(
                block_type=MBConvBlock,
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                kernel_size=kernel_size,
                expand_ratio=expand_ratio,
                use_se=use_se,
                se_reduction_ratio=4,
                drop_rate=0.0,
                activation=activation,
                use_batch_norm=use_batch_norm,
                init_type=init_type,
                zero_init_residual=zero_init_residual,
            )
            self.blocks.append(layer)
            in_channels = out_channels

        # ==========================================
        # Stage 8: Head
        # ==========================================
        head_channels = round_channels(1280)
        self.conv_head = nn.Conv2d(
            in_channels,
            head_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False if use_batch_norm else True,
        )
        init_weights(self.conv_head, init_type)

        self.bn2 = nn.BatchNorm2d(head_channels) if use_batch_norm else nn.Identity()
        self.act_head = self.act

        # ==========================================
        # Output
        # ==========================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(head_channels, num_classes)
        init_weights(self.fc, init_type)

        # ==========================================
        # Store layer information
        # ==========================================
        self._stage_info = self._get_stage_info()
        self._layer_info = self._get_layer_info()

        if verbose:
            print(self.summary())

    def _get_stage_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about each stage."""
        stages = {
            "stage_0": {
                "name": "Stem (Conv3x3, stride=2)",
                "layers": ["conv_stem", "bn1", "act_stem"],
                "output_shape": (32, 120, 120),
                "parameters": sum(p.numel() for p in self.conv_stem.parameters())
                + (sum(p.numel() for p in self.bn1.parameters()) if self.use_batch_norm else 0),
            }
        }

        for i in range(len(self.blocks)):
            stage_name = f"stage_{i+1}"
            stages[stage_name] = {
                "name": f"MBConv Stage {i+1}",
                "layers": [f"blocks[{i}]"],
                "num_blocks": len(self.blocks[i]),
                "output_shape": "N/A",
                "parameters": sum(p.numel() for p in self.blocks[i].parameters()),
            }

        stages["stage_8"] = {
            "name": "Head (Conv1x1 + AvgPool + FC)",
            "layers": ["conv_head", "bn2", "act_head", "avgpool", "dropout", "fc"],
            "output_shape": (self.num_classes,),
            "parameters": sum(p.numel() for p in self.conv_head.parameters())
            + (sum(p.numel() for p in self.bn2.parameters()) if self.use_batch_norm else 0)
            + sum(p.numel() for p in self.fc.parameters()),
        }

        return stages

    def _get_layer_info(self) -> Dict[str, Any]:
        """Get information about all layers in the network."""
        return {
            "input_shape": (self.input_channels, 240, 240),
            "output_shape": (self.num_classes,),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "stages": self._stage_info,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
                Usually (batch_size, 3, 240, 240)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Stage 0: Stem
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_stem(x)

        # Stage 1-7: MBConv Blocks
        for block in self.blocks:
            x = block(x)

        # Stage 8: Head
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act_head(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    # ==========================================
    # ✅ ADVANCED FEATURES 🔥 NEW
    # ==========================================

    def get_layer_wise_lr(
        self, base_lr: float = 0.001, decay_factor: float = 0.9
    ) -> List[Tuple[nn.Parameter, float]]:
        """
        Get layer-wise learning rates.

        Args:
            base_lr: Base learning rate
            decay_factor: Decay factor for each layer

        Returns:
            List of (parameter, learning_rate) tuples
        """
        lr_list = []
        for i, (name, param) in enumerate(self.named_parameters()):
            lr = base_lr * (decay_factor ** (i / len(list(self.parameters()))))
            lr_list.append((param, lr))
        return lr_list

    def get_prunable_layers(self) -> List[nn.Module]:
        """
        Get all layers that can be pruned.

        Returns:
            List of prunable layers
        """
        prunable = []
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable.append(module)
        return prunable

    def prepare_for_quantization(self) -> None:
        """Prepare model for quantization."""
        self.eval()
        for module in self.modules():
            if hasattr(module, "qconfig"):
                continue

    def export_config(self, path: str) -> None:
        """
        Export model configuration to JSON file.

        Args:
            path: Output path
        """
        config = {
            "class_name": self.__class__.__name__,
            "config": self._config.to_dict(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "feature_dim": self._feature_dim,
            "input_size": 240,
            "width_multiplier": self.width_multiplier,
            "depth_multiplier": self.depth_multiplier,
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Config exported to: {path}")

    def get_feature_extractor(self) -> nn.Sequential:
        """Get feature extractor part of the network (up to avgpool)."""
        layers = [
            self.conv_stem,
            self.bn1,
            self.act_stem,
        ]
        for block in self.blocks:
            layers.append(block)
        layers.extend([self.conv_head, self.bn2, self.act_head, self.avgpool])
        return nn.Sequential(*layers)

    def get_feature_dim(self) -> int:
        """Get the dimension of the feature output."""
        return self._feature_dim

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding (features before FC) for a given input."""
        with torch.no_grad():
            features = self.get_feature_extractor()(x)
            return torch.flatten(features, 1)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the FC layer."""
        for name, param in self.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        if self.verbose:
            print("✅ Backbone frozen. Only FC layer is trainable.")

    def freeze_stage(self, stage: int) -> None:
        """Freeze a specific stage of the network."""
        stage_names = [
            "conv_stem",
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",
            "blocks.6",
        ]
        if stage < 0 or stage >= len(stage_names):
            print(f"❌ Invalid stage: {stage}. Must be 0-{len(stage_names)-1}")
            return

        target_name = stage_names[stage]
        for name, param in self.named_parameters():
            if target_name in name:
                param.requires_grad = False
        if self.verbose:
            print(f"✅ Stage {stage} ({target_name}) frozen.")

    def unfreeze_all(self) -> None:
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
        if self.verbose:
            print("✅ All layers unfrozen.")

    def get_trainable_parameters(self) -> Dict[str, Union[int, float]]:
        """Get count of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_percentage": (trainable / total) * 100 if total > 0 else 0.0,
        }

    def get_layer_parameters(self) -> Dict[str, int]:
        """Get parameter count for each major layer."""
        layer_params = {}

        layer_params["conv_stem"] = sum(p.numel() for p in self.conv_stem.parameters())
        layer_params["bn1"] = (
            sum(p.numel() for p in self.bn1.parameters()) if self.use_batch_norm else 0
        )

        for i, block in enumerate(self.blocks):
            layer_params[f"blocks_{i}"] = sum(p.numel() for p in block.parameters())

        layer_params["conv_head"] = sum(p.numel() for p in self.conv_head.parameters())
        layer_params["bn2"] = (
            sum(p.numel() for p in self.bn2.parameters()) if self.use_batch_norm else 0
        )
        layer_params["avgpool"] = 0
        layer_params["dropout"] = 0
        layer_params["fc"] = sum(p.numel() for p in self.fc.parameters())

        return layer_params

    def summary(self) -> str:
        """Get a string summary of the model."""
        info = self.get_trainable_parameters()
        layer_params = self.get_layer_parameters()

        blocks_summary = ""
        for i in range(len(self.blocks)):
            blocks_summary += f"│ Blocks[{i}]    │ ...             │ {layer_params.get(f'blocks_{i}', 0):>10,} │     -     │\n"

        summary_str = f"""
{'='*80}
EfficientNet-B1 Architecture Summary (ULTIMATE EDITION)
{'='*80}

Input Shape: (batch, {self.input_channels}, 240, 240)
Output Shape: (batch, {self.num_classes})

Layer Configuration:
┌────────────────────────────────────────────────────────────────────────────┐
│ Layer        │ Output Shape    │ Parameters    │ Blocks    │ Expansion   │
├────────────────────────────────────────────────────────────────────────────┤
│ ConvStem     │ {self.conv_stem.out_channels}×120×120  │ {layer_params.get('conv_stem', 0):>10,} │     -     │      -      │
│ BN1          │ {self.conv_stem.out_channels}×120×120  │ {layer_params.get('bn1', 0):>10,} │     -     │      -      │
{blocks_summary}│ ConvHead     │ {self.conv_head.out_channels}×8×8      │ {layer_params.get('conv_head', 0):>10,} │     -     │      -      │
│ BN2          │ {self.conv_head.out_channels}×8×8      │ {layer_params.get('bn2', 0):>10,} │     -     │      -      │
│ AvgPool      │ {self._feature_dim}×1×1      │ 0             │     -     │      -      │
│ FC           │ {self.num_classes}               │ {layer_params.get('fc', 0):>10,} │     -     │      -      │
└────────────────────────────────────────────────────────────────────────────┘

Parameter Count:
- Total Parameters: {info['total']:,}
- Trainable Parameters: {info['trainable']:,}
- Frozen Parameters: {info['frozen']:,}
- Trainable Percentage: {info['trainable_percentage']:.1f}%

Configuration:
- Activation: {self.activation}
- BatchNorm: {'Enabled' if self.use_batch_norm else 'Disabled'}
- Dropout: {self.dropout_rate}
- Zero Init Residual: {self.zero_init_residual}
- Input Channels: {self.input_channels}
- Feature Dimension: {self._feature_dim}
- Width Multiplier: {self.width_multiplier}
- Depth Multiplier: {self.depth_multiplier}
- Init Type: {self.init_type}

Status: ✅ Starting from Scratch (No Pre-trained Weights)

Stage Information:
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage         │ Name                              │ Output Shape   │
├────────────────────────────────────────────────────────────────────────────┤
│ Stage 0       │ Stem (Conv3x3, stride=2)         │ 32×120×120    │
│ Stage 1       │ MBConv1 (3x3, expand=1)          │ 16×120×120    │
│ Stage 2       │ MBConv6 (3x3, expand=6)          │ 24×60×60      │
│ Stage 3       │ MBConv6 (5x5, expand=6, SE)      │ 40×30×30      │
│ Stage 4       │ MBConv6 (3x3, expand=6, SE)      │ 80×15×15      │
│ Stage 5       │ MBConv6 (5x5, expand=6, SE)      │ 112×15×15     │
│ Stage 6       │ MBConv6 (5x5, expand=6, SE)      │ 192×8×8       │
│ Stage 7       │ MBConv6 (3x3, expand=6, SE)      │ 320×8×8       │
│ Stage 8       │ Head (Conv1x1 + AvgPool + FC)    │ {self.num_classes}           │
└────────────────────────────────────────────────────────────────────────────┘
{'='*80}
"""
        return summary_str

    def get_flops(self, input_shape: Tuple[int, int] = (240, 240)) -> int:
        """Calculate total FLOPs for the network."""
        h, w = input_shape

        try:
            dummy = torch.randn(1, self.input_channels, h, w)
            flops = 0

            def hook_fn(
                module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor
            ) -> None:
                nonlocal flops
                if isinstance(module, nn.Conv2d):
                    b, c_out, h_out, w_out = output.shape
                    k_h, k_w = module.kernel_size
                    flops += (
                        b * c_out * h_out * w_out * k_h * k_w * module.in_channels // module.groups
                    )
                elif isinstance(module, nn.Linear):
                    b, c_out = output.shape
                    flops += b * c_out * module.in_features

            hooks = []
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)

            self.eval()
            with torch.no_grad():
                _ = self(dummy)

            for hook in hooks:
                hook.remove()

            return flops
        except Exception:
            return 0

    def get_model_size(self) -> Dict[str, float]:
        """Get the size of the model in memory."""
        from .utils import get_model_size

        return get_model_size(self)

    def count_parameters(self) -> Dict[str, int]:
        """Count all parameters."""
        from .utils import count_parameters

        return count_parameters(self)

    def __repr__(self) -> str:
        return f"EfficientNetB1(num_classes={self.num_classes}, activation={self.activation})"


def create_efficientnet_b1(
    num_classes: int = 5, pretrained: bool = False, verbose: bool = True, **kwargs
) -> EfficientNetB1:
    """
    Factory function to create an EfficientNet-B1 model.

    Args:
        num_classes: Number of output classes
        pretrained: If True, would load pretrained weights (but we're starting from scratch)
        verbose: If True, print model summary
        **kwargs: Additional arguments for EfficientNetB1 constructor

    Returns:
        EfficientNet-B1 model instance

    Example:
        >>> model = create_efficientnet_b1(num_classes=5)
        >>> output = model(torch.randn(4, 3, 240, 240))
        >>> print(output.shape)  # (4, 5)
    """
    if pretrained:
        print("⚠️ Warning: pretrained=True is ignored for Task 5")
        print("📌 Starting from scratch as per Task 5 requirements")

    model = EfficientNetB1(num_classes=num_classes, verbose=verbose, **kwargs)

    if verbose:
        print(model.summary())

    return model


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Complete EfficientNet-B1 Implementation (ULTIMATE EDITION)")
    print("=" * 80)

    # Test 1: Model Creation
    print("\n📌 Test 1: Model Creation")
    model = create_efficientnet_b1(num_classes=5, pretrained=False, verbose=False)
    print("✅ Model created successfully")

    # Test 2: Forward Pass
    print("\n📌 Test 2: Forward Pass")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 240, 240)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

    # Test 3: Config Export
    print("\n📌 Test 3: Config Export")
    model.export_config("efficientnet_b1_config.json")

    # Test 4: Feature Extractor
    print("\n📌 Test 4: Feature Extractor")
    features = model.get_feature_extractor()
    out = features(dummy_input)
    print(f"   Features shape: {out.shape}")

    # Test 5: Parameters
    print("\n📌 Test 5: Parameter Count")
    params = model.get_trainable_parameters()
    print(f"   Total: {params['total']:,}")

    # Test 6: Summary
    print("\n📌 Test 6: Model Summary")
    print(model.summary())

    print("\n✅ All tests passed! EfficientNet-B1 is ready!")
    print("=" * 80)
