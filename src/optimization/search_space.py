# src/optimization/search_space.py
"""
Search Space for GA — 22 independent parameters.

Each parameter is INDEPENDENT — no overlap. Constraints are applied
post-hoc via `validate_params`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class SearchSpace:
    name: str
    type: str  # 'float' | 'int' | 'categorical' | 'log_float'
    low: Any = None
    high: Any = None
    choices: List[Any] = field(default_factory=list)

    def sample(self) -> Any:
        if self.type == "categorical":
            return self.choices[np.random.randint(len(self.choices))]
        if self.type == "int":
            return int(np.random.randint(self.low, self.high + 1))
        if self.type == "float":
            return float(np.random.uniform(self.low, self.high))
        if self.type == "log_float":
            return float(np.exp(np.random.uniform(np.log(self.low), np.log(self.high))))
        raise ValueError(self.type)

    def clip(self, v: Any) -> Any:
        if self.type == "categorical":
            return v if v in self.choices else self.sample()
        if self.type == "int":
            return int(np.clip(v, self.low, self.high))
        return float(np.clip(v, self.low, self.high))

    def mutate(self, v: Any, strength: float = 0.2) -> Any:
        if self.type == "categorical":
            if np.random.rand() < 0.3:
                return self.sample()
            return v
        if self.type == "int":
            delta = np.random.normal(0, max(1, strength * (self.high - self.low)))
            return int(np.clip(v + delta, self.low, self.high))
        if self.type == "float":
            delta = np.random.normal(0, strength * (self.high - self.low))
            return float(np.clip(v + delta, self.low, self.high))
        if self.type == "log_float":
            lo, hi = np.log(self.low), np.log(self.high)
            delta = np.random.normal(0, strength * (hi - lo))
            return float(np.exp(np.clip(np.log(v) + delta, lo, hi)))
        return v


# ================================================================
# ✅ SEARCH SPACE PER MODEL
# ================================================================

def get_search_space(
    model_name: str,
    hardware_max_batch: int = 32,
    hardware_max_workers: int = 4,
    hardware_mixed_precision: bool = True,
) -> List[SearchSpace]:
    """
    Return the 22 parameters for the given model.

    ResNet18/50   → input_size = 224 only
    EfficientNet  → input_size ∈ {224, 240}
    """
    # input_size depends on model family
    if model_name in ("resnet18", "resnet50"):
        input_sizes = [224]
    else:
        input_sizes = [224, 240]

    # Filter batch sizes by hardware
    batch_all = [16, 32, 64]
    batch = [b for b in batch_all if b <= hardware_max_batch] or [8]

    # Filter workers by hardware
    workers_all = [0, 2, 4]
    workers = [w for w in workers_all if w <= hardware_max_workers] or [0]

    return [
        # === Model architecture ===
        SearchSpace("input_size", "categorical", choices=input_sizes),
        SearchSpace("projection_dim", "categorical", choices=[128, 256, 512]),
        SearchSpace("hidden_dim", "categorical", choices=[1024, 2048, 4096]),

        # === Optimization ===
        SearchSpace("lr", "log_float", low=1e-5, high=1e-3),
        SearchSpace("weight_decay", "log_float", low=1e-6, high=1e-3),
        SearchSpace("batch_size", "categorical", choices=batch),
        SearchSpace("optimizer_name", "categorical",
                    choices=["adamw", "radam", "ranger"]),
        SearchSpace("use_lars", "categorical", choices=[False, True]),
        SearchSpace("use_mixed_precision", "categorical",
                    choices=[hardware_mixed_precision]),
        SearchSpace("gradient_accumulation_steps", "categorical",
                    choices=[1, 2, 4]),
        SearchSpace("clip_grad_norm", "float", low=0.1, high=2.0),
        SearchSpace("warmup_epochs", "int", low=3, high=10),
        SearchSpace("total_epochs", "categorical", choices=[15]),

        # === BYOL specific ===
        SearchSpace("momentum", "float", low=0.99, high=0.9995),
        SearchSpace("momentum_schedule", "categorical",
                    choices=["constant", "cosine"]),

        # === Data ===
        SearchSpace("num_workers", "categorical", choices=workers),
        SearchSpace("augmentation_method", "categorical",
                    choices=["byol", "simclr"]),
        SearchSpace("dropout", "float", low=0.0, high=0.3),
        SearchSpace("label_smoothing", "float", low=0.0, high=0.2),
        SearchSpace("use_sampler", "categorical", choices=[False, True]),
        SearchSpace("early_stopping_patience", "int", low=5, high=15),

        # === LR multiplier for encoder (relative to head) ===
        SearchSpace("base_lr_mult", "float", low=0.5, high=2.0),
    ]


# ================================================================
# ✅ CONSTRAINTS (fix dependencies between params)
# ================================================================

def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix all constraint violations.
    """
    p = dict(params)

    # 1. warmup < total_epochs
    if p.get("warmup_epochs", 0) >= p.get("total_epochs", 15):
        p["warmup_epochs"] = max(1, p["total_epochs"] // 3)

    # 2. batch_size * grad_accum ≤ 256
    bs = p.get("batch_size", 32)
    ga = p.get("gradient_accumulation_steps", 1)
    if bs * ga > 256:
        p["gradient_accumulation_steps"] = max(1, 256 // bs)

    # 3. use_lars requires batch ≥ 32
    if p.get("use_lars", False) and bs < 32:
        p["use_lars"] = False

    # 4. lr < 5e-4 if use_lars is True (LARS needs smaller LR)
    if p.get("use_lars", False) and p.get("lr", 1e-4) > 5e-4:
        p["lr"] = 5e-4

    # 5. dropout < 0.1 if label_smoothing > 0.1 (avoid over-regularization)
    if p.get("label_smoothing", 0) > 0.1 and p.get("dropout", 0) > 0.2:
        p["dropout"] = 0.2

    return p


# ================================================================
# ✅ VECTOR <-> DICT CONVERSION (for GA arithmetic)
# ================================================================

def params_to_vector(params: Dict[str, Any], spaces: List[SearchSpace]) -> np.ndarray:
    vec = []
    for s in spaces:
        v = params.get(s.name)
        if s.type == "categorical":
            try:
                idx = s.choices.index(v)
            except (ValueError, AttributeError):
                idx = 0
            vec.append(float(idx))
        else:
            vec.append(float(v) if v is not None else s.sample())
    return np.array(vec, dtype=np.float64)


def vector_to_params(vec: np.ndarray, spaces: List[SearchSpace]) -> Dict[str, Any]:
    params = {}
    for i, s in enumerate(spaces):
        v = float(vec[i])
        if s.type == "categorical":
            idx = int(np.clip(round(v), 0, len(s.choices) - 1))
            params[s.name] = s.choices[idx]
        elif s.type == "int":
            params[s.name] = int(np.clip(round(v), s.low, s.high))
        else:
            params[s.name] = float(np.clip(v, s.low, s.high))
    return params