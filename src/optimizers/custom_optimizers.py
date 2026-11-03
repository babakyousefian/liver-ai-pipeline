# src/optimizers/custom_optimizers.py
"""
Custom Optimizers for Liver AI Pipeline

Provides RAdam, Ranger, Lookahead, and a unified get_optimizer() interface.

Author: Babak Yousefian
"""

import math
from typing import Iterable

import torch
import torch.optim as optim


# ================================================================
# ✅ RAdam (Rectified Adam)
# ================================================================
class RAdam(optim.Optimizer):
    """
    RAdam: Rectified Adam optimizer.

    Reference: Liu et al., 2019 "On the Variance of the Adaptive Learning Rate
    and Beyond" (https://arxiv.org/abs/1908.03265)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.rho_inf = 2.0 / (1.0 - betas[1]) - 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction2 = 1.0 - beta2 ** state["step"]
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                rho_t = self.rho_inf - 2.0 * state["step"] * (beta2 ** state["step"]) / bias_correction2

                if rho_t > 4.0:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    adaptive_lr = (
                        math.sqrt((1.0 - beta2 ** state["step"]) / bias_correction1)
                        * math.sqrt(
                            (rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf
                            / ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t)
                        )
                    )
                    denom = exp_avg_sq_hat.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-lr * adaptive_lr)
                else:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    step_size = lr / bias_correction1
                    p.add_(exp_avg, alpha=-step_size)

        return loss


# ================================================================
# ✅ Lookahead (for Ranger)
# ================================================================
class Lookahead(optim.Optimizer):
    """
    Lookahead optimizer wrapper.

    Reference: Zhang et al., 2019
    (https://arxiv.org/abs/1907.08610)
    """

    def __init__(self, base_optimizer, alpha: float = 0.5, k: int = 6, **kwargs):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self._step_count = 0

        defaults = dict(alpha=alpha, k=k, **base_optimizer.defaults, **kwargs)
        super().__init__(base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    if "slow_params" not in state:
                        state["slow_params"] = p.detach().clone()
                    slow = state["slow_params"]
                    slow.mul_(1.0 - self.alpha).add_(p, alpha=self.alpha)
                    p.data.copy_(slow)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


def Ranger(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
           weight_decay=0.0, alpha=0.5, k=6, **kwargs):
    """Ranger = RAdam + Lookahead."""
    radam = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return Lookahead(radam, alpha=alpha, k=k)


# ================================================================
# ✅ Unified get_optimizer()
# ================================================================
def get_optimizer(
    name: str,
    params: Iterable,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
) -> optim.Optimizer:
    """
    Unified optimizer factory.

    Args:
        name: 'adamw' | 'adam' | 'sgd' | 'radam' | 'ranger'
        params: Model parameters
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer args

    Returns:
        torch.optim.Optimizer instance
    """
    name = name.lower().strip()

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)

    if name == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        return optim.SGD(params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, **kwargs)

    if name == "radam":
        return RAdam(params, lr=lr, weight_decay=weight_decay, **kwargs)

    if name == "ranger":
        return Ranger(params, lr=lr, weight_decay=weight_decay, **kwargs)

    raise ValueError(
        f"Unknown optimizer: '{name}'. "
        f"Available: adamw, adam, sgd, radam, ranger"
    )


def create_optimizer(name: str, params, lr: float = 1e-3,
                     weight_decay: float = 0.0, **kwargs):
    """Alias for get_optimizer."""
    return get_optimizer(name, params, lr=lr, weight_decay=weight_decay, **kwargs)


def list_optimizers() -> list:
    """Return list of available optimizers."""
    return ["adamw", "adam", "sgd", "radam", "ranger"]