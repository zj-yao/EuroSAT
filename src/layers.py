from __future__ import annotations

from typing import Iterable

import numpy as np

from .autograd import Tensor


class Module:
    def parameters(self) -> list[Tensor]:
        return []

    def state_dict(self) -> dict[str, np.ndarray]:
        return {}

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        return None


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "xavier_uniform",
    ) -> None:
        if init == "xavier_uniform":
            limit = np.sqrt(6.0 / (in_features + out_features))
        elif init == "he_uniform":
            limit = np.sqrt(6.0 / in_features)
        else:
            raise ValueError(f"Unsupported initialization: {init}")
        weight = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float32)
        bias = np.zeros((out_features,), dtype=np.float32)
        self.weight = Tensor(weight, requires_grad=True)
        self.bias = Tensor(bias, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.data + self.bias.data

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weight": self.weight.data.copy(), "bias": self.bias.data.copy()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        self.weight.data = state_dict["weight"].astype(np.float32).copy()
        self.bias.data = state_dict["bias"].astype(np.float32).copy()


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    batch_size = targets.shape[0]
    loss_value = -np.log(probs[np.arange(batch_size), targets] + 1e-12).mean().astype(np.float32)

    out = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op="softmax_ce")

    def _backward() -> None:
        if not logits.requires_grad:
            return
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1.0
        grad /= batch_size
        logits.grad += out.grad * grad

    out._backward = _backward
    return out


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return float(-np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12).mean())


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def iter_parameters(modules: Iterable[Module]) -> list[Tensor]:
    params: list[Tensor] = []
    for module in modules:
        params.extend(module.parameters())
    return params
