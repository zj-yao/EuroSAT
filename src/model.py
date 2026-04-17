from __future__ import annotations

from typing import Sequence

import numpy as np

from .autograd import Tensor
from .layers import Linear, Module


def _apply_activation_tensor(x: Tensor, name: str) -> Tensor:
    if name == "relu":
        return x.relu()
    if name == "tanh":
        return x.tanh()
    if name == "sigmoid":
        return x.sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def _apply_activation_numpy(x: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(x, 0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unsupported activation: {name}")


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        activation: str = "relu",
    ) -> None:
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two hidden layer sizes for a three-layer MLP")
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.num_classes = num_classes
        self.activation = activation.lower()

        self.fc1 = Linear(input_dim, self.hidden_dims[0])
        self.fc2 = Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.fc3 = Linear(self.hidden_dims[1], num_classes)
        self.layers = [self.fc1, self.fc2, self.fc3]

    def __call__(self, x: Tensor) -> Tensor:
        x = _apply_activation_tensor(self.fc1(x), self.activation)
        x = _apply_activation_tensor(self.fc2(x), self.activation)
        return self.fc3(x)

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        x = _apply_activation_numpy(self.fc1.forward_numpy(x), self.activation)
        x = _apply_activation_numpy(self.fc2.forward_numpy(x), self.activation)
        return self.fc3.forward_numpy(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_numpy(x).argmax(axis=1)

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def state_dict(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self.layers, start=1):
            layer_state = layer.state_dict()
            state[f"layer{index}.weight"] = layer_state["weight"]
            state[f"layer{index}.bias"] = layer_state["bias"]
        return state

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        for index, layer in enumerate(self.layers, start=1):
            layer.load_state_dict(
                {
                    "weight": state_dict[f"layer{index}.weight"],
                    "bias": state_dict[f"layer{index}.bias"],
                }
            )

