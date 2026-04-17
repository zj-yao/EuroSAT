from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def _ensure_array(data: np.ndarray | float | int) -> np.ndarray:
    return np.array(data, dtype=np.float32)


def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(
        self,
        data: np.ndarray | float | int,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar tensors")
            grad = np.ones_like(self.data, dtype=np.float32)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = grad.astype(np.float32)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other: "Tensor" | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: "Tensor" | float | int) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward() -> None:
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: "Tensor" | float | int) -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else -float(other))

    def __rsub__(self, other: "Tensor" | float | int) -> "Tensor":
        return other + (-self) if isinstance(other, Tensor) else Tensor(other) - self

    def __mul__(self, other: "Tensor" | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: "Tensor" | float | int) -> "Tensor":
        return self * other

    def __truediv__(self, other: "Tensor" | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted((a if a >= 0 else a + self.data.ndim for a in axes)):
                    grad = np.expand_dims(grad, axis=ax)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_data = np.tanh(self.data)
        out = Tensor(tanh_data, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - tanh_data**2)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * sig * (1.0 - sig)

        out._backward = _backward
        return out

