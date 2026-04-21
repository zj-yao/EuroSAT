from __future__ import annotations

from typing import Iterable

import numpy as np

from .autograd import Tensor


class SGD:
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocities = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocities):
            if parameter.grad is None:
                continue
            grad = parameter.grad
            if self.weight_decay and parameter.data.ndim > 1:
                grad = grad + self.weight_decay * parameter.data
            if self.momentum:
                velocity *= self.momentum
                velocity += grad
                grad = velocity
            parameter.data -= self.lr * grad


class StepLRScheduler:
    def __init__(self, optimizer: SGD, step_size: int, gamma: float) -> None:
        self.optimizer = optimizer
        self.step_size = max(1, step_size)
        self.gamma = gamma

    def step(self, epoch: int) -> None:
        if epoch > 0 and epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
