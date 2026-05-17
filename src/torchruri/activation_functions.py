from typing import TypeAlias

import numpy as np

from . import ufunc as uf
from .auto_grad.tensor import Tensor

T: TypeAlias = "Tensor"
Number: TypeAlias = np.number | int | float
TensorParent: TypeAlias = tuple["Tensor", "Tensor"] | tuple["Tensor"]

NUMBER_RUNTIME = (np.number, int, float)


class ActivationFunction:
    pass


class ReLU(ActivationFunction):
    @staticmethod
    def forward(t: T) -> T:
        if not isinstance(t, Tensor):
            raise RuntimeError("relu() can only be called on a tensor")
        return t.max(0)

    def __call__(self, t: T) -> T:
        return ReLU.forward(t)


class LeakyReLU(ActivationFunction):
    @staticmethod
    def forward(t: T, negative_slope: Number = 0.1) -> T:
        if not isinstance(t, Tensor):
            raise RuntimeError("leaky_relu() can only be called on a tensor")

        return t.max(t * negative_slope)

    def __call__(self, t: T, negative_slope: Number = 0.1) -> T:
        return LeakyReLU.forward(t, negative_slope)


class SoftMax(ActivationFunction):
    @staticmethod
    def forward(t: T) -> T:
        if not isinstance(t, Tensor):
            raise RuntimeError("leaky_relu() can only be called on a tensor")

        if len(t.shape) != 1:
            raise ValueError("SoftMax does not support more than 1 dim tensor yet")

        return uf.exp(t) / (uf.sum(uf.exp(t)))

    def __call__(self, t: T) -> T:
        return SoftMax.forward(t)


class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(t: T) -> T:
        if not isinstance(t, Tensor):
            raise RuntimeError("leaky_relu() can only be called on a tensor")

        if len(t.shape) != 1:
            raise ValueError("SoftMax does not support more than 1 dim tensor yet")

        return 1 / (1 + uf.exp(-t))

    def __call__(self, t: T) -> T:
        return Sigmoid.forward(t)
