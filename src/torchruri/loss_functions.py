from typing import TypeAlias

import numpy as np

from . import ufunc as uf
from .auto_grad.tensor import Tensor

T: TypeAlias = Tensor
Number: TypeAlias = float | int | np.number


class LossFn:
    pass


class MSELoss(LossFn):
    @staticmethod
    def forward(target: T, pred: T) -> T:
        if len(pred.shape) != 1:
            raise ValueError("A loss function can only be called on a 1D Tensor")

        if pred.shape != target.shape:
            raise ValueError("The shape of the target and the input should be the same")

        return uf.sum((pred - target) ** 2) / pred.size

    def __call__(self, target: T, pred: T) -> T:
        return self.forward(target, pred)


class L1Loss(LossFn):
    @staticmethod
    def forward(self, target: T, pred: T) -> T:
        if len(pred.shape) != 1:
            raise ValueError("A loss function can only be called on a 1D Tensor")

        if pred.shape != target.shape:
            raise ValueError("The shape of the target and the input should be the same")

        return uf.sum(uf.abs(pred - target)) / pred.size

    def __call__(self, target: T, pred: T) -> T:
        return self.forward(target, pred)


class CrossEntropy(LossFn):
    @staticmethod
    def forward(self, target: T, pred: T) -> T:
        if len(pred.shape) != 1:
            raise ValueError("A loss function can only be called on a 1D Tensor")

        if pred.shape != target.shape:
            raise ValueError("The shape of the target and the input should be the same")

        return uf.sum(target * uf.log(pred))

    def __call__(self, one_encoded_target: T, pred_probaility: T) -> T:
        return self.forward(one_encoded_target, pred_probaility)
