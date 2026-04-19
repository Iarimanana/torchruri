from typing import TypeAlias

import numpy as np

from .auto_grad.tensor import Tensor

T: TypeAlias = Tensor
Number: TypeAlias = float | int | np.number


class Optimizer:
    pass


class SGD(Optimizer):
    def __init__(
        self,
        param: list[tuple[T, T]],
        lr: Number = 1e-3,
        momentum: Number = 0,
        nesterov: bool = False,
    ) -> None:
        self.param = param
        self.lr = lr
        self.beta = momentum
        self.nesterov = nesterov

    def step(self) -> None:
        if not self.nesterov:
            previous_weight_change: tuple[T, T] = (Tensor(0), Tensor(0))
            for layer in self.param:
                if len(layer) == 1:
                    (w_t,) = layer
                    v_t = self.beta * previous_weight_change[0] + (1 - self.beta) * w_t.grad

                    previous_weight_change = (v_t,)
                    w_t -= self.lr * v_t

                else:
                    w_t, b_t = layer

                    w_v_t = self.beta * previous_weight_change[0] + (1 - self.beta) * w_t.grad
                    b_v_t = self.beta * previous_weight_change[1] + (1 - self.beta) * b_t.grad

                    previous_weight_change = (w_v_t, b_v_t)

                    w_t -= self.lr * w_v_t
                    b_t -= self.lr * b_v_t

        else:
            for layer in self.param:
                if len(layer) == 1:
                    (w_t,) = layer
                else:
                    pass

    def zero_grad(self) -> None:
        for layer in self.param:
            if len(layer) == 1:
                (weight_,) = layer
                weight_.zero_()
            elif len(layer) == 2:
                weight_, bias_ = layer
                weight_.zero_()
                bias_.zero_()
            else:
                raise ValueError("The nn does not contain any layer")


class Adam(Optimizer):
    def __init__(self) -> None:
        pass
