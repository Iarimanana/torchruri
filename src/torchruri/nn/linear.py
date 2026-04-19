from typing import Any, TypeAlias

import numpy as np

from ..auto_grad.tensor import Tensor
from ._layer import Layer

T: TypeAlias = Tensor


class Linear(Layer):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self._in = (input_features,)
        self._out = (output_features,)
        self._use_bias = bias

        self._weigths = Tensor(np.random.normal(size=(*self._out, *self._in)), require_grad=True)
        if self._use_bias:
            self._bias = Tensor(np.zeros(self._out), require_grad=True)

    @property
    def bias(self) -> T | None:
        if self._use_bias:
            return self._bias
        else:
            raise AttributeError("Use bias=True to use bias on this layer.")

    @bias.setter
    def bias(self, value: T) -> None:
        if value.shape != self._out:
            raise ValueError(f"The bias tensor should be of size {self._out}.")
        self._bias = value

    @property
    def weigths(self) -> T | None:
        return self._weigths

    @weigths.setter
    def weigths(self, value: T) -> None:
        if value.shape != self._weigths.shape:
            raise ValueError(f"The weights tensor should be of size {self._weigths.shape}")
        self._weigths = value

    def forward(self, input: T) -> T:
        if self._weigths is None:
            raise ValueError("The 'weigths' attribute was not set.")

        if self._use_bias:
            return input @ self._weigths.T_ + self._bias
        else:
            return input @ self._weigths.T_

    # NOTE: Every layers have a tuple of weigths and bias named "parameters"
    def parameters(self) -> tuple[T | None, T | None]:
        return self.weigths, self.bias

    def __call__(self, input_: T) -> T:
        if input_.shape[1:] != self._in:
            raise ValueError(
                f"The shape {input_.shape} of the input  is incompatible with {self._in}"
            )
        return self.forward(input_)

    def __str__(self) -> Any:
        return f"{self.__class__.__name__}{*self._in, *self._out}"

    def __repr__(self) -> Any:
        return self.__str__()
