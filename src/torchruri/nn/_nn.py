from typing import Any

from ..auto_grad import Tensor
from ..types import T
from . import Layer


class Module:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._forward_was_override = True
        self._param: list[tuple[T, T]] = []

    def forward(self, *args: Any, **kwargs: Any) -> T:
        self._forward_was_override = False
        return Tensor([])

    def param(self) -> list[tuple[T, T]]:
        layers = list(filter(lambda arr: isinstance(arr, Layer), self.__dict__.values()))
        for layer in layers:
            self._param.append(layer.parameters())

        return self._param

    def zero_grad(self) -> None:
        for layer in self._param:
            if len(layer) == 1:
                (weight_,) = layer
                weight_.zero_()
            elif len(layer) == 2:
                weight_, bias_ = layer
                weight_.zero_()
                bias_.zero_()
            else:
                raise ValueError("The nn does not contain any layer")

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        _out = self.forward(*args, **kwargs)
        if not self._forward_was_override:
            raise Exception("The forward method must be override.")
        return _out

    def __str__(self) -> Any:
        return str({k: v for k, v in self.__dict__.items() if isinstance(v, Layer)})

    def __repr__(self) -> str:
        return self.__str__()
