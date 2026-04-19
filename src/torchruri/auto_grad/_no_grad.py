from contextlib import ContextDecorator
from typing import Any, Self, TypeAlias

from .tensor import Tensor

T: TypeAlias = Tensor


class NoGrad(ContextDecorator):
    def __enter__(self) -> Self:
        Tensor._no_grad = True
        return self

    def __exit__(self, *_: Any) -> None:
        Tensor._no_grad = False


no_grad = NoGrad()
