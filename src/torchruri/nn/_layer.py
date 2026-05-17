from typing import TypeAlias

from ..auto_grad.tensor import Tensor

T: TypeAlias = Tensor


class Layer:
    def __init__(self) -> None:
        pass

    def parameters(self) -> tuple[T | None, T | None]:
        return Tensor([]), Tensor([])
