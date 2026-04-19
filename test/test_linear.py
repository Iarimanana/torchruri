from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
import torch

from ..src.torchruri.auto_grad import Tensor
from ..src.torchruri.nn.linear import Linear

RTOL = 1e-10
ATOL = 0.0


def assert_allclose(a: object, b: object) -> None:
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


TupleTensorOne: TypeAlias = tuple[Tensor, torch.Tensor]
TupleTensorTwo: TypeAlias = tuple[Tensor, Tensor, torch.Tensor, torch.Tensor]
T: TypeAlias = Tensor
TorchT: TypeAlias = torch.Tensor
Number: TypeAlias = int | float | np.number


@pytest.fixture
def one_tensor() -> Callable[[], TupleTensorOne]:
    def create_one_tensor(
        size: tuple[int] | tuple[int, ...] = (8,),
    ) -> TupleTensorOne:
        arr = np.random.rand(*size)
        u = Tensor(arr, require_grad=True)
        tu = torch.from_numpy(arr)
        tu.requires_grad_(True)
        return u, tu

    return create_one_tensor


@pytest.fixture
def two_tensor() -> Callable[[], TupleTensorTwo]:
    def create_two_tensor(
        size: tuple[int] | tuple[int, ...] = (8,),
    ) -> TupleTensorTwo:
        arr_one = np.random.rand(*size)
        array_two = np.random.rand(*size)

        u = Tensor(arr_one, require_grad=True)
        v = Tensor(array_two, require_grad=True)

        tu = torch.from_numpy(arr_one)
        tu.requires_grad_(True)

        tv = torch.from_numpy(array_two)
        tv.requires_grad_(True)

        return u, v, tu, tv

    return create_two_tensor


def test_linear(
    one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    size: tuple[int] | tuple[int, ...] = (100, 2),
) -> None:
    layer = Linear(2, 42)
    tlayer = torch.nn.Linear(2, 42, dtype=torch.float64)
    u, tu = one_tensor(size)

    out = layer(u)
    print(out)
    print(out.shape)
