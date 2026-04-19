from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
import torch

from ..src.torchruri.activation_functions import ReLU
from ..src.torchruri.loss_functions import L1Loss, MSELoss
from ..src.torchruri.nn.linear import Linear
from ..src.torchruri.auto_grad.tensor import Tensor

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
        arr = np.random.randn(*size)
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
        arr_one = np.random.randn(*size)
        array_two = np.random.randn(*size)

        u = Tensor(arr_one, require_grad=True)
        v = Tensor(array_two, require_grad=True)

        tu = torch.from_numpy(arr_one)
        tu.requires_grad_(True)

        tv = torch.from_numpy(array_two)
        tv.requires_grad_(True)

        return u, v, tu, tv

    return create_two_tensor


def test_mse_loss(one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne]) -> None:
    u, tu = one_tensor((42,))
    v, tv = one_tensor((73,))
    y, ty = one_tensor((73,))

    linear = Linear(42, 73)
    tlinear = torch.nn.Linear(42, 73, dtype=torch.float64)
    relu = ReLU()
    loss_fn = MSELoss()
    tloss_fn = torch.nn.MSELoss()

    w = loss_fn(y, relu(linear(u)))
    tw = tloss_fn(ty, torch.nn.functional.relu(tlinear(tu)))

    w.backward()
    tw.backward()

    x = loss_fn(y, v)
    tx = tloss_fn(ty, tv)

    x = x.sum()
    tx = tx.sum()

    x.backward()
    tx.backward()

    assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)


def test_l1_loss(one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne]) -> None:
    u, tu = one_tensor((42,))
    v, tv = one_tensor((73,))
    y, ty = one_tensor((73,))

    linear = Linear(42, 73)
    tlinear = torch.nn.Linear(42, 73, dtype=torch.float64)
    relu = ReLU()
    loss_fn = L1Loss()
    tloss_fn = torch.nn.L1Loss()

    w = loss_fn(y, relu(linear(u)))
    tw = tloss_fn(ty, torch.nn.functional.relu(tlinear(tu)))

    w.backward()
    tw.backward()

    x = loss_fn(y, v)
    tx = tloss_fn(ty, tv)

    x = x.sum()
    tx = tx.sum()

    x.backward()
    tx.backward()

    assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)
