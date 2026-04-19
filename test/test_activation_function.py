"""
Unit test for Loss Functions
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
import torch

from ..src.torchruri.activation_functions import LeakyReLU, ReLU, Sigmoid
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


def test_relu(
    one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    size: tuple[int] | tuple[int, ...] = (8,),
) -> None:
    u, tu = one_tensor(size)
    relu = ReLU()
    w = relu(u)
    tw = torch.nn.functional.relu(tu)

    w = w.sum()
    tw = tw.sum()

    w.backward()
    tw.backward()

    assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)


def test_leaky_relu(
    one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    size: tuple[int] | tuple[int, ...] = (8,),
) -> None:
    u, tu = one_tensor(size)
    leaky_relu = LeakyReLU()
    w = leaky_relu(u)
    tw = torch.nn.functional.leaky_relu(tu)

    w = w.sum()
    tw = tw.sum()

    w.backward()
    tw.backward()

    assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)


def test_sigmoid(
    one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    size: tuple[int] | tuple[int, ...] = (8,),
) -> None:
    u, tu = one_tensor(size)
    sigmoid = Sigmoid()
    w = sigmoid(u)
    tw = torch.nn.functional.sigmoid(tu)

    w = w.sum()
    tw = tw.sum()

    w.backward()
    tw.backward()

    assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
