from collections.abc import Callable
from typing import TypeAlias, overload

import numpy as np
import pytest
import torch

from ..src.torchruri import Tensor, ufunc

RTOL = 1e-15
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
    def create_one_tensor() -> TupleTensorOne:
        arr = np.random.rand(8)
        u = Tensor(arr, require_grad=True)
        tu = torch.tensor(arr, requires_grad=True)
        return u, tu

    return create_one_tensor


@pytest.fixture
def two_tensor() -> Callable[[], TupleTensorTwo]:
    def create_two_tensor() -> TupleTensorTwo:
        arr_one = np.random.rand(8)
        array_two = np.random.rand(8)
        u = Tensor(arr_one, require_grad=True)
        v = Tensor(array_two, require_grad=True)
        tu = torch.tensor(arr_one, requires_grad=True)
        tv = torch.tensor(array_two, requires_grad=True)
        return u, v, tu, tv

    return create_two_tensor


@overload
def _testing_two_operand_operator(
    op: Callable[[Number, T], T],
    torch_op: Callable[[Number, TorchT], TorchT],
    /,
    u: Number,
    v: T,
    tu: Number,
    tv: TorchT,
) -> tuple[T, TorchT]: ...


@overload
def _testing_two_operand_operator(
    op: Callable[[T, Number], T],
    torch_op: Callable[[TorchT, Number], TorchT],
    /,
    u: T,
    v: Number,
    tu: TorchT,
    tv: Number,
) -> tuple[T, TorchT]: ...


@overload
def _testing_two_operand_operator(
    op: Callable[[T, T], T],
    torch_op: Callable[[TorchT, TorchT], TorchT],
    /,
    u: T,
    v: T,
    tu: TorchT,
    tv: TorchT,
) -> tuple[T, TorchT]: ...


def _testing_two_operand_operator(  # type: ignore
    op: Callable[[T | Number, T | Number], T],
    torch_op: Callable[[TorchT | Number, TorchT | Number], TorchT],
    /,
    u: T | Number,
    v: T | Number,
    tu: TorchT | Number,
    tv: TorchT | Number,
) -> tuple[T, TorchT]:
    w = op(u, v)
    z = w.sum()
    tw = torch_op(tu, tv)
    tz = tw.sum()

    z.backward()
    tz.backward()

    return z, tz


def _testing_one_operand_operator(
    op: Callable[[T], T],
    torch_op: Callable[[TorchT], TorchT],
    /,
    u: T,
    tu: TorchT,
) -> tuple[T, TorchT]:
    w = op(u)
    z = w.sum()
    tw = torch_op(tu)
    tz = tw.sum()

    z.backward()
    tz.backward()

    return z, tz


class TestTensor:
    def test_T_(  # noqa: N802
        self, one_tensor: Callable[[], TupleTensorOne]
    ) -> None:
        with pytest.raises(
            Exception,
            match="The value of the T_ attribute cannot be set directly.",
        ):
            u, tu = one_tensor()
            u.T_ = Tensor(np.random.rand(*u.shape))
            u.T_ = u.T_

    def test_sum(self, one_tensor: Callable[[], TupleTensorOne]) -> None:
        u, tu = one_tensor()
        w = u.sum()
        tw = tu.sum()
        w.backward()
        tw.backward()
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)

    def test_add(
        self,
        two_tensor: Callable[[], TupleTensorTwo],
        one_tensor: Callable[[], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor()
        x, tx = one_tensor()
        y, ty = one_tensor()
        z, tz = one_tensor()
        scalar = 2

        a, ta = _testing_two_operand_operator(ufunc.add, torch.add, u, v, tu, tv)
        b, tb = _testing_two_operand_operator(ufunc.add, torch.add, x, scalar, tx, scalar)
        c, tc = _testing_two_operand_operator(ufunc.add, torch.add, y, y, ty, ty)
        d, td = _testing_two_operand_operator(ufunc.add, torch.add, scalar, z, scalar, tz)

        assert_allclose(tz.grad.detach().numpy(), z.grad.tensor)
        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)

    def test_mul(
        self,
        two_tensor: Callable[[], TupleTensorTwo],
        one_tensor: Callable[[], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor()
        x, tx = one_tensor()
        y, ty = one_tensor()
        scalar = 2

        a, ta = _testing_two_operand_operator(ufunc.mul, torch.mul, u, v, tu, tv)
        b, tb = _testing_two_operand_operator(ufunc.mul, torch.mul, x, scalar, tx, scalar)
        c, tc = _testing_two_operand_operator(ufunc.mul, torch.mul, y, y, ty, ty)

        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)
        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)

    def test_sub(
        self,
        two_tensor: Callable[[], TupleTensorTwo],
        one_tensor: Callable[[], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor()
        x, tx = one_tensor()
        y, ty = one_tensor()
        z, tz = one_tensor()
        scalar = 2

        a, ta = _testing_two_operand_operator(ufunc.sub, torch.sub, u, v, tu, tv)
        b, tb = _testing_two_operand_operator(ufunc.sub, torch.sub, x, scalar, tx, scalar)
        c, tc = _testing_two_operand_operator(ufunc.sub, torch.sub, y, y, ty, ty)
        d, td = _testing_two_operand_operator(ufunc.sub, torch.sub, scalar, z, tz, scalar)

        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)
        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)

    def test_neg(self, one_tensor: Callable[[], TupleTensorOne]) -> None:
        x, tx = one_tensor()
        a, ta = _testing_one_operand_operator(ufunc.neg, torch.neg, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    def test_truediv(
        self,
        two_tensor: Callable[[], TupleTensorTwo],
        one_tensor: Callable[[], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor()
        x, tx = one_tensor()
        y, ty = one_tensor()
        scalar = 2
        a, ta = _testing_two_operand_operator(ufunc.truediv, torch.div, u, v, tu, tv)
        b, tb = _testing_two_operand_operator(ufunc.truediv, torch.div, x, scalar, tx, scalar)
        c, tc = _testing_two_operand_operator(ufunc.truediv, torch.div, scalar, y, scalar, ty)

        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    def test_cos(self, one_tensor: Callable[[], TupleTensorOne]) -> None:
        x, tx = one_tensor()
        a, ta = _testing_one_operand_operator(ufunc.cos, torch.cos, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    def test_sin(self, one_tensor: Callable[[], TupleTensorOne]) -> None:
        x, tx = one_tensor()
        a, ta = _testing_one_operand_operator(ufunc.sin, torch.sin, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    def test_tan(self, one_tensor: Callable[[], TupleTensorOne]) -> None:
        x, tx = one_tensor()
        a, ta = _testing_one_operand_operator(ufunc.tan, torch.tan, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    def test_backward(self, two_tensor: Callable[[], TupleTensorTwo]) -> None:
        u, v, tu, tv = two_tensor()

        f = (ufunc.sin(u) + v * v) / 3
        tf = (torch.sin(tu) + tv * tv) / 3

        w = f.sum()
        tw = tf.sum()

        w.backward()
        tw.backward()

        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
