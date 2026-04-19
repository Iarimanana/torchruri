from collections.abc import Callable
from typing import TypeAlias, overload

import numpy as np
import pytest
import torch

from ..src.torchruri import ufunc
from ..src.torchruri.auto_grad import Tensor, no_grad

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
    _size = [(8,), (12, 12), (42, 73), (100,), (11, 13, 17, 19, 23)]

    def test_T_(  # noqa: N802
        self, one_tensor: Callable[[], TupleTensorOne]
    ) -> None:
        v, tv = one_tensor()
        assert_allclose(tv.detach().numpy(), v.T_.tensor)
        with pytest.raises(
            Exception,
            match="The value of the T_ attribute cannot be set directly.",
        ):
            u, tu = one_tensor()
            u.T_ = Tensor(np.random.rand(*u.shape))
            u.T_ = u.T_

    @pytest.mark.parametrize("size", _size)
    def test_sum(
        self,
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
        size: tuple[int] | tuple[int, ...],
    ) -> None:
        u, tu = one_tensor(size)
        w = u.sum()
        tw = tu.sum()
        w.backward()
        tw.backward()
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_add(
        self,
        size: tuple[int] | tuple[int, ...],
        two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor(size)
        x, tx = one_tensor(size)
        y, ty = one_tensor(size)
        z, tz = one_tensor(size)
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

    @pytest.mark.parametrize("size", _size)
    def test_mul(
        self,
        size: tuple[int] | tuple[int, ...],
        two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor(size)
        x, tx = one_tensor(size)
        y, ty = one_tensor(size)
        scalar = 2

        a, ta = _testing_two_operand_operator(ufunc.mul, torch.mul, u, v, tu, tv)
        b, tb = _testing_two_operand_operator(ufunc.mul, torch.mul, x, scalar, tx, scalar)
        c, tc = _testing_two_operand_operator(ufunc.mul, torch.mul, y, y, ty, ty)

        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)
        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_sub(
        self,
        size: tuple[int] | tuple[int, ...],
        two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor(size)
        x, tx = one_tensor(size)
        y, ty = one_tensor(size)
        z, tz = one_tensor(size)
        scalar = 2

        a, ta = _testing_two_operand_operator(ufunc.sub, torch.sub, u, v, tu, tv)
        b, tb = _testing_two_operand_operator(ufunc.sub, torch.sub, x, scalar, tx, scalar)
        c, tc = _testing_two_operand_operator(ufunc.sub, torch.sub, y, y, ty, ty)
        d, td = _testing_two_operand_operator(ufunc.sub, torch.sub, scalar, z, tz, scalar)

        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)
        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_neg(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        x, tx = one_tensor(size)
        a, ta = _testing_one_operand_operator(ufunc.neg, torch.neg, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_truediv(
        self,
        size: tuple[int] | tuple[int, ...],
        two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, v, tu, tv = two_tensor(size)
        x, tx = one_tensor(size)
        y, ty = one_tensor(size)
        scalar = 2
        a, ta = _testing_two_operand_operator(ufunc.truediv, torch.div, u, v, tu, tv)
        b, tb = _testing_two_operand_operator(ufunc.truediv, torch.div, x, scalar, tx, scalar)
        c, tc = _testing_two_operand_operator(ufunc.truediv, torch.div, scalar, y, scalar, ty)

        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_cos(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        x, tx = one_tensor(size)
        a, ta = _testing_one_operand_operator(ufunc.cos, torch.cos, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_sin(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        x, tx = one_tensor(size)
        a, ta = _testing_one_operand_operator(ufunc.sin, torch.sin, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_tan(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        x, tx = one_tensor(size)
        a, ta = _testing_one_operand_operator(ufunc.tan, torch.tan, x, tx)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)

    @pytest.mark.parametrize(
        "size,reverse_size", [((8,), (8,)), ((2, 3), (3,)), ((4, 4), (4, 4)), ((3, 7), (7, 4))]
    )
    def test_dot(
        self,
        size: tuple[int] | tuple[int, ...],
        reverse_size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, tu = one_tensor(size)
        v, tv = one_tensor(reverse_size)

        w = u @ v
        tw = tu @ tv

        x = w.sum()
        tx = tw.sum()

        x.backward()
        tx.backward()

        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)

    @pytest.mark.parametrize("size", [(8,)])
    def test_pow(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
        two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo],
    ) -> None:
        u, tu = one_tensor(size)
        x, y, tx, ty = two_tensor(size)

        a = ufunc.pow(u, 3)
        b = a.sum()
        ta = torch.pow(tu, 3)
        tb = ta.sum()

        c = ufunc.pow(x, y)
        tc = torch.pow(tx, ty)
        c = c.sum()
        tc = torch.sum(tc)

        b.backward()
        tb.backward()
        c.backward()
        tc.backward()

        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)
        assert_allclose(ty.grad.detach().numpy(), y.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_log(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int | tuple[int, ...]]], TupleTensorOne],
    ) -> None:
        u, tu = one_tensor(size)
        a, ta = _testing_one_operand_operator(ufunc.log, torch.log, u, tu)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)

    @pytest.mark.parametrize(
        "size",
        _size,
    )
    def test_abs(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, tu = one_tensor(size)
        x = Tensor([1, 2, -2, -9, 0, 0, 1], dtype=np.float64, require_grad=True)
        tx = torch.tensor(
            [1, 2, -2, -9, 0, 0, 1],
            dtype=torch.float64,
            requires_grad=True,
        )

        a, ta = _testing_one_operand_operator(ufunc.abs, torch.abs, u, tu)
        b, tb = _testing_one_operand_operator(ufunc.abs, torch.abs, x, tx)

        assert_allclose(tx.grad.detach().numpy(), x.grad.tensor)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_sqrt(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, tu = one_tensor(size)
        a, ta = _testing_one_operand_operator(ufunc.sqrt, torch.sqrt, u, tu)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)

    @pytest.mark.parametrize("size", _size)
    def test_max(
        self,
        size: tuple[int] | tuple[int, ...],
        one_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorOne],
    ) -> None:
        u, tu = one_tensor(size)
        v, tv = (0, torch.tensor(0))
        a, ta = _testing_two_operand_operator(ufunc.max, torch.maximum, u, v, tu, tv)
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)

    def test_zero_(
        self, two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo]
    ) -> None:
        u, v, tu, tv = two_tensor((8,))
        a, ta = _testing_two_operand_operator(ufunc.add, torch.add, u, v, tu, tv)
        u.zero_()
        v.zero_()
        tu.grad.zero_()
        tv.grad.zero_()
        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)

    @pytest.mark.parametrize("size", [(8,)])
    def test_backward(
        self,
        size: tuple[int] | tuple[int, ...],
        two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo],
    ) -> None:
        u, v, tu, tv = two_tensor(size)
        a, b, ta, tb = two_tensor(size)

        f = (ufunc.sin(u) - ufunc.tan(v) + v * v) / 3
        tf = (torch.sin(tu) - torch.tan(tv) + tv * tv) / 3

        g = ufunc.sin((b**3 + a**2) / 7)
        tg = torch.sin((tb**3 + ta**2) / 7)

        w = f @ g
        tw = tf @ tg

        w.backward()
        tw.backward()

        assert_allclose(tu.grad.detach().numpy(), u.grad.tensor)
        assert_allclose(tv.grad.detach().numpy(), v.grad.tensor)
        assert_allclose(ta.grad.detach().numpy(), a.grad.tensor)
        assert_allclose(tb.grad.detach().numpy(), b.grad.tensor)

    def test_no_grad(
        self,
        two_tensor: Callable[[tuple[int] | tuple[int, ...]], TupleTensorTwo],
    ) -> None:
        u, v, tu, tv = two_tensor((8,))
        a, ta = _testing_two_operand_operator(ufunc.add, torch.add, u, v, tu, tv)
        pass_ = False

        with no_grad:
            w = u + v
            w = ufunc.sum(w)
            try:
                w.backward()
            except Exception:
                pass_ = True

        @no_grad
        def decorator_test() -> None:
            w = u + v
            w = ufunc.sum(w)
            try:
                w.backward()
            except Exception:
                pass_ = True

        assert pass_
