from typing import TypeAlias, overload

import numpy as np

from .auto_grad.tensor import Tensor

T: TypeAlias = Tensor
Number: TypeAlias = int | float | np.number
NUMBER_RUNTIME = (int, float, np.number)


@overload
def add(u: T, v: Number) -> T: ...


@overload
def add(u: Number, v: T) -> T: ...


@overload
def add(u: T, v: T) -> T: ...


def add(u: T | Number, v: T | Number) -> T:
    if isinstance(u, NUMBER_RUNTIME) and isinstance(v, NUMBER_RUNTIME):
        raise TypeError("The add() does not support Number and Number operation")
    return u + v  # type: ignore


@overload
def mul(u: Number, v: T) -> T: ...


@overload
def mul(u: T, v: Number) -> T: ...


@overload
def mul(u: T, v: T) -> T: ...


def mul(u: T | Number, v: T | Number) -> T:
    if isinstance(u, NUMBER_RUNTIME) and isinstance(v, NUMBER_RUNTIME):
        raise TypeError("The add() does not support Number and Number operation")
    return u * v  # type: ignore


@overload
def sub(u: Number, v: T) -> T: ...


@overload
def sub(u: T, v: Number) -> T: ...


@overload
def sub(u: T, v: T) -> T: ...


def sub(u: T | Number, v: T | Number) -> T:
    if isinstance(u, NUMBER_RUNTIME) and isinstance(v, NUMBER_RUNTIME):
        raise TypeError("The add() does not support Number and Number operation")
    return u - v  # type: ignore


@overload
def truediv(u: T, v: Number) -> T: ...


@overload
def truediv(u: Number, v: T) -> T: ...


@overload
def truediv(u: T, v: T) -> T: ...


def truediv(u: T | Number, v: T | Number) -> T:
    if isinstance(u, NUMBER_RUNTIME) and isinstance(v, NUMBER_RUNTIME):
        raise TypeError("The truediv does not support Number and Number operation")
    return u / v  # type: ignore


@overload
def max(u: T, v: T) -> T: ...


@overload
def max(u: T, v: Number) -> T: ...


@overload
def max(u: Number, v: T) -> T: ...


def max(u: T | Number, v: T | Number = 0) -> T:
    if isinstance(u, Tensor):
        return u.max(v)
    elif isinstance(v, Tensor):
        return v.max(u)
    else:
        raise TypeError("Incompatible type for max()")


def dot(u: T, v: T) -> T:
    if not isinstance(u, T) or not isinstance(v, T):
        raise TypeError("The truediv does not support Number and Number operation")
    return u @ v


@overload
def pow(u: T, p: T) -> T: ...


@overload
def pow(u: Number, p: T) -> T: ...


@overload
def pow(u: T, p: Number) -> T: ...


def pow(u: T | Number, p: T | Number) -> T:
    if not isinstance(u, Tensor) and not isinstance(u, NUMBER_RUNTIME):
        raise TypeError("The first tensor must be a number or a tensor")

    if not isinstance(p, Tensor) and not isinstance(p, NUMBER_RUNTIME):
        raise TypeError("The second tensor must be a number or a tensor")

    if isinstance(u, NUMBER_RUNTIME) and isinstance(p, NUMBER_RUNTIME):
        raise TypeError("One of the argument of uf.pow must be a tensor")

    return u**p  # type: ignore


def abs(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The abs() does not support {type(u)}.")
    return u.abs()


def sin(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The sin() does not support {type(u)}.")
    return u.sin()


def cos(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The cos() does not support {type(u)}.")
    return u.cos()


def all(u: T) -> np.bool_:
    if not isinstance(u, Tensor):
        raise TypeError(f"The all() does not support {type(u)}.")
    return Tensor.all(u)


def neg(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The neg() does not support {type(u)}.")
    return -u


def tan(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The tan() does not support {type(u)}.")
    return u.tan()


def sqrt(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The tan() does not support {type(u)}.")
    return u.sqrt()


def log(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The log() does not support {type(u)}.")
    return u.log()


def sum(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The tan() does not support {type(u)}.")
    return u.sum()


def exp(u: T) -> T:
    if not isinstance(u, Tensor):
        raise TypeError(f"The tan() does not support {type(u)}.")
    return np.e**u
