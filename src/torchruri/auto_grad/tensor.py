from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

T: TypeAlias = "Tensor"
Number: TypeAlias = np.number | int | float
TensorParent: TypeAlias = tuple["Tensor", "Tensor"] | tuple["Tensor"]

NUMBER_RUNTIME = (np.number, int, float)


class Tensor:
    _no_grad = False

    def __init__(
        self,
        obj: Iterable[Number] | Number,
        require_grad: bool = False,
        *,
        _init_grad: bool = True,
        **kwargs: Any,
    ):
        self.tensor = np.array(obj, **kwargs)
        self.shape: tuple[int, ...] = self.tensor.shape
        self.size: int = self.tensor.size
        self.dtype = self.tensor.dtype

        self._require_grad: None | bool = None
        self._grad: Tensor
        if _init_grad and np.issubdtype(self.dtype, np.floating):
            self._grad = Tensor(
                np.zeros(self.shape, dtype=self.dtype),
                _init_grad=False,
            )

        self.parents: TensorParent = tuple()
        self.is_leaf = True
        self._back_propagate = False
        self._grad_fn: Callable[..., None] = Callable()
        self.require_grad = require_grad

    @property
    def T_(self) -> "T":  # noqa: N802
        return self._transpose()

    @T_.setter
    def T_(self, value: npt.NDArray) -> None:  # noqa: N802
        raise AttributeError("The value of the T_ attribute cannot be set directly.")

    @property
    def require_grad(self) -> bool | None:
        return self._require_grad

    @require_grad.setter
    def require_grad(self, param_require_grad: bool) -> None:
        if param_require_grad and not np.issubdtype(self.dtype, np.floating):
            raise ValueError(f"The require_grad can only be set on float tensor not {self.dtype}")
        self._require_grad = param_require_grad

    @property
    def grad(self) -> "T":
        return self._grad

    @grad.setter
    def grad(self, param_grad: "T") -> None:
        self._grad = param_grad

    def _transpose(self) -> "T":
        return Tensor(self.tensor.T)

    def backward(self) -> None:
        if self._back_propagate:
            raise RuntimeError("You can do a backward pass only once.")

        if self.size != 1:
            raise RuntimeError("You can only call backward() on a scalar vector.")

        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError("Can only call backward on a floating point tensor.")

        self._back_propagate = True
        self.grad = Tensor(np.ones(self.shape), require_grad=True)
        self.grad_fn(self.parents)

        layer: list[T] = []
        layer.extend(self.parents)

        while len(layer) != 0:
            layer = self._backward(layer)

    @staticmethod
    def _backward(layer: list[T]) -> list[T]:
        next_layer: list[T] = []
        while len(layer) != 0:
            node = layer.pop()

            if node.is_leaf:
                continue

            node.grad_fn(node.parents)
            next_layer.extend(node.parents)
        return next_layer

    @classmethod
    def _new_node(
        cls,
        t: "T",
        func: Callable[..., npt.NDArray],
        grad_fn: Callable[..., None],
        other: "T | None" = None,
    ) -> "Tensor":
        if other is None:
            next_node = Tensor(func(t.tensor), require_grad=True)
            next_node.parents = (t,)
        else:
            next_node = Tensor(func(t.tensor, other.tensor), require_grad=True)
            next_node.parents = (t, other)
        next_node._grad_fn = grad_fn
        next_node.is_leaf = False
        return next_node

    def zero_(self) -> None:
        self._grad.tensor = np.zeros(self._grad.tensor.shape)

    def grad_fn(self, parent: tuple[T, T] | tuple[T]) -> None:
        self._grad_fn(self, *parent)

    def _check_number_type(self, u: "T | Number", op_name: str) -> "T":
        if isinstance(u, Tensor):
            return u
        if isinstance(u, NUMBER_RUNTIME):
            return Tensor(u, dtype=self.dtype)
        raise TypeError(f"The type: {type(u)} is not supported by {op_name}")

    def __add__(self, other_tensor: "T | Number") -> "T":
        other: "T" = self._check_number_type(other_tensor, "__add__")
        if not self._require_grad or Tensor._no_grad:
            return Tensor(self.tensor + other.tensor)
        return Tensor._new_node(self, np.add, Tensor._add_backward, other)

    def __radd__(self, other: "T | Number") -> "T":
        return self + other

    @classmethod
    def _add_backward(cls, t: "T", u: "T", v: "T") -> None:
        if u.grad.shape == v.grad.shape:
            u.grad.tensor += t.grad.tensor
            v.grad.tensor += t.grad.tensor

        elif u.grad.shape and not v.grad.shape:
            u.grad.tensor += t.grad.tensor

        elif not u.grad.shape and v.grad.shape:
            v.grad.tensor += t.grad.tensor

        else:
            u_shape = u.grad.shape
            v_shape = v.grad.shape
            raise RuntimeError(f"tensor of shape {u_shape} and {v_shape} are incompatible")

    def __pow__(self, p: "T | Number") -> "T":
        if not isinstance(p, NUMBER_RUNTIME) and not isinstance(p, Tensor):
            raise TypeError("Only support number and tensor")

        if isinstance(p, NUMBER_RUNTIME):
            p: "T" = Tensor(p, dtype=self.dtype)

        if not self._require_grad or Tensor._no_grad:
            return Tensor(self.tensor**p.tensor)
        return Tensor._new_node(self, np.pow, Tensor._pow_backward, p)

    def __rpow__(self, p: "T | Number") -> "T":
        if not isinstance(p, NUMBER_RUNTIME) and not isinstance(p, Tensor):
            raise TypeError("Only support number and tensor")

        if isinstance(p, NUMBER_RUNTIME):
            p: "T" = Tensor(p, dtype=self.dtype)

        if not self._require_grad or Tensor._no_grad:
            return Tensor(self.tensor**p.tensor)
        return Tensor._new_node(p, np.pow, Tensor._pow_backward, self)

    @classmethod
    def _pow_backward(cls, t: "T", u: "T", v: "T") -> None:
        if len(v.shape) != 0 and len(u.shape) != 0:
            u.grad.tensor += t.grad.tensor * (v.tensor * u.tensor ** (v.tensor - 1))
            v.grad.tensor += t.grad.tensor * ((u.tensor**v.tensor) * np.log(u.tensor))
        elif len(u.shape) != 0 and len(v.shape) == 0:
            u.grad.tensor += t.grad.tensor * (v.tensor * u.tensor ** (v.tensor - 1))
            v.grad.tensor = v.grad.tensor
        elif len(v.shape) != 0 and len(u.shape) == 0:
            u.grad.tensor += u.grad.tensor
            v.grad.tensor += t.grad.tensor * ((u.tensor**v.tensor) * np.log(u.tensor))
        else:
            u.grad.tensor = u.grad.tensor
            v.grad.tensor = v.grad.tensor

    def sum(self) -> "T":
        if not self._require_grad or Tensor._no_grad:
            return Tensor(self.tensor.sum())
        return Tensor._new_node(self, np.sum, self._sum_backward)

    @classmethod
    def _sum_backward(cls, t: "T", u: "T") -> None:
        u.grad.tensor += t.grad.tensor * np.ones(u.shape)

    def __mul__(self, other: "T | Number") -> "T":
        other: "T" = self._check_number_type(other, "__mul__")
        if not self._require_grad or Tensor._no_grad:
            return Tensor(self.tensor * other.tensor)
        return Tensor._new_node(self, np.multiply, Tensor._mul_backward, other)

    def __rmul__(self, other: "T | Number") -> "T":
        return other * self

    @classmethod
    def _mul_backward(cls, t: "T", u: "T", v: "T") -> None:
        if u.grad.shape == v.grad.shape:
            u.grad.tensor += t.grad.tensor * v.tensor
            v.grad.tensor += t.grad.tensor * u.tensor

        elif u.grad.shape and not v.grad.shape:
            u.grad.tensor += t.grad.tensor * v.tensor

        elif not u.grad.shape and v.grad.shape:
            v.grad.tensor += t.grad.tensor * u.tensor

        else:
            u_shape = u.grad.shape
            v_shape = v.grad.shape
            raise RuntimeError(f"Tensor of shape {u_shape} and {v_shape} are incompatible")

    def cos(self) -> "T":
        if not self._require_grad or Tensor._no_grad:
            return Tensor(np.cos(self.tensor))
        return Tensor._new_node(self, np.cos, Tensor._cos_backward)

    @classmethod
    def _cos_backward(cls, t: "T", u: "T") -> None:
        u.grad.tensor += t.grad.tensor * (-np.sin(u.tensor))

    def sin(self) -> "T":
        if not self._require_grad or Tensor._no_grad:
            return Tensor(np.sin(self.tensor))
        return Tensor._new_node(self, np.sin, Tensor._sin_backward)

    @classmethod
    def _sin_backward(cls, t: "T", u: "T") -> None:
        u.grad.tensor += t.grad.tensor * np.cos(u.tensor)

    def tan(self) -> "T":
        if not self._require_grad or Tensor._no_grad:
            return Tensor(np.tan(self.tensor))
        return self._new_node(self, np.tan, Tensor._tan_backward)

    @classmethod
    def _tan_backward(cls, t: "T", u: "T") -> None:
        u.grad.tensor += t.grad.tensor * (1 / np.square(np.cos(u.tensor)))

    def __sub__(self, other: "T | Number") -> "T":
        other: "T" = self._check_number_type(other, "__sub__")
        if not self.require_grad or Tensor._no_grad:
            return Tensor(self.tensor - other.tensor)
        return Tensor._new_node(self, np.subtract, Tensor._sub_backward, other)

    def __rsub__(self, other: "T | Number") -> "T":
        other: "T" = self._check_number_type(other, "__rsub__")
        if not self._require_grad or Tensor._no_grad:
            return Tensor(other.tensor - self.tensor)
        return Tensor._new_node(other, np.subtract, Tensor._sub_backward, self)

    @classmethod
    def _sub_backward(cls, t: "T", u: "T", v: "T") -> None:
        if u.grad.shape == v.grad.shape:
            u.grad.tensor += t.grad.tensor
            v.grad.tensor -= t.grad.tensor

        elif u.grad.shape and not v.grad.shape:
            u.grad.tensor += t.grad.tensor

        elif not u.grad.shape and v.grad.shape:
            v.grad.tensor -= t.grad.tensor

        else:
            u_shape = u.grad.shape
            v_shape = v.grad.shape
            raise RuntimeError(f"tensor of shape {u_shape} and {v_shape} are incompatible")

    def __neg__(self) -> "T":
        if not self.require_grad or Tensor._no_grad:
            return Tensor(-self.tensor)
        return Tensor._new_node(self, np.negative, Tensor._neg_backward)

    @classmethod
    def _neg_backward(cls, t: "T", u: "T") -> None:
        u.grad.tensor -= t.grad.tensor

    def log(self) -> "T":
        if not self._require_grad or Tensor._no_grad:
            return Tensor(np.log(self.tensor))
        return Tensor._new_node(self, np.log, Tensor._log_backward)

    @classmethod
    def _log_backward(cls, t: "T", u: "T") -> None:
        epsilon = 1e-30
        u.grad.tensor += t.grad.tensor * 1 / (u.tensor + epsilon)

    def __truediv__(self, other: "T | Number") -> "T":
        other: "T" = self._check_number_type(other, "__truediv__")
        if not self._require_grad or Tensor._no_grad:
            return Tensor(self.tensor / other.tensor)
        return Tensor._new_node(self, np.true_divide, Tensor._truediv_backward, other)

    def __rtruediv__(self, other: "T | Number") -> "T":
        other: "T" = self._check_number_type(other, "__rtruediv__")
        if not self._require_grad or Tensor._no_grad:
            return Tensor(other.tensor / self.tensor)
        return Tensor._new_node(other, np.divide, Tensor._truediv_backward, self)

    @classmethod
    def _truediv_backward(cls, global_t: "T", global_u: "T", global_v: "T") -> None:
        epsilon = 1e-30

        def u_grad(t: "T", u: "T", v: "T") -> None:
            u.grad.tensor += t.grad.tensor * (1 / (v.tensor + epsilon))

        def v_grad(t: "T", u: "T", v: "T") -> None:
            v.grad.tensor += t.grad.tensor * (-u.tensor / (np.square(v.tensor) + epsilon))

        if global_u.grad.shape == global_v.grad.shape:
            u_grad(global_t, global_u, global_v)
            v_grad(global_t, global_u, global_v)

        elif not global_u.grad.shape and global_v.grad.shape:
            v_grad(global_t, global_u, global_v)

        elif global_u.grad.shape and not global_v.grad.shape:
            u_grad(global_t, global_u, global_v)

        else:
            u_shape = global_u.grad.shape
            v_shape = global_v.grad.shape
            raise RuntimeError(f"tensor of shape {u_shape} and {v_shape} are incompatible")

    def __matmul__(self, other: "T") -> "T":
        return self.dot(other)

    def __rmatmul__(self, other: "T") -> "T":
        return self.dot(other)

    def dot(self, other: "T") -> "T":
        if not isinstance(other, Tensor):
            raise TypeError("The dot() operation is only supported on tensor obj.")

        if (
            self.shape[-1] == other.shape[0]
            and len(self.shape) in (1, 2)
            and len(other.shape) in (1, 2)
        ):
            if not self._require_grad or Tensor._no_grad:
                return Tensor(np.dot(self.tensor, other.tensor))
            return Tensor._new_node(self, np.dot, Tensor._dot_backward, other)

        else:
            raise Exception(
                f"Please use torchruri.mathmul for tensor of shape: {self.shape} and {other.shape}"
            )

    @classmethod
    def _dot_backward(cls, t: "T", u: "T", v: "T") -> None:
        if len(u.shape) == 1 and len(v.shape) == 1:
            u.grad.tensor += t.grad.tensor * v.tensor
            v.grad.tensor += t.grad.tensor * u.tensor

        elif len(u.shape) == 2 and len(v.shape) == 1:
            u.grad.tensor += np.outer(t.grad.tensor, v.tensor)
            v.grad.tensor += u.tensor.T @ t.grad.tensor

        elif len(u.shape) == 2 and len(v.shape) == 2:
            u.grad.tensor += t.grad.tensor @ v.tensor.T
            v.grad.tensor += u.tensor.T @ t.grad.tensor

        else:
            raise RuntimeError(f"Unsupported Tensor shape: {u.shape} and {v.shape}")

    def abs(self) -> T:
        if not self._require_grad or Tensor._no_grad:
            return Tensor(np.abs(self.tensor))
        return Tensor._new_node(self, np.abs, Tensor._abs_backward)

    @classmethod
    def _abs_backward(cls, t: "T", u: "T") -> None:
        u.grad.tensor += t.grad.tensor * np.piecewise(
            u.tensor, [u.tensor < 0, u.tensor > 0, u.tensor == 0], [-1, 1, 0]
        )

    def max(self, other: "T | Number") -> "T":
        other: "T" = self._check_number_type(other, "max")
        if not self._require_grad or Tensor._no_grad:
            return Tensor(np.maximum(self.tensor, other.tensor))
        return Tensor._new_node(self, np.maximum, Tensor._max_backward, other)

    @classmethod
    def _max_backward(cls, t: "T", u: "T") -> None:
        u.grad.tensor += t.grad.tensor * (u.tensor > 0).astype(u.dtype)

    def sqrt(self) -> "T":
        return self**0.5

    def all(self) -> np.bool_:
        return self.tensor.all()

    def copy_(self) -> "T":
        return Tensor(self.tensor.copy())

    def __getitem__(self, key: Any) -> "T | Number":
        return Tensor(self.tensor[key])

    def __str__(self) -> Any:
        return self.tensor.__str__()

    def __repr__(self) -> Any:
        return self.tensor.__repr__()

    def __isub__(self, other: "T | Number") -> T:
        return self - other

    def __iadd__(self, other: "T | Number") -> T:
        return self + other
