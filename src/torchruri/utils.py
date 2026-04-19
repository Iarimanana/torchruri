from collections.abc import Sized

import numpy as np

from .types import T


class OneHotEncoder:
    def __init__(self, label: Sized) -> None:
        self.label = label
        self.state_dict = {k: v for v, k in enumerate(self.label)}  # type: ignore

    def forward(self, u: T) -> T:
        if len(u.shape) != 1:
            raise ValueError("OneHotEncoder does not support tensor other than 1 dim")

        m = np.zeros(len(self.label) * u.size).reshape((u.size, len(self.label)))
        i = 0
        while i < u.size:
            m[i][self.state_dict[u[i]]] = 1
            i += 1

        return T(m)

    def __call__(self, u: T) -> T:
        return self.forward(u)
