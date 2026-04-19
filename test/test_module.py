from pprint import pprint

import numpy as np
import torch

from ..src.torchruri.activation_functions import ReLU
from ..src.torchruri.auto_grad import Tensor
from ..src.torchruri.nn import Linear
from ..src.torchruri.nn._nn import Module
from ..src.torchruri.types import T


class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Linear(42, 73)
        self.relu = ReLU()
        self.l2 = Linear(73, 10)

    def forward(self, x: T) -> T:
        out = self.l2(self.relu(self.l1(x)))
        return out


class TModel(torch.nn.Module):  # type: ignore
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(42, 73, dtype=torch.float64)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(73, 10, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.l2(self.relu(self.l1(x)))
        return out


class TestModule:
    def test_weight(self) -> None:
        arr = np.random.standard_normal(42)

        u = Tensor(arr)
        tu = torch.from_numpy(arr)

        tmodel = TModel()
        model = Model()

        tv = tmodel(tu)
        v = model(u)

        pprint(model.param())
        print("\n\n\n\n\n")
        print(model)
