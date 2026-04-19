import numpy as np

from ..src.torchruri.auto_grad import Tensor
from ..src.torchruri.utils import OneHotEncoder


def test_one_hot_encoder() -> None:
    one = OneHotEncoder(list(range(10)))
    u = Tensor(np.random.choice(10, 1000))
    print(one(u)[9:15])
    print(u[9:15])
    print(one.label)
