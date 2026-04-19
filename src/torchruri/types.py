from typing import TypeAlias

import numpy as np

from .auto_grad import Tensor

T: TypeAlias = Tensor
Number: TypeAlias = float | int | np.number
