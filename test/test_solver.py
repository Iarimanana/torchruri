import numpy as np

from ..src.torchruri import SGD, CrossEntropy, ReLU, Sigmoid
from ..src.torchruri.auto_grad import Tensor
from ..src.torchruri.data_loader import DataLoader
from ..src.torchruri.nn import Linear, Module
from ..src.torchruri.types import T


class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Linear(2, 42)
        self.l2 = Linear(42, 1)
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

    def forward(self, x: T) -> T:
        x1 = self.relu(self.l1(x))
        x2 = self.sigmoid(self.l2(x1))
        return x2


class TestSolver:
    def test_sgd(self) -> None:
        a = np.random.rand(512, 2) * 4.5
        a_id = np.zeros((512, 2))

        b = np.random.rand(2048, 2) * 5 + 5
        b_id = np.ones((2048, 2))

        index = np.arange(a + b)
        np.random.shuffle(index)
        t_point = np.concatenate((a, b), dtype=np.float64)[index]
        label = np.concatenate((a_id, b_id))[index]

        dl = DataLoader((Tensor(t_point), Tensor(label)), batch_size=12)

        model = Model()

        optim = SGD(model.param(), momentum=0.9)
        loss_fn = CrossEntropy()

        epochs = 10
        for _ in range(epochs):
            total_loss = 0
            for x, y in dl:
                outputs = model(x)
                loss = loss_fn(outputs, y)
                optim.zero_grad()

                loss.backward()
                optim.step()

                total_loss += loss
