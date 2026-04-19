import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__(self)
        self.l1 = nn.Linear(10, 42)
        self.l2 = nn.Linear(42, 73)
        self.l3 = nn.Linear(73, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.softmax(self.l3(x2))
        return x3
