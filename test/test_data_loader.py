import numpy as np
import torch

from ..src.torchruri.auto_grad.tensor import Tensor
from ..src.torchruri.data_loader import DataLoader


def test_data_loader() -> None:
    arr = np.arange(512).reshape(64, 8)
    label = np.arange(64)
    x, y = Tensor(arr), Tensor(label)
    tx, ty = torch.from_numpy(arr), torch.from_numpy(label)

    epochs = 8
    batch_size = 8
    dataloader = DataLoader((x, y), batch_size=batch_size)
    for i, _ in enumerate(range(epochs)):
        for j, (a, b) in enumerate(dataloader):
            print(f"=============={i}==============")
            print(a.shape)
            print(b.shape)
            print("\n\n\n\n")
        else:
            j += 1
            print(j)

    print("With shuffle=False")
    dataloader = DataLoader((x, y), batch_size=batch_size, shuffle=False)
    for i, _ in enumerate(range(epochs)):
        for j, (a, b) in enumerate(dataloader):
            print(f"=============={i}==============")
            print(a.shape)
            print(b.shape)
            print("\n\n\n\n")
        else:
            j += 1
            print(j)

    # tdataset = torch.utils.data.TensorDataset(tx, ty)
    # tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=batch_size)
    # for i, _ in enumerate(range(epochs)):
    #     for j, (x, y) in enumerate(tdataloader):
    #         print(f"=============={i}==============")
    #         pprint(x)
    #         pprint(y)
    #         print("\n\n\n\n")
    #     else:
    #         j += 1
    #         print(f"The number of iteration: {j}")
