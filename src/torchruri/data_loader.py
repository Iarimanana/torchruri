from typing import Any, Self

import numpy as np

from .types import T


class DataLoader:
    def __init__(self, dataset: tuple[T, T], batch_size: int = 1, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle

        if len(dataset) != 2:
            raise TypeError(
                "The dataset parameter only accept a dataset of traning data and labels"
            )

        if self.dataset[0].shape[0] != self.dataset[1].shape[0]:
            raise Exception("The number of training sample does not match the number of labels.")

        self.y: T
        self.x: T

        self._size = self.dataset[1].shape[0]
        self._current = 0
        self._stop = False
        self._batch_size = batch_size

    def batch(self) -> None:
        if self.shuffle:
            index_batch = (np.random.choice(self._size, size=self.batch_size)).tolist()
            self.x = self.dataset[0][index_batch]  # type: ignore
            self.y = self.dataset[1][index_batch]  # type: ignore
        else:
            self.x = self.dataset[0][self._current : self._current + self.batch_size]  # type: ignore
            self.y = self.dataset[1][self._current : self._current + self.batch_size]  # type: ignore
        self._current += self.batch_size

    def __iter__(self) -> Self:
        return self

    def _clean(self) -> None:
        self.batch_size = self._batch_size
        self._current = 0
        self._stop = False
        raise StopIteration

    def __next__(self) -> Any:
        if self._stop:
            self._clean()

        if self._current > self._size:
            self.batch_size = self._current - self._size
            self._stop = True

        if self._current == self._size:
            self._clean()

        self.batch()
        return self.x, self.y
