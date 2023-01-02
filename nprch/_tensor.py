import numpy as np
from typing import Any, Optional, Callable
from abc import ABC, abstractmethod
import weakref

from .config import Config


class AbstractFunction(ABC):
    def __new__(cls, *args, **kwargs):
        cls.generation = 0

    def __call__(self, *args: "Tensor", **kwargs) -> tuple["Tensor", ...]:
        output: tuple["Tensor", ...] = self.forward(*args)

        if Config.enable_backprop:
            self.generation = max([x.generation for x in args])
            for o in output:
                o.creator = self

            self.input: tuple["Tensor", ...] = args
            self.output: tuple[weakref.ReferenceType["Tensor"], ...] = tuple([weakref.ref(o) for o in output])

        return output


    @abstractmethod
    def forward(self, x) -> tuple["Tensor", ...]:
        pass

    @abstractmethod
    def backward(self, grad_y):
        pass


class Tensor(np.ndarray):
    def __new__(cls, data: Any, dtype: Optional[str] = None, requires_grad: bool = False, *args, **kwargs):
        self: np.ndarray = np.asarray(data, dtype=dtype).view(cls)
        self._tensor_data = data
        return self

    def __init__(self, data, name=None, *args, **kwargs):
        self.name = name
        self.grad: Optional = None
        self._creator: Optional[Callable] = None
        self._generation: int = 0

    @property
    def data(self):
        return self._tensor_data

    @property
    def creator(self):
        return self._creator

    @creator.setter
    def creator(self, fn: "AbstractFunction"):
        self._creator = fn
        self._generation = fn.generation + 1

    @property
    def generation(self):
        return self._generation

    def __array_finalize__(self, obj):  # objからselfが作られたときに呼び出される
        if obj is None:
            return None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"nprch.tensor({self._tensor_data})"
