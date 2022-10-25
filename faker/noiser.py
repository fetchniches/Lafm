import numpy as np
from abc import ABC, abstractmethod

class BasicNoise(ABC):

    @abstractmethod
    def apply(self, data):
        raise NotImplemented

class GaussianNoise(BasicNoise):

    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def apply(self, data):
        return data + np.random.randn(*data.shape) * self.sigma + self.mu

class ConstantNoise(BasicNoise):

    def __init__(self, constant) -> None:
        super().__init__()
        self.c = constant

    def apply(self, data):
        return data + self.c