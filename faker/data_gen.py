import numpy as np
from .noiser import BasicNoise

class Linear:
    
    def __init__(self, w, b):
        self.w = w.reshape(-1, 1)
        self.b = b
        self.noiser = None

    def gen(self, xmin, xmax, nums):
        shape = self.w.shape[0]
        Xs = np.random.ranf(size=(nums, shape)).astype(np.float64) * (xmax - xmin) + xmin
        Ys = Xs @ self.w + self.b
        if self.noiser is not None:
            Ys = self.noiser.apply(Ys)
        return Xs, Ys
    
    def attach_noiser(self, noiser: BasicNoise):
        self.noiser = noiser


if __name__ == '__main__':
    ...
