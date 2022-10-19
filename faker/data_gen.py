import numpy as np

class Linear:
    
    def __init__(self, w, b):
        self.w = w.reshape(-1, 1)
        self.b = b

    def gen(self, xmin, xmax, nums):
        shape = self.w.shape[0]
        Xs = np.random.randint(xmin, xmax, size=(nums, shape)).astype(np.float64)
        Ys = Xs @ self.w + self.b
        return Xs, Ys

if __name__ == '__main__':
    ...
