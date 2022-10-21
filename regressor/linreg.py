import numpy as np

from ..autograd.graph import comp_graph
from ..linalg import Mat, ones



class LinearRegressor:
    
    def __init__(self, dims: int = 2, weight_init: Mat = None, bias_init: Mat = None, dtype=np.float32):
        if weight_init is None:
            weight_init = ones(shape=(dims, 1), dtype=dtype, with_grad=True)
        if bias_init is None:
            bias_init = ones(shape=(1), dtype=dtype, with_grad=True)
        self.weight = weight_init
        self.bias = bias_init
        self.graph = comp_graph.get_default_graph()
        
    
    def train(self, Xs, ys, epochs: int = 10, lr: float = .05):
        ys = ys.reshape(-1, 1)
        Xs = Xs.view(Mat)
        ys = ys.view(Mat)

        for epoch in range(epochs):
            with self.graph:
                loss = (Xs @ self.weight + self.bias - ys).norm(p=2)
            loss.backward()
            # grad cal here
            self.step(lr)
            self.graph.clear()
            yield epoch, loss


    def step(self, lr: float):
        self.weight -= self.weight._grad * lr
        self.bias -= self.bias._grad * lr

if __name__ == '__main__':
    ...