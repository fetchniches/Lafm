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
        self.last_grads = None
    
    def train(self, Xs, ys, epochs: int = 10, lr: float = .05, momentum: float = 0):
        N = Xs.shape[0]
        ys = ys.reshape(-1, 1)
        Xs = Xs.view(Mat)
        ys = ys.view(Mat)
        with self.graph:
            loss = (Xs @ self.weight + self.bias - ys).norm(p=2) / N
        for epoch in range(epochs):
            self.graph.backward()
            self.step(lr, momentum)
            self.graph.clean_grad()
            self.graph.forward()
            yield epoch, loss


    def step(self, lr: float, m: float):
        w_step = self.weight.grad * lr
        b_step = self.bias.grad * lr
        if self.last_grads is not None:
            w_step = (1-m) * w_step + m * self.last_grads[0]
            b_step = (1-m) * b_step + m * self.last_grads[1]
        self.last_grads = [w_step, b_step]
        self.weight -= w_step
        self.bias -= b_step



if __name__ == '__main__':
    ...