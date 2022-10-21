# LAFM

**L**ow-performance **A**utograd **F**ramework for **M**achine learning algorithm. **Lafm** uses `numpy` as the backend and all matrix calculations are done by it.

## User Guide

Here is a test code for autograd below:

```python
import lafm as lm

# creating array with gradient
a = lm.array([[1, 2.4, 3], [2.5, 3, 1], [2, 2, 1.6]], with_grad=True)
b = lm.array([[2.3, 1.1, 3], [3, 3, 2.]], with_grad=True)
c = lm.array([[2, 3, 1], [4, 1, 2.], [1, 1, 1]], with_grad=True)
# initialize default graph and starting recording
graph = lm.comp_graph.get_default_graph()
# computation process
d = (a - c) @ b.T
d = d / lm.array([2])
d = d * lm.array([1.5])
result = d.sum()
# backward for gradient
result.backward()
# print gradient
print(a._grad)
print(b._grad)
print(c._grad)
```

Here is a Linear Regression example code:

```python
import lafm
import numpy as np
import matplotlib.pyplot as plt

# generate fake data
true_weight = np.array([2.2]).reshape(-1, 1)
true_bias = np.array([3.2])
linear_gen = lafm.faker.data_gen.Linear(true_weight, true_bias)
Xs, ys = linear_gen.gen(-5, 5, 24)
# initialize linear regression model
LR = lafm.linreg.LinearRegressor(dims=1)
disp_X = np.linspace(-5, 5, 2).reshape(-1, 1)
# training step
for epoch, loss in LR.train(Xs, ys, 10, lr=.01):
    disp_y = disp_X @ LR.weight + LR.bias
    # plotting result
    plt.plot(disp_X, disp_y)
    plt.scatter(Xs, ys, c='orange')
    plt.ylabel('y')
    plt.xlabel('X')
    plt.xlim(-5, 5)
    plt.ylim(-10, 10)
    plt.title('Epoch {} Loss: {:.2f}'.format(epoch, loss))
    plt.show()
```

## Support Operation

- [x] addition

- [x] subtraction

- [x] multiply (element-wise)

- [x] division (element-wise)

- [ ] multiply (boardcast)

- [ ] division (boardcast)

- [x] matrix multiply

- [x] sum (reduce all dimension)

- [ ] primary function

- [ ] in-place operation

- [x] power operation (operator overload, only support gradient for base)

- [x] absolute value

- [x] p-norm (combination of the above two)

## Optimization

- [ ] topological sorting for computational graph

- [ ] cleaning non-leaf node's gradient

- [ ] computational boardcast

- [ ] topological sorting in forward step

## Basic Features

- [ ] gradient clear

- [ ] dynamic graph (clear graph in backward step?)

- [ ] static graph

## Bugs

## Implementation Details

coming soon...
