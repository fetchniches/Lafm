# LAFM

**L**ow-performance **A**utograd **F**ramework for **M**achine learning algorithm. **Lafm** uses `numpy` as the backend and all matrix calculations are done by it.

## User Guide

Here is a test code for autograd below:

```python
# OLD VERSION, WON'T WORK
import lafm as lm

# creating array with gradient
a = lm.array([[1, 2.4, 3], [2.5, 3, 1], [2, 2, 1.6]], with_grad=True)
b = lm.array([[2.3, 1.1, 3], [3, 3, 2.]], with_grad=True)
c = lm.array([[2, 3, 1], [4, 1, 2.], [1, 1, 1]], with_grad=True)
# initialize default graph and starting recording
graph = lm.comp_graph.get_default_graph()
# computation process
with graph:
    d = (a - c) @ b.T
    d = d / lm.array([2])
    d = d * lm.array([1.5])
    result = d.sum()
# backward for gradient
result.backward()
# print gradient
print(a.grad)
print(b.grad)
print(c.grad)
```

Here is a Linear Regression example code:

```python
import lafm
import numpy as np
import matplotlib.pyplot as plt


gt_weight = np.array([-2.2]).reshape(-1, 1)
gt_bias = np.array([3.2])
linear_gen = lafm.faker.data_gen.Linear(gt_weight, gt_bias)
noiser = lafm.faker.noiser.GaussianNoise(0, 1)
linear_gen.attach_noiser(noiser)
Xs, ys = linear_gen.gen(-5, 5, 24)
LR = lafm.linreg.LinearRegressor(dims=1)
disp_X = np.linspace(-5, 5, 2).reshape(-1, 1)
plt.ion()
plt.ylabel('y')
plt.xlabel('X')
for epoch, loss in LR.train(Xs, ys, 20, lr=.05):
    disp_y = disp_X @ LR.weight + LR.bias
    plt.plot(disp_X, disp_y)
    plt.scatter(Xs, ys, c='orange')
    plt.xlim(-5, 5)
    plt.ylim(-15, 15)
    plt.title('Epoch {}  Loss: {:.2f}'.format(epoch, loss))
    plt.pause(.5)
    plt.clf()
```

## Support Operation

- [x] addition

- [x] subtraction

- [x] multiply (boardcast)

- [x] division (boardcast)

- [x] multiply (boardcast)

- [x] division (boardcast)

- [x] matrix multiply

- [x] sum (reduce all dimension)

- [x] elementary function

- [ ] in-place operation

- [x] power operation (operator overload, only support gradient for base)

- [x] absolute value

- [x] p-norm (combination of the above two)

## Optimization

- [x] topological sorting for computational graph

- [x] computational boardcast

- [ ] topological sorting in forward step

- [ ] architecture reconstruction

- [ ] graph optimization

## Basic Features

- [x] gradient clear

- [ ] dynamic graph (clear graph in backward step?)

- [x] static graph

## Next Steps

- [ ] elementary function support

- [ ] complete graph part

- [ ] kernel methods

## Bugs

- [x] high dimention matrix multiplication would occupy too much memory. (due to a huge inner matrix `tmp_grad`)

## Implementation Details

coming soon...
