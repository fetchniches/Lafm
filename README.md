# LAFM

**L**ow-performance **A**utograd **F**ramework for **M**achine learning algorithm. **Lafm** uses `numpy` as the backend and all matrix calculations are done by it.

## User Guide

Here is a test code below.

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

- [ ] topology sort for computational graph

- [ ] cleaning non-leaf node's gradient

- [ ] computational boardcast

## Basic Features

- [ ] gradient clear

## Bugs

- [ ] `__add__` function error when compute gradient in linear regression.

## Implementation Details

coming soon...
