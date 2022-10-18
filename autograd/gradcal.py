import numpy as np
from functools import reduce
from operator import mul
from enum import Enum

from ..linalg import Mat


class _supported_operator(Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MAT_MUL = 4
    SUM = 5
    T = 6

_operator_mapping = {
    '__add__': _supported_operator.ADD,
    '__sub__': _supported_operator.SUB,
    '__mul__': _supported_operator.MUL,
    '__div__': _supported_operator.DIV,
    '__matmul__': _supported_operator.MAT_MUL,
    'sum': _supported_operator.SUM,
    'T': _supported_operator.T,
}

# def _add_grad(input_X: Mat, input_Y: Mat, result_Z: Mat): ...
# def _sub_grad(input_X: Mat, input_Y: Mat, result_Z: Mat): ...
# def _mul_grad(input_X: Mat, input_Y: Mat, result_Z: Mat): ...
# def _div_grad(input_X: Mat, input_Y: Mat, result_Z: Mat): ...
# def _matmul_grad(input_X: Mat, input_Y: Mat, result_Z: Mat): ...
# def _transpose_grad(input_X: Mat, result_Z: Mat): ...
# def _sum_grad(input_X: Mat, result_Z: Mat): ...

def _add_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int):
    for index, inp in enumerate((input_X, input_Y)):
        if inp.with_grad and index == cal_grad:
            inp._grad += result_Z._grad

def _sub_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int):
    for index, (inp, sign) in enumerate(zip((input_X, input_Y), (1, -1))):
        if inp.with_grad and index == cal_grad:
            inp._grad += sign * result_Z._grad

def _mul_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int):
    # NOTE: only element-wise multiply is supported.
    # TODO: element-wise matmul 
    inputs = (input_X, input_Y)
    for index, inp in enumerate(inputs):
        if inp.size == 1 and index == cal_grad:
            inp._grad += inputs[index-1].reshape(-1) @ result_Z._grad.reshape(-1).T
        elif index == cal_grad:
            shape = (*result_Z.shape, *inp.shape)
            gradient = np.zeros(shape, dtype=inp.dtype)
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    gradient[i, j, i, j] = inputs[index-1]
            inp._grad += (result_Z._grad.reshape(-1) * gradient).reshape(inp.shape)

def _div_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int):
    # NOTE: only element-wise division is supported.
    # TODO: element-wise division
    inputs = (input_X, input_Y)
    for index, inp in enumerate(inputs):
        # one element
        if inp.size == 1 and index == cal_grad:
            inp._grad += - np.sum(inputs[index-1] / inp ** 2)
        elif index == cal_grad:
            inp._grad += np.ones(inp.shape, dtype=inp.dtype) / inputs[index-1]

def _matmul_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int):
    if cal_grad == 0 and input_X.with_grad:
        shape = (*result_Z.shape, *input_X.shape)
        grad = np.zeros(shape, dtype=input_X.dtype)
        # 
        for i in range(result_Z.shape[0]):
            for j in range(result_Z.shape[1]):
                grad[i, j, i] += input_Y[:, j] 
        tmp_grad = grad.reshape(reduce(mul, result_Z.shape), reduce(mul, input_X.shape))
        input_X._grad = (result_Z._grad.reshape(-1) @ tmp_grad).reshape(input_X.shape)
    if cal_grad == 1 and input_Y.with_grad:
        shape = (*result_Z.shape, *input_Y.shape)
        grad = np.zeros(shape, dtype=input_Y.dtype)
        # 
        for i in range(result_Z.shape[0]):
            for j in range(result_Z.shape[1]):
                grad[i, j, :, j] += input_X[i, :] 
        tmp_grad = grad.reshape(reduce(mul, result_Z.shape), reduce(mul, input_Y.shape))
        input_Y._grad = (result_Z._grad.reshape(-1) @ tmp_grad).reshape(input_Y.shape)

def _transpose_grad(input_X: Mat, result_Z: Mat, cal_grad: int):
    input_X._grad = result_Z._grad.transpose()

def _sum_grad(input_X: Mat, result_Z: Mat, cal_grad: int):
    # NOTE: all dim reduceds
    input_X._grad = np.ones_like(input_X) * result_Z._grad

_op_grad_mapping = {
    _supported_operator.ADD: _add_grad,
    _supported_operator.SUB: _sub_grad,
    _supported_operator.MUL: _mul_grad,
    _supported_operator.DIV: _div_grad,
    _supported_operator.MAT_MUL: _matmul_grad,
    _supported_operator.SUM: _sum_grad,
    _supported_operator.T: _transpose_grad
}

def _calculate_gradient(inputs, output, op_type: _supported_operator, cal_grad: int):
    func = _op_grad_mapping[op_type]
    func(*inputs, result_Z=output, cal_grad=cal_grad)
