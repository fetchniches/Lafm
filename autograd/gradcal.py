from typing import Sequence
import numpy as np
from functools import reduce
from operator import mul

from ..linalg import Mat


default_dtype = np.float32


################################
###### backward calculate ######
################################

def _gen_elem_grad(elem_wise_grad):
    def _elem_grad(input_X: Mat, result_Z: Mat, cal_grad: int, **kwargs):
        input_X.grad += elem_wise_grad(input_X) * result_Z.grad

    return _elem_grad

def _add_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    inp = [input_X, input_Y]
    inp[cal_grad].grad += result_Z.grad


def _sub_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    inp = [input_X, input_Y]
    inp[cal_grad].grad += result_Z.grad * (-1) ** cal_grad


def _mul_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    inp = [input_X, input_Y]
    inp[cal_grad].grad += result_Z.grad * inp[1 - cal_grad]


def _div_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    if 0 == cal_grad:
        input_X.grad += result_Z.grad / input_Y
    elif 1 == cal_grad:
        input_Y.grad += - result_Z.grad * input_X / (input_Y ** 2)


def _matmul_grad(input_X, input_Y, result_Z, cal_grad: int, **kwargs):
    if cal_grad == 0:
        input_X.grad += result_Z.grad @ input_Y.T
    if cal_grad == 1:
        input_Y.grad += input_X.T @ result_Z.grad

def _transpose_grad(input_X: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    if cal_grad == 0:
        input_X.grad += result_Z.grad.T

def _getitem_grad(input_X: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    if cal_grad == 0:
        slices = kwargs['slices']
        input_X.grad[slices] += result_Z.grad
    else:
        raise NotImplementedError

def _repeat_grad(input_X: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    if cal_grad == 0:
        axis = kwargs['axis']
        input_X.grad += np.sum(result_Z.grad, axis=axis).reshape(input_X.grad.shape)

def _pow_grad(input_X: Mat, input_Y: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    if cal_grad == 0:
        input_X.grad += result_Z.grad * input_Y * input_X ** (input_Y - Mat(1).reshape(1,))

def _reshape_grad(input_X: Mat, result_Z: Mat, cal_grad: int, **kwargs):
    if cal_grad == 0:
        input_X.grad += result_Z.grad.reshape(input_X.shape)


_abs_grad = _gen_elem_grad(lambda x: np.sign(x))
_relu_grad = _gen_elem_grad(lambda x: np.where(x > 0, 1, 0))
_exp_grad = _gen_elem_grad(lambda x: np.exp(x))
_log_grad = _gen_elem_grad(lambda x: 1 / x)
_neg_grad = _gen_elem_grad(lambda x: -1)

_op_grad_mapping = {
    '__add__': _add_grad,
    '__sub__': _sub_grad,
    '__mul__': _mul_grad,
    '__truediv__': _div_grad,
    '__matmul__': _matmul_grad,
    'T': _transpose_grad,
    '__getitem__': _getitem_grad,
    'repeat': _repeat_grad,
    'reshape': _reshape_grad,
    
    '__pow__': _pow_grad,
    '__abs__': _abs_grad,
    'relu': _relu_grad,
    'exp': _exp_grad,
    'log': _log_grad,
    '__neg__': _neg_grad,
    
}

def _backward_calculate(inputs, output, op_node, cal_grad: int):
    func = _op_grad_mapping[op_node.op_type]
    if op_node.op_type == 'reshape':
        func(inputs[0], output, cal_grad=cal_grad, **op_node.kwargs)
    else:
        func(*inputs, result_Z=output, cal_grad=cal_grad, **op_node.kwargs)



def _forward_calculate(inputs: Sequence[Mat], output: Mat, op_node):
    func = getattr(Mat, op_node.op_type)
    output *= 0
    try:
        output += func(*inputs, **op_node.kwargs)
    except TypeError:
        output += getattr(inputs[0], op_node.op_type)

