from typing import List, Mapping, Union
import numpy as np
from pprint import pprint

from ..linalg import Mat, ones_like, ones
from .gradcal import (
    _backward_calculate, 
    _forward_calculate,
    _op_grad_mapping, 
    default_dtype
)

########################################
##### base operator implementation #####
########################################

def _boardcast(x, shape):
    if not comp_graph._graph_recording or x.shape == shape:
        return x
    if x.ndim != len(shape):
        if x.shape[0] == shape[0]:
            return x.reshape(-1, 1).repeat(shape[1], axis=1)
        elif x.shape[0] == shape[1]:
            return x.reshape(1, -1).repeat(shape[0], axis=0)
    else:
        if x.shape[0] != shape[0]:
            return x.repeat(shape[0], axis=0)
        elif x.shape[1] != shape[1]:
            return x.repeat(shape[1], axis=1)

def check_args_with_grad(args):
    for arg in args:
        if isinstance(arg, Mat) and arg.with_grad:
            return True
    return False


def _operator_wrapper(op):
    def _inner(*args, **kwargs):
        res = op(*args, **kwargs)
        # if op.__name__ == "__truediv__":
        #     print(args)
        if comp_graph._graph_recording:
            res.with_grad = check_args_with_grad(args)
            # create node
            g = comp_graph.get_default_graph()
            operands = [g.attach_node(data, is_data=True) for data in (*args, res)]
            operator = g.attach_node(op.__name__, is_data=False, kwargs=kwargs)
            # connect
            for node in operands[:-1]:
                node.link_to(operator)
            operator.link_to(operands[-1])

        return res
    _inner.__name__ = op.__name__
    return _inner

def _op_boardcast(op):
    def _inner(*args, **kwargs):
        args = list(args)
        if not isinstance(args[0], Mat) or args[0].ndim == 0:
            args[0] = Mat(args[0]) # .reshape(1,)
        if not isinstance(args[1], Mat) or args[1].ndim == 0:
            args[1] = Mat(args[1]) # .reshape(1,)
        if args[0].size == 1:
            args[0] = args[0].reshape(1, ).repeat(args[1].shape[0], axis=0)
            if args[1].ndim == 2:
                args[0] = args[0].reshape(-1, 1).repeat(args[1].shape[1], axis=1)
        elif args[1].size == 1:
            args[1] = args[1].reshape(1, ).repeat(args[0].shape[0], axis=0)
            if args[0].ndim == 2:
                args[1] = args[1].reshape(-1, 1).repeat(args[0].shape[1], axis=1)
        elif args[0].ndim < args[1].ndim:
            args[0] = _boardcast(args[0], args[1].shape)
        elif args[0].ndim > args[1].ndim:
            args[1] = _boardcast(args[1], args[0].shape)
        elif args[0].size < args[1].size:
            args[0] = _boardcast(args[0], args[1].shape)
        elif args[0].size > args[1].size:
            args[1] = _boardcast(args[1], args[0].shape)
        res = op(*args, **kwargs)
        return res
    _inner.__name__ = op.__name__
    return _inner


def _overload_sum(self, *args, **kwargs):
    """only accept axis and keepdims args"""
    axis = kwargs.get('axis', None)
    keepdims = kwargs.get('keepdims', False)
    if axis is not None:
        if axis == 0:
            res = (ones((1, self.shape[0])) @ self)
        elif axis == 1:
            res = (self @ ones((self.shape[1], 1)))
        else:
            raise NotImplementedError
        if not keepdims:
            res = res.reshape(-1)
    else:
        res = (ones((1, self.shape[0])) @ self) @ ones((self.shape[1], 1))
        if not keepdims:
            res = res[0, 0]
    return res

def _overload_T(self):
    value = self.transpose()
    if comp_graph._graph_recording:
        value.with_grad = True
        graph = comp_graph.get_default_graph()
        op_node = graph.attach_node('T', is_data=False)
        data_nodes = graph.attach_node(self), graph.attach_node(value)
        data_nodes[0].link_to(op_node)
        op_node.link_to(data_nodes[1])

    return value

def _overload_norm(self, p):
    p = Mat(p).reshape(1,)
    value = pow((pow(self, p)).sum(keepdims=True), (1/p))
    
    return value

def _overload_getitem(self, slices):
    value = super(Mat, self).__getitem__(slices)
    if value.size == 1:
        value = Mat(value) #.reshape(1,)
    if comp_graph._graph_recording:
        value.with_grad = True
        graph = comp_graph.get_default_graph()
        op_node = graph.attach_node('__getitem__', is_data=False, kwargs={'slices': slices})
        data_nodes = graph.attach_node(self), graph.attach_node(value)
        data_nodes[0].link_to(op_node)
        op_node.link_to(data_nodes[1])

    return value

def repeat(self, repeats, axis):
    res = super(Mat, self).repeat(repeats, axis=axis)
    if comp_graph._graph_recording:
        res.with_grad = True
        graph = comp_graph.get_default_graph()
        op_node = graph.attach_node('repeat', is_data=False, kwargs={'repeats': repeats, 'axis': axis})
        data_nodes = graph.attach_node(self), graph.attach_node(res)
        data_nodes[0].link_to(op_node)
        op_node.link_to(data_nodes[1])
    return res


######################################### 
##### extra operator implementation #####
######################################### 

def mean(self, axis=None, keepdims: bool=False):
    if axis is None:
        return self.sum().reshape(1,) / Mat(self.size).reshape(1,)
    else:
        return self.sum(axis=axis, keepdims=keepdims) / Mat(self.shape[axis]).reshape(1,)

def var(self, axis=None, keepdims: bool=False):
    if axis is None:
        return ((self - self.mean()) ** 2).sum().reshape(1,) / Mat(self.size - 1).reshape(1,)
    else:
        return ((self - self.mean(axis=axis, keepdims=True)) ** 2).sum(axis=axis, keepdims=keepdims) / Mat(self.shape[axis] - 1).reshape(1,)

def max(self, axis=None, keepdims: bool=False):
    if axis is not None:
        mask = np.argmax(self, axis=axis)
        slice_ = list(range(self.shape[1-axis]))
        if axis == 0:
            res = self[mask, slice_]
        elif axis == 1:
            res = self[slice_, mask]    
        if keepdims:
            res = res.reshape(*self.shape[:axis], 1, *self.shape[axis+1:])
    else:
        mask = np.argmax(self)
        row = mask // self.shape[1]
        col = mask % self.shape[1]
        res = self[row, col]
        if keepdims:
            res = res.reshape(1, 1)
    return res

def min(self, axis=None, keepdims: bool=False):
    if axis is not None:
        mask = np.argmin(self, axis=axis)
        slice_ = list(range(self.shape[1-axis]))
        if axis == 0:
            res = self[mask, slice_]
        elif axis == 1:
            res = self[slice_, mask]    
        if keepdims:
            res = res.reshape(*self.shape[:axis], 1, *self.shape[axis+1:])
    else:
        mask = np.argmin(self)
        row = mask // self.shape[1]
        col = mask % self.shape[1]
        res = self[row, col]
        if keepdims:
            res = res.reshape(1, 1)
    return res

def relu(self):
    return np.maximum(self, 0)

def softmax(self, axis=1):
    exp_ = (self - self.max(axis=axis, keepdims=True)).exp()
    res = (exp_ / exp_.sum(axis=axis, keepdims=True))

    return res

def sigmoid(self):
    return ones_like(self) / (ones_like(self) + (-self).exp())

def dropout(self, rate=0.1):
    one_mat = ones_like(self)
    for row in range(one_mat.shape[0]):
        one_mat[row] = np.random.binomial(1, 1-rate, size=one_mat.shape[1])
    return self * one_mat

def cross_entropy(self, y):
    return (-y * self.log()).sum()

#############################
##### operator overload #####
#############################

Mat.__add__ = _op_boardcast(_operator_wrapper(Mat.__add__))
Mat.__sub__ = _op_boardcast(_operator_wrapper(Mat.__sub__))
Mat.__mul__ = _op_boardcast(_operator_wrapper(Mat.__mul__)) 
Mat.__truediv__ = _op_boardcast(_operator_wrapper(Mat.__truediv__))
Mat.__matmul__ = _operator_wrapper(Mat.__matmul__)
Mat.__getitem__ = _overload_getitem
Mat.repeat = repeat
Mat.T = property(_overload_T)

Mat.__pow__ = _operator_wrapper(Mat.__pow__)
Mat.__abs__ = _operator_wrapper(Mat.__abs__)
Mat.exp = _operator_wrapper(np.exp)
Mat.log = _operator_wrapper(np.log)
Mat.__neg__ = _operator_wrapper(Mat.__neg__)

Mat.sum = _overload_sum
Mat.norm = _overload_norm
Mat.reshape = _operator_wrapper(Mat.reshape)
Mat.mean = mean
Mat.var = var
Mat.max = max
Mat.min = min


Mat.relu = _operator_wrapper(relu)

Mat.softmax = softmax
Mat.sigmoid = sigmoid
Mat.dropout = dropout
Mat.cross_entropy = cross_entropy

###############################
##### computational graph #####
###############################

def no_record(func):
    def inner(*args, **kwargs):
        gr = comp_graph._graph_recording
        comp_graph._graph_recording = False
        ret = func(*args, **kwargs)
        comp_graph._graph_recording = gr
        return ret 
    return inner

class comp_graph(object):
    DEFAULT_GRAPH = None
    graphs = []
    ctx_stack = []
    _graph_recording = False
    def __init__(self):
        self._nodes: Mapping[str, node] = {}
        self._need_sort = True
        self._sorted_nodes: List[_operator_node] = []
        comp_graph.graphs.append(self)

    def set_default_graph(self):
        comp_graph.DEFAULT_GRAPH = self

    @staticmethod
    def get_default_graph():
        if comp_graph.DEFAULT_GRAPH is None:
            comp_graph.DEFAULT_GRAPH = comp_graph()
        return comp_graph.DEFAULT_GRAPH

    @property
    def nodes(self):
        return self._nodes

    def __enter__(self):
        comp_graph.ctx_stack.append(comp_graph.DEFAULT_GRAPH)
        comp_graph.DEFAULT_GRAPH = self
        comp_graph._graph_recording = True

    def __exit__(self, *args, **kwargs):
        comp_graph.DEFAULT_GRAPH = comp_graph.ctx_stack.pop()
        comp_graph._graph_recording = False

    def attach_node(self, op_or_value: Union[Mat, str], is_data: bool = True, kwargs={}):
        """
        TODO: delete usage of `_node`
        """
        if is_data:
            if not isinstance(op_or_value, Mat):
                op_or_value = Mat(op_or_value, with_grad=True)
            try:
                node = op_or_value._node
            except AttributeError:
                node = _data_node(op_or_value, self)
                op_or_value._node = node
        else:
            op = op_or_value
            if _op_grad_mapping.get(op, None) is None:
                raise RuntimeWarning("Operator {} may not support now.".format(op_or_value))
            node = _operator_node(op, self, kwargs=kwargs)

        self._need_sort = True

        return node

    def clear_grad(self):
        """set grad to zero"""
        for nname in self._nodes:
            _node = self._nodes[nname]
            if isinstance(_node, _data_node):
                if _node._data.with_grad:
                    try:
                        _node._data.grad *= 0
                    except Exception:
                        pass
    
    def clear(self):
        """clear every node in the graph"""
        for nname in self._nodes:
            _node = self._nodes[nname]
            if isinstance(_node, _data_node):
                del _node._data._node
            del _node
        self._nodes = {}

    def topo_sort(self):
        if not self._need_sort: 
            return
        self._need_sort = False
        topo_nodes = {}
        for key in self.nodes:
            nn = self.nodes[key]
            topo_nodes.setdefault(nn._id, [nn, len(nn._ins)])
        self._sorted_nodes.clear()
        while len(topo_nodes) > 0:
            for key in topo_nodes:
                if topo_nodes[key][1] == 0:
                    del_node = topo_nodes[key][0]
                    # update out node
                    for out_node in del_node._outs:
                        topo_nodes[out_node._id][1] -= 1
                    del topo_nodes[key]
                    break
            if isinstance(del_node, _operator_node):
                self._sorted_nodes.append(del_node)
    
    @no_record
    def backward(self):
        self.topo_sort()
        sorted_nodes = tuple(reversed(self._sorted_nodes))
        output = sorted_nodes[0]._outs[0]._data
        output.grad = np.ones(output.shape, dtype=default_dtype)
        for op_node in sorted_nodes:
            inputs = [node._data for node in op_node._ins]
            output = op_node._outs[0]._data
            for i, input in enumerate(inputs):
                if input.with_grad:
                    if input.grad is None:
                        input.grad = np.zeros(input.shape, dtype=default_dtype)
                    _backward_calculate(inputs, output, op_node, i)

    @no_record 
    def forward(self):
        self.topo_sort()
        for op_node in self._sorted_nodes:
            inputs = [node._data for node in op_node._ins]
            output = op_node._outs[0]._data
            _forward_calculate(inputs, output, op_node)



class node(object):

    def __init__(self, identifier: str, context: comp_graph):
        self._id = identifier
        self._ins: List[node] = []
        self._outs: List[node] = []
        if context is None:
            context = comp_graph.get_default_graph()
        self._ctx = context
        self._ctx._nodes.setdefault(identifier, self)
        

    def link_to(self, node):
        self._outs.append(node)
        node._ins.append(self)

class _data_node(node):
    NODE_COUNTS = 0
    def __init__(self, data, context: comp_graph = None, producer = None):
        super().__init__("datanode_{}".format(_data_node.NODE_COUNTS), context)
        self._data = data
        self.producer = producer
        _data_node.NODE_COUNTS += 1

    def link_to(self, node):
        if not isinstance(node, _operator_node):
            raise RuntimeError("Data node should link to operator node, no node will be added.")
        else:
            super().link_to(node)

    def __repr__(self):
        return "<data_node: Size {}, produced by: {}>".format(self._data.shape, self.producer)


class _operator_node(node):
    NODE_COUNTS = 0
    def __init__(self, op: str, context: comp_graph = None, kwargs={}):
        super().__init__("opnode_{}".format(_operator_node.NODE_COUNTS), context)
        self._op_name = op
        self.kwargs = kwargs
        _operator_node.NODE_COUNTS += 1

    @property
    def op_type(self):
        return self._op_name

    def link_to(self, node):
        if not isinstance(node, _data_node):
            raise RuntimeError("Operator node should link to data node, no node will be added.")
        elif len(self._outs) > 0:
            raise RuntimeError('Only one data node conld be the output of current operator node.')
        else:
            super().link_to(node)
            node.producer = self

    def __repr__(self):
        return "<op_node: {}, kwargs: {}>".format(self._op_name, self.kwargs)
