from typing import List, Mapping, Sequence, Union
import numpy as np

from ..linalg import ones_like, Mat
from .gradcal import (
    _backward_calculate, 
    _forward_calculate,
    _operator_mapping, 
    _supported_operator, 
    default_dtype
)

#############################
##### operator overload #####
#############################

def _operator_wrapper(op):
    def _inner(*args):
        res = op(*args)
        if comp_graph._graph_recording:
            res.with_grad = True
            # create node
            g = comp_graph.get_default_graph()
            operands = [g.attach_node(data, is_data=True) for data in (*args, res)]
            operator = g.attach_node(op.__name__, is_data=False)
            # connect
            for node in operands[:-1]:
                node.link_to(operator)
            operator.link_to(operands[-1])

        return res
    return _inner

# TODO: dim args
def _overload_sum(self, *args, **kwargs):
    res = super(Mat, self).sum()
    if comp_graph._graph_recording:
        res.with_grad = True
        graph = comp_graph.get_default_graph()
        op_node = graph.attach_node('sum', is_data=False)
        data_nodes = graph.attach_node(self), graph.attach_node(res)
        data_nodes[0].link_to(op_node)
        op_node.link_to(data_nodes[1])

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
    value = (self ** p).sum() ** (1/p)
    
    return value


Mat.__add__ = _operator_wrapper(Mat.__add__)
Mat.__sub__ = _operator_wrapper(Mat.__sub__)
Mat.__mul__ = _operator_wrapper(Mat.__mul__) 
Mat.__truediv__ = _operator_wrapper(Mat.__truediv__)
Mat.__matmul__ = _operator_wrapper(Mat.__matmul__)
Mat.sum = _overload_sum
Mat.T = property(_overload_T)
Mat.__pow__ = _operator_wrapper(Mat.__pow__)
Mat.__abs__ = _operator_wrapper(Mat.__abs__)
Mat.norm = _overload_norm

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
    def clean_state():
        comp_graph.DEFAULT_GRAPH = None
        comp_graph._graph_recording = False

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

    def attach_node(self, op_or_value: Union[Mat, str], is_data: bool = True):
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
            op = _operator_mapping.get(op_or_value, None)
            if op is None:
                raise RuntimeWarning("Operator {} does not support now.".format(op_or_value))
            node = _operator_node(op, self)

        self._need_sort = True
        # self._nodes.setdefault(node._id, node) attached in _node `__init__`

        return node

    def clean_grad(self):
        for nname in self._nodes:
            _node = self._nodes[nname]
            if isinstance(_node, _data_node):
                if _node._data.with_grad:
                    _node._data.grad *= 0
    
    def clear(self):
        for nname in self._nodes:
            _node = self._nodes[nname]
            _node._ins.clear()
            _node._outs.clear()
            if isinstance(_node, _data_node) and _node._data.with_grad:
                _node._data.grad *= 0 
        self._nodes = {}

    def topo_sort(self):
        if not self._need_sort: 
            return
        topo_nodes = {}
        for key in self._nodes:
            nn = self._nodes[key]
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
            op_type = op_node._op_type
            for i, input in enumerate(inputs):
                if input.with_grad:
                    if input.grad is None:
                        input.grad = np.zeros(input.shape, dtype=default_dtype)
                    _backward_calculate(inputs, output, op_type, i)

    @no_record 
    def forward(self):
        self.topo_sort()
        for op_node in self._sorted_nodes:
            inputs = [node._data for node in op_node._ins]
            output = op_node._outs[0]._data
            op_type = op_node._op_type
            _forward_calculate(inputs, output, op_type)


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

    @no_record
    def backward(self):
        raise NotImplemented

    @no_record
    def forward(self):
        raise NotImplemented

class _data_node(node):
    NODE_COUNTS = 0
    def __init__(self, data, with_grad: bool, context: comp_graph = None):
        super().__init__("datanode_{}".format(_data_node.NODE_COUNTS), context)
        self._data = data
        _data_node.NODE_COUNTS += 1

    def link_to(self, node):
        if not isinstance(node, _operator_node):
            raise RuntimeError("Data node should link to operator node, no node will be added.")
        else:
            super().link_to(node)

    def __repr__(self) -> str:
        return "<data_node: Size {}>".format(self._data.shape)


class _operator_node(node):
    NODE_COUNTS = 0
    def __init__(self, op: _supported_operator, context: comp_graph = None, **kwargs):
        super().__init__("opnode_{}".format(_operator_node.NODE_COUNTS), context)
        self._op_type = op
        self.kwargs = kwargs
        _operator_node.NODE_COUNTS += 1

    @property
    def op_type(self):
        return self._op_type

    def link_to(self, node):
        if not isinstance(node, _data_node):
            raise RuntimeError("Operator node should link to data node, no node will be added.")
        elif len(self._outs) > 0:
            raise RuntimeError('Only one data node conld be the output of current operator node.')
        else:
            super().link_to(node)

    def __repr__(self) -> str:
        return "<op_node: {}>".format(self._op_type)
