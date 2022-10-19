from typing import Sequence, Union
import numpy as np

from ..linalg import ones_like, Mat
from .gradcal import _calculate_gradient, _operator_mapping, _supported_operator

##### operation override #####
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


class comp_graph(object):
    DEFAULT_GRAPH = None
    graphs = []
    _graph_recording = False
    def __init__(self):
        self._nodes = {}
        comp_graph.graphs.append(self)

    def set_default_graph(self):
        comp_graph.DEFAULT_GRAPH = self
        comp_graph._graph_recording = True

    @staticmethod
    def clean():
        comp_graph.DEFAULT_GRAPH = None
        comp_graph._graph_recording = False

    @staticmethod
    def get_default_graph():
        if comp_graph.DEFAULT_GRAPH is None:
            comp_graph.DEFAULT_GRAPH = comp_graph()
        comp_graph._graph_recording = True
        return comp_graph.DEFAULT_GRAPH

    @staticmethod
    def no_record(func):
        
        def inner(*args, **kwargs):
            gr = comp_graph._graph_recording
            comp_graph._graph_recording = False
            ret = func(*args, **kwargs)
            comp_graph._graph_recording = gr
            return ret 
        
        return inner

    @property
    def nodes(self):
        return self._nodes

    def __enter__(self):
        self.last_graph = comp_graph.DEFAULT_GRAPH
        comp_graph.DEFAULT_GRAPH = self
        comp_graph._graph_recording = True

    def __exit__(self, *args, **kwargs):
        comp_graph.DEFAULT_GRAPH = self.last_graph
        self.last_graph = None
        if comp_graph.DEFAULT_GRAPH is None:
            comp_graph._graph_recording = False

    def attach_node(self, op_or_value: Union[Mat, str], is_data: bool = True):
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

        self._nodes.setdefault(node._id, node)

        return node

class node(object):

    def __init__(self, identifier: str, context: comp_graph):
        self._id = identifier
        self._ins: Sequence[node] = []
        self._outs: Sequence[node] = []
        if context is None:
            context = comp_graph.get_default_graph()
        self._ctx = context
        self._ctx._nodes.setdefault(identifier, self)

    def link_to(self, node):
        self._outs.append(node)
        node._ins.append(self)

    @comp_graph.no_record
    def backward(self):
        raise NotImplemented

class _data_node(node):
    NODE_COUNTS = 0
    def __init__(self, data, context: comp_graph = None):
        super().__init__("datanode_{}".format(_data_node.NODE_COUNTS), context)
        self._data = data
        _data_node.NODE_COUNTS += 1

    def link_to(self, node):
        if not isinstance(node, _operator_node):
            raise RuntimeError("Data node should link to operator node, no node will be added.")
        # elif len(self._ins) > 0:
        #     raise RuntimeError('Only one operator conld be link to current data node.')
        else:
            super().link_to(node)

    @comp_graph.no_record
    def backward(self):
        # init grad val
        if self._data.with_grad and self._data._grad is None:
            self._data._grad = np.zeros_like(self._data)
        # leaf node
        if len(self._outs) == 0:
            if self._data.size != 1:
                raise TypeError("Only constant could be used to start calculating gradient.")
            self._data._grad = ones_like(self._data)
        if len(self._ins) == 0 and not self._data.with_grad:
            pass
        # non-leaf node
        else:
            # iterate each operator type
            for op in self._outs:
                result = op._outs[0]
                inputs = [node._data for node in op._ins]
                for idx, node in enumerate(op._ins):
                    if node._id == self._id:
                        cal_grad = idx
                op_type = op.op_type
                _calculate_gradient(inputs, result._data, op_type, cal_grad)
        # depth first order
        for op_node in self._ins:
            op_node.backward()

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

    @comp_graph.no_record
    def backward(self):
        for data_node in self._ins:
            data_node.backward()

    