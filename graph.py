from typing import Union

from ._operators import _supported_operator, _operator_mapping
from .linalg import Mat, array, ones_like
from ._autograd import _calculate_gradient



class comp_graph(object):
    DEFAULT_GRAPH = None
    _graph_recording = False
    def __init__(self):
        if comp_graph.DEFAULT_GRAPH is None:
            comp_graph.DEFAULT_GRAPH = self
        self._nodes = {}

    @staticmethod
    def get_default_graph():
        if comp_graph.DEFAULT_GRAPH is None:
            comp_graph.DEFAULT_GRAPH = comp_graph()
        return comp_graph.DEFAULT_GRAPH

    @property
    def nodes(self):
        return self._nodes

    def __enter__(self):

        comp_graph._graph_recording = True

    def __exit__(self, *args, **kwargs):
        
        comp_graph._graph_recording = False

    def attach_node(self, op_or_value: Union[Mat, str], is_data: bool = True):
        if is_data:
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
        self._ins = []
        self._outs = []
        if context is None:
            context = comp_graph.get_default_graph()
        self._ctx = context
        self._ctx._nodes.setdefault(identifier, self)

    def link_to(self, node):
        self._outs.append(node)
        node._ins.append(self)

    def backward(self):
        raise NotImplementedError

class _data_node(node):
    NODE_COUNTS = 0
    def __init__(self, data, context: comp_graph = None):
        super().__init__("datanode_{}".format(_data_node.NODE_COUNTS), context)
        self._data = data
        _data_node.NODE_COUNTS += 1

    def link_to(self, node):
        if not isinstance(node, _operator_node):
            raise RuntimeError("Data node should link to operator node, no node will be added.")
        elif len(self._ins) > 0:
            raise RuntimeError('Only one operator conld be link to current data node.')
        else:
            super().link_to(node)

    def backward(self):
        # leaf node
        if len(self._outs) == 0:
            self._data._grad = ones_like(self._data)
        # non-leaf node
        else:
            # iterate each operator type
            for op in self._outs:
                result = op._outs[0]
                inputs = op._ins
                op_type = op.op_type
                _calculate_gradient(inputs, result, op_type)

class _operator_node(node):
    NODE_COUNTS = 0
    def __init__(self, op: _supported_operator, context: comp_graph = None):
        super().__init__("opnode_{}".format(_operator_node.NODE_COUNTS), context)
        self._op_type = op
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

# m x n @ n x 1 = m x 1, sum(m x 1) = 1
# g0: 1, g1: ones(m x 1), g2: m x n () n x 1 ()