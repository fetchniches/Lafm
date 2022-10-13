from .graph import comp_graph
from .linalg import Mat

def _calculate_gradient(inputs, output, op_type):
    ...


def _binary_operator_wrapper(op):
    def _inner(self, other):
        res = op(self, other)
        if comp_graph._graph_recording:
            res.with_grad = True
            # create node
            g = comp_graph.get_default_graph()
            operands = [g.attach_node(data, is_data=True) for data in (self, other, res)]
            operator = g.attach_node(op.__name__, is_data=False)
            # connect
            for node in operands[:2]:
                node.link_to(operator)
            operator.link_to(operands[-1])

        return res
    return _inner

# add gradient
Mat.__add__ = _binary_operator_wrapper(Mat.__add__)
Mat.__sub__ = _binary_operator_wrapper(Mat.__sub__)
Mat.__mul__ = _binary_operator_wrapper(Mat.__mul__) 
Mat.__truediv__ = _binary_operator_wrapper(Mat.__truediv__)
Mat.__matmul__ = _binary_operator_wrapper(Mat.__matmul__)


