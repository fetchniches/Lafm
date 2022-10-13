from enum import Enum


class _supported_operator(Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MAT_MUL = 4
    SUM = 5

_operator_mapping = {
    '__add__': _supported_operator.ADD,
    '__sub__': _supported_operator.SUB,
    '__mul__': _supported_operator.MUL,
    '__div__': _supported_operator.DIV,
    '__matmul__': _supported_operator.MAT_MUL
}