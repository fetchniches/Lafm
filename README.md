# LAFM

[English Version](./README-en.md)

**L**ow-performance **A**utograd **F**ramework for **M**achine learning algorithm. **Lafm**使用`numpy`作为后端，所有矩阵计算都由它完成。

## 代码示例

如下是一个简单的自动微分代码示例

```python
# OLD VERSION, WON'T WORK
import lafm

graph = lafm.comp_graph.get_default_graph()
 
with graph:
    a = lafm.randn(3, 4, with_grad=True)
    c = lafm.randn(4, with_grad=True)
    d = a / c
    f = d.var()
    e = f.reshape(1, -1).sum()
t1 = time.time()
graph.backward()
t2 = time.time()
graph.clear()
print(a.grad)
```

## 实现的算子

- [x] addition (boardcast)

- [x] subtraction (boardcast)

- [x] multiply (boardcast)

- [x] division (boardcast)

- [x] multiply (boardcast)

- [x] division (boardcast)

- [x] matrix multiply

- [x] sum

- [x] elementary function

- [ ] in-place operation

- [x] power operation (operator overload, only support gradient for base)

- [x] absolute value

- [x] p-norm (combination of the above two)

## 优化

- [x] topological sorting for computational graph

- [x] computational boardcast

- [ ] topological sorting in forward step

- [ ] architecture reconstruction

- [ ] graph optimization

## 基础功能

- [x] gradient clear

- [ ] dynamic graph (clear graph in backward step?)

- [x] static graph

## 下一步?

- [x] elementary function support

- [ ] complete graph part

- [ ] kernel methods

## Bugs

- [x] high dimention matrix multiplication would occupy too much memory. (due to a huge inner matrix `tmp_grad`)

## 实现细节

[实现细节](./details/details.md)
