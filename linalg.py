import numpy as np


class Mat(np.ndarray): 

    def __new__(cls, inputs_array, with_grad: bool = False):
        if not isinstance(inputs_array, np.ndarray):
            inputs_array = np.asarray(inputs_array)
        obj = inputs_array.view(cls)
        obj.with_grad = with_grad
        obj.grad = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return 
        # check if it is a newly created        
        # if self.base is not obj or isinstance(obj, np.ndarray):
        if isinstance(obj, np.ndarray):
            self.with_grad = False
        else:
            self.with_grad = obj.with_grad

        self.grad = None

    def norm(self, p): ...


def mat_wrapper(func):
    def inner(*args, **kwargs):
        mat_kwarg_keys = ["with_grad"]
        mat_kwargs = {}
        for key in mat_kwarg_keys:
            try:
                val = kwargs.pop(key)
            except KeyError:
                continue
            mat_kwargs.setdefault(key, val)
            
        array = func(*args, **kwargs)
        array = Mat(array, **mat_kwargs)
        return array
    return inner

array = mat_wrapper(np.array)
ones_like = mat_wrapper(np.ones_like)
zeros_like = mat_wrapper(np.zeros_like)
ones = mat_wrapper(np.ones)
zeros = mat_wrapper(np.zeros)