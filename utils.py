from numba import njit
import numpy as np
import torch
from abc import ABC, abstractmethod

from functools import update_wrapper, partial




def np_one_hot(values, dim):
    res = np.zeros(shape=(dim,))
    res[np.array(values)] = 1
    return res


@njit
def one_hot_to_int(np_arr):
    return np.where(np_arr == 1)[0]


@njit
def one_hot_arrays_to_list_of_ints(arrays):
    ones = np.argwhere(arrays == 1)
    return ones[:, 1]
    # return np.array([one_hot_to_int(arr) for arr in arrays])


@njit
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class Decorator(ABC):
    def __init__(self, f):
        self.func = f
        update_wrapper(self, f, updated=[])  # updated=[] so that 'self' attributes are not overwritten

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __get__(self, instance, owner):
        new_f = partial(self.__call__, instance)
        update_wrapper(new_f, self.func)
        return new_f


def to_tensor(x):
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return torch.from_numpy(np.array(x)).float()
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x


# noinspection PyPep8Naming
class input_to_tensors(Decorator):
    def __call__(self, *args, **kwargs):
        new_args = [to_tensor(arg) for arg in args]
        new_kwargs = {key: to_tensor(value) for key, value in kwargs.items()}
        return self.func(*new_args, **new_kwargs)


# noinspection PyPep8Naming
class output_to_tensors(Decorator):
    def __call__(self, *args, **kwargs):
        outputs = self.func(*args, **kwargs)
        if isinstance(outputs, np.ndarray):
            return to_tensor(outputs)
        if isinstance(outputs, tuple):
            new_outputs = tuple([to_tensor(item) for item in outputs])
            return new_outputs
        return outputs


# noinspection PyPep8Naming
class input_to_numpy(Decorator):
    def __call__(self, *args, **kwargs):
        new_args = [to_numpy(arg) for arg in args]
        new_kwargs = {key: to_numpy(value) for key, value in kwargs.items()}
        return self.func(*new_args, **new_kwargs)


# noinspection PyPep8Naming
class output_to_numpy(Decorator):
    def __call__(self, *args, **kwargs):
        outputs = self.func(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return to_numpy(outputs)
        if isinstance(outputs, tuple):
            new_outputs = tuple([to_numpy(item) for item in outputs])
            return new_outputs
        return outputs
