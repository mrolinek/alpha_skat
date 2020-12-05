import numpy as np
from numba import njit

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
    #return np.array([one_hot_to_int(arr) for arr in arrays])

