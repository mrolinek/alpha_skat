import numpy as np
from numba import njit

def np_one_hot(values, dim):
    res = np.zeros(shape=(dim,))
    res[np.array(values)] = 1
    return res

@njit
def one_hot_to_int(np_arr):
    return np.where(np_arr == 1)[0]

