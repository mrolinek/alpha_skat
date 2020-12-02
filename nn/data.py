from abc import abstractmethod
from collections import deque, OrderedDict

import numpy as np
import torch


def efficient_from_numpy(x, device):
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)


class DatasetIterator(object):
    def __init__(self, *, dataset, **kwargs):
        data_dict = dataset.data
        data_point_idx = np.arange(len(list(data_dict.values())[0]))[:, None]
        data_dict['datapoint_idx'] = data_point_idx
        zipped_data = list(zip(*data_dict.values()))

        self.dataset = dataset

        self.dtype = [(key, "f4", value[0].shape) for key, value in data_dict.items()]
        # PyTorch works with 32-bit floats by default

        self.array = np.array(zipped_data, dtype=self.dtype)
        self._size = None

    @abstractmethod
    def get_epoch_iterator(self, batch_size, number_of_epochs, device='cpu', shuffle=True):
        raise NotImplementedError()


class BatchNumpyIterator(DatasetIterator):
    def __init__(self, *, dataset, **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    def get_epoch_iterator(self, batch_size, number_of_epochs, device='cpu', preload=False, shuffle=True):
        def iterator():
            for i in range(number_of_epochs):
                if shuffle:
                    np.random.shuffle(self.array)
                for j in range(1 + len(self.array) // batch_size):
                    numpy_batch = self.array[j * batch_size: (j + 1) * batch_size]
                    batch = {k: numpy_batch[k] for k in numpy_batch.dtype.names}
                    yield batch

        return iterator()


class BatchIterator(DatasetIterator):
    def __init__(self, *, dataset, **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    def get_epoch_iterator(self, batch_size, number_of_epochs, device='cpu', preload=False, shuffle=True):
        def iterator():
            if preload:
                preload_deque = deque(maxlen=2)
            for i in range(number_of_epochs):
                if shuffle:
                    np.random.shuffle(self.array)
                for j in range(1 + len(self.array) // batch_size):
                    numpy_batch = self.array[j * batch_size: (j + 1) * batch_size]
                    torch_batch = OrderedDict(
                        list([(key, efficient_from_numpy(numpy_batch[key], device=device)) for key in
                              numpy_batch.dtype.names]))

                    if numpy_batch.size:
                        if j == 0 and preload:
                            preload_deque.appendleft(torch_batch)
                            continue
                        if preload:
                            preload_deque.appendleft(torch_batch)
                            yield preload_deque.pop()
                        else:
                            yield torch_batch
                if preload:
                    while len(preload_deque) > 0:
                        yield preload_deque.pop()

        return iterator()
