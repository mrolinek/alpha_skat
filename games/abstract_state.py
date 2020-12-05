from abc import ABC, abstractmethod
import numpy as np

class ArraySlice(object):
    def __init__(self, slice_in_array):
        self.slice_in_array = slice_in_array

    def __set__(self, instance, value):
        instance.full_state[self.slice_in_array] = value

    def __get__(self, instance, cls):
        return instance.full_state[self.slice_in_array]


class GameState(ABC):
    name = "Abstract"
    status_rows = None
    hand_rows = None
    gameplay_rows = None
    redundant_rows = None
    implication_rows = None
    columns = 32

    def __init__(self, full_state=None):
        self.total_rows = (self.status_rows+
                           self.hand_rows+
                           self.gameplay_rows+
                           self.redundant_rows+
                           self.implication_rows)
        if full_state is not None:
            if not full_state.shape == (self.total_rows, self.columns):
                raise TypeError(f"State for game {self.name} must have shape"
                                f"{(self.total_rows, self.columns)}, not {self.columns}")
            self.full_state = full_state
        else:
            self.full_state = np.zeros(shape=(self.total_rows, self.columns), dtype=np.int16)

    @classmethod
    @abstractmethod
    def from_initial_hands(cls, initial_hands, dealer):
        pass


    def current_hand_for_player(self, player_id):  # skat will have ID 3 (if applicable)
        return self.full_state[self.status_rows + player_id]

    @abstractmethod
    def state_for_nn(self):
        pass

    @abstractmethod
    def state_for_player(self, player_id):
        pass
