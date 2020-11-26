from abc import ABC, abstractmethod


class Game(object):
    @abstractmethod
    def available_actions(self, state=None):
        pass

    @abstractmethod
    @property
    def result(self):
        return None

    @abstractmethod
    def state_of_player(self, player_id):
        pass

    @abstractmethod
    def get_implications(self):
        pass

    @abstractmethod
    def visualize(self):
        pass

