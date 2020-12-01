from abc import ABC, abstractmethod


class Ruleset(ABC):

    @staticmethod
    @abstractmethod
    def generate_init_state(last_game_final_state):
        pass

    @staticmethod
    @abstractmethod
    def available_actions(state):
        pass

    @staticmethod
    @abstractmethod
    def final_rewards(state):
        pass

    @staticmethod
    @abstractmethod
    def is_finished(state):
        pass


    @staticmethod
    @abstractmethod
    def do_action(state, action):
        pass
