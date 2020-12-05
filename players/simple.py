from abc import ABC, abstractmethod
import random
import torch

from train_model import TrainSkatModel
from utils import np_one_hot


class Player(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def play(self, state, available_actions, ruleset):
        pass

    def save_data(self, working_dir, player_id):
        pass


class RandomPlayer(Player):
    def play(self, state, available_actions, ruleset):
        return random.choice(available_actions)


class NNPlayer(Player):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = TrainSkatModel.load_from_checkpoint(checkpoint_path)
        self.model.eval()

    def play(self, state, available_actions, ruleset):
        action_mask = np_one_hot(available_actions, dim=32)

        nn_state = state.state_for_nn[None, ...]
        nn_state = torch.Tensor(nn_state)
        with torch.no_grad():
            probs = self.model(nn_state)[0].data * action_mask
        action = int(torch.argmax(probs, dim=-1).item())
        assert action in available_actions
        return action


class HardcodedRamschPlayer(Player):
    def play(self, state, available_actions, ruleset):
        played_cards = state.played_cards.sum(axis=0)
        scores = [(self.score_card(card, played_cards, available_actions), card) for card in available_actions]
        top_action = sorted(scores)[-1][1]
        assert top_action in available_actions
        return top_action

    def score_card(self, card, played_cards, my_hand):
        cards_under = HardcodedRamschPlayer.cards_under(card)
        gone = sum([played_cards[_card] for _card in cards_under])
        in_hand = len([_card for _card in my_hand if _card in cards_under])
        left = len(cards_under) - gone - in_hand
        return left - 0.9*in_hand

    @staticmethod
    def cards_under(card):
        if HardcodedRamschPlayer.suit(card) == 0:
            return [_card for _card in range(32) if HardcodedRamschPlayer.suit(_card) != 0 or _card < card]
        else:
            return [_card for _card in range(32) if HardcodedRamschPlayer.suit(_card) == HardcodedRamschPlayer.suit(card) and
                    HardcodedRamschPlayer.smaller(_card, card)]

    @staticmethod
    def suit(card):
        if (card % 8) == 4:
            return 0  # Trumf

        return 1 + card // 8  # Other suits in order

    @staticmethod
    def smaller(card1, card2):  # Assumes same suit (not jack)
        c1 = card1 % 8
        c1 = c1 if c1 != 3 else c1 + 3.5
        c2 = card2 % 8
        c2 = c2 if c2 != 3 else c2 + 3.5
        return c1 < c2
