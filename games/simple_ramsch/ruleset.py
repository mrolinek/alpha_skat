import random

from games.abstract_ruleset import Ruleset
import numpy as np

from games.simple_ramsch.state import RamschState
from utils import one_hot_to_int

dealing_pattern = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2,
                                3, 3,
                                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2])
card_values = [0, 0, 0, 10, 2, 3, 4, 11]

from numba import jit


@jit(nopython=True)
def suit(card):
    if (card % 8) == 4:
        return 0  # Trumf

    return 1 + card // 8  # Other suits in order


@jit(nopython=True)
def card_playabilities(trick):
    if len(trick) == 0 or len(trick) == 3:
        return [0] * 32  # Every card can be played initially
    starting_suit = suit(trick[0])

    def priority(card):
        return 1 if suit(card) == starting_suit else 0  # Follow suit!

    return [priority(card) for card in range(32)]


@jit(nopython=True)
def card_priorities(starting_card):
    assert starting_card is not None
    starting_suit = suit(starting_card)

    def priority(card):
        if suit(card) == 0:
            return card + 100  # Jacks have highet priority (and are sorted in order)
        elif suit(card) != starting_suit:  # Wrong suits have min priority
            return -1
        else:
            value = card % 8
            if value == 3:  # It is a ten
                value = 6.5  # Put it between king and ace
            return value

    return [priority(card) for card in range(32)]



class RamschRuleset(Ruleset):

    @staticmethod
    def generate_init_state(last_game_final_state):
        if last_game_final_state is None:
            dealer = 0
        else:
            dealer = (one_hot_to_int(last_game_final_state.dealer)[0] + 1) % 3

        deck = np.array(list(range(32)))
        random.shuffle(deck)
        hands = [list(deck[dealing_pattern == 0]),
                 list(deck[dealing_pattern == 1]),
                 list(deck[dealing_pattern == 2]),
                 list(deck[dealing_pattern == 3])]

        return RamschState.from_initial_hands(hands, dealer)

    @staticmethod
    def rotate_init_state(init_state):
        new_state = RamschState(init_state.full_state.copy())
        new_state.dealer = init_state.dealer[np.array([2, 0, 1])]
        new_state.active_player = (init_state.active_player + 1) % 3

        new_state.all_hands = init_state.all_hands[np.array([2, 0, 1, 3])]
        return new_state



    @staticmethod
    def available_actions(state):
        active_player_hand = one_hot_to_int(state.all_hands[state.active_player])
        trick = state.current_trick_as_ints
        playabilities = card_playabilities(trick)
        playability_on_hand = [(playabilities[card], card) for card in active_player_hand]

        max_playability = max(playability_on_hand)[0]
        equal_to_max = [card for val, card in playability_on_hand if val == max_playability]
        ans = np.array(equal_to_max)
        return ans

    @staticmethod
    def trick_winner(trick):
        # returns index of card winning the trick (index 0 is first card played)
        assert len(trick) == 3

        priorities = card_priorities(starting_card=trick[0])
        priorities_in_trick = [priorities[card] for card in trick]
        max_priority = max(priorities_in_trick)

        equal_to_max = [i for i, prio in enumerate(priorities_in_trick) if prio == max_priority]
        assert len(equal_to_max) == 1  # Unique trick winner
        return equal_to_max[0]

    @staticmethod
    def do_action(state, action):
        new_state = RamschState(state.full_state.copy())
        trick = new_state.current_trick_as_ints
        implications = RamschRuleset.action_public_implications(trick, action)
        new_state.apply_public_implications(*implications)
        new_state.play_card(action)

        if new_state.num_played_cards % 3 == 0:  # End of trick
            trick = new_state.current_trick_as_ints
            winning_card = RamschRuleset.trick_winner(trick)
            winning_player = (winning_card + new_state.active_player) % 3
            current_scores = new_state.current_scores
            current_scores[winning_player] += sum([card_values[c % 8] for c in trick])

            if new_state.num_played_cards == 30:
                skat_cards = one_hot_to_int(state.skat)
                current_scores[winning_player] += sum([card_values[c % 8] for c in skat_cards])
            else:
                new_state.active_player = winning_player

            new_state.current_scores = current_scores

        return new_state

    @staticmethod
    def action_public_implications(trick, action):
        has_cards = [action]
        playabilities = card_playabilities(trick)
        doesnt_have_cards = [card for card in range(32) if playabilities[card] > playabilities[action]]
        return has_cards, doesnt_have_cards


    @staticmethod
    def finalize_scores(state):
        scores = state.current_scores
        assert scores.sum() == 120
        if 120 in scores:  # Durchmarsch
            final_scores = scores
        else:
            coef = 2 if 0 in scores else 1  # Jungfrau
            final_scores = [-coef * c if c == max(scores) else 0 for c in scores]
        new_state = RamschState(state.full_state.copy())
        new_state.current_scores = final_scores
        return new_state

    @staticmethod
    def is_finished(state):
        return state.num_played_cards == 30

    @staticmethod
    def final_rewards(state):
        if RamschRuleset.is_finished(state):
            return RamschRuleset.finalize_scores(state).current_scores
        else:
            return None




