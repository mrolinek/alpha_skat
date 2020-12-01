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
    def suit(card):
        if (card % 8) == 4:
            return 0  # Trumf

        return 1 + card // 8  # Other suits in order

    @staticmethod
    def available_actions(state):
        active_player_hand = one_hot_to_int(state.all_hands[state.active_player])
        #print(f"Player {state.active_player} has hand {}")
        trick = state.current_trick_as_ints
        if not trick or len(trick) == 3:
            return active_player_hand

        leading_suit = RamschRuleset.suit(trick[0])
        same_suit = [card for card in active_player_hand if RamschRuleset.suit(card) == leading_suit]
        if same_suit:
            return np.array(same_suit)

        return active_player_hand

    @staticmethod
    def trick_winner(trick):
        # returns index of card winning the trick (index 0 is first card played)
        assert len(trick) == 3
        suits = [RamschRuleset.suit(c) for c in trick]
        if 0 in suits:  # trumf was played
            candidates = [(c, ind) for ind, c in enumerate(trick) if RamschRuleset.suit(c) == 0]
            assert len(candidates) > 0
            winning_card = sorted(candidates)[-1][1]  # index of winner card (suits are sorted)
        else:
            candidate_values = [(c % 8, ind) for ind, c in enumerate(trick) if RamschRuleset.suit(c) == suits[0]]

            # Make ten high
            candidate_values = [(c if c != 3 else c + 3.5, ind) for c, ind in candidate_values]
            # Those that follow the first suit
            assert len(candidate_values) > 0
            winning_card = sorted(candidate_values)[-1][1]  # index of winner card (suits are sorted)

        return winning_card

    @staticmethod
    def do_action(state, action):
        new_state = RamschState(state.full_state.copy())
        new_state.play_card(action)

        if new_state.num_played_cards % 3 == 0:  # End of trick
            active_trick = new_state.current_trick
            trick = [one_hot_to_int(card)[0] for card in active_trick]
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




