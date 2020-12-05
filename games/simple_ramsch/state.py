from games.abstract_state import GameState, ArraySlice
from utils import np_one_hot, one_hot_to_int, one_hot_arrays_to_list_of_ints
import numpy as np
from numba import jit


@jit(nopython=True)
def extract_tricks(full_state, first_row, num_played_cards):
    finished_tricks = max(0, (num_played_cards - 1) // 3)
    first_row_of_trick = first_row + 3 * finished_tricks
    num_cards_in_trick = num_played_cards - 3 * finished_tricks
    return full_state[first_row_of_trick:first_row_of_trick + num_cards_in_trick]



class RamschState(GameState):
    status_rows = 2
    hand_rows = 4
    gameplay_rows = 30
    implication_rows = 4
    redundant_rows = 3

    dealer = ArraySlice(slice_in_array=(0, slice(0, 3)))
    num_played_cards = ArraySlice(slice_in_array=(-1, 0))
    current_scores = ArraySlice(slice_in_array=(-1, slice(1, 4)))
    player_id_sequence = ArraySlice(slice_in_array=-2)
    active_player = ArraySlice(slice_in_array=(-1, 4))
    all_hands = ArraySlice(slice_in_array=slice(status_rows, status_rows + hand_rows))
    skat = ArraySlice(slice_in_array=status_rows + hand_rows-1)
    implications = ArraySlice(slice_in_array=slice(status_rows + hand_rows+ gameplay_rows,
                                                  status_rows + hand_rows+ gameplay_rows + implication_rows))
    played_cards = ArraySlice(slice_in_array=slice(status_rows + hand_rows, status_rows + hand_rows+gameplay_rows))

    @classmethod
    def from_initial_hands(cls, initial_hands, dealer):
        instance = cls(full_state=None)
        np_hands = [np_one_hot(hand, 32)[None, :] for hand in initial_hands]
        instance.all_hands = np.concatenate(np_hands, axis=0)
        instance.dealer = np_one_hot(dealer, 3)
        instance.active_player = (dealer + 1) % 3
        return instance

    def state_for_player(self, player_id):
        new_state = RamschState(self.full_state.copy())
        new_state.add_private_implications(player_id)
        for i in range(self.hand_rows):
            if i != player_id:
                new_state.full_state[self.status_rows + i] = 0
        return new_state

    @property
    def state_for_nn(self):
        return self.full_state[:-self.redundant_rows]

    @staticmethod
    def full_state_from_partial_and_initial_hands(partial_state, initial_hands):
        inconsistencies = partial_state.all_hands * (1-initial_hands)
        assert inconsistencies.sum() == 0  # guess on initial hands shouldn't be inconsistent
        still_in_play = 1 - np.sum(partial_state.played_cards, axis=0)
        full_state = RamschState(partial_state.full_state.copy())
        full_state.all_hands = initial_hands * still_in_play[None, :]
        return full_state

    def add_private_implications(self, player_id):
        active_row = self.status_rows + self.hand_rows + self.gameplay_rows + self.active_player
        for card in self.hands_as_ints[player_id]:
            assert self.full_state[active_row][card] != -1
            self.full_state[active_row][card] = 1

    @property
    def hands_as_ints(self):
        return [one_hot_to_int(hand) for hand in self.all_hands]

    @property
    def skat_as_ints(self):
        return one_hot_to_int(self.skat)


    @property
    def current_trick_as_ints(self):
        # returns a list of ints
        return one_hot_arrays_to_list_of_ints(self.current_trick)

    @property
    def played_cards_as_ints(self):
        played_cards = [one_hot_to_int(card) for card in self.played_cards]
        return [item[0] for item in played_cards[:self.num_played_cards]]

    @property
    def players_of_all_played_cards(self):
        return self.player_id_sequence[:self.num_played_cards]

    @property
    def starter_of_current_trick(self):
        finished_tricks = max(0, (self.num_played_cards - 1) // 3)
        return self.player_id_sequence[3 * finished_tricks]

    def play_card(self, card):
        assert self.all_hands[self.active_player][card] == 1  # The player has the card
        card_as_one_hot = np_one_hot(card, 32)

        # Add card to new trick
        current_row = self.status_rows + self.hand_rows + self.num_played_cards
        self.full_state[current_row] = card_as_one_hot

        # Take the card from player
        player_row = self.status_rows + self.active_player
        self.full_state[player_row][card] = 0

        sequence = self.player_id_sequence
        sequence[self.num_played_cards] = self.active_player
        self.player_id_sequence = sequence

        self.num_played_cards = self.num_played_cards + 1
        self.active_player = (self.active_player + 1) % 3

    def apply_public_implications(self, has_cards, doesnt_have_cards):
        current_row = self.status_rows + self.hand_rows + self.gameplay_rows + self.active_player
        for card in has_cards:
            assert self.full_state[current_row][card] != -1
            self.full_state[current_row][card] = 1

        for card in doesnt_have_cards:
            if self.full_state[current_row][card] == 0:  # If status unknown
                self.full_state[current_row][card] = -1

    @property
    def current_trick(self):
        return extract_tricks(self.full_state, self.status_rows + self.hand_rows, self.num_played_cards)
        #finished_tricks = max(0, (self.num_played_cards-1) // 3)
        #first_row = self.status_rows + self.hand_rows + 3*finished_tricks
        #num_cards_in_trick = self.num_played_cards - 3*finished_tricks
        #return self.full_state[first_row:first_row+num_cards_in_trick]

    def check_sound(self):
        card_row_start = self.status_rows
        num_card_rows = self.hand_rows + self.gameplay_rows
        card_sums = np.sum(self.full_state[card_row_start:card_row_start+num_card_rows], axis=0)
        assert np.abs(card_sums - 1).sum() == 0, card_sums
