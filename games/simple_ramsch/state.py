from games.abstract_state import GameState, ArraySlice
from utils import np_one_hot, one_hot_to_int
import numpy as np

class RamschState(GameState):
    status_rows = 2
    hand_rows = 4
    gameplay_rows = 30
    redundant_rows = 3

    dealer = ArraySlice(slice_in_array=(0, slice(0, 3)))
    num_played_cards = ArraySlice(slice_in_array=(-1, 0))
    current_scores = ArraySlice(slice_in_array=(-1, slice(1, 4)))
    player_id_sequence = ArraySlice(slice_in_array=-2)
    active_player = ArraySlice(slice_in_array=(-1, 4))
    all_hands = ArraySlice(slice_in_array=slice(status_rows, status_rows + hand_rows))
    skat = ArraySlice(slice_in_array=status_rows + hand_rows-1)
    played_cards = ArraySlice(slice_in_array=slice(status_rows + hand_rows, status_rows + hand_rows+gameplay_rows))

    @classmethod
    def from_initial_hands(cls, initial_hands, dealer):
        instance = cls(full_state=None)
        np_hands = [np_one_hot(hand, 32)[None, :] for hand in initial_hands]
        instance.all_hands = np.concatenate(np_hands, axis=0)
        instance.dealer = np_one_hot(dealer, 3)
        instance.active_player = (dealer + 1) % 3
        return instance

    def nn_state_for_player_view(self, player_id):
        copied_state = self.full_state.copy()
        for i in range(self.hand_rows):
            if i != player_id:
                copied_state[self.status_rows + i] = 0
        return copied_state

    @property
    def hands_as_ints(self):
        return [one_hot_to_int(hand) for hand in self.all_hands]

    @property
    def skat_as_ints(self):
        return one_hot_to_int(self.skat)


    @property
    def current_trick_as_ints(self):
        # returns a list of ints
        return sum([list(one_hot_to_int(card)) for card in self.current_trick], [])

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

    @property
    def current_trick(self):
        finished_tricks = max(0, (self.num_played_cards-1) // 3)
        first_row = self.status_rows + self.hand_rows + 3*finished_tricks
        return self.full_state[first_row:first_row+3]

    def check_sound(self):
        card_row_start = self.status_rows
        num_card_rows = self.hand_rows + self.gameplay_rows
        card_sums = np.sum(self.full_state[card_row_start:card_row_start+num_card_rows], axis=0)
        assert np.abs(card_sums - 1).sum() == 0, card_sums
