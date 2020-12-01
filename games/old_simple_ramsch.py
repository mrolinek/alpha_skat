
import pickle

import numpy as np
import random


class Ramsch(object):
    def __init__(self):
        self.deck = np.array(list(range(32)))
        self.hands = [[], [], []]
        self.skat = []
        self.hashable_state = None
        self.active_player = None
        self.taken_actions = None
        self.dealing_pattern = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2,
                                3, 3,
                                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2])

        self.card_values = [0, 0, 0, 10, 2, 3, 4, 11]

        assert len(self.dealing_pattern) == 32

    def start_new(self):
        random.shuffle(self.deck)
        self.hands[0] = list(self.deck[self.dealing_pattern == 0])
        self.hands[1] = list(self.deck[self.dealing_pattern == 1])
        self.hands[2] = list(self.deck[self.dealing_pattern == 2])
        self.skat = self.deck[self.dealing_pattern == 3]
        self.init_hands = [list(item) for item in self.hands] + [list(self.skat)]
        self.historical_scores = []
        self.active_trick = []
        self.tricks = []
        self.scores = [0, 0, 0]
        self.done = False
        self.final_scores = [None, None, None]
        self.taken_actions = []

        assert len(self.hands[0]) + len(self.hands[1]) + len(self.hands[2]) + len(self.skat) == 32, self.hands
        assert len(set(self.hands[0]) | set(self.hands[1]) | set(self.hands[2]) | set(self.skat)) == 32, self.hands

        self.active_player = 0


    def suit(self, card):
        if (card % 8) == 4:
            return 0  # Trumf

        return 1 + card // 8  # Other suits in order

    def available_actions(self):
        if not self.active_trick:
            return np.array(self.hands[self.active_player])

        leading_suit = self.suit(self.active_trick[0])
        same_suit = [card for card in self.hands[self.active_player] if self.suit(card) == leading_suit]
        if same_suit:
            return np.array(same_suit)

        return np.array(self.hands[self.active_player])


    def trick_winner(self, trick):
        assert len(trick) == 3
        suits = [self.suit(c) for c in trick]
        if 0 in suits:  # trumf was played
            candidates = [(c, ind) for ind, c in enumerate(trick) if self.suit(c) == 0]
            assert len(candidates) > 0
            winning_card = sorted(candidates)[-1][1]  # index of winner card (suits are sorted)
        else:
            candidate_values = [(c%8, ind) for ind, c in enumerate(trick) if self.suit(c) == suits[0]]

            # Make ten high
            candidate_values = [(c if c != 3 else c+3.5, ind) for c, ind in candidate_values]
            # Those that follow the first suit
            assert len(candidate_values) > 0
            winning_card = sorted(candidate_values)[-1][1]  # index of winner card (suits are sorted)

        trick_started_by = (self.active_player + 1) % 3
        winning_player = (winning_card + trick_started_by) % 3
        return winning_player

    def finalize_scores(self):
        assert sum(self.scores) == 120
        if 120 in self.scores:
            self.final_scores = list(self.scores)  # Durchmarsch
        else:
            coef = 2 if 0 in self.scores else 1  # Jungfrau
            self.final_scores = [-coef*c if c == max(self.scores) else 0 for c in self.scores]


    def take_action(self, action):
        assert len(self.active_trick) < 3

        self.active_trick.append(action)
        assert action in self.hands[self.active_player]

        self.hands[self.active_player].remove(action)
        self.taken_actions.append((action, self.active_player))

        assert action not in self.hands[self.active_player]

        if len(self.active_trick) < 3:
            self.active_player = (self.active_player + 1) % 3
        else:
            winner = self.trick_winner(self.active_trick)
            self.tricks.append(list(self.active_trick))
            self.scores[winner] += sum([self.card_values[c % 8] for c in self.active_trick])
            self.active_player = winner
            self.active_trick = []

            if sum([len(x) for x in self.hands]) == 0:  # Game over
                self.scores[winner] += sum([self.card_values[c % 8] for c in self.skat])
                self.done = True
                self.finalize_scores()

            self.historical_scores.append(list(self.scores))

    def save(self, filename):
        to_save = dict(init_hands=self.init_hands,
                       actions=self.taken_actions,
                       historical_scores=self.historical_scores,
                       final_scores=self.final_scores)
        with open(filename, 'wb') as f:
            pickle.dump(to_save, f)



game = Ramsch()


for i in range(200):
    game.start_new()

    while not game.done:
        choices = game.available_actions()
        action = random.choice(choices)
        game.take_action(action)

    if max(game.scores) > 100:
        print(game.final_scores)
        game.save("durchmarsch.gm")
        exit(0)










