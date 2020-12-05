import os
from collections import defaultdict

from algorithms.mcts_basic import MCTS, CardGameNode
from players.simple import Player
from sat_solver import solve_sat_for_init_hands
from utils import np_one_hot

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class MCTSPlayer(Player):
    def __init__(self, num_mcts_rollouts, exploration_weight, softmax_temperature, guessed_hands):
        super().__init__()
        self.guessed_hands = guessed_hands
        self.softmax_temperature = softmax_temperature
        self.exploration_weight = exploration_weight
        self.num_mcts_rollouts = num_mcts_rollouts

        self.action_masks = []
        self.predicted_probabilities = []
        self.input_states = []

    def collect_data(self, available_actions, scores, state):
        action_mask = np_one_hot(available_actions, dim=32)
        learned_values = [(val, action) for (action, val) in scores.items()]
        probabilities = softmax(self.softmax_temperature * np.array([it[0] for it in learned_values]))
        masked_probabilities = np.zeros_like(action_mask)
        for prob, (logit, action) in zip(probabilities, learned_values):
            masked_probabilities[action] = prob

        self.action_masks.append(action_mask[None, ...])
        self.predicted_probabilities.append(masked_probabilities[None, ...])
        self.input_states.append(state.state_for_nn[None, ...])

    def play(self, state, available_actions, ruleset):

        init_hands = solve_sat_for_init_hands(state.implications, self.guessed_hands)
        init_full_states = [state.full_state_from_partial_and_initial_hands(state, init_hand)
                            for init_hand in init_hands]

        def values_for_starting_state(init_full_state, iterations, scores):
            starting_node = CardGameNode(ruleset, init_full_state)
            mcts_runner = MCTS(self.exploration_weight)

            for i in range(iterations):
                mcts_runner.do_rollout(starting_node)

            learned_values = mcts_runner.choose(starting_node)
            for child, val in learned_values:
                scores[child.current_state.current_trick_as_ints[-1]] += val

        iteration_per_sol = self.num_mcts_rollouts // len(init_full_states)
        scores = defaultdict(int)
        for init_state in init_full_states:
            values_for_starting_state(init_state, iteration_per_sol, scores)

        for action, value in scores.items():
            scores[action] = scores[action] / len(init_full_states)

        print(init_hands[0])
        print(len(init_hands))
        print(scores)
        print('-----------------------------------------------------------------')
        top_action = max(scores, key=scores.get)  # key with maximal value

        self.collect_data(available_actions, scores, state)
        assert top_action in available_actions
        return top_action

    def save_data(self, working_dir, player_id):
        state_file = os.path.join(working_dir, f"inputs_{player_id}.npy")
        all_states = np.concatenate(self.input_states, axis=0)
        np.save(state_file, all_states)

        all_masks_file = os.path.join(working_dir, f"masks_{player_id}.npy")
        all_masks = np.concatenate(self.action_masks, axis=0)
        np.save(all_masks_file, all_masks)

        all_probs_file = os.path.join(working_dir, f"probs_{player_id}.npy")
        all_probs = np.concatenate(self.predicted_probabilities, axis=0)
        np.save(all_probs_file, all_probs)
