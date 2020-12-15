import os
from collections import defaultdict

from algorithms.mcts_basic import MCTS, CardGameNode
from algorithms.mcts_parallel_value import MCTS_parallel
from players.simple import Player
from sat_solver import solve_sat_for_init_hands, top_k_likely_hands
from train_model import TrainSkatModel
from utils import np_one_hot, softmax

import numpy as np


class MCTSPlayer(Player):
    def __init__(self, num_mcts_rollouts, exploration_weight, guessed_hands, softmax_temperature_for_saving,
                 use_policy_for_init_hands,
                 use_policy_for_ucb,
                 init_hands_to_sample,
                 epsilon,
                 value_function_checkpoint=None,
                 policy_function_checkpoint=None):
        super().__init__()
        self.epsilon = epsilon
        self.init_hands_to_sample = init_hands_to_sample
        self.use_policy_for_ucb = use_policy_for_ucb
        self.use_policy_for_init_hands = use_policy_for_init_hands
        self.guessed_hands = guessed_hands
        self.exploration_weight = exploration_weight
        self.num_mcts_rollouts = num_mcts_rollouts
        self.softmax_temperature_for_saving = softmax_temperature_for_saving
        if value_function_checkpoint:
            self.value_model = TrainSkatModel.load_from_checkpoint(value_function_checkpoint)
        else:
            self.value_model = None

        if policy_function_checkpoint:
            self.policy_model = TrainSkatModel.load_from_checkpoint(policy_function_checkpoint)
        else:
            self.policy_model = None

        self.action_masks = []
        self.policy_probs = []
        self.input_states = []
        self.state_values = []

    def collect_data(self, available_actions, scores, state, active_player_scores):
        action_mask = np_one_hot(available_actions, dim=32)
        probabilities = np.zeros_like(action_mask)
        state_value = np.array([0.0, 0.0, 0.0])

        for (action, val) in active_player_scores.items():
            probabilities[action] = val * self.softmax_temperature_for_saving
        probabilities = softmax(probabilities+1000*(action_mask-1))*action_mask

        for action, vals in scores.items():
            state_value += probabilities[action] * vals

        self.action_masks.append(action_mask[None, ...])
        self.policy_probs.append(probabilities[None, ...])
        self.input_states.append(state.state_for_nn[None, ...])
        self.state_values.append(state_value[None, ...])

    def play(self, state, available_actions, ruleset):

        if self.use_policy_for_init_hands:
            assert self.policy_model is not None
            init_hands = top_k_likely_hands(ruleset, state, self.guessed_hands, self.policy_model,
                                            epsilon=self.epsilon,
                                            init_hands_to_sample=self.init_hands_to_sample)
        else:
            init_hands = solve_sat_for_init_hands(state.implications, self.guessed_hands)
        init_full_states = [state.full_state_from_partial_and_initial_hands(state, init_hand)
                            for init_hand in init_hands]

        def values_for_starting_state(init_full_state, iterations, scores):
            starting_node = CardGameNode(ruleset, init_full_state)
            if self.value_model:
                mcts_runner = MCTS_parallel(value_function_model=self.value_model,
                                            exploration_weight=self.exploration_weight,
                                            policy_model=self.policy_model if self.use_policy_for_ucb else None)
                for i in range(4):
                    mcts_runner.do_rollouts(starting_node, iterations // 4)
            else:
                mcts_runner = MCTS(self.exploration_weight, self.policy_model if self.use_policy_for_ucb else None)
                for i in range(iterations):
                    mcts_runner.do_rollout(starting_node)

            learned_values = mcts_runner.choose(starting_node)
            for child, val in learned_values:
                scores[child.last_played_card] += val / len(init_full_states)  # normalize

        iteration_per_sol = self.num_mcts_rollouts // len(init_full_states)
        scores = defaultdict(int)
        for init_state in init_full_states:
            values_for_starting_state(init_state, iteration_per_sol, scores)

        print(sorted(scores.items()))
        print('-----------------------------------------------------------------')
        player = int(state.active_player)
        active_player_scores = {action: score[player] for action, score in scores.items()}
        top_action = max(active_player_scores, key=active_player_scores.get)  # key with maximal value

        self.collect_data(available_actions, scores, state, active_player_scores)
        assert top_action in available_actions
        return top_action

    def save_data(self, working_dir, player_id):

        def save_arr(arr, file_name):
            full_filename = os.path.join(working_dir, f"{file_name}_{player_id}.npy")
            concat = np.concatenate(arr, axis=0)
            np.save(full_filename,concat)

        save_arr(self.input_states, "inputs")
        save_arr(self.policy_probs, "policy_probs")
        save_arr(self.action_masks, "masks")
        save_arr(self.state_values, "state_values")

