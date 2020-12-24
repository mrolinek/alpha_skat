import os
from collections import defaultdict

from algorithms.mcts_basic import MCTS, CardGameNode
from algorithms.mcts_parallel_value import MCTS_parallel
from players.simple import Player
from sat_solver import solve_sat_for_init_hands, top_k_likely_hands
from train_model import PolicyModel, ValueModel
from utils import np_one_hot

import numpy as np


class MCTSPlayer(Player):
    def __init__(self, num_mcts_rollouts, exploration_weight, guessed_hands,
                 use_policy_for_init_hands,
                 use_policy_for_ucb,
                 init_hands_to_sample,
                 epsilon,
                 policy_ucb_coef,
                 policy_simulation_steps,
                 use_policy_for_opponents,
                 value_function_checkpoint=None,
                 policy_function_checkpoint=None):
        super().__init__()
        self.policy_simulation_steps = policy_simulation_steps
        self.policy_ucb_coef = policy_ucb_coef
        self.epsilon = epsilon
        self.use_policy_for_opponents = use_policy_for_opponents
        self.init_hands_to_sample = init_hands_to_sample
        self.use_policy_for_ucb = use_policy_for_ucb
        self.use_policy_for_init_hands = use_policy_for_init_hands
        self.guessed_hands = guessed_hands
        self.exploration_weight = exploration_weight
        self.num_mcts_rollouts = num_mcts_rollouts
        if value_function_checkpoint:
            self.value_model = ValueModel.load_from_checkpoint(value_function_checkpoint)
        else:
            self.value_model = None

        if policy_function_checkpoint:
            self.policy_model = PolicyModel.load_from_checkpoint(policy_function_checkpoint)
        else:
            self.policy_model = None

        self.action_masks = []
        self.policy_probs = []
        self.input_states = []
        self.full_state_values = []
        self.full_states = []

    def collect_data(self, available_actions, state_values, state, visitations):
        action_mask = np_one_hot(available_actions, dim=32)
        probabilities = np.zeros_like(action_mask, dtype=np.float32)

        for (action, num_visits) in visitations.items():
            probabilities[action] = num_visits
        probabilities = probabilities / probabilities.sum()

        for child, vals in state_values.items():
            self.full_states.append(child.current_state.state_for_nn[None, ...])
            self.full_state_values.append(vals[None, ...])

        self.action_masks.append(action_mask[None, ...])
        self.policy_probs.append(probabilities[None, ...])
        self.input_states.append(state.state_for_nn[None, ...])

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

        def values_for_starting_state(init_full_state, iterations, state_values, visitations, action_values):
            starting_node = CardGameNode(ruleset, init_full_state)
            print("Trying init state:")
            print([list(item) for item in init_full_state.hands_as_ints])
            if self.value_model:
                mcts_runner = MCTS_parallel(value_model=self.value_model,
                                            exploration_weight=self.exploration_weight,
                                            policy_model=self.policy_model,
                                            policy_ucb_coef=self.policy_ucb_coef,
                                            policy_simulation_steps=self.policy_simulation_steps,
                                            use_policy_for_opponents=self.use_policy_for_opponents)
                mcts_runner.root_node = starting_node
                for i in range(4):
                    mcts_runner.do_rollouts(iterations // 4)
            else:
                mcts_runner = MCTS(self.exploration_weight, self.policy_model,
                                   policy_ucb_coef=self.policy_ucb_coef,
                                   value_model=self.value_model,
                                   policy_simulation_steps=self.policy_simulation_steps,
                                   use_policy_for_opponents=self.use_policy_for_opponents)
                mcts_runner.root_node = starting_node
                for i in range(iterations):
                    mcts_runner.do_rollout()

            learned_values, learned_visitations = mcts_runner.choose()
            print("Values:")
            print({c.last_played_card: list(val) for c, val in learned_values})
            visitation_dict = dict(learned_visitations)
            print("Visitations:")
            print({c.last_played_card: val for c, val in learned_visitations})
            print('-----------------------------------------------------------------')
            for child, val in learned_values:
                if visitation_dict[child] > iterations // 4:
                    state_values[child] = val
                action_values[child.last_played_card] += val / len(init_full_states)
            for child, val in learned_visitations:
                visitations[child.last_played_card] += val

        iteration_per_sol = self.num_mcts_rollouts // len(init_full_states)
        state_values = dict()
        visitations = defaultdict(int)
        action_values = defaultdict(int)

        for init_state in init_full_states:
            values_for_starting_state(init_state, iteration_per_sol, state_values, visitations, action_values)

        print("Aggregate evaluation:")
        print(sorted(visitations.items()))
        print(sorted(action_values.items()))
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        player = int(state.active_player)
        active_player_scores = {action: score[player] for action, score in action_values.items()}
        top_action = max(active_player_scores, key=active_player_scores.get)  # key with maximal value

        self.collect_data(available_actions, state_values, state, visitations)
        assert top_action in available_actions
        return top_action

    def save_data(self, working_dir, player_id):

        def save_arr(arr, file_name):
            full_filename = os.path.join(working_dir, f"{file_name}_{player_id}.npy")
            concat = np.concatenate(arr, axis=0)
            np.save(full_filename, concat)

        save_arr(self.input_states, "inputs")
        save_arr(self.policy_probs, "policy_probs")
        save_arr(self.action_masks, "masks")
        save_arr(self.full_state_values, "full_state_values")
        save_arr(self.full_states, "full_states")
