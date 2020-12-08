import random
from collections import defaultdict

import torch
import numpy as np

from algorithms.mcts_basic import MCTS, scaling_constant
from utils import np_one_hot


class MCTS_parallel(MCTS):
    def __init__(self, exploration_weight, value_function_model):
        super().__init__(exploration_weight)
        self.model = value_function_model
        self.N = defaultdict(lambda: 1)

    def do_rollouts(self, node, how_many):
        paths, leafs = [], []
        for i in range(how_many):
            path = self._select(node)
            leaf = path[-1]
            self._expand(leaf)
            if leaf.is_terminal():
                reward = leaf.rewards()
                self._backpropagate(path, reward)
            else:
                paths.append(path)
                leafs.append(leaf)

        if not paths:
            return

        nodes_to_evaluate, node_mapping = self._backpropagate_batch(paths)
        list_to_evaluate = list(nodes_to_evaluate)
        values = evaluate_batch_of_nodes(list_to_evaluate, self.model)
        value_mapping = dict(zip(list_to_evaluate, values))

        for node, eval_node in node_mapping.items():
            self.Q[node] += value_mapping[eval_node]



    def _backpropagate_batch(self, paths):
        nodes_to_evaluate, node_mapping = set(), dict()
        for path in paths:
            nodes_keeping_reward = [None, None, None]
            active_players = [path[0].active_player] + [node.active_player for node in path[:-1]]
            for node, active_player in zip(reversed(path), reversed(active_players)):
                if nodes_keeping_reward[active_player] is None:
                    nodes_keeping_reward[active_player] = node
                    nodes_to_evaluate.add(node)
                self.N[node] += 1
                node_mapping[node] = nodes_keeping_reward[active_player]

        return nodes_to_evaluate, node_mapping


def evaluate_batch_of_nodes(list_of_nodes, model):
    print(f"Evaluating NN on {len(list_of_nodes)} nodes")
    with torch.no_grad():
        nn_states = [node.current_state.state_for_player(node.active_player).state_for_nn[None, ...]
                     for node in list_of_nodes]
        nn_states = torch.Tensor(np.concatenate(nn_states, axis=0))
        q_values = model(nn_states).numpy()
        one_hot_actions = [np_one_hot(node.actions, dim=32)[None, ...] for node in list_of_nodes]
        one_hot_actions = np.concatenate(one_hot_actions, axis=0)
        values = scaling_constant * (q_values + 1000 * (one_hot_actions - 1)).max(axis=1)
        return values

