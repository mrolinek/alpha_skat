import random
from collections import defaultdict

import torch
import numpy as np

from algorithms.mcts_basic import MCTS
from utils import np_one_hot


class MCTS_parallel(MCTS):
    def __init__(self, *, value_function_model, **kwargs):
        super().__init__(**kwargs)
        self.model = value_function_model

    def do_rollouts(self, node, how_many):
        paths = []
        for i in range(how_many):
            path = self._select(node)
            leaf = path[-1]
            self._expand(leaf)
            if leaf.is_terminal():
                reward = leaf.rewards()
                self._backpropagate(path, reward)
            else:
                paths.append(path)
                for n in path:
                    self.N[n] += 1

        if not paths:
            return

        nodes_to_evaluate, node_mapping = self._backpropagate_batch(paths)
        list_to_evaluate = list(nodes_to_evaluate)
        values = evaluate_batch_of_nodes(list_to_evaluate, self.model)
        value_mapping = dict(zip(list_to_evaluate, values))

        for node, list_to_update in node_mapping.items():
            for updated_node in list_to_update:
                self.Q[updated_node] += value_mapping[node]

    def _backpropagate_batch(self, paths):
        nodes_to_evaluate, node_mapping = set(), defaultdict(list)
        for path in paths:
            leaf = path[-1]
            nodes_to_evaluate.add(leaf)
            node_mapping[leaf].extend(path)

        return nodes_to_evaluate, node_mapping


def evaluate_batch_of_nodes(list_of_nodes, model):
    nn_states = [node.current_state.state_for_player(node.active_player).state_for_nn[None, ...]
                 for node in list_of_nodes]
    nn_states = np.concatenate(nn_states, axis=0)
    policy, value = model.get_policy_and_value(nn_states, cuda=True)
    return value

