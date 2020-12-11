import random
import torch
from abc import ABC, abstractmethod
from collections import defaultdict
import math

from utils import np_one_hot, softmax
import numpy as np


class MCTS(object):
    def __init__(self, exploration_weight, policy_model):
        self.policy_model = policy_model
        self.Q = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def reset(self):
        self.children = dict()

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            raise RuntimeError(f"No iterations performed {node}")

        def all_value_functions(n):
            assert self.N[n] > 0
            return self.Q[n] / self.N[n]  # average reward

        res = [(child, all_value_functions(child)) for child in self.children[node]]
        return res

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if node.is_terminal():
            self.children[node] = []
        else:
            self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                rewards = node.rewards()
                return rewards
            node = node.find_random_child()


    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward


    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        assert all(n in self.N for n in self.children[node])

        if len(self.children[node]) == 1:
            return list(self.children[node])[0]
        
        sqrt_parent_visits = math.sqrt(self.N[node])
        player = node.active_player
        avgs = {n: self.Q[n][player] / self.N[n] for n in self.children[node]}

        if self.policy_model is not None:
            policy_probabilities = node.policy_estimate(self.policy_model)
        else:
            policy_probabilities = [1.0/len(self.children[node])] * 32

        def uct(n):
            "Upper confidence bound for trees"
            policy_value = policy_probabilities[n.last_played_card]
            exploration_value = self.exploration_weight * policy_value * sqrt_parent_visits / (self.N[n])
            return avgs[n] + exploration_value

        return max(self.children[node], key=uct)


class MultiplayerMCTSNode(ABC):
    num_players = 3

    @abstractmethod
    def current_player(self):
        return None

    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_random_child(self):
        return None

    @abstractmethod
    def is_terminal(self):
        return True

    @abstractmethod
    def rewards(self):
        return [0.0, 0.0, 0.0]

    @abstractmethod
    def value_function_estimate(self, model):
        pass

    @abstractmethod
    def policy_estimate(self, model):
        pass


    @abstractmethod
    def __hash__(self):
        return None

    @abstractmethod
    def __eq__(self, node2):
        "Nodes must be comparable"
        return True

scaling_constant = 50.0

class CardGameNode(MultiplayerMCTSNode):
    def __init__(self, ruleset, current_state):
        self.ruleset = ruleset
        self.current_state = current_state
        self.active_player = self.current_state.active_player
        self.hashed = None
        self._actions = None
        self._value = None
        self._policy_probabilities = None


    @property
    def actions(self):
        if self._actions is None:
            self._actions = self.ruleset.available_actions(self.current_state)
        return self._actions


    def current_player(self):
        return self.active_player

    @property
    def last_played_card(self):
        return self.current_state.current_trick_as_ints[-1]

    def find_children(self):
        next_states = [self.ruleset.do_action(self.current_state, action) for action in self.actions]
        return set([CardGameNode(self.ruleset, next_state) for next_state in next_states])

    def find_random_child(self):
        next_state = self.ruleset.do_action(self.current_state, random.choice(self.actions))
        return CardGameNode(self.ruleset, next_state)

    def is_terminal(self):
        return self.ruleset.is_finished(self.current_state)

    def value_function_estimate(self, model):
        if self._value is not None:
            return self._value
        with torch.no_grad():
            nn_state = self.current_state.state_for_player(self.active_player).state_for_nn
            nn_state = torch.Tensor(nn_state[None, ...])
            q_values = model(nn_state)[0].numpy()
            one_hot_actions = np_one_hot(self.actions, dim=32)
            self._value = scaling_constant * (q_values + 1000 * (one_hot_actions - 1)).max()
            return self._value

    def policy_estimate(self, model):
        if self._policy_probabilities is not None:
            return self._policy_probabilities
        with torch.no_grad():
            nn_state = self.current_state.state_for_player(self.active_player).state_for_nn
            nn_state = torch.Tensor(nn_state[None, ...])
            q_values = model(nn_state)[0].numpy()
            one_hot_actions = np_one_hot(self.actions, dim=32)
            self._policy_probabilities = softmax(q_values + 1000 * (one_hot_actions - 1)) * one_hot_actions
            return self._policy_probabilities

    def rewards(self):
        rewards = self.ruleset.final_rewards(self.current_state)
        return rewards - rewards.mean()  # zero sum rewards

    def __hash__(self):
        if self.hashed:
            return self.hashed
        else:
            self.hashed = hash(self.current_state.full_state.data.tobytes())
            return self.hashed

    def __eq__(self, node2):
        return hash(self) == hash(node2)





