import random
from abc import ABC, abstractmethod
from collections import defaultdict
import math


class MCTS(object):
    def __init__(self, exploration_weight):
        self.Q = defaultdict(int)  # total reward of each node
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
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return [(child, score(child)) for child in self.children[node]]

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
        active_players = [path[0].active_player] + [node.active_player for node in path[:-1]]
        for node, active_player in zip(reversed(path), reversed(active_players)):
            self.N[node] += 1
            self.Q[node] += reward[active_player]


    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])
        avgs = [self.Q[n] / self.N[n] for n in self.children[node]]
        rng = max(1.0, max(avgs) - min(avgs))

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + rng*self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

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
    def __hash__(self):
        return None

    @abstractmethod
    def __eq__(self, node2):
        "Nodes must be comparable"
        return True


class CardGameNode(MultiplayerMCTSNode):
    def __init__(self, ruleset, current_state):
        self.ruleset = ruleset
        self.current_state = current_state
        self.active_player = self.current_state.active_player
        self.hashed = None

    def current_player(self):
        return self.active_player

    def find_children(self):
        actions = self.ruleset.available_actions(self.current_state)
        next_states = [self.ruleset.do_action(self.current_state, action) for action in actions]
        return set([CardGameNode(self.ruleset, next_state) for next_state in next_states])

    def find_random_child(self):
        actions = self.ruleset.available_actions(self.current_state)
        next_state = self.ruleset.do_action(self.current_state, random.choice(actions))
        return CardGameNode(self.ruleset, next_state)

    def is_terminal(self):
        return self.ruleset.is_finished(self.current_state)

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





