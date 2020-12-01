import pickle
import random

from algorithms.mcts_basic import MCTS, CardGameNode
from games.simple_ramsch.ruleset import RamschRuleset

name_to_ruleset = dict(simple_ramsch=RamschRuleset)


class RandomPlayer(object):
    def play(self, state, available_actions):
        return random.choice(available_actions)


class FullStateMCTSPlayer(object):
    def __init__(self, ruleset, num_mcts_rollouts, exploration_weight):
        self.exploration_weight = exploration_weight
        self.ruleset = ruleset
        self.num_mcts_rollouts = num_mcts_rollouts
        self.mcts_object = MCTS(self.exploration_weight)

    def play(self, state, available_actions):
        starting_node = CardGameNode(self.ruleset, state)
        self.mcts_object.reset()

        for i in range(self.num_mcts_rollouts):
            self.mcts_object.do_rollout(starting_node)

        learned_values = self.mcts_object.choose(starting_node)
        learned_values = [(val, child.current_state.current_trick_as_ints[-1]) for (child, val) in learned_values]
        print(learned_values)

        top_action = sorted(learned_values)[-1][1]
        assert top_action in available_actions
        return top_action



class Game(object):
    def __init__(self, name, last_game_final_state):
        self.ruleset = name_to_ruleset[name]
        self.state = self.ruleset.generate_init_state(last_game_final_state)
        self.status_storage = [self.state]

    def play(self, players):
        while not self.ruleset.is_finished(self.state):
            self.state.check_sound()
            player = self.state.active_player
            #state_for_player = self.state.nn_state_for_player_view(player)
            available_actions = self.ruleset.available_actions(self.state)
            if len(available_actions) == 1:
                action = available_actions[0]
            else:
                action = players[player].play(self.state, available_actions)
            print(player, ": ", action)
            self.state = self.ruleset.do_action(self.state, action)
            self.status_storage.append(self.state)

        self.state = self.ruleset.finalize_scores(self.state)
        self.status_storage.append(self.state)
        print(self.state.current_scores)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.status_storage, f)


num_rollouts = 2000
exploration_weight = 10.0



g = Game("simple_ramsch", None)

players = [FullStateMCTSPlayer(num_mcts_rollouts=num_rollouts, exploration_weight=exploration_weight, ruleset=g.ruleset),
           FullStateMCTSPlayer(num_mcts_rollouts=num_rollouts, exploration_weight=exploration_weight, ruleset=g.ruleset),
           FullStateMCTSPlayer(num_mcts_rollouts=num_rollouts, exploration_weight=exploration_weight, ruleset=g.ruleset)]
           #RandomPlayer(),
           #RandomPlayer()]


g.play(players)
g.save("some_game2.gm")
