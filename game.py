import os
import pickle
import random

from cluster import cluster_main

import numpy as np

from tqdm import tqdm

from algorithms.mcts_basic import MCTS, CardGameNode
from games.simple_ramsch.ruleset import RamschRuleset

name_to_ruleset = dict(simple_ramsch=RamschRuleset)


class RandomPlayer(object):
    def play(self, state, available_actions, ruleset):
        return random.choice(available_actions)


class FullStateMCTSPlayer(object):
    def __init__(self, num_mcts_rollouts, exploration_weight):
        self.exploration_weight = exploration_weight
        self.num_mcts_rollouts = num_mcts_rollouts
        self.mcts_object = MCTS(self.exploration_weight)

    def play(self, state, available_actions, ruleset):
        starting_node = CardGameNode(ruleset, state)
        self.mcts_object.reset()

        for i in range(self.num_mcts_rollouts):
            self.mcts_object.do_rollout(starting_node)

        learned_values = self.mcts_object.choose(starting_node)
        learned_values = [(val, child.current_state.current_trick_as_ints[-1]) for (child, val) in learned_values]
        #print(learned_values)

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
                action = players[player].play(self.state, available_actions, self.ruleset)
            self.state = self.ruleset.do_action(self.state, action)
            self.status_storage.append(self.state)

        self.state = self.ruleset.finalize_scores(self.state)
        self.status_storage.append(self.state)
        return self.state

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.status_storage, f)


player_dict = dict(FullStateMCTSPlayer=FullStateMCTSPlayer, RandomPlayer=RandomPlayer)


def spawn_player(name, **kwargs):
    return player_dict[name](**kwargs)


@cluster_main
def main(player_params, num_games, game_name, working_dir, save_games):
    players = [spawn_player(**player) for player in player_params]

    total_scores = np.array([0., 0., 0.])
    last_state = None
    for i in tqdm(range(num_games)):
        new_game = Game(game_name, last_state)
        last_state = new_game.play(players)
        total_scores += last_state.current_scores
        if save_games:
            new_game.save(os.path.join(working_dir, f"game_{i+1}.gm"))

    mean_scores = total_scores / num_games
    result_dict = {f"player_{i+1}_avg_score": score for i, score in enumerate(mean_scores)}
    print(result_dict)
    return result_dict


if __name__ == '__main__':
    main()
