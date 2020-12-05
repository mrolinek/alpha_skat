import os
import pickle
import random
import torch

from cluster import cluster_main

import numpy as np

from tqdm import tqdm

from games.simple_ramsch.ruleset import RamschRuleset
from players import get_player

name_to_ruleset = dict(simple_ramsch=RamschRuleset)


class Game(object):
    def __init__(self, name, last_game_final_state):
        self.ruleset = name_to_ruleset[name]
        self.state = self.ruleset.generate_init_state(last_game_final_state)
        self.status_storage = [self.state]

    def play(self, players):
        while not self.ruleset.is_finished(self.state):
            self.state.check_sound()
            player = self.state.active_player
            state_for_player = self.state.state_for_player(player)
            available_actions = self.ruleset.available_actions(self.state)
            if len(available_actions) == 1:
                action = available_actions[0]
            else:
                action = players[player].play(state_for_player, available_actions, self.ruleset)
            self.state = self.ruleset.do_action(self.state, action)
            self.status_storage.append(self.state)

        self.state = self.ruleset.finalize_scores(self.state)
        self.status_storage.append(self.state)
        return self.state

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.status_storage, f)


@cluster_main
def main(player_params, num_games, game_name, working_dir, save_games):
    players = [get_player(**player_params.player_one),
               get_player(**player_params.player_two),
               get_player(**player_params.player_three)]

    total_scores = np.array([0., 0., 0.])
    last_state = None
    for i in tqdm(range(num_games)):
        new_game = Game(game_name, last_state)
        last_state = new_game.play(players)
        total_scores += last_state.current_scores - last_state.current_scores.mean()
        if save_games:
            new_game.save(os.path.join(working_dir, f"game_{i+1}.gm"))

    for i, player in enumerate(players):
        player.save_data(working_dir, player_id=i)
    mean_scores = total_scores / num_games
    result_dict = {f"player_{i+1}_avg_score": score for i, score in enumerate(mean_scores)}
    print(result_dict)
    return result_dict


if __name__ == '__main__':
    main()
