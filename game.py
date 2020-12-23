import os
import pickle
from cluster import cluster_main

import numpy as np

from tqdm import tqdm

from games.simple_ramsch.ruleset import RamschRuleset
from players import get_player

name_to_ruleset = dict(simple_ramsch=RamschRuleset)


class Game(object):
    counter = 0
    last_initial_state = None

    def __init__(self, name, last_game_final_state, play_all_hand_rotations, players):
        self.ruleset = name_to_ruleset[name]

        if not play_all_hand_rotations or Game.counter % 3 == 0:
            Game.last_initial_state = self.ruleset.generate_init_state(last_game_final_state)
            self.state = Game.last_initial_state
        else:
            Game.last_initial_state = self.ruleset.rotate_init_state(Game.last_initial_state)
            self.state = Game.last_initial_state

        Game.counter += 1
        self.status_storage = [self.state]
        self.players = players

    def play(self):
        while not self.ruleset.is_finished(self.state):
            self.state.check_sound()
            player = self.state.active_player
            state_for_player = self.state.state_for_player(player)
            available_actions = self.ruleset.available_actions(self.state)
            if len(available_actions) == 1:
                action = available_actions[0]
            else:
                action = self.players[player].play(state_for_player, available_actions, self.ruleset)
            self.state = self.ruleset.do_action(self.state, action)
            self.status_storage.append(self.state)

        self.state = self.ruleset.finalize_scores(self.state)
        self.status_storage.append(self.state)
        return self.state

    def save(self, filename):
        names = [p.name for p in self.players]
        with open(filename, 'wb') as f:
            pickle.dump((self.status_storage, names), f)


@cluster_main
def main(player_params, num_games, game_name, working_dir, save_games, play_all_hand_rotations):
    players = [get_player(**player_params.player_one),
               get_player(**player_params.player_two),
               get_player(**player_params.player_three)]

    total_scores = np.array([0., 0., 0.])
    last_state = None
    for i in tqdm(range(num_games)):
        new_game = Game(game_name, last_state, play_all_hand_rotations, players)
        last_state = new_game.play()
        total_scores += last_state.current_scores - last_state.current_scores.mean()
        print("###########################################################")
        print("Scores: ", total_scores[0], total_scores[1], total_scores[2])
        print("###########################################################")
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
