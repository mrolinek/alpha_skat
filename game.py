import os
import pickle
import random
import torch

from cluster import cluster_main

import numpy as np

from tqdm import tqdm

from algorithms.mcts_basic import MCTS, CardGameNode
from games.simple_ramsch.ruleset import RamschRuleset
from train_model import TrainSkatModel
from utils import np_one_hot


name_to_ruleset = dict(simple_ramsch=RamschRuleset)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class RandomPlayer(object):
    def play(self, state, available_actions, ruleset):
        return random.choice(available_actions)

    def save_data(self, working_dir, player_id):
        pass


class NNPlayer(object):
    def __init__(self, checkpoint_path):
        self.model = TrainSkatModel.load_from_checkpoint(checkpoint_path)

    def play(self, state, available_actions, ruleset):
        action_mask = np_one_hot(available_actions, dim=32)

        nn_state = state.nn_state_for_player_view(state.active_player)[None, ...]
        nn_state = torch.Tensor(nn_state)
        probs = self.model(nn_state)[0].data * action_mask
        action = int(torch.max(probs, dim=-1)[1].item())
        assert action in available_actions
        return action

    def save_data(self, working_dir, player_id):
        pass


class FullStateMCTSPlayer(object):
    def __init__(self, num_mcts_rollouts, exploration_weight, softmax_temperature):
        self.softmax_temperature = softmax_temperature
        self.exploration_weight = exploration_weight
        self.num_mcts_rollouts = num_mcts_rollouts
        self.mcts_object = MCTS(self.exploration_weight)

        self.action_masks = []
        self.predicted_probabilities = []
        self.input_states = []

    def collect_data(self, available_actions, learned_values, state):
        action_mask = np_one_hot(available_actions, dim=32)
        probabilities = softmax(self.softmax_temperature * np.array([it[0] for it in learned_values]))
        masked_probabilities = np.zeros_like(action_mask)
        for prob, (logit, action) in zip(probabilities, learned_values):
            masked_probabilities[action] = prob

        self.action_masks.append(action_mask[None, ...])
        self.predicted_probabilities.append(masked_probabilities[None, ...])
        self.input_states.append(state.nn_state_for_player_view(state.active_player)[None, ...])


    def play(self, state, available_actions, ruleset):
        starting_node = CardGameNode(ruleset, state)
        self.mcts_object.reset()

        for i in range(self.num_mcts_rollouts):
            self.mcts_object.do_rollout(starting_node)

        learned_values = self.mcts_object.choose(starting_node)
        learned_values = [(val, child.current_state.current_trick_as_ints[-1]) for (child, val) in learned_values]

        top_action = sorted(learned_values)[-1][1]

        self.collect_data(available_actions, learned_values, state)
        assert top_action in available_actions
        return top_action

    def save_data(self, working_dir, player_id):
        state_file = os.path.join(working_dir, f"inputs_{player_id}.npy")
        all_states = np.concatenate(self.input_states, axis=0)
        np.save(state_file, all_states)
        print(all_states.shape)

        all_masks_file = os.path.join(working_dir, f"masks_{player_id}.npy")
        all_masks = np.concatenate(self.action_masks, axis=0)
        np.save(all_masks_file, all_masks)
        print(all_masks.shape)

        all_probs_file = os.path.join(working_dir, f"probs_{player_id}.npy")
        all_probs = np.concatenate(self.predicted_probabilities, axis=0)
        np.save(all_probs_file, all_probs)
        print(all_probs.shape)





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


player_dict = dict(FullStateMCTSPlayer=FullStateMCTSPlayer, RandomPlayer=RandomPlayer, NNPlayer=NNPlayer)


def spawn_player(name, **kwargs):
    return player_dict[name](**kwargs)


@cluster_main
def main(player_params, num_games, game_name, working_dir, save_games):
    players = [spawn_player(**player_params.player_one),
               spawn_player(**player_params.player_two),
               spawn_player(**player_params.player_three)]

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
