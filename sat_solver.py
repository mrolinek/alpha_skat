import random
from itertools import islice
from pysat.solvers import Minicard
import numpy as np

from numba import njit, int16

from utils import np_one_hot, softmax


def add_equal(solver, literals, k):
    solver.add_atmost(literals, k=k)
    solver.add_atmost([-l for l in literals], k=len(literals) - k)


def add_basic_clauses(solver):
    for i in range(1, 33):
        add_equal(solver, [i, i+32, i+64, i+96], k=1)

    add_equal(solver, list(range(1, 33)), k=10)
    add_equal(solver, list(range(33, 65)), k=10)
    add_equal(solver, list(range(65, 97)), k=10)
    add_equal(solver, list(range(97, 129)), k=2)


#@njit(int16[:,:](int16[:]))
def sol_to_numpy(sol):
    np_sol = np.zeros(shape=32 * 4, dtype=np.int16)
    np_sol[[i - 1 for i in sol if i > 0]] = 1
    return np_sol.reshape(4, 32)


def solve_sat_for_init_hands(public_state_array, num_solutions):
    assert public_state_array.shape == (4, 32), public_state_array.shape
    solver = Minicard()
    add_basic_clauses(solver)

    positives = np.argwhere(public_state_array == 1)
    positive_literals = [int(1+ 32*i+j) for (i,j) in positives]


    negatives = np.argwhere(public_state_array == -1)
    negative_literals = [int(-(1 + 32 * i + j)) for (i, j) in negatives]

    solutions = solver.enum_models(assumptions=positive_literals+negative_literals)
    sols = list(islice(solutions, 2000))  # High number to ensure sufficient randomness

    if len(sols) > num_solutions:
        sols = random.sample(sols, num_solutions)

    result = [sol_to_numpy(sol) for sol in sols]
    return result


def top_k_likely_hands(ruleset, current_state, k, policy_model, init_hands_to_sample=200, epsilon=1e-4):
    top_candidates = solve_sat_for_init_hands(current_state.implications, init_hands_to_sample)
    assert top_candidates
    num_sampled = len(top_candidates)

    init_states = [current_state.recover_init_state(initial_hands) for initial_hands in top_candidates]
    action_sequence = current_state.actions_taken

    all_states, all_actions, all_masks = [], [], []

    actions_per_init_state = None
    for init_state in init_states:
        added_states = 0
        state = init_state
        for action in action_sequence:
            available_actions = ruleset.available_actions(state)
            assert action in available_actions
            if state.active_player != current_state.active_player:
                all_states.append(state)
                all_actions.append(action)
                all_masks.append(available_actions)
                added_states += 1
            state = ruleset.do_action(state, action)

        actions_per_init_state = actions_per_init_state or added_states
        assert actions_per_init_state == added_states  # Every init hand has equal number of actions to evaluate
        actions_per_init_state = added_states

    if not all_states:
        if len(top_candidates) > k:
            top_candidates = random.sample(top_candidates, k)
        return top_candidates

    # Run NN
    nn_states = [state.state_for_player(state.active_player).state_for_nn[None, ...] for state in all_states]
    nn_states = np.concatenate(nn_states, axis=0)
    all_masks_numpy = np.concatenate([np_one_hot(mask, 32)[None, ...] for mask in all_masks], axis=0)
    policy_logits, value = policy_model.get_policy_and_value(nn_states)
    assert policy_logits.shape == all_masks_numpy.shape

    # Collect probabilities of init_hands
    policy_probabilities = softmax(policy_logits + 1000*(all_masks_numpy - 1))
    log_probabilities = np.log(policy_probabilities + epsilon)
    log_probs_of_taken_actions = log_probabilities[np.arange(len(all_actions)), np.array(all_actions)]
    log_probs_by_init_state = log_probs_of_taken_actions.reshape((num_sampled, actions_per_init_state))
    log_probs_by_init_state = np.sum(log_probs_by_init_state, axis=1)
    probabilities_of_init_hands = softmax(log_probs_by_init_state)

    sampled_indices = np.random.choice(np.arange(len(top_candidates)), size=k,
                                       replace=True, p=probabilities_of_init_hands)
    sampled_hands = [top_candidates[int(i)] for i in sampled_indices]


    # Compute and return top_k
    #top_k_init_state_indices = np.argsort(-log_probs_by_init_state)[:k]
    #top_k_init_hands = [top_candidates[i] for i in top_k_init_state_indices]
    print("Guessed hands:")
    for hand in sampled_hands:
        print(hand)
    return sampled_hands





