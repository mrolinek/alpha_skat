import random
from itertools import islice
from pysat.solvers import Minicard
import numpy as np


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




def solve_sat_for_init_hands(public_state_array, num_solutions):
    solver = Minicard()
    add_basic_clauses(solver)

    positives = np.argwhere(public_state_array == 1)
    positive_literals = [int(1+ 32*i+j) for (i,j) in positives]


    negatives = np.argwhere(public_state_array == -1)
    negative_literals = [int(-(1 + 32 * i + j)) for (i, j) in negatives]

    solutions = solver.enum_models(assumptions=positive_literals+negative_literals)
    sols = list(islice(solutions, 1000))  # High number to ensure sufficient randomness

    if len(sols) > num_solutions:
        sols = random.sample(sols, num_solutions)

    def sol_to_numpy(sol):
        np_sol = np.zeros(shape=32 * 4, dtype=np.int16)
        np_sol[[i - 1 for i in sol if i > 0]] = 1
        return np_sol.reshape(4, 32)

    result = [sol_to_numpy(sol) for sol in sols]
    return result
