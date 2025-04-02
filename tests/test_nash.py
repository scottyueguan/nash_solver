import pathlib
import pytest
import numpy as np
from nash_solver.solver import NashSolver
from nash_solver.base_game import BaseGame


def get_rps_result(a1, a2):
    """
    Result for player 1 do a1 and player 2 do a2
    :param a1: player 1's action
    :param a2: player 2's action
    :return: result of the game, 0 for draw, 1 for p1 win, -1 for p2 win
    """
    if a1 == 0:
        if a2 == 0:
            return 0
        if a2 == 1:
            return -1
        if a2 == 2:
            return 1
    if a1 == 1:
        if a2 == 0:
            return 1
        if a2 == 1:
            return 0
        if a2 == 2:
            return -1
    if a1 == 2:
        if a2 == 0:
            return -1
        if a2 == 1:
            return 1
        if a2 == 2:
            return 0


seq_rps = BaseGame(n_states=9,
                   n_actions_1=3, n_actions_2=3)
transitions = []
for a1 in range(3):
    transitions.append([])
    for a2 in range(3):
        transition_matrix = np.zeros((9, 9))
        action_result = get_rps_result(a1, a2)
        for s in range(9):
            if s == 0:
                if action_result == 0:
                    transition_matrix[s, s] = 1
                if action_result == 1:
                    transition_matrix[s, 1] = 1
                if action_result == -1:
                    transition_matrix[s, 5] = 1
            elif s in [1, 2, 3]:
                if action_result == 0:
                    transition_matrix[s, s] = 1
                if action_result == 1:
                    transition_matrix[s, s + 1] = 1
                if action_result == -1:
                    transition_matrix[s, 5] = 1
            elif s in [5, 6, 7]:
                if action_result == 0:
                    transition_matrix[s, s] = 1
                if action_result == 1:
                    transition_matrix[s, 1] = 1
                if action_result == -1:
                    transition_matrix[s, s + 1] = 1
            else:
                transition_matrix[s, s] = 1
        transitions[a1].append(transition_matrix)

seq_rps.set_transitions(transitions)

rewards = np.zeros(9)
rewards[4] = 1
rewards[8] = -1
seq_rps.set_rewards(rewards)


def test_sequential_rps():
    # test the nash solver for the sequential rock paper scissor game
    # https://arxiv.org/pdf/2009.00162.pdf
    # a0-R, a1-P, a2-S

    solver = NashSolver(seq_rps, eps=1e-3, verbose=False, n_workers=2)
    solver.solve()
    p1, p2 = solver.policy_1, solver.policy_2

    for s in [0, 1, 2, 3, 5, 6, 7]:
        assert np.allclose(p1[s], np.array([1 / 3, 1 / 3, 1 / 3]))
        assert np.allclose(p2[s], np.array([1 / 3, 1 / 3, 1 / 3]))

def test_compressed_transition():
    #todo: implement test for compressed transition
    pass

def test_policy_eval():
    # todo: test the policy evaluation function
    pass

def test_save_load():
    # test the save and load function of the nash solver
    path = pathlib.Path(__file__).parent / 'test_data'

    solver = NashSolver(seq_rps, eps=1e-3, verbose=False, save_path=path, save_checkpoint=True, n_workers=1)
    solver.solve()
    solver.save(check_point=False)

    solver.load_checkpoint(save_path_model=path / 'model.pkl', save_path_log=path / 'log.pkl')


if __name__ == "__main__":
    pytest.main()
