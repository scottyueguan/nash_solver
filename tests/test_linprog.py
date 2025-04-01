import pytest
import numpy as np
from nash_solver.nash_utils import linprog_solver_col, linprog_solver_row, linprog_solve


def test_linprog_rps():
    # test the linear program matrix game solver on the rock-paper-scissors game
    # https://cs.stanford.edu/people/eroberts/courses/soco/projects/1998-99/game-theory/psr.html
    r = [
        # R  P  S
        [0, 1, -1],  # R
        [-1, 0, 1],  # P
        [1, -1, 0],  # S
    ]
    r = np.array(r)
    v, px, py = linprog_solve(r)
    px_row, v_row = linprog_solver_row(r)
    py_col, v_col = linprog_solver_col(r)
    assert max(abs(v), abs(v_row), abs(v_col)) < 1e-4
    assert np.linalg.norm(px - np.ones(3) / 3) == np.linalg.norm(px_row - np.ones(3) / 3) < 1e-4
    assert np.linalg.norm(py - np.ones(3) / 3) == np.linalg.norm(py_col - np.ones(3) / 3) < 1e-4


def test_linprog_bluffing():
    # test the linear program matrix game solver on the bluffing game
    #https://www.dam.brown.edu/people/huiwang/classes/am121/Archive/game_121_2.pdf

    a, b= 2, 5
    r = [
        # Call Fold
        [0, a],  # Bluff
        [(b-a)/2, 0],  # Not Bluff
    ]
    r = np.array(r)
    v, px, py = linprog_solve(r)
    px_row, v_row = linprog_solver_row(r)
    py_col, v_col = linprog_solver_col(r)

    predicted_value = a * (b-a)/(b+a)
    assert max(abs(v - predicted_value), abs(v_row - predicted_value), abs(v_col - predicted_value)) < 1e-4
    assert np.linalg.norm(px - np.array([(b-a)/(b+a), 2*a/(b+a)])) < 1e-4


if __name__ == "__main__":
    pytest.main()
