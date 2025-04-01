import pytest
import numpy as np
from nash_solver.nash_utils import linprog_solve


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
    assert abs(v) < 1e-4
    assert np.linalg.norm(px - np.ones(3) / 3) < 1e-4
    assert np.linalg.norm(py - np.ones(3) / 3) < 1e-4


def test_linprog_bluffing():
    # test the linear program matrix game solver on the bluffing game
    # https://www.dam.brown.edu/people/huiwang/classes/am121/Archive/game_121_2.pdf

    a, b = 2, 5
    r = [
        # Call Fold
        [0, a],  # Bluff
        [(b - a) / 2, 0],  # Not Bluff
    ]

    r = np.array(r)
    v, px, py = linprog_solve(r)

    predicted_value = a * (b - a) / (b + a)
    assert abs(v - predicted_value) < 1e-4
    assert np.linalg.norm(px - np.array([(b - a) / (b + a), 2 * a / (b + a)])) < 1e-4

def test_linprog_battleship():
    # test the linear program matrix game solver on the battleship game
    # https://www.matem.unam.mx/~omar/math340/matrix-games.html
    r = [
        # H1G1 H1G2 H2G1 H2G2
        [0,    2,   -3,     0], # H1G1
        [-2,   0,    0,     3], # H1G2
        [3,    0,    0,    -4], # H2G1
        [0,   -3,    4,     0], # H2G2
        [0,    0,    -3,    3], # H1GS
        [-2,   2,     0,    0], # H2GD
        [3,   -3,     0,    0], # H2GS
        [0,    0,     4,   -4], # H2GD
    ]
    r = np.array(r)
    v, px, py = linprog_solve(r)
    assert abs(v - 4/99) < 1e-4
    assert np.linalg.norm(px - np.array([0., 0.565657, 0.404040, 0., 0.,0.020202, 0., 0.010101])) < 1e-4
    assert np.linalg.norm(py - np.array([0.282828, 0.303030, 0.212121, 0.202020])) < 1e-4


if __name__ == "__main__":
    pytest.main()
