import numpy as np
from scipy.optimize import linprog

# warnings.filterwarnings('ignore', '.*Ill-conditioned*')
DEBUG = True


def linprog_solve(reward_matrix:np.ndarray, precision:int=4)->(float, np.ndarray, np.ndarray):
    '''
    :param reward_matrix: M X N Matrix. The entry of the jointly selected row and column represents the winnings of the
                         row player and the loss of the column player
                         i.e. row maximizes and column minimizes
    :param precision: precision of the matrix solver
    :return: the value of the game, the optimal policy for the row player and the optimal policy for the column player
    '''

    reward_matrix = -np.nan_to_num(np.round(reward_matrix, precision))
    m, n = reward_matrix.shape[0], reward_matrix.shape[1]

    # save computation if the value matrix is zeros, meaning that the reward hasn't propagated to the state yet
    if np.allclose(reward_matrix, 0):
        return 0, np.ones(m) / m, np.ones(n) / n

    # Linear program for solving column player's problem
    # Decision variables y1, y2, ..., yn, v
    # Objective is to minimize v, thus objective vector c is 0*y1+0*y2+...+0*yn+v
    C = []
    for i in range(n):
        C.append(0)
    C.append(-1)

    #   Optimality constraints: value_matrix[i_row, :] @ Y >= v for all i_row
    A = []
    for i_row in range(m):
        col = reward_matrix[i_row, :]
        constraint_row = []
        for item in col:
            constraint_row.append(-item)
        constraint_row.append(1)
        A.append(constraint_row)
    B = []
    for i in range(m):
        B.append(0)

    # Probability constraint: y1 + y2 + ... + yn = 1
    A_eq = []
    A_eq_row = []
    for i in range(n):
        A_eq_row.append(1)
    A_eq_row.append(0)
    A_eq.append(A_eq_row)
    B_eq = [1]

    bounds = []
    for i in range(n):
        bounds.append((0, 1))
    bounds.append((None, None))

    # Linprog solves the problem:
    # Minimize c ^ T * x
    # Subject to:
    #   A_ub * x <= b_ub
    #   A_eq * x == b_eq
    #   l <= x <= u
    res = linprog(C, A_ub=A, b_ub=B, A_eq=A_eq, b_eq=B_eq, bounds=bounds, method='highs')

    # Col player's policy is retrieved from the primal variable
    policy_col = res['x'][:-1]
    for i, p in enumerate(policy_col):
        if p < 0:
            policy_col[i] = 0
    policy_col /= sum(policy_col)

    # Row player's policy is retrieved from the dual variable
    policy_row = -res.ineqlin.marginals

    return res['fun'], policy_row, policy_col


def linprog_solve_value(reward_matrix:np.ndarray, precision:int=4) -> float:
    """
    Wrapper of the linprog_solve to only return the value of the game
    :param reward_matrix: M X N Matrix, row maximizes and column minimizes
    :param precision: precision of the matrix solver
    :return: value of the matrix game
    """
    value, _, _ = linprog_solve(reward_matrix=reward_matrix, precision=precision)
    return value
