import numpy as np
from scipy.optimize import linprog

# warnings.filterwarnings('ignore', '.*Ill-conditioned*')
DEBUG = True


def linprog_solve(value_matrix, precision=4):
    '''
    :param value_matrix: M X N Matrix. The entry of the jointly selected row and column represents the winnings of the
                         row player and the loss of the column player
                         i.e. row maximizes and column minimizes
    :param precision: precision of the matrix solver
    :return: policy of the column player, the value of the game
    '''


    value_matrix = -np.nan_to_num(np.round(value_matrix, precision))
    m, n = value_matrix.shape

    # save computation if the value matrix is zeros, meaning that the reward hasn't propagated to the state yet
    if np.allclose(value_matrix, 0):
        return 0, np.ones(m) / m, np.ones(n) / n

    # solve col
    # objectif vector c is 0*x1+0*x2+...+0*xn+v
    C = []
    for i in range(n):
        C.append(0)
    C.append(-1)
    A = []
    for i_row in range(m):
        col = value_matrix[i_row, :]
        constraint_row = []
        for item in col:
            constraint_row.append(-item)
        constraint_row.append(1)
        A.append(constraint_row)
    B = []
    for i in range(m):
        B.append(0)

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

    res = linprog(C, A_ub=A, b_ub=B, A_eq=A_eq, b_eq=B_eq, bounds=bounds, method='highs')
    policy_col = res['x'][:-1]
    for i, p in enumerate(policy_col):
        if p < 0:
            policy_col[i] = 0
    policy_col /= sum(policy_col)

    policy_row = -res.ineqlin.marginals

    return res['fun'], policy_row, policy_col


def linprog_solve_value(value_matrix, precision=4):
    value, _, _ = linprog_solve(value_matrix, precision)
    return value

# def policy_eval()