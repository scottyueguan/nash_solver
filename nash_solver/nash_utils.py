import numpy as np
from scipy.optimize import linprog


# warnings.filterwarnings('ignore', '.*Ill-conditioned*')
DEBUG=True

def linprog_solver_col(value_matrix, precision=4):
    '''
    :param value_matrix: M X N Matrix. The entry of the jointly selected row and column represents the winnings of the
                         row player and the loss of the column player
                         i.e. row maximizes and column minimizes
    :param precision: precision of the matrix solver
    :return: policy of the column player, the value of the game
    '''

    value_matrix = -np.nan_to_num(np.round(value_matrix, precision))
    m, n = value_matrix.shape

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

    options = {'cholesky': False,
               'sym_pos': False,
               'lstsq': True,
               'presolve': True},

    res = linprog(C, A_ub=A, b_ub=B, A_eq=A_eq, b_eq=B_eq, bounds=bounds)
    policy = res['x'][:-1]
    for i, p in enumerate(policy):
        if p < 0:
            policy[i] = 0
    policy /= sum(policy)
    return policy, res['fun']


def linprog_solver_row(value_matrix, precision=4):
    policy, value = linprog_solver_col(-value_matrix.T, precision)
    return policy, -value


def linprog_solve_value(value_matrix, precision=4):
    _, value = linprog_solver_row(value_matrix, precision=4)
    if DEBUG and np.isnan(value):
        print(value_matrix)
    value = np.nan_to_num(np.round(value, precision))
    return value

def linprog_solve_policy_x(value_matrix, precision=4):
    px, _ = linprog_solver_row(value_matrix, precision)
    return px

def linprog_solve_policy_y(value_matrix, precision=4):
    py, _ = linprog_solver_col(value_matrix, precision)
    return py

def linprog_solve(value_matrix, precision=4):
    value_matrix = np.round(value_matrix, precision)
    py, value = linprog_solver_col(value_matrix, precision)
    px, _ = linprog_solver_row(value_matrix, precision)
    for i, x in enumerate(px):
        if np.fabs(x) < 10e-6:
            px[i] = 0
    for i, y in enumerate(py):
        if np.fabs(y) < 10e-6:
            py[i] = 0
    px = np.divide(px, np.sum(px))
    py = np.divide(py, np.sum(py))
    return value, px, py