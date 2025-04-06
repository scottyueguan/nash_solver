import pathlib

from nash_solver.base_game import BaseGame
import numpy as np
from copy import deepcopy
from nash_solver.nash_utils import linprog_solve
import pickle
import time
from multiprocessing import Pool
from functools import partial
from typing import List


class NashSolver:
    def __init__(self, game: BaseGame):
        """
        Nash equilibrium solver using value iteration and linear programming.
        Player 1 (row) maximizes while Player 2 (column) minimizes.
        :param game: BaseGame object that provides the necessary information of the game
        """
        self.game = game

        # V stores the old value function and V_ stores the updated value function
        self.V, self.V_ = np.zeros(self.game.get_n_states()), np.zeros(self.game.get_n_states())
        # Q stores the Q matrix for each state
        self.Q = [np.zeros((self.game.get_n_action1(s), self.game.get_n_action2(s))) for s in
                  range(self.game.get_n_states())]

        # store the policy as a list of numpy arrays, such that policy[s] gives the action distribution
        self.policy_1 = [None for _ in range(self.game.get_n_states())]
        self.policy_2 = [None for _ in range(self.game.get_n_states())]

        # initialize logging variables
        self.iter_counter = 0
        self.error = []
        self.time = []

        # saving path
        self.save_path = None

    def solve(self,
              eps: float = 1e-3,
              n_policy_eval: int = 0,
              n_workers: int = 1,
              save_path: pathlib.Path = None,
              save_checkpoint: bool = False,
              verbose: bool = False) -> (List[np.ndarray], List[np.ndarray], np.ndarray, List[np.ndarray]):
        """
        Solve the Nash equilibrium of the zero-sum stochastic game using value iteration and linear programming
        :param eps: float, the convergence threshold of the l2-norm difference between old and new value function. `
        :param n_policy_eval: int, the number of policy evaluation step before the LP step. Default 0 to skip.
        :param n_workers: int, the number of workers to use for parallel computation. Default 1 for no parallelization.
        :param save_path: Path to save the model and log. If None, no saving is performed.
        :param save_checkpoint: Bool, whether to save the model and log at each iteration. Default False.
                                Recommended for large games in case solver crashed.
        :param verbose: Bool, whether to print the log. Default False.
        :return: Policy for player 1, Policy for player 2, Value function, Q function
        """

        self._initialize_saving(save_path, save_checkpoint)

        tic = time.time()
        diff = 1000
        self._print("Solving Nash equilibrium of a game with {} states".format(self.game.get_n_states()), verbose)
        self._print(f"{'Iter' : <10} {'Difference': <15} {'Time': <10}", verbose)

        if n_workers > 1:
            pool = Pool(n_workers)
        else:
            pool = None

        while diff > eps:
            # update value function with the current policies
            self._policy_eval(n_policy_eval=n_policy_eval, pool=pool)

            # update the q function with the current value function
            self._update_q(pool=pool)

            # compute the new value function with the nash matrix game LP solver
            self._update_v_(pool=pool)

            toc = time.time()

            diff = np.round(np.linalg.norm(self.V_ - self.V), 4)
            self.error.append(diff)
            self.time.append(toc - tic)

            # copy the new value function to the old value function
            self._update_v()

            self._print(f"{self.iter_counter : <10} {diff: <15} {np.round(toc - tic, 4): <10}", verbose)
            self.iter_counter += 1

            if self.iter_counter % 5 == 0 and self.iter_counter > 0 and self.save_checkpoint:
                self.save(check_point=True)

        self._print("Value iterations converged!", verbose)
        # self.policy_1, self.policy_2 = self._generate_policy_parallel()

        n_matrix_game_solver_called = int(self.game.get_n_states() * self.iter_counter)
        self._print("Matrix Game solver called {} times".format(n_matrix_game_solver_called), verbose)
        return self.policy_1, self.policy_2, self.V, self.Q

    def _policy_eval(self, n_policy_eval: int, pool: Pool=None):
        # evaluate the value function under the current policy to help speed up the convergence.
        # if the n_policy_eval is 0, then this step is skipped.
        for _ in range(n_policy_eval):
            self._update_q(pool=pool)
            self._eval_v()

    def _update_q(self, pool:Pool=None):
        if pool is not None:
            s_list = list(range(self.game.get_n_states()))
            func = partial(compute_q_s, game=self.game, v=self.V)
            res = pool.map(func, s_list)
            self.Q = list(res)
        else:
            for s in range(self.game.get_n_states()):
                self.Q[s] = compute_q_s(s, self.game, self.V)

    def _eval_v(self):
        if self.policy_1[0] is not None or self.policy_2[0] is not None:
            v = [self.policy_1[s] @ self.Q[s] @ self.policy_2[s] for s in range(self.game.get_n_states())]
            self.V = np.array(v)

    def _update_v_(self, pool:Pool=None):
        if pool is not None:
            results = pool.map(linprog_solve, [self.Q[s] for s in range(self.game.get_n_states())])
            self.V_ = np.array([res[0] for res in results])
            self.policy_1 = [res[1] for res in results]
            self.policy_2 = [res[2] for res in results]
        else:
            for s in range(self.game.get_n_states()):
                self.V_[s], self.policy_1[s], self.policy_2[s] = linprog_solve(self.Q[s])

    def _update_v(self):
        self.V = deepcopy(self.V_)

    def get_policy(self):
        return self.policy_1, self.policy_2

    def get_v(self):
        return self.V

    def get_q(self):
        return self.Q

    def save(self, check_point=False):
        save_path = self.save_path
        self.save_policy(save_path, check_point)
        self.save_model(save_path, check_point)
        self.save_log(save_path, check_point)

    def save_policy(self, save_path: pathlib.Path, check_point):
        if not check_point:
            with open(save_path / "policy.pkl", "wb") as f:
                data = [self.policy_1, self.policy_2]
                pickle.dump(data, f)

    def save_model(self, save_path: pathlib.Path, check_point):
        if check_point:
            file_name = save_path / "model_check_point.pkl"
        else:
            file_name = save_path / "model.pkl"
        with open(file_name, "wb") as f:
            data = [self.V, self.Q]
            pickle.dump(data, f)

    def save_log(self, save_path: pathlib.Path, check_point):
        if check_point:
            file_name = save_path / "log_check_point.pkl"
        else:
            file_name = save_path / "log.pkl"
        with open(file_name, "wb") as f:
            data = [self.error, self.time]
            pickle.dump(data, f)

    def load_checkpoint(self, save_path_model: pathlib.Path, save_path_log: pathlib.Path):
        with open(save_path_model, "rb") as f_model:
            self.V, self.Q = deepcopy(pickle.load(f_model))
        with open(save_path_log, "rb") as f_log:
            self.error, self.time = deepcopy(pickle.load(f_log))

        self.V_ = deepcopy(self.V)
        self.iter_counter = len(self.error)

    def _print(self, text: str, verbose: bool = True):
        if verbose:
            print(text)

    def _initialize_saving(self, save_path: pathlib.Path, save_checkpoint: bool):
        self.save_checkpoint = save_checkpoint
        if self.save_checkpoint:
            assert save_path is not None, "Requested to save checkpoints, but no save path provided!"
        self.save_path = save_path

        if save_path is not None and not save_path.exists():
            save_path.mkdir(parents=True)


def compute_q_s(s: int, game: BaseGame, v: np.ndarray):
    q = np.zeros((game.get_n_action1(s), game.get_n_action2(s)))
    for a1 in range(game.get_n_action1(s)):
        for a2 in range(game.get_n_action2(s)):
            if game.has_compressed_transition:
                to_states, to_states_prob = game.get_compressed_transitions_s(s, a1, a2)
                q[a1, a2] = game.get_rewards(s, a1, a2) + game.gamma * np.array(to_states_prob).dot(
                    np.array(v[to_states]))
            else:
                T_s = game.get_transitions_s(s=s, a1=a1, a2=a2)
                q[a1, a2] = game.get_rewards(s, a1, a2) + game.gamma * T_s.dot(v)
    return q
