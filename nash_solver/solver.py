import pathlib

from nash_solver.base_game import BaseGame
import numpy as np
from copy import deepcopy
from nash_solver.nash_utils import linprog_solve, linprog_solve_value
import pickle
import time
from multiprocessing import Pool
from functools import partial


class NashSolver:
    def __init__(self, game: BaseGame,
                 eps: float = 1e-2,
                 save_path: pathlib.Path = None,
                 n_workers: int = 1,
                 verbose: bool = True,
                 save_checkpoint: bool = False):
        """
        Nash equilibrium solver using value iteration and linear programming
        :param game: BaseGame object that provides the necessary information of the game
        :param eps: the difference threshold for convergence. Using l2 norm between two value iterations
        :param save_path: pathlib.Path determines the directory to save the model and log
        :param n_workers: number of worker used for parallelization. Default is 1 for sequential.
        :param verbose: Whether to print out the solver progress: iteration, difference between value iterations, time
        :param save_checkpoint: Whether to save check points to restart the solver recommended for large games
        """
        self.game = game
        self.eps = eps

        # V stores the old value function and V_ stores the updated value function
        self.V, self.V_ = np.zeros(self.game.get_n_states()), np.zeros(self.game.get_n_states())
        # Q stores the Q matrix for each state
        self.Q = [np.zeros((self.game.get_n_action1(s), self.game.get_n_action2(s))) for s in
                  range(self.game.get_n_states())]

        self.policy_1 = [None for _ in range(self.game.get_n_states())]
        self.policy_2 = [None for _ in range(self.game.get_n_states())]
        self.error = []
        self.time = []
        self.verbose = verbose
        self.counter = 0

        self.save_checkpoint = save_checkpoint
        if self.save_checkpoint:
            assert save_path is not None
        self.save_path = save_path

        if save_path is not None and not save_path.exists():
            save_path.mkdir(parents=True)

        self.n_workers = n_workers

    def solve(self, **options):
        np.set_printoptions(precision=options.get("precision", 4))

        n_policy_eval = options.get("n_policy_eval", 0)

        tic = time.time()
        diff = 1000
        self._print("Solving Nash equilibrium of a game with {} states".format(self.game.get_n_states()))
        self._print(f"{'Iter' : <10} {'Difference': <15} {'Time': <10}")
        while diff > self.eps:
            self._policy_eval(n_policy_eval=n_policy_eval)

            self._update_q()

            self._update_v_()

            toc = time.time()

            diff = np.round(np.linalg.norm(self.V_ - self.V), 3)
            self.error.append(diff)
            self.time.append(toc - tic)

            self._update_v()
            self._print(f"{self.counter : <10} {diff: <15} {np.round(toc - tic, 1): <10}")
            self.counter += 1

            if self.counter % 5 == 0 and self.counter > 0 and self.save_checkpoint:
                self.save(check_point=True)

        self._print("Value iterations converged!")
        # self.policy_1, self.policy_2 = self._generate_policy_parallel()

        n_matrix_game_solver_called = int(self.game.get_n_states() * self.counter)
        self._print("Matrix Game solver called {} times".format(n_matrix_game_solver_called))
        return self.policy_1, self.policy_2, self.V, self.Q, n_matrix_game_solver_called

    def _policy_eval(self, n_policy_eval=0):
        for _ in range(n_policy_eval):
            self._update_q()
            self._eval_v()

    def _update_q(self):
        if self.n_workers > 1:
            s_list = list(range(self.game.get_n_states()))
            func = partial(compute_q_s, game=self.game, v=self.V)
            with Pool(self.n_workers) as pool:
                res = pool.map(func, s_list)
            self.Q = list(res)
        else:
            for s in range(self.game.get_n_states()):
                self.Q[s] = compute_q_s(s, self.game, self.V)

    def _eval_v(self):
        if self.policy_1[0] is not None or self.policy_2[0] is not None:
            v = [self.policy_1[s] @ self.Q[s] @ self.policy_2[s] for s in range(self.game.get_n_states())]
            self.V = np.array(v)

    def _update_v_(self):
        if self.n_workers > 1:
            with Pool(self.n_workers) as pool:
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
        self.counter = len(self.error)

    def _print(self, text: str):
        if self.verbose:
            print(text)


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