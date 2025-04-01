import numpy as np
from typing import List, Tuple, Union
from copy import deepcopy


class BaseGame:
    def __init__(self, n_states: Union[int],
                 n_actions_1: Union[List[int], int], n_actions_2: Union[List[int], int],
                 transitions: Union[List[List[np.ndarray]], None] = None,
                 compressed_transition: Union[
                     Tuple[List[List[List[List[int]]]], List[List[List[np.ndarray]]]], None] = None,
                 rewards: np.ndarray = None,
                 terminal_states: List[int] = None, gamma: float = 0.95):
        """
        The base game class used by the Nash solver.
        :param n_states: int, number of (joint) states of the game
        :param n_actions_1: int or List[int], number of actions for player 1 (in each state)
        :param n_actions_2: int or List[int], number of actions for player 2 (in each state)
        :param transitions: List[List[np.ndarray]], transitions[a1][a2] gives the n_states x n_states transition matrix with row sum to 1
        :param compressed_transition:
               compressed_transition[0][a1][a2][s] gives the state indices that the joint state will transition to
               compressed_transition[1][a1][a2][s] gives the probability of transitioning to the corresponding state
        :param rewards: np.ndarray, either 1d or 3d.
               if 1d, rewards[s] gives the reward of state s,
               if 3d, rewards[s, a1, a2] gives the reward of state s with action a1 and a2
        :param terminal_states: List[int], list of terminal states
        :param gamma: float, discount factor in (0,1)
        """
        self._n_states = n_states
        self._n_actions_1 = n_actions_1
        self._n_actions_2 = n_actions_2
        self._transitions = transitions

        self._rewards = rewards
        self.gamma = gamma
        self.terminal_states = terminal_states

        self._transitions_to_state, self._transitions_prob = None, None
        if compressed_transition is not None:
            self.set_compressed_transitions(*compressed_transition)

    def set_rewards(self, rewards: np.ndarray):
        """
        Set the rewards of the game. Either r(s) or r(s, a1, a2)
        :param rewards: np.ndarray, either 1d or 3d.
        :return: None
        """
        assert len(rewards.shape) == 1 or len(rewards.shape) == 3, "rewards should be 1d or 3d"
        self._rewards = rewards

    def set_transitions(self, transitions: List[List[np.ndarray]]):
        """
        Set the transitions of the game.
        :param transitions: List[List[np.ndarray]],
               transitions[a1][a2] gives the n_states x n_states transition matrix with row sum to 1
        :return: None
        """
        for a1 in range(self.get_max_n_action1()):
            for a2 in range(self.get_max_n_action2()):
                assert transitions[a1][a2].shape == (self._n_states, self._n_states), "transitions shape mismatch"
                assert transitions[a1][a2].sum(axis=1).all() == 1, "transitions should sum to 1"
        self._transitions = transitions

    def set_terminal_states(self, terminal_states: List):
        """
        Set the terminal states of the game
        :param terminal_states: List of terminal states
        :return: None
        """
        self.terminal_states = terminal_states

    def set_compressed_transitions(self,
                                   transition_to_state: List[List[List[int]]],
                                   transitions_prob: List[List[List[int]]]):
        """
        Set the compressed transitions of the game
        :param transition_to_state: transition_to_state[a1][a2][s] gives the state indices that the joint state will transition to
        :param transitions_prob: transitions_prob[a1][a2][s] gives the probability of transitioning to the corresponding state
        :return: None
        """
        self._transitions_to_state = deepcopy(transition_to_state)
        self._transitions_prob = deepcopy(transitions_prob)

    @property
    def has_compressed_transition(self):
        return self._transitions_to_state is not None and self._transitions_prob is not None

    def get_transitions_s(self, s: int, a1: int, a2: int) -> np.ndarray:
        """
        Get the transition probability from state s with action a1 and a2
        :param s: state
        :param a1: action 1
        :param a2: action 2
        :return: p_s_prime[s_prime] = P(s_prime | s, a1, a2)
        """
        return self._transitions[a1][a2][s, :]

    def get_compressed_transitions_s(self, s: int, a1: int, a2: int) -> Tuple[List[int], List[np.ndarray]]:
        """
        Get the compressed representation of the transition matrix from state s with action a1 and a2
        :param s: state
        :param a1: action 1
        :param a2: action 2
        :return:
            to_states: a list of state indices that the joint state will transition to
            to_states_prob: a 1d array of probabilities of transitioning to the corresponding state
        """
        assert self._transitions_to_state is not None and self._transitions_prob is not None
        to_states = self._transitions_to_state[a1][a2][s]
        to_states_prob = self._transitions_prob[a1][a2][s]
        return to_states, to_states_prob

    def get_rewards(self, s, a1=None, a2=None) -> float:
        """
        Get the reward of the game
        :param s: state
        :param a1: action 1
        :param a2: action 2
        :return: reward, either state dependent or state-action dependent
        """
        if len(self._rewards.shape) == 3:
            assert a1 is not None and a2 is not None
            return float(self._rewards[s, a1, a2])
        else:
            return float(self._rewards[s])

    def get_n_states(self) -> int:
        return self._n_states

    def get_n_action1(self, state) -> int:
        if isinstance(self._n_actions_1, int):
            return self._n_actions_1
        else:  # state dependent action set
            return self._n_actions_1[state]

    def get_n_action2(self, state) -> int:
        if isinstance(self._n_actions_2, int):
            return self._n_actions_2
        else:  # state dependent action set
            return self._n_actions_2[state]

    def get_max_n_action1(self) -> int:
        if isinstance(self._n_actions_1, int):
            return self._n_actions_1
        else:
            return max(self._n_actions_1)

    def get_max_n_action2(self) -> int:
        if isinstance(self._n_actions_2, int):
            return self._n_actions_2
        else:
            return max(self._n_actions_2)
