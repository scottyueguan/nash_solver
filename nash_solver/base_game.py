import numpy as np
from typing import List, Tuple, Union
from copy import deepcopy


class BaseGame:
    def __init__(self, n_states: Union[int],
                 n_actions_1: Union[List[int], int], n_actions_2: Union[List[int], int], gamma: float = 0.95):
        """
        The base game class used by the Nash solver.
        :param n_states: int, number of (joint) states of the game
        :param n_actions_1: int or List[int], number of actions for player 1 (in each state)
        :param n_actions_2: int or List[int], number of actions for player 2 (in each state)
        :param gamma: float, discount factor in (0,1)
        """
        self._n_states = n_states
        self._n_actions_1 = n_actions_1
        self._n_actions_2 = n_actions_2
        self.gamma = gamma

        self._rewards = None
        self._terminal_states = None
        self._transitions = None
        self._transitions_to_state, self._transitions_prob = None, None
        self._padded_transitions_to_state, self._padded_transitions_prob = None, None

    def set_rewards(self, rewards: np.ndarray):
        """
        Set the rewards of the game. Either r(s) or r(s, a1, a2)
        :param rewards: np.ndarray, either 1d or 3d.
               if 1d, rewards[s] gives the reward of state s,
               if 3d, rewards[s, a1, a2] gives the reward of state s with action a1 and a2
        :return: None
        """
        assert len(rewards.shape) == 1 or len(rewards.shape) == 3, "rewards should be 1d or 3d"
        if len(rewards.shape) == 1:
            rewards = rewards.reshape((self._n_states, 1, 1))
        self._rewards = rewards

    def set_terminal_states(self, terminal_states: List):
        """
        Set the terminal states of the game
        :param terminal_states: List[int], list of terminal states
        :return: None
        """
        self._terminal_states = deepcopy(terminal_states)

    def set_transitions(self, transitions: List[List[np.ndarray]]):
        """
        Set the transitions of the game.
        If action space is state dependent, one should use max number of actions as the size of the lists.
        :param transitions: List[List[np.ndarray]],
               transitions[a1][a2] gives the n_states x n_states transition matrix with row sum to 1
        :return: None
        """
        for a1 in range(self.get_max_n_action1()):
            for a2 in range(self.get_max_n_action2()):
                assert transitions[a1][a2].shape == (self._n_states, self._n_states), "transitions shape mismatch"
                assert transitions[a1][a2].sum(axis=1).all() == 1, "transitions should sum to 1"
        self._transitions = transitions

    def set_compressed_transitions(self,
                                   transition_to_state: List[List[List[List[int]]]],
                                   transitions_prob: List[List[List[List[int]]]]):
        """
        Set the compressed transitions of the game
        :param transition_to_state: transition_to_state[s][a1][a2] gives the state indices that the joint state will transition to
        :param transitions_prob: transitions_prob[s][a1][a2] gives the probability of transitioning to the corresponding state
        :return: None
        """
        assert len(transition_to_state) == self._n_states and len(transitions_prob) == self._n_states, \
            ("Compressed transitions size error: "
             "use transitions_prob[s][a1][a2] to store the probabilities of transitioning to the corresponding state.")
        self._transitions_to_state = deepcopy(transition_to_state)
        self._transitions_prob = deepcopy(transitions_prob)

        self._generate_padded_compressed_transitions()

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
        assert self.has_compressed_transition, "Compressed transitions not set for the game!"
        to_states = self._transitions_to_state[s][a1][a2]
        to_states_prob = self._transitions_prob[s][a1][a2]
        return to_states, to_states_prob

    def get_all_compressed_transitions_s(self, s: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get compressed transitions from state s for all actions
        :param s: state
        :return:
            to_states: ndarray with to_states[a1, a2] gives the state indices that the joint state will transition to
            to_states_prob: ndarray with to_states_prob[a1, a2] gives the probability of transitioning
        """
        assert self.has_compressed_transition, "Compressed transitions not set for the game!"
        to_states = np.array(self._transitions_to_state[s]).astype(int)
        to_states_prob = np.array(self._transitions_prob[s])
        return to_states, to_states_prob

    def get_all_transitions_s(self, s: int) -> np.ndarray:
        """
        Get transitions from state s for all actions
        :param s: state
        :return: transitions [a1, a2] gives the transition vector that sum to 1
        """
        trans_prob = np.array(self._transitions)[:, :, s, :]
        return trans_prob

    def get_all_transitions(self) -> np.ndarray:
        """
        Get transitions for all states
        :return: transitions [s, a1, a2] gives the transition vector that sum to 1
        """
        return np.array(self._transitions).transpose((3, 0, 1, 2))

    def get_padded_transitions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the padded transitions for all states size with (S, MAX_A1, MAX_A2, MAX_NEXT_STATE)
        :return:
            to_states: ndarray with to_states[s, a1, a2] gives the state indices that the joint state will transition, padded with -1
            to_states_prob: ndarray with to_states_prob[s, a1, a2] gives the probability of transitioning, padded with -1
        """
        return self._padded_transitions_to_state, self._padded_transitions_prob

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

    def get_all_rewards_s(self, s) -> Union[np.ndarray, float]:
        if len(self._rewards.shape) == 3:
            return self._rewards[s, :, :]
        else:
            return self._rewards[s]

    def get_all_rewards(self) -> Union[np.ndarray, float]:
        """
        Get rewards at state s for all actions
        :return: SXA1XA2 ndarray with rewards R[s, a1, a2] gives the reward of state s with action a1 and a2
            or   S ndarray if reward is not action dependent
        """
        return self._rewards


    def is_terminal(self, state: int) -> bool:
        if self._terminal_states is not None:
            return state in self._terminal_states
        else:
            return False

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

    def _generate_padded_compressed_transitions(self):
        n_states = self._n_states
        n_a1_max = self.get_max_n_action1()
        n_a2_max = self.get_max_n_action2()
        n_next_state_max = max([len(self._transitions_to_state[s][a1][a2]) for s in range(n_states)
                                for a1 in range(self.get_n_action1(s)) for a2 in range(self.get_n_action2(s))])

        self._padded_transitions_to_state = np.full((n_states, n_a1_max, n_a2_max, n_next_state_max), 0, dtype=int)
        self._padded_transitions_prob = np.full((n_states, n_a1_max, n_a2_max, n_next_state_max), 0, dtype=float)
        for s in range(n_states):
            n_a1, n_a2 = self.get_n_action1(s), self.get_n_action2(s)
            for a1 in range(n_a1):
                for a2 in range(n_a2):
                    to_states, to_states_prob = self.get_compressed_transitions_s(s, a1, a2)
                    n_next = len(to_states)
                    self._padded_transitions_to_state[s, a1, a2, :n_next] = to_states
                    self._padded_transitions_prob[s, a1, a2, :n_next] = to_states_prob
