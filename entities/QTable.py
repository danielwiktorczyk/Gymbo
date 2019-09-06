import random
from sys import getsizeof
from typing import Dict, Any  # For Q_table mapping

import cv2
import numpy as np  # For arrays


class QTable:
    dictionary: Dict[Any, np.ndarray]  # to contain state - action_value pairs
    image_downscaling: bool
    image_grayscaling: bool
    action_count: int

    def __init__(self, image_downscaling: bool, image_grayscaling: bool, action_count: int):
        self.image_downscaling = image_downscaling
        self.image_grayscaling = image_grayscaling
        self.action_count = action_count
        self.dictionary = {}

    def print_statistics(self):
        """Print some information about the dictionary contents"""

        print("Size of Q table in bytes : {}".format(getsizeof(self.dictionary)))
        print("Size of Q table in kb    : {}".format(getsizeof(self.dictionary) / 1000))
        print("Size of Q table in mb    : {}".format(getsizeof(self.dictionary) / 1000000))
        print("States in Q table:       : {}".format(len(self.dictionary)))

    def initial_action_values(self):
        return np.zeros(self.action_count, dtype=float)

    def add(self, state):
        state_as_key = self.state_to_key(state)
        self.dictionary.update({state_as_key: self.initial_action_values()})

    def action_values(self, state: np.ndarray):
        """Get action values of state. If state not in dictionary, add it"""
        state_as_key = self.state_to_key(state)

        if state_as_key not in self.dictionary:
            self.add(state)

        return self.dictionary.get(state_as_key)

    def preprocess(self, state: np.ndarray):
        """Preprocessing of a state"""

        # if self.image_downscaling:
        #     state = cv2.resize(state, dsize=(80, 80))

        if self.image_grayscaling:
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

        return state

    def state_to_key(self, state: np.ndarray):
        """Hashing of a state, s(tate)-hash"""

        preprocessed_state = self.preprocess(state)

        return preprocessed_state.tobytes()

    def greedy_action(self, state: object) -> object:
        """Returns the greedy action corresponding to a given state. Ties broken randomly"""

        if self.state_to_key(state) not in self.dictionary:
            self.add(state)

        action_values = self.dictionary[self.state_to_key(state)]

        best_actions = np.argwhere(action_values == np.amax(action_values)).flatten().tolist()

        return random.choice(best_actions)

    def update(self, state: np.ndarray, action_values: np.ndarray):
        self.dictionary.update({state: action_values})
