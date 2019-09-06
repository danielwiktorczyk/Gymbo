import random
from sys import getsizeof
from typing import Dict, Any  # For Q_table mapping

import cv2
import numpy as np  # For arrays

from agents.Agent import Agent
from modules.q_learning_methods import greediness_linear, create_noise


class QTableAgent(Agent):
    agent_name = "Q Table Agent"

    q_table: Dict[Any, np.ndarray] = {}  # to contain state - action_value pairs
    greedy_run_list: [] = []

    discount_factor = 0.99  # Discount factor to be applied during the Target updating
    learning_rate = 0.9  # Learning factor
    image_downscaling: bool = True
    image_grayscaling: bool = True
    noise_factor = 500
    noise_exponent = 2
    e_greedy_initial = .8
    e_greedy_final = .9
    e_greedy_step = .0001

    def __init__(self, env):
        super().__init__(env)
        greedy_run_list: [] = []
        for i in range(25, 100000):
            if i % 200 == 0:
                greedy_run_list.extend(range(i, i+25))
        self.greedy_run_list = greedy_run_list

    def determine_action(self):
        """Select an Action with Noise or e-Greedy"""
        if self.latest_episode.number in self.greedy_run_list:
            if np.max(self.action_values(self.latest_state)) != 0:
                self.latest_playback.append("G")
                self.latest_episode.experience += 1
            else:
                self.latest_playback.append("U")
            return self.greedy_action(self.latest_state)
        elif random.uniform(0, 1) < greediness_linear(self.latest_episode.number, self.e_greedy_initial,
                                                      self.e_greedy_final,
                                                      self.e_greedy_step):
            action_values = self.action_values(self.latest_state)
            noise = create_noise(self.env.action_space.n, self.latest_episode.number, self.noise_factor,
                                 self.noise_exponent)
            if np.max(action_values) != 0:
                self.latest_playback.append("G")
                self.latest_episode.experience += 1
            else:
                self.latest_playback.append("U")
            return np.argmax(action_values + noise)
        else:
            return self.env.action_space.sample()

    def learn(self):
        action_values_new_state = self.action_values(self.latest_state)
        next_state_value = np.max(action_values_new_state)
        target = self.latest_reward + self.discount_factor * next_state_value

        state_value = self.action_values(self.latest_state)[self.latest_action]
        self.action_values(self.previous_state)[self.latest_action] = (
                (1 - self.learning_rate) * state_value + (self.learning_rate * target))

    def __str__(self):
        """Print some information about the dictionary contents"""
        print("Size of Q table in bytes : {}".format(getsizeof(self.q_table)))
        print("Size of Q table in kb    : {}".format(getsizeof(self.q_table) / 1000))
        print("Size of Q table in mb    : {}".format(getsizeof(self.q_table) / 1000000))
        print("States in Q table:       : {}".format(len(self.q_table)))

    def initial_action_values(self):
        return np.zeros(self.env.action_space.n, dtype=float)

    def add(self, state):
        state_as_key = self.state_to_key(state)
        self.q_table.update({state_as_key: self.initial_action_values()})

    def action_values(self, state: np.ndarray):
        """Get action values of state. If state not in dictionary, add it"""
        state_as_key = self.state_to_key(state)

        if state_as_key not in self.q_table:
            self.add(state)

        return self.q_table.get(state_as_key)

    def preprocess(self, state: np.ndarray):
        """Preprocessing of a state"""

        if self.image_downscaling:
            state = cv2.resize(state, dsize=(80, 80))

        if self.image_grayscaling:
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

        return state

    def state_to_key(self, state: np.ndarray):
        """Hashing of a state, s(tate)-hash"""

        preprocessed_state = self.preprocess(state)

        return preprocessed_state.tobytes()

    def greedy_action(self, state: np.ndarray) -> int:
        """Returns the greedy action corresponding to a given state. Ties broken randomly"""

        if self.state_to_key(state) not in self.q_table:
            self.add(state)

        action_values = self.q_table[self.state_to_key(state)]

        best_actions = np.argwhere(action_values == np.amax(action_values)).flatten().tolist()

        return random.choice(best_actions)

    def update(self, state: np.ndarray, action_values: np.ndarray):
        self.q_table.update({state: action_values})

    def print_latest_episode(self):
        print("{:15} | "
              "Episode {:>3} :  "
              "Score | {:>6} "
              "Time | {:04.2f} "
              "Steps | {:4d} "
              "Cached | {:6d} "
              "Greediness | {:04.2f} %"
              "{}".format(self.agent_name, self.latest_episode.number,
                          sum(self.latest_episode.rewards),
                          self.latest_episode.runtime, self.latest_episode.number_of_steps,
                          len(self.q_table),
                          float(self.latest_episode.experience) / float(self.latest_episode.number_of_steps) * 100.0,
                          "   !Greedy Run!" if self.latest_episode.number in self.greedy_run_list else ""))
