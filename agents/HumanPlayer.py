import numpy as np

from agents.Agent import Agent
from modules.state_display import display_state


class HumanPlayer(Agent):
    """TODO Class to be implemented

    HumanPlayer is an agent that allows a human to test a game. Will implement different state renderings"""

    def play_an_episode(self):
        state = self.env.reset()
        self.env.render()

        action = self.determine_action()
        next_state, reward, done, info = self.env.step(action)
        self.learn()

    def determine_action(self):
        key = input()
        # map keys to int in action space
        return key

    def learn(self):
        pass
