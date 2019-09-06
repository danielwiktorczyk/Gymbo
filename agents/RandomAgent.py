import random

from agents.Agent import Agent, Session, Episode


class RandomAgent(Agent):

    agent_name = "Random Agent"

    def learn(self):  # Does not learn
        pass

    def determine_action(self):
        return random.randint(0, self.env.action_space.n) - 1
