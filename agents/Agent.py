import time
from abc import abstractmethod
from typing import List

import numpy as np
from gym.wrappers import TimeLimit


class Episode:
    number: int
    rewards: List[float]
    runtime: float
    experience: int
    number_of_steps: int

    def __init__(self):
        self.number = 0
        self.rewards = []
        self.runtime = 0.0
        self.experience = 0
        self.number_of_steps = 0


class Session:
    episodes: List[type('Episode')]
    run_times: List[float]
    rewards: List[float]

    def __init__(self):
        self.episodes = []
        self.run_times = []
        self.rewards = []


class Agent:
    """
    An agent has its own policy for determining actions and updating itself based on experience.
    It interacts with a session object
    """

    agent_name: str = "Generic Agent"

    env: TimeLimit
    session: Session
    latest_episode: Episode

    latest_state: np.ndarray
    previous_state: np.ndarray
    latest_action: int = -1
    previous_action: int = -1
    latest_reward: float = 0.0
    episode_currently_done: bool = False
    latest_info: object
    latest_playback: []

    def __init__(self, env):
        self.env = env
        self.session = Session()
        self.latest_episode = Episode()
        self.latest_playback = []

    def play_an_episode(self):
        episode = Episode()
        episode.number = self.latest_episode.number + 1
        self.latest_episode = episode
        self.latest_episode.rewards = []
        self.latest_episode.experience = 0
        self.latest_state = self.env.reset()
        self.previous_state = self.latest_state
        self.latest_action = -1
        self.previous_action = -1
        self.latest_reward = 0.0
        self.episode_currently_done = False
        self.latest_info = 0
        self.latest_playback = []
        self.latest_episode.runtime = 0.0
        done = False
        runtime_start = time.time()

        while not done:
            self.latest_episode.number_of_steps += 1

            "Determine an action"
            action = self.determine_action()
            self.previous_action = self.latest_action
            self.latest_action = action

            "Apply action, get results"
            state, reward, done, info = self.env.step(action)
            self.previous_state = self.latest_state
            self.latest_state = state
            self.latest_reward = reward
            self.latest_episode.rewards.append(reward)
            self.episode_currently_done = done
            self.latest_info = info

            "Learn from results"
            self.learn()

        self.session.episodes.append(self.latest_episode)
        self.session.rewards.append(sum(self.latest_episode.rewards))
        self.latest_episode.runtime = time.time() - runtime_start

        self.print_latest_episode()

    @abstractmethod
    def determine_action(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    def __str__(self):
        return self.agent_name

    def compare_performance_to(self, other_agent: 'Agent', interval: int):
        this_average_performance = np.mean(self.session.rewards[-1000:])
        other_agent_performance = np.mean(other_agent.session.rewards[-1000:])
        performance_ratio: float
        if other_agent_performance > 0:
            performance_ratio = this_average_performance / other_agent_performance * 100
        else:
            performance_ratio = 100.0

        this_average_100_episode_performance = np.mean(self.session.rewards[-interval:])
        other_agent_100_episode_performance = np.mean(other_agent.session.rewards[-interval:])
        interval_ratio: float
        if other_agent_performance > 0:
            interval_ratio = this_average_100_episode_performance / other_agent_100_episode_performance * 100.0
        else:
            interval_ratio = 100.0

        print("Comparing the {} to the {}: \n"
              .format(self.agent_name, other_agent.agent_name))
        print("Last 1000 Episode Average Performances are {:04.2f} and {:04.2f} ({:04.2f}%)"
              .format(this_average_performance,
                      other_agent_performance,
                      performance_ratio))
        print("Last {} Episode Average Performances are {:04.2f} and {:04.2f} ({:04.2f}%)"
              .format(interval,
                      this_average_100_episode_performance,
                      other_agent_100_episode_performance,
                      interval_ratio))

    def print_latest_episode(self):
        print("{:15} | "
              "Episode {:>3} :  "
              "Score | {:>6} "
              "Time | {:04.2f} "
              "Steps | {:4d}".format(self.agent_name, self.latest_episode.number, sum(self.latest_episode.rewards),
                                     self.latest_episode.runtime, self.latest_episode.number_of_steps))
