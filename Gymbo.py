"""This is the updated version of the Q-Learning agent for the Gym environment (Q-Bert)"""

"""TODO try adding -loss of life reward (-100 points)-gameover reward (-500 points)"""

import gym  # The RL environment with Atari support

from agents.QTableAgent import QTableAgent
from agents.RandomAgent import RandomAgent


def main():
    """Running parameters"""
    atari_game: str = "Qbert-v4"
    # atari_game: str = "Pong-v4"
    number_of_episodes: int = 20000  # Number of episodes to run
    episode_number: int = 1
    env = gym.make(atari_game)

    "Create agents"
    random_agent: RandomAgent = RandomAgent(env)
    q_table_agent: QTableAgent = QTableAgent(env)

    while episode_number <= number_of_episodes:
        random_agent.play_an_episode()
        q_table_agent.play_an_episode()
        print()

        if episode_number % 100 == 0:
            q_table_agent.compare_performance_to(random_agent, 100)
            print()

        episode_number += 1

    print(q_table_agent)


if __name__ == '__main__':
    main()
