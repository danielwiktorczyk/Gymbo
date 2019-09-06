"""

This is the updated version of the Q-Learning agent for the Gym environment (Q-Bert)

Q-Learning should not be the optimal approach to this, but let's explore how effective it can be for ourselves

"""

from typing import Any, List  # For Q_table mapping

import random
import time
import gym  # The RL environment with Atari support
import numpy as np  # For arrays

from entities.QTable import QTable
from modules.q_learning_methods import create_noise, greediness_linear
from modules.state_display import display_state


def main():
    """Running parameters"""
    episode_number: int = 1  # Starting episode number
    episodes = 2000  # Number of episodes to run
    display = False  # Display the different preprocessed states
    atari_game = "Qbert-v4"
    # atari_game = "Qbert-v4"
    max_steps = 6000
    loss_of_life_knowledge = True
    loss_of_life_reward = -100

    """Agent information"""
    discount_factor = 0.99  # Discount factor to be applied during the Target updating
    learning_rate = 0.9  # Learning factor
    image_downscaling = True
    image_grayscaling = True
    noise_factor = 500
    noise_exponent = 2
    e_greedy_initial = .8
    e_greedy_final = .9
    e_greedy_step = .0001

    "Create the gym environment"
    env = gym.make(atari_game)
    state = env.reset()

    "Let's try a Q table approach. It will map from state_ids to actions (6)"
    q_table: QTable = QTable(image_downscaling, image_grayscaling, env.action_space.n)

    times = []
    runtime = 0
    rewards: List[Any]
    greedy_run_list = []
    for i in range(1, episodes):
        if i % 50 == 0:
            greedy_run_list.extend(range(i, i + 5))
    rewards = []

    while episode_number <= episodes:

        state = env.reset()
        done = False
        step_number = 0
        experience = 0
        episode_time = time.time()
        rewards = []
        playback = []
        lives_remaining = {'ale.lives': 4}

        while not done and step_number < max_steps:
            "Update step number"
            step_number = step_number + 1

            """Select an Action with Noise or e-Greedy"""
            if episode_number in greedy_run_list:
                action = q_table.greedy_action(state)
                if np.max(q_table.action_values(state)) != 0:
                    playback.append("G")
                    experience = experience + 1
                else:
                    playback.append("U")
            elif random.uniform(0, 1) < greediness_linear(episode_number, e_greedy_initial, e_greedy_final,
                                                          e_greedy_step):
                action_values = q_table.action_values(state)
                noise = create_noise(env.action_space.n, episode_number, noise_factor, noise_exponent)
                action = np.argmax(action_values + noise)
                if np.max(action_values) != 0:
                    playback.append("G")
                    experience = experience + 1
                else:
                    playback.append("U")
            else:
                action = env.action_space.sample()

            """Result of Action. Get next state, the reward, and whether the episode has concluded"""
            reward: int
            next_state, reward, done, info = env.step(action)
            if loss_of_life_knowledge and lives_remaining != info:
                reward = loss_of_life_reward
                lives_remaining = info
            else:
                rewards.append(reward)

            if reward != 0:
                playback.append(reward)

            action_values_new_state = q_table.action_values(next_state)
            next_state_value = np.max(action_values_new_state)
            target = reward + discount_factor * next_state_value

            state_value = q_table.action_values(state)[action]
            q_table.action_values(state)[action] = ((1 - learning_rate) * state_value + (learning_rate * target))

            state = next_state

            if display & (step_number > 200):
                display_state(state)

        runtime = time.time() - episode_time
        times.append(runtime)

        print("Episode {:>5}     :  "
              "Score | {:>6} "
              "Cached | {:>8} "
              "Greedy | {:>3} "
              "Runtime | {:04.2f} "
              "Steps | {:4d}"
              "{} ".format(episode_number, sum(rewards), len(q_table.dictionary), experience, runtime, step_number,
                           "GREEDY RUN" if episode_number in greedy_run_list else ""))
        print(playback)
        episode_number = episode_number + 1

    print()
    q_table.print_statistics()
    print("Runtime Average:       : {}"
          "Average Score: {}".format(np.mean(runtime), np.mean(rewards)))


if __name__ == '__main__':
    main()
