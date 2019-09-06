import numpy as np


def create_noise(action_count: int, episode_number: int, noise_factor: float, noise_exponent: float) -> np.ndarray:
    """
    Create noise to encourage learning for earlier episodes
    Exponentially lower this modifier with advanced episodes
    """
    return np.random.random((1, action_count)) * noise_factor / (episode_number ** noise_exponent)


def greediness_linear(episode, e_greedy_initial, e_greedy_final, e_greedy_step) -> float:
    """Increase Greediness as episodes advance linearly, up to a specified final value approaching 1.0"""
    greediness = e_greedy_initial + e_greedy_step * (episode - 1)
    if greediness > e_greedy_final:
        return e_greedy_final
    return greediness
