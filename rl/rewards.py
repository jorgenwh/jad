"""
Reward computation for Jad RL environment.
Shared between train.py and jad_gymnasium_env.py.
"""

from env import Observation, TerminationState


def compute_reward(
    obs: Observation,
    prev_obs: Observation | None,
    termination: TerminationState,
    episode_length: int = 0,
) -> float:
    """
    Compute reward for a single step.

    Args:
        obs: Current observation (after action)
        prev_obs: Previous observation (before action)
        termination: How the episode ended (or ONGOING if still running)
        episode_length: Current episode length (for time-based penalties)

    Returns:
        Raw reward value (not scaled)
    """
    reward = 0.0

    # Prayer switching feedback
    if prev_obs is not None and prev_obs.jad_attack != 0:
        if obs.active_prayer == prev_obs.jad_attack:
            reward += 1
        else:
            reward -= 1

    # Survival reward
    reward += 0.1

    # Per-step rewards (only if we have previous observation to compare)
    if prev_obs is not None:
        # Damage taken penalty
        damage_taken = prev_obs.player_hp - obs.player_hp
        if damage_taken > 0:
            reward -= damage_taken * 0.1

        # Damage dealt reward
        damage_dealt = prev_obs.jad_hp - obs.jad_hp
        if damage_dealt > 0:
            reward += damage_dealt * 0.1

    # Terminal rewards
    match termination:
        case TerminationState.JAD_KILLED:
            reward += 100.0
            reward -= episode_length * 0.25  # Faster kills are better
        case TerminationState.PLAYER_DIED | TerminationState.TRUNCATED:
            reward -= 100.0  # Lose penalty

    return reward
