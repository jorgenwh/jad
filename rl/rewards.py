"""
Reward computation for Jad RL environment.
Shared between train.py and jad_gymnasium_env.py.

Healer aggro values:
  0 = NOT_PRESENT
  1 = JAD (healing Jad)
  2 = PLAYER (tagged, attacking player)
"""

from env import Observation, TerminationState


# Healer aggro constants (must match HealerAggro enum in TypeScript)
HEALER_AGGRO_NOT_PRESENT = 0
HEALER_AGGRO_JAD = 1
HEALER_AGGRO_PLAYER = 2


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

    # Penalty for not being in combat (encourages attacking)
    # player_aggro is now an int: 0=none, 1=jad, 2-4=healer
    if obs.player_aggro == 0:
        reward -= 0.5

    # Per-step rewards (only if we have previous observation to compare)
    if prev_obs is not None:
        # Damage taken penalty
        damage_taken = prev_obs.player_hp - obs.player_hp
        if damage_taken > 0:
            reward -= damage_taken * 0.1

        # Damage dealt reward - only for Jad damage, NOT healer damage
        # This encourages focusing on Jad rather than killing healers
        damage_dealt = prev_obs.jad_hp - obs.jad_hp
        if damage_dealt > 0:
            reward += damage_dealt * 0.2

        # Jad healing penalty - punishes letting healers heal Jad
        jad_healed = obs.jad_hp - prev_obs.jad_hp
        if jad_healed > 0:
            reward -= jad_healed * 0.3

        # Healer tagging reward - one-time bonus when healer aggro changes from JAD to PLAYER
        # This directly incentivizes the optimal strategy: tag healers to pull them off Jad
        reward += _healer_tag_reward(obs, prev_obs)

    # Terminal rewards
    match termination:
        case TerminationState.JAD_KILLED:
            reward += 100.0
            reward -= episode_length * 0.1  # Faster kills are better
        case TerminationState.PLAYER_DIED:
            reward -= 50.0  # Dying while trying is acceptable
        case TerminationState.TRUNCATED:
            reward -= 150.0  # Timeout is worse than dying - discourages passivity

    return reward


def _healer_tag_reward(obs: Observation, prev_obs: Observation) -> float:
    """
    Compute reward for tagging healers (pulling them off Jad).

    Returns +5 for each healer whose aggro transitions from JAD to PLAYER.
    This is a one-time reward per healer per spawn.
    """
    reward = 0.0
    tag_reward = 5.0

    # Check each healer for JAD -> PLAYER transition
    if (prev_obs.healer_1_aggro == HEALER_AGGRO_JAD and
            obs.healer_1_aggro == HEALER_AGGRO_PLAYER):
        reward += tag_reward

    if (prev_obs.healer_2_aggro == HEALER_AGGRO_JAD and
            obs.healer_2_aggro == HEALER_AGGRO_PLAYER):
        reward += tag_reward

    if (prev_obs.healer_3_aggro == HEALER_AGGRO_JAD and
            obs.healer_3_aggro == HEALER_AGGRO_PLAYER):
        reward += tag_reward

    return reward
