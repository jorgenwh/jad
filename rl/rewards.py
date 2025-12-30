"""
Reward computation for Jad RL environment.
Supports multiple reward functions via decorator registry.
"""

from typing import Callable
from env import Observation, TerminationState


# Healer aggro constants (must match HealerAggro enum in TypeScript)
HEALER_AGGRO_NOT_PRESENT = 0
HEALER_AGGRO_JAD = 1
HEALER_AGGRO_PLAYER = 2

REWARD_FUNCTIONS: dict[str, Callable] = {}


def reward_function(name: str):
    """Decorator to register a reward function."""
    def decorator(fn: Callable) -> Callable:
        REWARD_FUNCTIONS[name] = fn
        return fn
    return decorator


def list_reward_functions() -> list[str]:
    """Return list of available reward function names."""
    return list(REWARD_FUNCTIONS.keys())


def compute_reward(
    obs: Observation,
    prev_obs: Observation | None,
    termination: TerminationState,
    episode_length: int = 0,
    reward_type: str = "default",
) -> float:
    """
    Compute reward for a single step using the specified reward function.

    Args:
        obs: Current observation (after action)
        prev_obs: Previous observation (before action)
        termination: How the episode ended (or ONGOING if still running)
        episode_length: Current episode length (for time-based penalties)
        reward_type: Name of the reward function to use (default: "default")

    Returns:
        Raw reward value (not scaled)
    """
    fn = REWARD_FUNCTIONS.get(reward_type)
    if fn is None:
        available = list_reward_functions()
        raise ValueError(f"Unknown reward type: '{reward_type}'. Available: {available}")
    return fn(obs, prev_obs, termination, episode_length)


def healer_tag_reward(obs: Observation, prev_obs: Observation, tag_reward: float = 5.0) -> float:
    """
    Compute reward for tagging healers (pulling them off Jad).

    Returns <tag_reward> for each healer whose aggro transitions from JAD to PLAYER.
    This is a one-time reward per healer per spawn.
    """
    reward = 0.0

    for healer, prev_healer in zip(obs.healers, prev_obs.healers):
        if (prev_healer.aggro == HEALER_AGGRO_JAD and
                healer.aggro == HEALER_AGGRO_PLAYER):
            reward += tag_reward

    return reward


def prayer_landing_reward(obs: Observation, prev_obs: Observation,
                          correct: float = 2.5, wrong: float = -7.5) -> float:
    """
    Compute prayer switching reward on the tick damage lands.
    Attack landing = was visible in prev, now cleared.
    """
    reward = 0.0
    for jad, prev_jad in zip(obs.jads, prev_obs.jads):
        if prev_jad.attack != 0 and jad.attack == 0:
            if obs.active_prayer == prev_jad.attack:
                reward += correct
            else:
                reward += wrong
    return reward


def terminal_reward(obs: Observation, termination: TerminationState,
                    episode_length: int,
                    kill: float = 100.0, death: float = -200.0,
                    timeout: float = -150.0, time_penalty: float = 0.1) -> float:
    """Compute terminal reward based on episode outcome."""
    match termination:
        case TerminationState.JAD_KILLED:
            return kill * obs.jad_count - episode_length * time_penalty
        case TerminationState.PLAYER_DIED:
            return death
        case TerminationState.TRUNCATED:
            return timeout
        case _:
            return 0.0


@reward_function("default")
def reward_default(
    obs: Observation,
    prev_obs: Observation | None,
    termination: TerminationState,
    episode_length: int,
) -> float:
    """
    Default reward function with balanced shaping.
    - Prayer switching on landing tick
    - Combat engagement penalties
    - Buff maintenance (rigour, ranged stat)
    - Healer management
    """
    reward = 0.0

    # Prayer switching - only on landing tick
    if prev_obs is not None:
        reward += prayer_landing_reward(obs, prev_obs, correct=2.5, wrong=-7.5)

    # Penalty for not being in combat
    if obs.player_aggro == 0:
        reward -= 0.5

    # Penalty for rigour not active
    if not obs.rigour_active:
        reward -= 0.2

    # Penalty for low ranged stat (encourages bastion)
    reward -= (1 - (obs.player_ranged / 112.0))

    # Per-step rewards
    if prev_obs is not None:
        # Damage taken penalty
        damage_taken = prev_obs.player_hp - obs.player_hp
        if damage_taken > 0:
            reward -= damage_taken * 0.1

        # Jad healing penalty
        for jad, prev_jad in zip(obs.jads, prev_obs.jads):
            jad_healed = jad.hp - prev_jad.hp
            if jad_healed > 0:
                reward -= jad_healed * 0.3

        # Healer tagging reward
        reward += healer_tag_reward(obs, prev_obs)

    # Terminal rewards
    match termination:
        case TerminationState.JAD_KILLED:
            reward += 100.0 * obs.jad_count
            reward -= episode_length * 0.1
        case TerminationState.PLAYER_DIED:
            reward -= 200
        case TerminationState.TRUNCATED:
            reward -= 150

    return reward


@reward_function("sparse")
def reward_sparse(
    obs: Observation,
    prev_obs: Observation | None,
    termination: TerminationState,
    episode_length: int,
) -> float:
    """
    Sparse reward - only terminal rewards.
    Good for testing if agent can learn from pure outcome signal.
    """
    match termination:
        case TerminationState.JAD_KILLED:
            return 1
        case TerminationState.PLAYER_DIED:
            return -1
        case TerminationState.TRUNCATED:
            return -1
        case _:
            return 0.0

