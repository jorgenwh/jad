import { Observation, TerminationState } from '../types';
import {
    healerTagReward,
    healerTargetingJadPenalty,
    prayerLandingReward,
    jadKillReward,
    jadDamageReward,
} from './helpers';

export type RewardFunction = (
    obs: Observation,
    prevObs: Observation | null,
    termination: TerminationState,
    episodeLength: number
) => number;

const REWARD_FUNCTIONS: Map<string, RewardFunction> = new Map();

function register(name: string, fn: RewardFunction): void {
    REWARD_FUNCTIONS.set(name, fn);
}

export function listRewardFunctions(): string[] {
    return Array.from(REWARD_FUNCTIONS.keys());
}

export function computeReward(
    obs: Observation,
    prevObs: Observation | null,
    termination: TerminationState,
    episodeLength: number,
    rewardFunc: string
): number {
    const fn = REWARD_FUNCTIONS.get(rewardFunc);
    if (!fn) {
        const available = listRewardFunctions();
        throw new Error(`Unknown reward function: '${rewardFunc}'. Available: ${available.join(', ')}`);
    }
    return fn(obs, prevObs, termination, episodeLength);
}

register('sparse', (
    _obs: Observation,
    _prevObs: Observation | null,
    termination: TerminationState,
    _episodeLength: number
) => {
    switch (termination) {
        case TerminationState.JAD_KILLED:
            return 1;
        case TerminationState.PLAYER_DIED:
            return -1;
        default:
            return 0;
    }
});

register('jad1', (
    obs: Observation,
    prevObs: Observation | null,
    termination: TerminationState,
    episodeLength: number
) => {
    if (prevObs === null) {
        return 0;
    }

    let reward = 0;

    reward += prayerLandingReward(obs, prevObs, 2.5, -7.5);

    if (obs.player_target === 0) {
        reward -= 0.5;
    }

    if (!obs.rigour_active) {
        reward -= 0.2;
    }

    reward -= (1 - (obs.player_ranged / 112.0));

    const damageTaken = prevObs.player_hp - obs.player_hp;
    if (damageTaken > 0) {
        reward -= damageTaken * 0.1;
    }

    reward += jadDamageReward(obs, prevObs, 0.2);
    reward -= healerTargetingJadPenalty(obs, 0.3);
    reward += healerTagReward(obs, prevObs);

    switch (termination) {
        case TerminationState.JAD_KILLED:
            reward += 100.0;
            reward -= episodeLength * 0.1;
            break;
        case TerminationState.PLAYER_DIED:
            reward -= 200.0;
            break;
    }

    return reward;
});

register('jad2', (
    obs: Observation,
    prevObs: Observation | null,
    termination: TerminationState,
    _episodeLength: number
) => {
    let reward = 0;

    if (prevObs === null) {
        return reward;
    }

    reward += prayerLandingReward(obs, prevObs, 2.0, -2.0);

    if (obs.player_target === 0) {
        reward -= 1.0;
    }

    if (!obs.rigour_active) {
        reward -= 0.1;
    }

    reward += jadDamageReward(obs, prevObs, 0.3);
    reward += healerTagReward(obs, prevObs);
    reward += jadKillReward(obs, prevObs, 200.0);

    switch (termination) {
        case TerminationState.JAD_KILLED:
            reward += 100.0;
            break;
        case TerminationState.PLAYER_DIED:
            const totalJadHpRemaining = obs.jads.reduce((sum, jad) => sum + Math.max(0, jad.hp), 0);
            const maxJadHp = obs.jads.length * 350;
            const damageDealt = maxJadHp - totalJadHpRemaining;
            const progressBonus = damageDealt * 0.1;
            reward -= 100.0;
            reward += progressBonus;
            break;
    }

    return reward;
});

register('jad3', (
    obs: Observation,
    prevObs: Observation | null,
    termination: TerminationState,
    _episodeLength: number
) => {
    let reward = 0;

    if (prevObs === null) {
        return reward;
    }

    reward += prayerLandingReward(obs, prevObs, 8.0, -8.0);

    if (obs.player_target === 0) {
        reward -= 1.0;
    }

    if (!obs.rigour_active) {
        reward -= 0.1;
    }

    reward += jadDamageReward(obs, prevObs, 0.3);
    reward += healerTagReward(obs, prevObs);
    reward += jadKillReward(obs, prevObs, 200.0);

    switch (termination) {
        case TerminationState.JAD_KILLED:
            reward += 100.0;
            break;
        case TerminationState.PLAYER_DIED:
            const totalJadHpRemaining = obs.jads.reduce((sum, jad) => sum + Math.max(0, jad.hp), 0);
            const maxJadHp = obs.jads.length * 350;
            const damageDealt = maxJadHp - totalJadHpRemaining;
            const progressBonus = damageDealt * 0.1;
            reward -= 100.0;
            reward += progressBonus;
            break;
    }

    return reward;
});
