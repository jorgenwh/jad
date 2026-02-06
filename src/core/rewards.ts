import { Observation, HealerTarget, TerminationState } from './types';

export type RewardFunction = (
    obs: Observation,
    prevObs: Observation | null,
    termination: TerminationState,
    episodeLength: number
) => number;

const REWARD_FUNCTIONS: Map<string, RewardFunction> = new Map();

function registerRewardFunction(name: string, fn: RewardFunction): void {
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
    rewardFunc: string = 'default'
): number {
    const fn = REWARD_FUNCTIONS.get(rewardFunc);
    if (!fn) {
        const available = listRewardFunctions();
        throw new Error(`Unknown reward function: '${rewardFunc}'. Available: ${available.join(', ')}`);
    }
    return fn(obs, prevObs, termination, episodeLength);
}

function healerTagReward(obs: Observation, prevObs: Observation): number {
    let reward = 0;

    for (let i = 0; i < obs.healers.length; i++) {
        const healer = obs.healers[i];
        const prevHealer = prevObs.healers[i];
        if (prevHealer.target === HealerTarget.JAD && healer.target === HealerTarget.PLAYER) {
            reward += 5.0;
        }
    }

    return reward;
}

function healerTargetingJadPenalty(obs: Observation, penaltyPerHealer: number): number {
    let penalty = 0;

    for (const healer of obs.healers) {
        if (healer.target === HealerTarget.JAD) {
            penalty += penaltyPerHealer;
        }
    }

    return penalty;
}

function prayerLandingReward(
    obs: Observation,
    prevObs: Observation,
    correct: number,
    wrong: number
): number {
    let reward = 0;

    for (let i = 0; i < obs.jads.length; i++) {
        const jad = obs.jads[i];
        const prevJad = prevObs.jads[i];
        if (prevJad.attack !== 0 && jad.attack === 0) {
            if (obs.active_prayer === prevJad.attack) {
                reward += correct;
            } else {
                reward += wrong;
            }
        }
    }

    return reward;
}

function jadKillReward(obs: Observation, prevObs: Observation, killReward: number): number {
    let reward = 0;

    for (let i = 0; i < obs.jads.length; i++) {
        const jad = obs.jads[i];
        const prevJad = prevObs.jads[i];
        if (prevJad.hp > 0 && jad.hp <= 0) {
            reward += killReward;
        }
    }

    return reward;
}

registerRewardFunction(
    'default',
    (
        obs: Observation,
        prevObs: Observation | null,
        termination: TerminationState,
        episodeLength: number
    ) => {
    if (prevObs === null) {
        return 0;
    }

    let reward = 0;

    // Penalty/reward for praying correctly on tick when Jad attack lands
    reward += prayerLandingReward(obs, prevObs, 2.5, -7.5);

    // Penalty for not being in combat
    if (obs.player_target === 0) {
        reward -= 0.5;
    }

    // Penalty for rigour not active
    if (!obs.rigour_active) {
        reward -= 0.2;
    }

    // Penalty for low ranged stat
    reward -= (1 - (obs.player_ranged / 112.0));

    // Penalty for damage taken
    const damageTaken = prevObs.player_hp - obs.player_hp;
    if (damageTaken > 0) {
        reward -= damageTaken * 0.1;
    }

    // Reward for dealing damage to Jads
    for (let i = 0; i < obs.jads.length; i++) {
        const jadDamage = prevObs.jads[i].hp - obs.jads[i].hp;
        if (jadDamage > 0) {
            reward += jadDamage * 0.2;
        }
    }

    // Penalty for each healer targeting Jad (encourages tagging healers)
    reward -= healerTargetingJadPenalty(obs, 0.3);

    // Reward for tagging healers (one-time bonus when healer switches from Jad to player)
    reward += healerTagReward(obs, prevObs);

    // Terminal rewards
    switch (termination) {
        case TerminationState.JAD_KILLED:
            reward += 100.0;
            reward -= episodeLength * 0.1; // Faster kills are better
            break;
        case TerminationState.PLAYER_DIED:
            reward -= 200.0;
            break;
    }

    return reward;
});

registerRewardFunction(
    'sparse',
    (
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

registerRewardFunction(
    'multijad',
    (
        obs: Observation,
        prevObs: Observation | null,
        termination: TerminationState,
        _episodeLength: number
    ) => {
    let reward = 0;

    if (prevObs === null) {
        return reward;
    }

    // Prayer reward/penalty (triggered when projectile lands)
    reward += prayerLandingReward(obs, prevObs, 2.0, -2.0);

    // Penalty for not being in combat - prevents stalling
    if (obs.player_target === 0) {
        reward -= 1.0;
    }

    // Penalty for rigour not active
    if (!obs.rigour_active) {
        reward -= 0.1;
    }

    // Reward for dealing damage to any Jad (positive only - no penalty for healer healing)
    for (let i = 0; i < obs.jads.length; i++) {
        const hpDelta = prevObs.jads[i].hp - obs.jads[i].hp;
        if (hpDelta > 0) {
            reward += hpDelta * 0.3;
        }
    }

    // Reward for tagging healers off Jad (one-time bonus per healer)
    reward += healerTagReward(obs, prevObs);

    // Large reward for each Jad killed
    reward += jadKillReward(obs, prevObs, 200.0);

    // Terminal rewards
    switch (termination) {
        case TerminationState.JAD_KILLED:
            reward += 100.0;
            break;
        case TerminationState.PLAYER_DIED:
            // Penalty reduced by progress made (damage dealt to Jads)
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
