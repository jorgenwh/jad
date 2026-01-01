import { Observation, HealerAggro } from './types';

export enum TerminationState {
    ONGOING = 'ongoing',
    PLAYER_DIED = 'player_died',
    JAD_KILLED = 'jad_killed',
    TRUNCATED = 'truncated',
}

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

/**
 * Compute reward for tagging healers (pulling them off Jad).
 * Returns +5 for each healer whose aggro transitions from JAD to PLAYER.
 */
function healerTagReward(obs: Observation, prevObs: Observation): number {
    let reward = 0;
    const tagReward = 5.0;

    for (let i = 0; i < obs.healers.length; i++) {
        const healer = obs.healers[i];
        const prevHealer = prevObs.healers[i];
        if (prevHealer.aggro === HealerAggro.JAD && healer.aggro === HealerAggro.PLAYER) {
            reward += tagReward;
        }
    }

    return reward;
}

/**
 * Compute prayer switching reward on the tick damage lands.
 * Attack landing = was visible in prev, now cleared.
 */
function prayerLandingReward(
    obs: Observation,
    prevObs: Observation,
    correct: number = 2.5,
    wrong: number = -7.5
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

/**
 * Compute reward for killing individual Jads.
 * Detects when a Jad's HP transitions from >0 to <=0.
 */
function jadKillReward(obs: Observation, prevObs: Observation, killReward: number = 100.0): number {
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

// Default reward function
registerRewardFunction('default', (obs, prevObs, termination, episodeLength) => {
    let reward = 0;

    if (prevObs === null) {
        return reward;
    }

    // Prayer switching - only on landing tick
    reward += prayerLandingReward(obs, prevObs, 2.5, -7.5);

    // Penalty for not being in combat
    if (obs.player_aggro === 0) {
        reward -= 0.5;
    }

    // Penalty for rigour not active
    if (!obs.rigour_active) {
        reward -= 0.2;
    }

    // Penalty for low ranged stat (encourages bastion)
    reward -= (1 - (obs.player_ranged / 112.0));

    // Damage taken penalty
    const damageTaken = prevObs.player_hp - obs.player_hp;
    if (damageTaken > 0) {
        reward -= damageTaken * 0.1;
    }

    // Jad healing penalty
    for (let i = 0; i < obs.jads.length; i++) {
        const jadHealed = obs.jads[i].hp - prevObs.jads[i].hp;
        if (jadHealed > 0) {
            reward -= jadHealed * 0.3;
        }
    }

    // Healer tagging reward
    reward += healerTagReward(obs, prevObs);

    // Terminal rewards
    switch (termination) {
        case TerminationState.JAD_KILLED:
            reward += 100.0;
            reward -= episodeLength * 0.1; // Faster kills are better
            break;
        case TerminationState.PLAYER_DIED:
            reward -= 200.0; // Death is catastrophic
            break;
        case TerminationState.TRUNCATED:
            reward -= 150.0; // Timeout is worse than dying
            break;
    }

    return reward;
});

// Sparse reward function
registerRewardFunction('sparse', (_obs, _prevObs, termination, _episodeLength) => {
    switch (termination) {
        case TerminationState.JAD_KILLED:
            return 1;
        case TerminationState.PLAYER_DIED:
        case TerminationState.TRUNCATED:
            return -1;
        default:
            return 0;
    }
});

// Multi-jad reward function
registerRewardFunction('multijad', (obs, prevObs, termination, episodeLength) => {
    let reward = 0;

    if (prevObs === null) {
        return reward;
    }

    // Prayer switching - only on landing tick (reduced values for multi-jad)
    reward += prayerLandingReward(obs, prevObs, 1.25, -3.75);

    // Penalty for not being in combat
    if (obs.player_aggro === 0) {
        reward -= 0.5;
    }

    // Penalty for rigour not active
    if (!obs.rigour_active) {
        reward -= 0.2;
    }

    // Penalty for low ranged stat (encourages bastion)
    reward -= (1 - (obs.player_ranged / 112.0));

    // Damage taken penalty
    const damageTaken = prevObs.player_hp - obs.player_hp;
    if (damageTaken > 0) {
        reward -= damageTaken * 0.1;
    }

    // Jad healing penalty
    for (let i = 0; i < obs.jads.length; i++) {
        const jadHealed = obs.jads[i].hp - prevObs.jads[i].hp;
        if (jadHealed > 0) {
            reward -= jadHealed * 0.3;
        }
    }

    // Healer tagging reward
    reward += healerTagReward(obs, prevObs);

    // Per-Jad kill reward (incremental, not just at end)
    reward += jadKillReward(obs, prevObs);

    // Terminal rewards
    switch (termination) {
        case TerminationState.JAD_KILLED:
            reward -= episodeLength * 0.1;
            break;
        case TerminationState.PLAYER_DIED:
            reward -= 200.0;
            break;
        case TerminationState.TRUNCATED:
            reward -= 150.0;
            break;
    }

    return reward;
});
