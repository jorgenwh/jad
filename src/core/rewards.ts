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
        episodeLength: number
    ) => {
    let reward = 0;

    if (prevObs === null) {
        return reward;
    }

    // Count how many Jads attacked this tick (for scaling penalties)
    let attackCount = 0;
    for (let i = 0; i < prevObs.jads.length; i++) {
        if (prevObs.jads[i].attack !== 0 && obs.jads[i].attack === 0) {
            attackCount++;
        }
    }

    // Prayer reward/penalty - reduced penalty when multiple Jads attack (unavoidable damage)
    const prayerWrongPenalty = attackCount > 1 ? -1.5 : -3.0;
    reward += prayerLandingReward(obs, prevObs, 1.5, prayerWrongPenalty);

    // Significant penalty for not being in combat - prevents stalling
    if (obs.player_target === 0) {
        reward -= 1.0;
    }

    // Penalty for rigour not active
    if (!obs.rigour_active) {
        reward -= 0.1;
    }

    // Time pressure: small increasing penalty to discourage stalling
    // At tick 100: -0.2, at tick 300: -0.6, at tick 500: -1.0
    reward -= episodeLength * 0.002;

    // Count healers targeting Jad vs player
    let healersOnJad = 0;
    for (const healer of obs.healers) {
        if (healer.target === HealerTarget.JAD) {
            healersOnJad++;
        }
    }

    // Find the Jad with lowest HP (to encourage focusing)
    let lowestHpJadIndex = 0;
    let lowestHp = obs.jads[0]?.hp ?? Infinity;
    for (let i = 1; i < obs.jads.length; i++) {
        if (obs.jads[i].hp > 0 && obs.jads[i].hp < lowestHp) {
            lowestHp = obs.jads[i].hp;
            lowestHpJadIndex = i;
        }
    }

    // Reward for dealing damage to Jads - bonus for focusing lowest HP Jad
    // Scale damage reward based on healer situation: more valuable when healers are tagged
    const healerMultiplier = healersOnJad === 0 ? 1.5 : 1.0;
    for (let i = 0; i < obs.jads.length; i++) {
        const jadDamage = prevObs.jads[i].hp - obs.jads[i].hp;
        if (jadDamage > 0) {
            // Base damage reward (higher when all healers tagged)
            reward += jadDamage * 0.3 * healerMultiplier;
            // Bonus for damaging the lowest HP Jad (encourages focus fire)
            if (i === lowestHpJadIndex && obs.jads.length > 1) {
                reward += jadDamage * 0.2;
            }
        }
    }

    // Each healer heals ~1 HP/tick, so 6 healers = 6 HP/tick = massive DPS loss
    // Penalty scales with number of healers to make it urgent
    const healerPenaltyPerHealer = 1.5 + (healersOnJad * 0.3); // 1.5 base, +0.3 per additional
    reward -= healersOnJad * healerPenaltyPerHealer;

    // One-time reward for tagging healers (when healer switches from Jad to player)
    reward += healerTagReward(obs, prevObs) * 3.0; // 15 per healer tagged

    // NOTE: Removed continuous +2.0 bonus for healers tagged - it allowed farming without killing Jads

    // Large reward for each Jad killed
    reward += jadKillReward(obs, prevObs, 200.0);

    // Terminal rewards
    switch (termination) {
        case TerminationState.JAD_KILLED:
            // Big bonus for winning, with time bonus for faster kills
            reward += 100.0;
            reward += Math.max(0, (600 - episodeLength) * 0.2); // Up to +120 for fast kills
            break;
        case TerminationState.PLAYER_DIED:
            // Penalty reduced by progress made (damage dealt to Jads)
            const totalJadHpRemaining = obs.jads.reduce((sum, jad) => sum + Math.max(0, jad.hp), 0);
            const maxJadHp = obs.jads.length * 350;
            const damageDealt = maxJadHp - totalJadHpRemaining;
            const progressBonus = damageDealt * 0.1; // Up to 70 bonus for dealing 700 damage
            reward -= 100.0;
            reward += progressBonus;
            break;
    }

    return reward;
});
