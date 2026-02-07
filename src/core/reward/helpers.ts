import { Observation, HealerTarget } from '../types';

export function healerTagReward(obs: Observation, prevObs: Observation): number {
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

export function healerTargetingJadPenalty(obs: Observation, penaltyPerHealer: number): number {
    let penalty = 0;

    for (const healer of obs.healers) {
        if (healer.target === HealerTarget.JAD) {
            penalty += penaltyPerHealer;
        }
    }

    return penalty;
}

export function prayerLandingReward(
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

export function jadKillReward(obs: Observation, prevObs: Observation, killReward: number): number {
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

export function jadDamageReward(obs: Observation, prevObs: Observation, scale: number): number {
    let reward = 0;

    for (let i = 0; i < obs.jads.length; i++) {
        const hpDelta = prevObs.jads[i].hp - obs.jads[i].hp;
        if (hpDelta > 0) {
            reward += hpDelta * scale;
        }
    }

    return reward;
}
