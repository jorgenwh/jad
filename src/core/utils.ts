import { Player } from 'osrs-sdk';
import { JadConfig, TerminationState } from './types';
import { JadRegion } from './jad-region';

/**
 * Get action count for given Jad configuration
 * Actions: DO_NOTHING + N*AGGRO_JAD + N*H*AGGRO_HEALER + 7 prayers/potions
 */
export function getActionCount(config: JadConfig): number {
    return 1 + config.jadCount + config.jadCount * config.healersPerJad + 7;
}

export function getActionName(action: number, config: JadConfig): string {
    if (action === 0) return 'DO_NOTHING';

    let idx = 1;

    // Aggro Jad actions
    for (let i = 0; i < config.jadCount; i++) {
        if (action === idx) return `AGGRO_JAD_${i + 1}`;
        idx++;
    }

    // Aggro healer actions
    for (let jadIdx = 0; jadIdx < config.jadCount; jadIdx++) {
        for (let healerIdx = 0; healerIdx < config.healersPerJad; healerIdx++) {
            if (action === idx) return `AGGRO_H${jadIdx + 1}.${healerIdx + 1}`;
            idx++;
        }
    }

    // Prayer/potion actions
    const fixedActions = [
        'TOGGLE_PROTECT_MELEE',
        'TOGGLE_PROTECT_MISSILES',
        'TOGGLE_PROTECT_MAGIC',
        'TOGGLE_RIGOUR',
        'DRINK_BASTION',
        'DRINK_SUPER_RESTORE',
        'DRINK_SARA_BREW',
    ];
    const fixedIdx = action - idx;
    if (fixedIdx >= 0 && fixedIdx < fixedActions.length) {
        return fixedActions[fixedIdx];
    }

    return 'UNKNOWN';
}

export function checkTermination(
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig
): TerminationState {
    // Check player death
    if (player.dying > 0 || player.currentStats.hitpoint <= 0) {
        return TerminationState.PLAYER_DIED;
    }

    // Check if all Jads are dead
    for (let i = 0; i < config.jadCount; i++) {
        const jad = jadRegion.getJad(i);
        if (jad && jad.currentStats.hitpoint > 0 && jad.dying === -1) {
            return TerminationState.ONGOING;
        }
    }

    return TerminationState.JAD_KILLED;
}
