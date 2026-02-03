import { Player } from 'osrs-sdk';
import { JadConfig, TerminationState } from './types';
import { JadRegion } from './jad-region';

/**
 * Get human-readable name for a multi-head action.
 */
export function getActionName(action: number[], config: JadConfig): string {
    const parts: string[] = [];

    // Protection prayer (head 0)
    const protectionNames = ['', 'MAGE', 'RANGE', 'MELEE'];
    if (action[0] > 0) {
        parts.push(`PROTECT_${protectionNames[action[0]]}`);
    }

    // Offensive prayer (head 1)
    if (action[1] === 1) {
        parts.push('TOGGLE_RIGOUR');
    }

    // Potion (head 2)
    const potionNames = ['', 'BASTION', 'SUPER_RESTORE', 'SARA_BREW'];
    if (action[2] > 0) {
        parts.push(`DRINK_${potionNames[action[2]]}`);
    }

    // Target (head 3)
    const target = action[3];
    if (target > 0) {
        const numJads = config.jadCount;
        if (target <= numJads) {
            parts.push(`AGGRO_JAD_${target}`);
        } else {
            const healerIdx = target - numJads - 1;
            const jadIdx = Math.floor(healerIdx / config.healersPerJad);
            const hIdx = healerIdx % config.healersPerJad;
            parts.push(`AGGRO_H${jadIdx + 1}.${hIdx + 1}`);
        }
    }

    if (parts.length === 0) {
        return 'NO_OP';
    }

    return parts.join(' + ');
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
