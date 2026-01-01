export interface JadConfig {
    jadCount: number; // 1-6, (default: 1)
    healersPerJad: number; // 0-5, (default: 3)
}

export const DEFAULT_CONFIG: JadConfig = {
    jadCount: 1,
    healersPerJad: 3,
};

/**
 * Get action count for given Jad configuration.
 * Actions: DO_NOTHING + N*AGGRO_JAD + N*H*AGGRO_HEALER + 7 prayers/potions
 */
export function getActionCount(config: JadConfig): number {
    return 1 + config.jadCount + config.jadCount * config.healersPerJad + 7;
}

export interface JadState {
    hp: number;
    attack: number;  // 0=none, 1=mage, 2=range, 3=melee
    x: number;
    y: number;
    alive: boolean;
}

export enum HealerAggro {
    NOT_PRESENT = 0,
    JAD = 1,
    PLAYER = 2,
}

export interface HealerState {
    hp: number;
    x: number;
    y: number;
    aggro: HealerAggro;
}

export interface Observation {
    player_hp: number;
    player_prayer: number;
    player_ranged: number;
    player_defence: number;
    player_location_x: number;
    player_location_y: number;

    // Player's current attack target. N is number of Jads
    // - 0: none
    // - 1..N: Jad 1 through Jad N
    // - N+1..N+N*H: Healer (jadIdx * healersPerJad + healerIdx + N + 1)
    player_aggro: number;

    active_prayer: number;  // 0=none, 1=mage, 2=range, 3=melee
    rigour_active: boolean;

    jads: JadState[];
    healers: HealerState[];
    healers_spawned: boolean;

    bastion_doses: number;
    sara_brew_doses: number;
    super_restore_doses: number;

    starting_bastion_doses: number;
    starting_sara_brew_doses: number;
    starting_super_restore_doses: number;
}

