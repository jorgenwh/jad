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

// Single Jad state in observation
export interface JadState {
    hp: number;
    attack: number;  // 0=none, 1=mage, 2=range, 3=melee
    x: number;
    y: number;
    alive: boolean;
}

// Healer aggro state
export enum HealerAggro {
    NOT_PRESENT = 0,
    JAD = 1,
    PLAYER = 2,
}

// Single healer state in observation
export interface HealerState {
    hp: number;
    x: number;
    y: number;
    aggro: HealerAggro;
}

export interface Observation {
    // Player state
    player_hp: number;
    player_prayer: number;
    player_ranged: number;
    player_defence: number;
    player_x: number;
    player_y: number;
    player_aggro: number;  // 0=none, 1..N=jad_N, N+1..=healer

    // Prayer state
    active_prayer: number;  // 0=none, 1=mage, 2=range, 3=melee
    rigour_active: boolean;

    // Inventory
    bastion_doses: number;
    sara_brew_doses: number;
    super_restore_doses: number;

    // Dynamic Jad state array
    jads: JadState[];

    // Dynamic healer state array (flattened)
    healers: HealerState[];

    // Whether any healers have spawned
    healers_spawned: boolean;

    // Starting doses for normalization
    starting_bastion_doses: number;
    starting_sara_brew_doses: number;
    starting_super_restore_doses: number;
}

