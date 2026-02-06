export interface JadConfig {
    jadCount: number; // 1-6, (default: 1)
    healersPerJad: number; // 0-5, (default: 3)
}

export const DEFAULT_CONFIG: JadConfig = {
    jadCount: 1,
    healersPerJad: 3,
};

export interface JadState {
    hp: number;
    attack: number;              // 0=none, 1=mage, 2=range, 3=melee (non-zero while projectile in flight)
    ticks_until_impact: number;  // 0=none, 1-3=ticks remaining until projectile hits
    x: number;
    y: number;
    alive: boolean;
}

export enum HealerTarget {
    NOT_PRESENT = 0,
    JAD = 1,
    PLAYER = 2,
}

export interface HealerState {
    hp: number;
    x: number;
    y: number;
    target: HealerTarget;
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
    player_target: number;

    active_prayer: number;  // 0=none, 1=mage, 2=range, 3=melee
    rigour_active: boolean;

    jads: JadState[];
    healers: HealerState[];
    healers_spawned: boolean;

    // Derived: the soonest-landing projectile across all Jads
    next_projectile_type: number;   // 0=none, 1=mage, 2=range, 3=melee
    next_projectile_ticks: number;  // 0=none, 1-3=ticks remaining

    bastion_doses: number;
    sara_brew_doses: number;
    super_restore_doses: number;
    starting_bastion_doses: number;
    starting_sara_brew_doses: number;
    starting_super_restore_doses: number;
}

export enum TerminationState {
    ONGOING = 'ongoing',
    PLAYER_DIED = 'player_died',
    JAD_KILLED = 'jad_killed',
}

export interface StepResult {
    observation: Observation;
    reward: number;
    terminated: boolean;
    valid_action_mask: boolean[][];
}

// MultiDiscrete action head indices
export const enum ActionHead {
    PROTECTION_PRAYER = 0,
    OFFENSIVE_PRAYER = 1,
    POTION = 2,
    TARGET = 3,
}

// Protection prayer head values (head 0)
export const enum ProtectionPrayerAction {
    NO_OP = 0,
    PROTECT_MAGIC = 1,
    PROTECT_RANGE = 2,
    PROTECT_MELEE = 3,
}

// Offensive prayer head values (head 1)
export const enum OffensivePrayerAction {
    NO_OP = 0,
    TOGGLE_RIGOUR = 1,
}

// Potion head values (head 2)
export const enum PotionAction {
    NONE = 0,
    BASTION = 1,
    SUPER_RESTORE = 2,
    SARA_BREW = 3,
}

// Target head values (head 3)
// 0 = no-op, 1..N = jad, N+1.. = healer (dynamic based on config)
export const TARGET_NO_OP = 0;

// Action space dimensions for each head
export const ACTION_HEAD_COUNT = 4;
export const PROTECTION_PRAYER_SIZE = 4;  // no-op + 3 prayers
export const OFFENSIVE_PRAYER_SIZE = 2;   // no-op + rigour
export const POTION_SIZE = 4;             // none + 3 potions

// Helper to get target head size: 1 (no-op) + N (jads) + N*H (healers)
export function getTargetHeadSize(config: JadConfig): number {
    return 1 + config.jadCount + config.jadCount * config.healersPerJad;
}
