import { Player, Potion } from 'osrs-sdk';
import { JadRegion } from './jad-region';
import { JadConfig, JadObservation, JadState, HealerState, JadAttackState, HealerAggro } from './types';

/**
 * Count potion doses in player inventory.
 */
export function countPotionDoses(player: Player): {
    bastionDoses: number;
    saraBrewDoses: number;
    superRestoreDoses: number;
} {
    let bastionDoses = 0;
    let saraBrewDoses = 0;
    let superRestoreDoses = 0;

    if (player && player.inventory) {
        for (const item of player.inventory) {
            if (item && item instanceof Potion && item.doses > 0) {
                const itemName = item.itemName?.toString().toLowerCase() || '';
                if (itemName.includes('bastion')) {
                    bastionDoses += item.doses;
                } else if (itemName.includes('saradomin brew')) {
                    saraBrewDoses += item.doses;
                } else if (itemName.includes('restore')) {
                    superRestoreDoses += item.doses;
                }
            }
        }
    }

    return { bastionDoses, saraBrewDoses, superRestoreDoses };
}

/**
 * Get active prayer state from player.
 */
export function getActivePrayer(player: Player): { activePrayer: number; rigourActive: boolean } {
    let activePrayer = 0;
    let rigourActive = false;

    const prayerController = player.prayerController;
    if (prayerController) {
        const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
        const rangePrayer = prayerController.findPrayerByName('Protect from Range');
        const meleePrayer = prayerController.findPrayerByName('Protect from Melee');
        const rigourPrayer = prayerController.findPrayerByName('Rigour');

        if (magicPrayer?.isActive) {
            activePrayer = 1;
        } else if (rangePrayer?.isActive) {
            activePrayer = 2;
        } else if (meleePrayer?.isActive) {
            activePrayer = 3;
        }

        rigourActive = rigourPrayer?.isActive ?? false;
    }

    return { activePrayer, rigourActive };
}

/**
 * Determine player's current aggro target encoded as observation value.
 * 0=none, 1..N=jad_N, N+1..N+N*H=healer
 */
export function getPlayerAggroTarget(
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig
): number {
    let playerAggro = 0;
    const playerAggroTarget = player.aggro;

    if (playerAggroTarget) {
        // Check if aggro is one of the Jads
        for (let i = 0; i < config.jadCount; i++) {
            const jad = jadRegion.getJad(i);
            if (jad && playerAggroTarget === jad) {
                playerAggro = i + 1;
                break;
            }
        }

        // If not a Jad, check if it's a healer
        if (playerAggro === 0) {
            for (let jadIdx = 0; jadIdx < config.jadCount; jadIdx++) {
                for (let healerIdx = 0; healerIdx < config.healersPerJad; healerIdx++) {
                    const healer = jadRegion.getHealer(jadIdx, healerIdx);
                    if (healer && playerAggroTarget === healer) {
                        playerAggro = config.jadCount + jadIdx * config.healersPerJad + healerIdx + 1;
                        break;
                    }
                }
                if (playerAggro !== 0) break;
            }
        }
    }

    return playerAggro;
}

/**
 * Build Jad state array for observation.
 */
export function buildJadStates(
    jadRegion: JadRegion,
    config: JadConfig,
    attackStates: JadAttackState[]
): JadState[] {
    const jads: JadState[] = [];

    for (let i = 0; i < config.jadCount; i++) {
        const jad = jadRegion.getJad(i);
        const attackState = attackStates[i];

        if (jad) {
            jads.push({
                hp: jad.currentStats?.hitpoint ?? 0,
                attack: attackState?.attack ?? 0,
                x: jad.location.x,
                y: jad.location.y,
                alive: jad.dying === -1 && (jad.currentStats?.hitpoint ?? 0) > 0,
            });
        } else {
            jads.push({ hp: 0, attack: 0, x: 0, y: 0, alive: false });
        }
    }

    return jads;
}

/**
 * Build healer state array for observation (flattened).
 */
export function buildHealerStates(
    jadRegion: JadRegion,
    config: JadConfig
): { healers: HealerState[]; healersSpawned: boolean } {
    const healers: HealerState[] = [];
    let healersSpawned = false;

    for (let jadIdx = 0; jadIdx < config.jadCount; jadIdx++) {
        for (let healerIdx = 0; healerIdx < config.healersPerJad; healerIdx++) {
            const healer = jadRegion.getHealer(jadIdx, healerIdx);
            if (healer) {
                healersSpawned = true;
                healers.push({
                    hp: healer.currentStats?.hitpoint ?? 0,
                    x: healer.location.x,
                    y: healer.location.y,
                    aggro: jadRegion.getHealerAggro(jadIdx, healerIdx),
                });
            } else {
                healers.push({
                    hp: 0,
                    x: 0,
                    y: 0,
                    aggro: HealerAggro.NOT_PRESENT,
                });
            }
        }
    }

    return { healers, healersSpawned };
}

/**
 * Update Jad attack tracking state based on current game state.
 * Detects when a Jad starts an attack based on attackDelay changes.
 */
export function updateJadAttackTracking(
    jadRegion: JadRegion,
    config: JadConfig,
    attackStates: JadAttackState[]
): void {
    for (let i = 0; i < config.jadCount; i++) {
        const jad = jadRegion.getJad(i);
        if (!jad) continue;

        const state = attackStates[i];
        const currentDelay = jad.attackDelay;

        // Detect actual attack: attackDelay was low (0-1), now it's high (just reset after attacking)
        if (state.prevAttackDelay <= 1 && currentDelay > 1) {
            const style = jad.attackStyle;
            if (style === 'magic') {
                state.attack = 1;
                state.ticksRemaining = 4;
            } else if (style === 'range') {
                state.attack = 2;
                state.ticksRemaining = 4;
            } else {
                state.attack = 3;
                state.ticksRemaining = 2;
            }
        }

        // Decrement visibility timer
        if (state.ticksRemaining > 0) {
            state.ticksRemaining--;
            if (state.ticksRemaining === 0) {
                state.attack = 0;
            }
        }

        state.prevAttackDelay = currentDelay;
    }
}

/**
 * Build complete observation from game state.
 */
export function buildObservation(
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig,
    attackStates: JadAttackState[],
    startingDoses: { bastion: number; saraBrew: number; superRestore: number }
): JadObservation {
    const { activePrayer, rigourActive } = getActivePrayer(player);
    const { bastionDoses, saraBrewDoses, superRestoreDoses } = countPotionDoses(player);
    const playerAggro = getPlayerAggroTarget(player, jadRegion, config);
    const jads = buildJadStates(jadRegion, config, attackStates);
    const { healers, healersSpawned } = buildHealerStates(jadRegion, config);

    return {
        jad_count: config.jadCount,
        healers_per_jad: config.healersPerJad,

        player_hp: player?.currentStats?.hitpoint ?? 0,
        player_prayer: player?.currentStats?.prayer ?? 0,
        player_ranged: player?.currentStats?.range ?? 99,
        player_defence: player?.currentStats?.defence ?? 99,
        player_x: player?.location?.x ?? 0,
        player_y: player?.location?.y ?? 0,
        player_aggro: playerAggro,

        active_prayer: activePrayer,
        rigour_active: rigourActive,

        bastion_doses: bastionDoses,
        sara_brew_doses: saraBrewDoses,
        super_restore_doses: superRestoreDoses,

        jads,
        healers,
        healers_spawned: healersSpawned,

        starting_bastion_doses: startingDoses.bastion,
        starting_sara_brew_doses: startingDoses.saraBrew,
        starting_super_restore_doses: startingDoses.superRestore,
    };
}

export function initializeAttackStates(jadRegion: JadRegion, config: JadConfig): JadAttackState[] {
    const states: JadAttackState[] = [];
    for (let i = 0; i < config.jadCount; i++) {
        const jad = jadRegion.getJad(i);
        states.push({
            attack: 0,
            ticksRemaining: 0,
            prevAttackDelay: jad?.attackDelay ?? 0,
        });
    }
    return states;
}
