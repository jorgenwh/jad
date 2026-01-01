import { Player, Potion } from 'osrs-sdk';
import { JadRegion } from './jad-region';
import { JadConfig, Observation, JadState, HealerState, HealerAggro } from './types';

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
 * Get attack value for a Jad.
 * Attack is only reported on the tick it fires (when attackDelay just reset to attackSpeed).
 */
function getJadAttack(jad: { attackDelay: number; attackSpeed: number; attackStyle: string }): number {
    // Attack just fired this tick if delay equals attack speed (just reset)
    if (jad.attackDelay === jad.attackSpeed) {
        switch (jad.attackStyle) {
            case 'magic': return 1;
            case 'range': return 2;
            default: return 3; // melee
        }
    }
    return 0; // No attack this tick
}

export function buildJadStates(
    jadRegion: JadRegion,
    config: JadConfig
): JadState[] {
    const jads: JadState[] = [];

    for (let i = 0; i < config.jadCount; i++) {
        const jad = jadRegion.getJad(i);

        if (jad) {
            jads.push({
                hp: jad.currentStats?.hitpoint ?? 0,
                attack: getJadAttack(jad),
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

export function buildObservation(
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig,
    startingDoses: { bastion: number; saraBrew: number; superRestore: number }
): Observation {
    const { activePrayer, rigourActive } = getActivePrayer(player);
    const { bastionDoses, saraBrewDoses, superRestoreDoses } = countPotionDoses(player);
    const playerAggro = getPlayerAggroTarget(player, jadRegion, config);
    const jads = buildJadStates(jadRegion, config);
    const { healers, healersSpawned } = buildHealerStates(jadRegion, config);

    return {
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

