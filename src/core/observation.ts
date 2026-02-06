import { Player, Potion } from 'osrs-sdk';
import { JadRegion } from './jad-region';
import { JadConfig, Observation, JadState, HealerState, HealerTarget } from './types';

export function countPotionDoses(player: Player): {
    bastionDoses: number;
    saraBrewDoses: number;
    superRestoreDoses: number;
} {
    if (!player || !player.inventory) {
        return { bastionDoses: 0, saraBrewDoses: 0, superRestoreDoses: 0 };
    }

    let bastionDoses = 0;
    let saraBrewDoses = 0;
    let superRestoreDoses = 0;

    for (const item of player.inventory) {
        if (!item || !(item instanceof Potion) || item.doses <= 0) {
            continue;
        }

        const itemName = item.itemName?.toString().toLowerCase() || '';
        if (itemName.includes('bastion')) {
            bastionDoses += item.doses;
        } else if (itemName.includes('saradomin brew')) {
            saraBrewDoses += item.doses;
        } else if (itemName.includes('restore')) {
            superRestoreDoses += item.doses;
        }
    }

    return { bastionDoses, saraBrewDoses, superRestoreDoses };
}

export function getActivePrayer(player: Player): { activePrayer: number; rigourActive: boolean } {
    const prayerController = player.prayerController;
    if (!prayerController) {
        return { activePrayer: 0, rigourActive: false };
    }

    let activePrayer = 0;
    let rigourActive = false;

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

    return { activePrayer, rigourActive };
}

export function getPlayerTarget(
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig
): number {
    const target = player.aggro;
    if (!target) {
        return 0; // no target
    }

    // Check if target is one of the Jads in our region
    for (let i = 0; i < config.jadCount; i++) {
        const jad = jadRegion.getJad(i);
        if (jad && jad === target) {
            return i + 1;
        }
    }

    // Check if target is one of the healers in our region
    for (let jadIdx = 0; jadIdx < config.jadCount; jadIdx++) {
        for (let healerIdx = 0; healerIdx < config.healersPerJad; healerIdx++) {
            const healer = jadRegion.getHealer(jadIdx, healerIdx);
            if (healer && healer === target) {
                return config.jadCount + jadIdx * config.healersPerJad + healerIdx + 1;
            }
        }
    }

    // Target not found among alive entities - player may be targeting a dead entity
    return 0;
}

export function getJadStates(
    jadRegion: JadRegion,
    config: JadConfig
): JadState[] {
    const jads: JadState[] = [];

    for (let i = 0; i < config.jadCount; i++) {
        const jad = jadRegion.getJad(i);
        if (jad) {
            const projectile = jadRegion.getProjectileState(i);
            jads.push({
                hp: jad.currentStats?.hitpoint ?? 0,
                attack: projectile.type,
                ticks_until_impact: projectile.ticksRemaining,
                x: jad.location.x,
                y: jad.location.y,
                alive: jad.dying === -1 && (jad.currentStats?.hitpoint ?? 0) > 0,
            });
        } else {
            jads.push({ hp: 0, attack: 0, ticks_until_impact: 0, x: 0, y: 0, alive: false });
        }
    }

    return jads;
}

export function getHealerStates(
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
                    target: jadRegion.getHealerTarget(jadIdx, healerIdx),
                });
            } else {
                healers.push({
                    hp: 0,
                    x: 0,
                    y: 0,
                    target: HealerTarget.NOT_PRESENT,
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
    const target = getPlayerTarget(player, jadRegion, config);
    const { activePrayer, rigourActive } = getActivePrayer(player);
    const jads = getJadStates(jadRegion, config);
    const { healers, healersSpawned } = getHealerStates(jadRegion, config);
    const { bastionDoses, saraBrewDoses, superRestoreDoses } = countPotionDoses(player);

    // Find the soonest-landing projectile across all Jads
    let nextProjectileType = 0;
    let nextProjectileTicks = 0;
    for (const jad of jads) {
        if (jad.ticks_until_impact > 0) {
            if (nextProjectileTicks === 0 || jad.ticks_until_impact < nextProjectileTicks) {
                nextProjectileType = jad.attack;
                nextProjectileTicks = jad.ticks_until_impact;
            }
        }
    }

    return {
        player_hp: player?.currentStats?.hitpoint ?? 0,
        player_prayer: player?.currentStats?.prayer ?? 0,
        player_ranged: player?.currentStats?.range ?? 99,
        player_defence: player?.currentStats?.defence ?? 99,
        player_location_x: player?.location?.x ?? 0,
        player_location_y: player?.location?.y ?? 0,
        player_target: target,

        active_prayer: activePrayer,
        rigour_active: rigourActive,

        jads,
        healers,
        healers_spawned: healersSpawned,

        next_projectile_type: nextProjectileType,
        next_projectile_ticks: nextProjectileTicks,

        bastion_doses: bastionDoses,
        sara_brew_doses: saraBrewDoses,
        super_restore_doses: superRestoreDoses,
        starting_bastion_doses: startingDoses.bastion,
        starting_sara_brew_doses: startingDoses.saraBrew,
        starting_super_restore_doses: startingDoses.superRestore,
    };
}
