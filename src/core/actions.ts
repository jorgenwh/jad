import { Player, Potion } from 'osrs-sdk';
import { JadRegion } from './jad-region';
import { JadConfig } from './types';

export function togglePrayer(player: Player, prayerName: string): void {
    const prayerController = player.prayerController;
    if (!prayerController) {
        return;
    }

    const targetPrayer = prayerController.findPrayerByName(prayerName);
    if (!targetPrayer) {
        return;
    }

    if (targetPrayer.isActive) {
        targetPrayer.deactivate();
    } else {
        // Can't activate prayers with 0 prayer points
        if ((player.currentStats?.prayer ?? 0) > 0) {
            targetPrayer.activate(player);
        }
    }
}

export function drinkPotion(player: Player, potionType: string): void {
    if (!player || !player.inventory) {
        return;
    }

    for (const item of player.inventory) {
        if (!item) {
            continue;
        }
        if (!(item instanceof Potion)) {
            continue;
        }
        if (item.doses <= 0) {
            continue;
        }

        const itemName = item.itemName?.toString().toLowerCase() || '';
        if (!itemName.includes(potionType)) {
            continue;
        }

        item.inventoryLeftClick(player);
        break;
    }
}

/**
 * Execute action based on dynamic action space.
 *
 * Action structure:
 * - 0: DO_NOTHING
 * - 1..N: AGGRO_JAD_1 through AGGRO_JAD_N
 * - N+1..N+N*H: AGGRO_HEALER (encoded)
 * - N+N*H+1..N+N*H+7: prayers/potions
 *   - +0: TOGGLE_PROTECT_MELEE
 *   - +1: TOGGLE_PROTECT_MISSILES
 *   - +2: TOGGLE_PROTECT_MAGIC
 *   - +3: TOGGLE_RIGOUR
 *   - +4: DRINK_BASTION
 *   - +5: DRINK_SUPER_RESTORE
 *   - +6: DRINK_SARA_BREW
 */
export function executeAction(
    action: number,
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig
): void {
    // Action 0: DO_NOTHING
    if (action === 0) {
        return;
    }

    const numJads = config.jadCount;
    const healersPerJad = config.healersPerJad;
    const totalHealers = numJads * healersPerJad;

    // Actions 1..N: AGGRO_JAD_1 through AGGRO_JAD_N
    if (action >= 1 && action <= numJads) {
        const jadIndex = action - 1;
        const jad = jadRegion.getJad(jadIndex);
        if (jad) {
            console.log(`[AGGRO] Targeting Jad ${jadIndex} (HP: ${jad.currentStats.hitpoint})`);
            player.setAggro(jad);
        } else {
            // Debug: check why jad is null
            const rawJad = (jadRegion as any)._jads[jadIndex];
            if (!rawJad) {
                console.log(`[AGGRO] Jad ${jadIndex} not found (never spawned)`);
            } else {
                console.log(`[AGGRO] Jad ${jadIndex} is dead (HP: ${rawJad.currentStats.hitpoint}, dying: ${rawJad.dying})`);
            }
        }
        return;
    }

    // Actions N+1..N+N*H: AGGRO_HEALER
    const healerActionStart = numJads + 1;
    const healerActionEnd = numJads + totalHealers;
    if (action >= healerActionStart && action <= healerActionEnd) {
        const healerActionIndex = action - healerActionStart;
        const jadIndex = Math.floor(healerActionIndex / healersPerJad);
        const healerIndex = healerActionIndex % healersPerJad;
        const healer = jadRegion.getHealer(jadIndex, healerIndex);
        if (healer) {
            console.log(`[AGGRO] Targeting Healer [${jadIndex}][${healerIndex}] (HP: ${healer.currentStats.hitpoint})`);
            player.setAggro(healer);
        } else {
            // Debug: check why healer is null
            const healerTuple = (jadRegion as any)._healers.get(jadIndex);
            const rawHealer = healerTuple?.[healerIndex];
            if (!rawHealer) {
                console.log(`[AGGRO] Healer [${jadIndex}][${healerIndex}] not spawned yet`);
            } else {
                console.log(`[AGGRO] Healer [${jadIndex}][${healerIndex}] is dead (HP: ${rawHealer.currentStats.hitpoint}, dying: ${rawHealer.dying})`);
            }
        }
        return;
    }

    // Prayer/potion actions (offset by aggro actions)
    const prayerPotionActionStart = numJads + totalHealers + 1;
    const relativeAction = action - prayerPotionActionStart;

    switch (relativeAction) {
        case 0: // TOGGLE_PROTECT_MELEE
            togglePrayer(player, 'Protect from Melee');
            break;
        case 1: // TOGGLE_PROTECT_MISSILES
            togglePrayer(player, 'Protect from Range');
            break;
        case 2: // TOGGLE_PROTECT_MAGIC
            togglePrayer(player, 'Protect from Magic');
            break;
        case 3: // TOGGLE_RIGOUR
            togglePrayer(player, 'Rigour');
            break;
        case 4: // DRINK_BASTION
            drinkPotion(player, 'bastion');
            break;
        case 5: // DRINK_SUPER_RESTORE
            drinkPotion(player, 'restore');
            break;
        case 6: // DRINK_SARA_BREW
            drinkPotion(player, 'saradomin brew');
            break;
    }
}
