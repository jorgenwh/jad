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

function attackJad(player: Player, jadRegion: JadRegion, jadIndex: number): void {
    const jad = jadRegion.getJad(jadIndex);
    if (!jad) {
        // Jad not present or dead; no action
        return;
    }
    player.setAggro(jad);
}

function attackHealer(player: Player, jadRegion: JadRegion, jadIndex: number, healerIndex: number): void {
    const healer = jadRegion.getHealer(jadIndex, healerIndex);
    if (!healer) {
        // Healer not present or dead; no action
        return;
    }
    player.setAggro(healer);
}

/**
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
    if (action === 0) {
        return;
    }

    let relativeAction = action - 1;

    const numJads = config.jadCount;
    const healersPerJad = config.healersPerJad;
    const numHealers = numJads * healersPerJad;

    if (relativeAction < numJads) {
        attackJad(player, jadRegion, relativeAction);
        return;
    }
    relativeAction -= numJads;

    if (relativeAction < numHealers) {
        const jadIndex = Math.floor(relativeAction / healersPerJad);
        const healerIndex = relativeAction % healersPerJad;
        attackHealer(player, jadRegion, jadIndex, healerIndex);
        return;
    }
    relativeAction -= numHealers;

    switch (relativeAction) {
        case 0:
            togglePrayer(player, 'Protect from Melee');
            break;
        case 1:
            togglePrayer(player, 'Protect from Range');
            break;
        case 2:
            togglePrayer(player, 'Protect from Magic');
            break;
        case 3:
            togglePrayer(player, 'Rigour');
            break;
        case 4:
            drinkPotion(player, 'bastion');
            break;
        case 5:
            drinkPotion(player, 'restore');
            break;
        case 6:
            drinkPotion(player, 'saradomin brew');
            break;
    }
}
