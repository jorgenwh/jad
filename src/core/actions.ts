import { Player, Potion } from 'osrs-sdk';
import { JadRegion } from './jad-region';
import { JadConfig, Observation } from './types';

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

export function getActionSpaceSize(config: JadConfig): number {
    const numJads = config.jadCount;
    const numHealers = numJads * config.healersPerJad;
    // 1 (DO_NOTHING) + numJads + numHealers + 7 (prayers/potions)
    return 1 + numJads + numHealers + 7;
}

/**
 * Builds a boolean mask indicating which actions are valid for the current state.
 *
 * Action structure:
 * - 0: DO_NOTHING (always valid)
 * - 1..N: AGGRO_JAD_1 through AGGRO_JAD_N (valid if Jad is alive)
 * - N+1..N+N*H: AGGRO_HEALER (valid if healer is spawned and alive)
 * - N+N*H+1..N+N*H+7: prayers/potions
 *   - +0: TOGGLE_PROTECT_MELEE (always valid)
 *   - +1: TOGGLE_PROTECT_MISSILES (always valid)
 *   - +2: TOGGLE_PROTECT_MAGIC (always valid)
 *   - +3: TOGGLE_RIGOUR (always valid)
 *   - +4: DRINK_BASTION (valid if doses > 0)
 *   - +5: DRINK_SUPER_RESTORE (valid if doses > 0)
 *   - +6: DRINK_SARA_BREW (valid if doses > 0)
 */
export function buildValidActionMask(
    jadRegion: JadRegion,
    config: JadConfig,
    observation: Observation
): boolean[] {
    const numJads = config.jadCount;
    const healersPerJad = config.healersPerJad;
    const numHealers = numJads * healersPerJad;
    const actionSpaceSize = getActionSpaceSize(config);

    const mask: boolean[] = new Array(actionSpaceSize).fill(false);

    // Action 0: DO_NOTHING - always valid
    mask[0] = true;

    // Actions 1..numJads: AGGRO_JAD - valid if Jad is alive
    for (let i = 0; i < numJads; i++) {
        const jad = jadRegion.getJad(i);
        mask[1 + i] = jad !== null;
    }

    // Actions numJads+1..numJads+numHealers: AGGRO_HEALER - valid if healer exists
    for (let jadIdx = 0; jadIdx < numJads; jadIdx++) {
        for (let healerIdx = 0; healerIdx < healersPerJad; healerIdx++) {
            const healer = jadRegion.getHealer(jadIdx, healerIdx);
            const actionIdx = 1 + numJads + jadIdx * healersPerJad + healerIdx;
            mask[actionIdx] = healer !== null;
        }
    }

    const prayerPotionBase = 1 + numJads + numHealers;

    // Prayer toggles - always valid (can toggle on/off regardless of prayer points)
    mask[prayerPotionBase + 0] = true; // TOGGLE_PROTECT_MELEE
    mask[prayerPotionBase + 1] = true; // TOGGLE_PROTECT_MISSILES
    mask[prayerPotionBase + 2] = true; // TOGGLE_PROTECT_MAGIC
    mask[prayerPotionBase + 3] = true; // TOGGLE_RIGOUR

    // Potions - valid if doses > 0
    mask[prayerPotionBase + 4] = observation.bastion_doses > 0;
    mask[prayerPotionBase + 5] = observation.super_restore_doses > 0;
    mask[prayerPotionBase + 6] = observation.sara_brew_doses > 0;

    return mask;
}
