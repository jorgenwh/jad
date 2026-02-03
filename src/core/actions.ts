import { Player, Potion } from 'osrs-sdk';
import { JadRegion } from './jad-region';
import {
    JadConfig,
    Observation,
    ProtectionPrayerAction,
    OffensivePrayerAction,
    PotionAction,
    TARGET_NO_OP,
    PROTECTION_PRAYER_SIZE,
    OFFENSIVE_PRAYER_SIZE,
    POTION_SIZE,
    getTargetHeadSize,
} from './types';

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
 * Execute protection prayer action (head 0).
 * Uses toggle semantics - click to toggle the prayer on/off.
 */
function executeProtectionPrayer(action: number, player: Player): void {
    switch (action) {
        case ProtectionPrayerAction.NO_OP:
            break;
        case ProtectionPrayerAction.PROTECT_MAGIC:
            togglePrayer(player, 'Protect from Magic');
            break;
        case ProtectionPrayerAction.PROTECT_RANGE:
            togglePrayer(player, 'Protect from Range');
            break;
        case ProtectionPrayerAction.PROTECT_MELEE:
            togglePrayer(player, 'Protect from Melee');
            break;
    }
}

/**
 * Execute offensive prayer action (head 1).
 * Uses toggle semantics - click to toggle Rigour on/off.
 */
function executeOffensivePrayer(action: number, player: Player): void {
    switch (action) {
        case OffensivePrayerAction.NO_OP:
            break;
        case OffensivePrayerAction.TOGGLE_RIGOUR:
            togglePrayer(player, 'Rigour');
            break;
    }
}

/**
 * Execute potion action (head 2).
 */
function executePotion(action: number, player: Player): void {
    switch (action) {
        case PotionAction.NONE:
            break;
        case PotionAction.BASTION:
            drinkPotion(player, 'bastion');
            break;
        case PotionAction.SUPER_RESTORE:
            drinkPotion(player, 'restore');
            break;
        case PotionAction.SARA_BREW:
            drinkPotion(player, 'saradomin brew');
            break;
    }
}

/**
 * Execute target action (head 3).
 * 0 = no-op (keep current target)
 * 1..N = attack Jad 1..N
 * N+1.. = attack healer
 */
function executeTarget(
    action: number,
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig
): void {
    if (action === TARGET_NO_OP) {
        return;
    }

    const numJads = config.jadCount;
    const healersPerJad = config.healersPerJad;

    // Actions 1..numJads are Jad targets
    if (action <= numJads) {
        const jadIndex = action - 1;
        attackJad(player, jadRegion, jadIndex);
        return;
    }

    // Actions numJads+1.. are healer targets
    const healerAction = action - numJads - 1;
    const jadIndex = Math.floor(healerAction / healersPerJad);
    const healerIndex = healerAction % healersPerJad;
    attackHealer(player, jadRegion, jadIndex, healerIndex);
}

/**
 * Execute a MultiDiscrete action (array of 4 head values).
 *
 * Action heads:
 * - [0] protection_prayer: 0=no-op, 1=mage, 2=range, 3=melee
 * - [1] offensive_prayer: 0=no-op, 1=toggle_rigour
 * - [2] potion: 0=none, 1=bastion, 2=super_restore, 3=sara_brew
 * - [3] target: 0=no-op, 1..N=jad, N+1..=healer
 */
export function executeAction(
    action: number[],
    player: Player,
    jadRegion: JadRegion,
    config: JadConfig
): void {
    executeProtectionPrayer(action[0], player);
    executeOffensivePrayer(action[1], player);
    executePotion(action[2], player);
    executeTarget(action[3], player, jadRegion, config);
}

/**
 * Get the dimensions of the MultiDiscrete action space.
 * Returns [protection_prayer_size, offensive_prayer_size, potion_size, target_size].
 */
export function getActionSpaceDims(config: JadConfig): number[] {
    return [
        PROTECTION_PRAYER_SIZE,
        OFFENSIVE_PRAYER_SIZE,
        POTION_SIZE,
        getTargetHeadSize(config),
    ];
}

/**
 * Builds per-head boolean masks indicating which actions are valid for each head.
 *
 * Returns array of 4 boolean arrays, one per head:
 * - [0] protection_prayer: always all valid
 * - [1] offensive_prayer: always all valid
 * - [2] potion: [true, bastion>0, restore>0, brew>0]
 * - [3] target: [true, jad1_alive, ..., healer1_alive, ...]
 */
export function buildValidActionMask(
    jadRegion: JadRegion,
    config: JadConfig,
    observation: Observation
): boolean[][] {
    const numJads = config.jadCount;
    const healersPerJad = config.healersPerJad;
    const targetHeadSize = getTargetHeadSize(config);

    // Head 0: Protection prayer - always valid
    const protectionMask = new Array(PROTECTION_PRAYER_SIZE).fill(true);

    // Head 1: Offensive prayer - always valid
    const offensiveMask = new Array(OFFENSIVE_PRAYER_SIZE).fill(true);

    // Head 2: Potions - valid if doses > 0
    const potionMask = [
        true,  // NONE always valid
        observation.bastion_doses > 0,
        observation.super_restore_doses > 0,
        observation.sara_brew_doses > 0,
    ];

    // Head 3: Target - valid if target is alive
    const targetMask = new Array(targetHeadSize).fill(false);

    // Action 0: no-op always valid
    targetMask[0] = true;

    // Actions 1..numJads: Jads - valid if alive
    for (let i = 0; i < numJads; i++) {
        const jad = jadRegion.getJad(i);
        targetMask[1 + i] = jad !== null;
    }

    // Actions numJads+1..: Healers - valid if exists
    for (let jadIdx = 0; jadIdx < numJads; jadIdx++) {
        for (let healerIdx = 0; healerIdx < healersPerJad; healerIdx++) {
            const healer = jadRegion.getHealer(jadIdx, healerIdx);
            const actionIdx = 1 + numJads + jadIdx * healersPerJad + healerIdx;
            targetMask[actionIdx] = healer !== null;
        }
    }

    return [protectionMask, offensiveMask, potionMask, targetMask];
}
