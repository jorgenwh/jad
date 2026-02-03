import { Mob, BasePrayer, Item, Potion } from 'osrs-sdk';
import { JadRegion } from '../../core/jad-region';
import { JadConfig } from '../../core/types';
import { Jad } from '../../core/jad';

/**
 * Records player actions in MultiDiscrete format.
 * Each tick produces a 4-element action array:
 * [protection_prayer, offensive_prayer, potion, target]
 */
export class ActionRecorder {
    private jadRegion: JadRegion;
    private config: JadConfig;

    // Pending actions for current tick (one per head)
    private pendingProtectionPrayer: number = 0;
    private pendingOffensivePrayer: number = 0;
    private pendingPotion: number = 0;
    private pendingTarget: number = 0;

    constructor(jadRegion: JadRegion, config: JadConfig) {
        this.jadRegion = jadRegion;
        this.config = config;
    }

    recordTargetChange(target: Mob | null): void {
        if (!target) {
            return;
        }

        const actionIndex = this.getTargetActionIndex(target);
        if (actionIndex !== null) {
            this.pendingTarget = actionIndex;
        }
    }

    recordPrayerToggle(prayer: BasePrayer): void {
        const prayerName = prayer.name.toLowerCase();

        // Protection prayers (head 0)
        if (prayerName.includes('magic')) {
            this.pendingProtectionPrayer = 1;
        } else if (prayerName.includes('range') || prayerName.includes('missiles')) {
            this.pendingProtectionPrayer = 2;
        } else if (prayerName.includes('melee')) {
            this.pendingProtectionPrayer = 3;
        }
        // Offensive prayer (head 1)
        else if (prayerName.includes('rigour')) {
            this.pendingOffensivePrayer = 1;
        }
    }

    recordPotionUse(item: Item): void {
        if (!(item instanceof Potion)) {
            return;
        }

        const itemName = item.itemName?.toString().toLowerCase() || '';

        if (itemName.includes('bastion')) {
            this.pendingPotion = 1;
        } else if (itemName.includes('restore')) {
            this.pendingPotion = 2;
        } else if (itemName.includes('saradomin brew')) {
            this.pendingPotion = 3;
        }
    }

    /**
     * Consume and return the action for the current tick as a 4-element array.
     * Format: [protection_prayer, offensive_prayer, potion, target]
     */
    consumePendingAction(): number[] {
        const action = [
            this.pendingProtectionPrayer,
            this.pendingOffensivePrayer,
            this.pendingPotion,
            this.pendingTarget,
        ];

        // Reset for next tick
        this.pendingProtectionPrayer = 0;
        this.pendingOffensivePrayer = 0;
        this.pendingPotion = 0;
        this.pendingTarget = 0;

        return action;
    }

    /**
     * @deprecated Use consumePendingAction() instead
     */
    consumePendingActions(): number[] {
        return this.consumePendingAction();
    }

    private getTargetActionIndex(target: Mob): number | null {
        // Check if target is a Jad (by checking getJadIndex method)
        if (target instanceof Jad) {
            const jadIndex = target.getJadIndex();
            if (jadIndex >= 0 && jadIndex < this.config.jadCount) {
                return 1 + jadIndex;
            }
        }

        // Check if target is a Jad by reference (fallback)
        for (let i = 0; i < this.config.jadCount; i++) {
            const jad = this.jadRegion.getJad(i);
            if (jad && jad === target) {
                return 1 + i;
            }
        }

        // Check if target is a healer by iterating through all registered healers
        for (let jadIdx = 0; jadIdx < this.config.jadCount; jadIdx++) {
            for (let healerIdx = 0; healerIdx < this.config.healersPerJad; healerIdx++) {
                const healer = this.jadRegion.getHealer(jadIdx, healerIdx);
                if (healer && healer === target) {
                    // Action index: 1 + jadCount + jadIdx * healersPerJad + healerIdx
                    return 1 + this.config.jadCount + jadIdx * this.config.healersPerJad + healerIdx;
                }
            }
        }

        return null;
    }
}
