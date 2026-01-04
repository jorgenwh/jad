import { Mob, BasePrayer, Item, Potion } from 'osrs-sdk';
import { JadRegion } from '../../core/jad-region';
import { JadConfig } from '../../core/types';
import { Jad } from '../../core/jad';

export class ActionRecorder {
    private jadRegion: JadRegion;
    private config: JadConfig;
    private pendingActions: number[] = [];

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
            this.pendingActions.push(actionIndex);
        }
    }

    recordPrayerToggle(prayer: BasePrayer): void {
        const actionIndex = this.getPrayerActionIndex(prayer);
        if (actionIndex !== null) {
            this.pendingActions.push(actionIndex);
        }
    }

    recordPotionUse(item: Item): void {
        const actionIndex = this.getPotionActionIndex(item);
        if (actionIndex !== null) {
            this.pendingActions.push(actionIndex);
        }
    }

    consumePendingActions(): number[] {
        const actions = [...this.pendingActions];
        this.pendingActions = [];
        return actions;
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

    private getPrayerActionIndex(prayer: BasePrayer): number | null {
        const baseIndex = 1 + this.config.jadCount + this.config.jadCount * this.config.healersPerJad;
        const prayerName = prayer.name.toLowerCase();

        if (prayerName.includes('melee')) {
            return baseIndex + 0;
        } else if (prayerName.includes('range') || prayerName.includes('missiles')) {
            return baseIndex + 1;
        } else if (prayerName.includes('magic')) {
            return baseIndex + 2;
        } else if (prayerName.includes('rigour')) {
            return baseIndex + 3;
        }

        return null;
    }

    private getPotionActionIndex(item: Item): number | null {
        if (!(item instanceof Potion)) {
            return null;
        }

        const baseIndex = 1 + this.config.jadCount + this.config.jadCount * this.config.healersPerJad;
        const itemName = item.itemName?.toString().toLowerCase() || '';

        if (itemName.includes('bastion')) {
            return baseIndex + 4;
        } else if (itemName.includes('restore')) {
            return baseIndex + 5;
        } else if (itemName.includes('saradomin brew')) {
            return baseIndex + 6;
        }

        return null;
    }
}
