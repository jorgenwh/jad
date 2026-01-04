import { Player, World, Unit, BasePrayer, Item, Potion } from 'osrs-sdk';
import { JadRegion } from '../../core/jad-region';
import { JadConfig } from '../../core/types';
import { ActionRecorder } from './action-recorder';
import { DataRecorder } from './data-recorder';

export class RecordingController {
    private player: Player;
    private world: World;

    private actionRecorder: ActionRecorder;
    private dataRecorder: DataRecorder;

    private setAggro: ((target: Unit) => void) | null = null;
    private prayerHooks: Map<BasePrayer, { toggle: (p: Player) => void; activate: (p: Player) => void }> = new Map();
    private potionHooks: Map<Item, (p: Player) => void> = new Map();

    constructor(player: Player, jadRegion: JadRegion, world: World, config: JadConfig) {
        this.player = player;
        this.world = world;

        this.actionRecorder = new ActionRecorder(jadRegion, config);
        this.dataRecorder = new DataRecorder(player, jadRegion, config, this.actionRecorder);

        this.setupHooks();
        this.setupUI();
        this.setupBeforeUnload();
    }

    start(): void {
        this.dataRecorder.startRecording();
        this.hookWorldTick();
    }

    private setupHooks(): void {
        this.hookSetAggro();
        this.hookPrayers();
        this.hookCurrentInventory();
    }

    private hookSetAggro(): void {
        const self = this;
        this.setAggro = this.player.setAggro.bind(this.player);

        this.player.setAggro = function(target: Unit) {
            // Only record if target is a Mob (Jad or Healer)
            if (target && 'mobName' in target) {
                self.actionRecorder.recordTargetChange(target as any);
            }
            self.setAggro!(target);
        };
    }

    private hookPrayers(): void {
        const self = this;
        const prayerController = this.player.prayerController;

        if (!prayerController) {
            return;
        }

        for (const prayer of prayerController.prayers) {
            const toggle = prayer.toggle.bind(prayer);
            const activate = prayer.activate.bind(prayer);

            this.prayerHooks.set(prayer, { toggle: toggle, activate: activate });

            prayer.toggle = function(player: Player) {
                self.actionRecorder.recordPrayerToggle(prayer);
                toggle(player);
            };

            prayer.activate = function(player: Player) {
                self.actionRecorder.recordPrayerToggle(prayer);
                activate(player);
            };
        }
    }

    private hookCurrentInventory(): void {
        const self = this;

        if (!this.player.inventory) {
            return;
        }

        for (const item of this.player.inventory) {
            if (!item || !(item instanceof Potion)) {
                continue;
            }

            // Skip if already hooked
            if (this.potionHooks.has(item)) {
                continue;
            }

            const leftClick = item.inventoryLeftClick.bind(item);
            this.potionHooks.set(item, leftClick);

            item.inventoryLeftClick = function(player: Player) {
                self.actionRecorder.recordPotionUse(item);
                leftClick(player);
            };
        }
    }

    private hookWorldTick(): void {
        const self = this;
        const tickWorld = this.world.tickWorld.bind(this.world);

        this.world.tickWorld = function(ticks: number) {
            tickWorld(ticks);

            // Re-hook potions in case inventory changed
            self.hookCurrentInventory();

            // Record this tick
            const termination = self.dataRecorder.tick();

            // Update UI
            self.updateUI();
        };
    }

    private setupUI(): void {
        // Create recording indicator
        const indicator = document.createElement('div');
        indicator.id = 'recording-indicator';
        indicator.style.cssText = `
            position: fixed;
            top: 60px;
            left: 10px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 14px;
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        indicator.innerHTML = `
            <span style="width: 12px; height: 12px; background: white; border-radius: 50%; animation: pulse 1s infinite;"></span>
            <span id="recording-status">REC</span>
            <span id="recording-ticks">0</span>
        `;

        // Add pulse animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
        `;
        document.head.appendChild(style);
        document.body.appendChild(indicator);
    }

    private updateUI(): void {
        const stats = this.dataRecorder.getStats();
        const ticksElement = document.getElementById('recording-ticks');
        if (ticksElement) {
            ticksElement.textContent = `${stats.stepCount} steps`;
        }
    }

    private setupBeforeUnload(): void {
        window.addEventListener('beforeunload', (e) => {
            const stats = this.dataRecorder.getStats();
            if (stats.isRecording && stats.stepCount > 0) {
                // Export data before page closes
                this.dataRecorder.exportCurrentData();

                // Show confirmation dialog
                e.preventDefault();
                e.returnValue = 'Recording data will be exported. Are you sure you want to leave?';
            }
        });
    }

    cleanup(): void {
        // Restore original methods
        if (this.setAggro) {
            this.player.setAggro = this.setAggro;
        }

        for (const [prayer, hooks] of this.prayerHooks) {
            prayer.toggle = hooks.toggle;
            prayer.activate = hooks.activate;
        }

        for (const [item, original] of this.potionHooks) {
            item.inventoryLeftClick = original;
        }

        this.dataRecorder.stopRecording();
    }
}
