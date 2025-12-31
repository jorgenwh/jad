/**
 * Headless environment wrapper for RL training.
 * Supports 1-6 Jads with per-Jad healers.
 */

import { World, Region, Player, Settings, Trainer } from 'osrs-sdk';
import {
    JadRegion,
    JadConfig,
    DEFAULT_CONFIG,
    getActionCount,
    JadObservation,
    JadAttackState,
    countPotionDoses,
    buildObservation,
    updateJadAttackTracking,
    initializeAttackStates,
    executeAction,
} from '../core';

// Initialize settings from storage (uses mock localStorage)
Settings.readFromStorage();

// Re-export for convenience
export { getActionCount };
export type { JadObservation };

export interface StepResult {
    observation: JadObservation;
    terminated: boolean;
}

export interface ResetResult {
    observation: JadObservation;
}

export class HeadlessEnv {
    private world: World;
    private region: Region;
    private player!: Player;
    private config: JadConfig;
    private createRegion: (config: JadConfig) => Region;

    // Track attack state per Jad
    private jadAttackStates: JadAttackState[] = [];

    // Track starting potion doses for normalization
    private startingDoses = { bastion: 0, saraBrew: 0, superRestore: 0 };

    constructor(createRegion: (config: JadConfig) => Region, config: JadConfig = DEFAULT_CONFIG) {
        this.config = config;
        this.createRegion = createRegion;
        this.world = new World();
        this.region = createRegion(config);
        this.initializeRegion();
    }

    private initializeRegion(): void {
        this.region.world = this.world;
        this.world.addRegion(this.region);

        // Initialize region and get player
        const result = (this.region as { initialiseRegion(): { player: Player } }).initialiseRegion();
        this.player = result.player;

        // Set player in Trainer (used by osrs-sdk internals)
        Trainer.setPlayer(this.player);

        // Initialize per-Jad attack tracking
        this.jadAttackStates = initializeAttackStates(this.region as JadRegion, this.config);

        // Capture starting potion doses for normalization
        this.captureStartingDoses();
    }

    private captureStartingDoses(): void {
        const doses = countPotionDoses(this.player);
        this.startingDoses = {
            bastion: doses.bastionDoses,
            saraBrew: doses.saraBrewDoses,
            superRestore: doses.superRestoreDoses,
        };
    }

    reset(): ResetResult {
        // Clear and recreate
        this.world = new World();
        this.region = this.createRegion(this.config);
        this.initializeRegion();

        return {
            observation: this.getObservation(),
        };
    }

    step(action: number): StepResult {
        // Execute action before tick
        executeAction(action, this.player, this.region as JadRegion, this.config);

        // Update Jad attack tracking before tick
        updateJadAttackTracking(this.region as JadRegion, this.config, this.jadAttackStates);

        // Advance simulation by one tick
        this.world.tickWorld(1);

        // Get observation after tick
        const observation = this.getObservation();

        // Check termination - player dead or ALL Jads dead
        const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
        const jadRegion = this.region as JadRegion;
        let allJadsDead = true;
        for (let i = 0; i < this.config.jadCount; i++) {
            const jad = jadRegion.getJad(i);
            if (jad && jad.currentStats.hitpoint > 0 && jad.dying === -1) {
                allJadsDead = false;
                break;
            }
        }
        const terminated = playerDead || allJadsDead;

        return {
            observation,
            terminated,
        };
    }

    private getObservation(): JadObservation {
        return buildObservation(
            this.player,
            this.region as JadRegion,
            this.config,
            this.jadAttackStates,
            this.startingDoses
        );
    }
}
