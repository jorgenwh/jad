/**
 * Episode state tracking for RL training.
 * Manages attack states, termination detection, and observation building.
 */

import { Player } from 'osrs-sdk';
import { JadRegion } from './jad-region';
import { JadConfig, JadObservation, JadAttackState } from './types';
import {
    countPotionDoses,
    buildObservation,
    updateJadAttackTracking,
    initializeAttackStates,
} from './observation';

export type TerminationResult = 'player_died' | 'jad_killed' | null;

export class EpisodeState {
    private player: Player;
    private jadRegion: JadRegion;
    private config: JadConfig;

    private jadAttackStates: JadAttackState[] = [];
    private startingDoses = { bastion: 0, saraBrew: 0, superRestore: 0 };

    private _cumulativeReward = 0;
    private _episodeLength = 0;
    private _terminated = false;
    private _terminationResult: TerminationResult = null;

    constructor(player: Player, jadRegion: JadRegion, config: JadConfig) {
        this.player = player;
        this.jadRegion = jadRegion;
        this.config = config;
        this.initialize();
    }

    get cumulativeReward(): number {
        return this._cumulativeReward;
    }

    set cumulativeReward(value: number) {
        this._cumulativeReward = value;
    }

    get episodeLength(): number {
        return this._episodeLength;
    }

    set episodeLength(value: number) {
        this._episodeLength = value;
    }

    get terminated(): boolean {
        return this._terminated;
    }

    get terminationResult(): TerminationResult {
        return this._terminationResult;
    }

    /**
     * Initialize or reset episode state.
     */
    initialize(): void {
        this._cumulativeReward = 0;
        this._episodeLength = 0;
        this._terminated = false;
        this._terminationResult = null;

        this.jadAttackStates = initializeAttackStates(this.jadRegion, this.config);
        this.captureStartingDoses();
    }

    /**
     * Reset for a new episode.
     */
    reset(): void {
        this.initialize();
    }

    /**
     * Update attack tracking for all Jads.
     * Should be called each tick before building observation.
     */
    updateAttackTracking(): void {
        updateJadAttackTracking(this.jadRegion, this.config, this.jadAttackStates);
    }

    /**
     * Check if the episode has terminated.
     * Returns the termination result, or null if still running.
     */
    checkTermination(): TerminationResult {
        if (this._terminated) {
            return this._terminationResult;
        }

        // Check player death
        const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
        if (playerDead) {
            this._terminated = true;
            this._terminationResult = 'player_died';
            return this._terminationResult;
        }

        // Check if all Jads are dead
        let allJadsDead = true;
        for (let i = 0; i < this.config.jadCount; i++) {
            const jad = this.jadRegion.getJad(i);
            if (jad && jad.currentStats.hitpoint > 0 && jad.dying === -1) {
                allJadsDead = false;
                break;
            }
        }

        if (allJadsDead) {
            this._terminated = true;
            this._terminationResult = 'jad_killed';
            return this._terminationResult;
        }

        return null;
    }

    /**
     * Build the current observation.
     */
    getObservation(): JadObservation {
        return buildObservation(
            this.player,
            this.jadRegion,
            this.config,
            this.jadAttackStates,
            this.startingDoses
        );
    }

    private captureStartingDoses(): void {
        const doses = countPotionDoses(this.player);
        this.startingDoses = {
            bastion: doses.bastionDoses,
            saraBrew: doses.saraBrewDoses,
            superRestore: doses.superRestoreDoses,
        };
    }
}
