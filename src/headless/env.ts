import { World, Region, Player, Trainer } from 'osrs-sdk';
import {
    JadRegion,
    JadConfig,
    DEFAULT_CONFIG,
    JadObservation,
    JadAttackState,
    countPotionDoses,
    buildObservation,
    updateJadAttackTracking,
    initializeAttackStates,
    executeAction,
    computeReward,
    TerminationState,
} from '../core';

export interface EnvConfig {
    rewardFunc?: string;
}

const DEFAULT_ENV_CONFIG: EnvConfig = {
    rewardFunc: 'default',
};

export interface StepResult {
    observation: JadObservation;
    reward: number;
    terminated: boolean;
}

export class HeadlessEnv {
    private world: World;
    private region: Region;
    private player!: Player;
    private jadConfig: JadConfig;
    private envConfig: EnvConfig;
    private createRegionFunc: (config: JadConfig) => Region;

    // Track attack state per Jad
    private jadAttackStates: JadAttackState[] = [];

    // Track starting potion doses for normalization
    private startingDoses = { bastion: 0, saraBrew: 0, superRestore: 0 };

    // Track for reward computation
    private prevObservation: JadObservation | null = null;
    private episodeLength: number = 0;

    constructor(
        createRegionFunc: (config: JadConfig) => Region,
        jadConfig: JadConfig = DEFAULT_CONFIG,
        envConfig: EnvConfig = DEFAULT_ENV_CONFIG
    ) {
        this.jadConfig = jadConfig;
        this.envConfig = envConfig;
        this.createRegionFunc = createRegionFunc;

        this.world = new World();
        this.region = createRegionFunc(jadConfig);
        this.initialize();
    }

    private initialize(): void {
        this.region.world = this.world;
        this.world.addRegion(this.region);
        Trainer.setPlayer(this.player);

        const result = (this.region as { initialiseRegion(): { player: Player } }).initialiseRegion();
        this.player = result.player;

        this.jadAttackStates = initializeAttackStates(this.region as JadRegion, this.jadConfig);

        this.captureStartingDoses();

        this.prevObservation = null;
        this.episodeLength = 0;
    }

    private captureStartingDoses(): void {
        const doses = countPotionDoses(this.player);
        this.startingDoses = {
            bastion: doses.bastionDoses,
            saraBrew: doses.saraBrewDoses,
            superRestore: doses.superRestoreDoses,
        };
    }

    reset(): StepResult {
        this.world = new World();
        this.region = this.createRegionFunc(this.jadConfig);
        this.initialize();

        const observation = this.getObservation();

        return {
            observation,
            reward: 0,
            terminated: false,
        };
    }

    step(action: number): StepResult {
        // Store previous observation for reward computation
        this.prevObservation = this.getObservation();

        // Execute action before tick
        executeAction(action, this.player, this.region as JadRegion, this.jadConfig);

        // Update Jad attack tracking before tick
        updateJadAttackTracking(this.region as JadRegion, this.jadConfig, this.jadAttackStates);

        // Advance simulation by one tick
        this.world.tickWorld(1);

        // Increment episode length
        this.episodeLength++;

        // Get observation after tick
        const observation = this.getObservation();

        // Determine termination state
        const termination = this.getTerminationState();

        // Compute reward
        const reward = computeReward(
            observation,
            this.prevObservation,
            termination,
            this.episodeLength,
            this.envConfig.rewardFunc
        );

        return {
            observation,
            reward,
            terminated: termination === TerminationState.PLAYER_DIED || termination === TerminationState.JAD_KILLED,
        };
    }

    private getTerminationState(): TerminationState {
        // Check player death
        const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
        if (playerDead) {
            return TerminationState.PLAYER_DIED;
        }

        // Check if ALL Jads are dead
        const jadRegion = this.region as JadRegion;
        let allJadsDead = true;
        for (let i = 0; i < this.jadConfig.jadCount; i++) {
            const jad = jadRegion.getJad(i);
            if (jad && jad.currentStats.hitpoint > 0 && jad.dying === -1) {
                allJadsDead = false;
                break;
            }
        }
        if (allJadsDead) {
            return TerminationState.JAD_KILLED;
        }

        return TerminationState.ONGOING;
    }

    private getObservation(): JadObservation {
        return buildObservation(
            this.player,
            this.region as JadRegion,
            this.jadConfig,
            this.jadAttackStates,
            this.startingDoses
        );
    }
}
