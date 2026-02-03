import { World, Region, Player, Trainer } from 'osrs-sdk';
import {
    JadRegion,
    JadConfig,
    DEFAULT_CONFIG,
    Observation,
    StepResult,
    countPotionDoses,
    buildObservation,
    executeAction,
    buildValidActionMask,
    computeReward,
    checkTermination,
    TerminationState,
} from '../core';

export interface EnvConfig {
    rewardFunc?: string;
}

const DEFAULT_ENV_CONFIG: EnvConfig = {
    rewardFunc: 'default',
};

export class HeadlessEnv {
    private world: World;
    private region: Region;
    private player!: Player;
    private jadConfig: JadConfig;
    private envConfig: EnvConfig;
    private createRegionFunc: (config: JadConfig) => Region;

    // Track starting potion doses for observation normalization
    private startingDoses = { bastion: 0, saraBrew: 0, superRestore: 0 };

    private prevObservation: Observation | null = null;
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

        const result = (this.region as { initialiseRegion(): { player: Player } }).initialiseRegion();
        this.player = result.player;

        Trainer.setPlayer(this.player);

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

        const jadRegion = this.region as JadRegion;
        const observation = buildObservation(
            this.player,
            jadRegion,
            this.jadConfig,
            this.startingDoses
        );
        const validActionMask = buildValidActionMask(jadRegion, this.jadConfig, observation);

        return {
            observation,
            reward: 0,
            terminated: false,
            valid_action_mask: validActionMask,
        };
    }

    step(action: number[]): StepResult {
        const jadRegion = this.region as JadRegion;

        this.prevObservation = buildObservation(
            this.player,
            jadRegion,
            this.jadConfig,
            this.startingDoses
        );

        executeAction(action, this.player, jadRegion, this.jadConfig);

        this.world.tickWorld(1);
        this.episodeLength++;

        const observation = buildObservation(
            this.player,
            jadRegion,
            this.jadConfig,
            this.startingDoses
        );
        const termination = checkTermination(this.player, jadRegion, this.jadConfig);
        const reward = computeReward(
            observation,
            this.prevObservation,
            termination,
            this.episodeLength,
            this.envConfig.rewardFunc
        );
        const validActionMask = buildValidActionMask(jadRegion, this.jadConfig, observation);

        return {
            observation,
            reward,
            terminated: termination !== TerminationState.ONGOING,
            valid_action_mask: validActionMask,
        };
    }
}
