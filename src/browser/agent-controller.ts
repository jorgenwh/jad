import { Player } from 'osrs-sdk';
import { AgentWebSocket } from './agent-websocket';
import { AgentUI } from './agent-ui';
import {
    JadRegion,
    JadConfig,
    Observation,
    executeAction,
    countPotionDoses,
    buildObservation,
    buildValidActionMask,
    computeReward,
    TerminationState,
    getActionName,
    checkTermination,
} from '../core';

export class AgentController {
    private ws: AgentWebSocket;
    private ui: AgentUI;

    private player: Player;
    private jadRegion: JadRegion;
    private config: JadConfig;

    private startingDoses = { bastion: 0, saraBrew: 0, superRestore: 0 };
    private prevObservation: Observation | null = null;
    private cumulativeReward = 0;
    private episodeLength = 0;
    private terminationState = TerminationState.ONGOING;

    constructor(player: Player, jadRegion: JadRegion, config: JadConfig) {
        this.player = player;
        this.jadRegion = jadRegion;
        this.config = config;

        this.ws = new AgentWebSocket();
        this.ui = new AgentUI();

        this.captureStartingDoses();
        this.setupWebSocketHandlers();
    }

    private setupWebSocketHandlers(): void {
        this.ws.onConnectionChange = (connected: boolean) => {
            this.ui.showAgentInfo(connected);
        };

        this.ws.onAction = (action, value) => {
            console.log(`Received action: ${action} (${getActionName(action, this.config)})`);

            executeAction(action, this.player, this.jadRegion, this.config);

            const obs = buildObservation(
                this.player,
                this.jadRegion,
                this.config,
                this.startingDoses
            );
            const reward = computeReward(
                obs,
                this.prevObservation,
                TerminationState.ONGOING,
                this.episodeLength
            );
            this.cumulativeReward += reward;
            this.episodeLength++;
            this.prevObservation = obs;

            this.ui.updateAction(action, value, this.config);
            this.ui.updateObservation(obs);
            this.ui.updateReward(this.cumulativeReward, this.episodeLength);
        };
    }

    connect(): Promise<void> {
        return this.ws.connect().then(() => {
            this.resetEpisode();
            this.ws.sendReset();
        });
    }

    tick(): void {
        if (!this.ws.connected || this.terminationState !== TerminationState.ONGOING) {
            return;
        }

        const obs = buildObservation(
            this.player,
            this.jadRegion,
            this.config,
            this.startingDoses
        );

        const state = checkTermination(this.player, this.jadRegion, this.config);

        // Handle ongoing episode
        if (state === TerminationState.ONGOING) {
            const mask = buildValidActionMask(this.jadRegion, this.config, obs);
            this.ws.sendStep({
                observation: obs,
                reward: 0,
                terminated: false,
                valid_action_mask: mask,
            });
        } else { // Handle episode termination
            this.terminationState = state;
            const reward = computeReward(
                obs,
                this.prevObservation,
                state,
                this.episodeLength
            );
            this.cumulativeReward += reward

            this.ui.updateObservation(obs);
            this.ui.updateActionStatus(state === TerminationState.PLAYER_DIED ? 'DEAD' : 'WIN');
            this.ui.updateReward(this.cumulativeReward, this.episodeLength);

            console.log(`Episode ${state}: terminal_reward=${reward.toFixed(1)}, total=${this.cumulativeReward.toFixed(1)}`);

            const mask = buildValidActionMask(this.jadRegion, this.config, obs);
            this.ws.sendStep({
                observation: obs,
                reward: reward,
                terminated: true,
                valid_action_mask: mask,
            });
        }
    }

    private captureStartingDoses(): void {
        const doses = countPotionDoses(this.player);
        this.startingDoses = {
            bastion: doses.bastionDoses,
            saraBrew: doses.saraBrewDoses,
            superRestore: doses.superRestoreDoses,
        };
    }

    private resetEpisode(): void {
        this.prevObservation = null;
        this.cumulativeReward = 0;
        this.episodeLength = 0;
        this.terminationState = TerminationState.ONGOING;
        this.captureStartingDoses();
    }
}
